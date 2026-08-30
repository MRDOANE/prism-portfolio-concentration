#!/usr/bin/env python3
"""Run the prespecified E4-E7 portfolio decision-frontier experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import yaml
from scipy.special import ndtri
from scipy.stats import binom

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from prism_sim.advanced import (
    build_economic_basis,
    capacity_requirement_for_gain,
    cvar_lower,
    direct_funding_metrics,
    launch_summary,
    linear_components,
    mc_mean_se,
    paired_mean_interval,
    path_value_summary,
    peak_annual_development,
)
from prism_sim.config import load_config, scenario_hash
from prism_sim.outcomes import generate_outcomes
from prism_sim.policy import schedule_parallel, schedule_staged
from prism_sim.randomness import RandomBlock, generate_random_block


STRATEGIES = (
    ("Parallel PIAP", "same_asset", "piap_rhos"),
    ("MSOG", "same_indication", "msog_rhos"),
)


@dataclass
class ReferenceStore:
    by_loe: dict[float, dict[str, list[np.ndarray]]]
    per_seed_runs: int

    def array(self, loe: float, key: str, total_runs: int) -> np.ndarray:
        take = total_runs // len(self.by_loe[loe][key])
        if take > self.per_seed_runs:
            raise ValueError("Reference store does not contain enough runs")
        return np.concatenate([values[:take] for values in self.by_loe[loe][key]])


def load_plan(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def numeric_grid(spec: dict[str, float]) -> np.ndarray:
    start = float(spec["start"])
    stop = float(spec["stop"])
    step = float(spec["step"])
    count = int(round((stop - start) / step))
    return np.round(np.linspace(start, stop, count + 1), 10)


def slice_block(block: RandomBlock, n_runs: int) -> RandomBlock:
    return RandomBlock(
        seed=block.seed,
        block_index=block.block_index,
        residuals=block.residuals[:n_runs],
        cluster_factors=block.cluster_factors[:n_runs],
        shock_uniform=block.shock_uniform[:n_runs],
    )


def concat(parts: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(parts, axis=0)


def max_probability_half_width(probability: float, n_runs: int) -> float:
    return float(1.96 * math.sqrt(max(probability * (1.0 - probability), 0.0) / n_runs))


def run_reference(
    config: dict,
    blocks: list[RandomBlock],
    loe_years: list[float],
) -> tuple[ReferenceStore, pd.DataFrame]:
    probabilities = np.array(
        [
            config["development"]["phase2"]["pass_probability"],
            config["development"]["phase3"]["pass_probability"],
            config["development"]["regulatory"]["pass_probability"],
        ],
        dtype=float,
    )
    store = {
        loe: {
            "npv": [],
            "pvdev": [],
            "launch_time": [],
            "peak_rnd": [],
            **{f"funding_{capacity}": [] for capacity in config["enterprise"]["capacity_profiles"]},
        }
        for loe in loe_years
    }
    for block in blocks:
        outcomes = generate_outcomes(block, probabilities, rho=0.0)
        schedule = schedule_parallel(outcomes, config)
        basis = build_economic_basis(schedule, config)
        for loe in loe_years:
            components = linear_components(basis, config, loe, commercial_clustered=False)
            store[loe]["npv"].append(components.base_npv)
            store[loe]["pvdev"].append(components.base_pv_development)
            store[loe]["launch_time"].append(schedule.launch_time)
            store[loe]["peak_rnd"].append(peak_annual_development(basis.development_base, basis.times))
            for capacity in config["enterprise"]["capacity_profiles"]:
                funding = direct_funding_metrics(components.base_cash_flow, config, capacity)["funding_need"]
                store[loe][f"funding_{capacity}"].append(funding)
    reference = ReferenceStore(store, blocks[0].n_runs)
    rows = []
    for n_runs in (100000, 250000):
        for loe in loe_years:
            npv = reference.array(loe, "npv", n_runs)
            pvdev = reference.array(loe, "pvdev", n_runs)
            row = {
                "strategy": "Diversified",
                "rho": 0.0,
                "loe_year": loe,
                **path_value_summary(npv, pvdev),
                **launch_summary(reference.array(loe, "launch_time", n_runs), loe, 6),
                "mean_peak_annual_RnD_usd_m": float(np.mean(reference.array(loe, "peak_rnd", n_runs))),
            }
            for capacity, profile in config["enterprise"]["capacity_profiles"].items():
                funding = reference.array(loe, f"funding_{capacity}", n_runs)
                financing = float(profile["financing_capacity_usd_m"])
                row[f"{capacity}_capacity_breach_probability"] = float(np.mean(funding > financing + 1e-10))
                row[f"{capacity}_mean_funding_need_usd_m"] = float(np.mean(funding))
                row[f"{capacity}_P95_funding_need_usd_m"] = float(np.quantile(funding, 0.95))
            rows.append(row)
    return reference, pd.DataFrame(rows)


def _empty_accumulator(loe_years: list[float], cannibalization: list[float], capacities: list[str]) -> dict:
    result = {}
    for loe in loe_years:
        result[loe] = {
            "base_npv": [],
            "rnd_npv_gain": [],
            "commercial_npv_gain": [],
            "overlap_npv_loss": [],
            "base_pvdev": [],
            "rnd_pvdev_saving": [],
            "launch_time": [],
            "peak_rnd_base": [],
            "peak_rnd_gain": [],
            "required_rnd": {
                c: {capacity: [] for capacity in capacities} for c in cannibalization
            },
            "required_commercial": {capacity: [] for capacity in capacities},
        }
    return result


def simulate_e4_structure(
    config: dict,
    blocks: list[RandomBlock],
    total_runs: int,
    strategy: str,
    topology: str,
    rho: float,
    acceleration: float,
    loe_years: list[float],
    cannibalization: list[float],
    include_commercial_requirement: bool,
) -> dict:
    capacities = list(config["enterprise"]["capacity_profiles"])
    accum = _empty_accumulator(loe_years, cannibalization, capacities)
    take = total_runs // len(blocks)
    probabilities = np.array(
        [
            config["development"]["phase2"]["pass_probability"],
            config["development"]["phase3"]["pass_probability"],
            config["development"]["regulatory"]["pass_probability"],
        ]
    )
    for full_block in blocks:
        block = slice_block(full_block, take)
        outcomes = generate_outcomes(block, probabilities, rho=rho)
        schedule = schedule_parallel(outcomes, config, acceleration_years=acceleration)
        basis = build_economic_basis(schedule, config)
        for loe in loe_years:
            components = linear_components(basis, config, loe, commercial_clustered=True)
            target = accum[loe]
            target["base_npv"].append(components.base_npv)
            target["rnd_npv_gain"].append(components.rnd_npv_gain)
            target["commercial_npv_gain"].append(components.commercial_npv_gain)
            target["overlap_npv_loss"].append(components.cannibalization_npv_loss)
            target["base_pvdev"].append(components.base_pv_development)
            target["rnd_pvdev_saving"].append(components.rnd_pv_development_saving)
            target["launch_time"].append(schedule.launch_time)
            target["peak_rnd_base"].append(
                peak_annual_development(components.development_base, components.times)
            )
            target["peak_rnd_gain"].append(
                peak_annual_development(components.follower_development, components.times)
            )
            for c in cannibalization:
                adjusted = components.base_cash_flow - c * components.cannibalization_cash_loss
                for capacity in capacities:
                    required = capacity_requirement_for_gain(
                        adjusted, components.rnd_cash_gain, config, capacity
                    )
                    target["required_rnd"][c][capacity].append(required)
            if include_commercial_requirement:
                for capacity in capacities:
                    required = capacity_requirement_for_gain(
                        components.base_cash_flow,
                        components.commercial_cash_gain,
                        config,
                        capacity,
                    )
                    target["required_commercial"][capacity].append(required)
    for target in accum.values():
        for key in (
            "base_npv",
            "rnd_npv_gain",
            "commercial_npv_gain",
            "overlap_npv_loss",
            "base_pvdev",
            "rnd_pvdev_saving",
            "launch_time",
            "peak_rnd_base",
            "peak_rnd_gain",
        ):
            target[key] = concat(target[key])
        for c in cannibalization:
            for capacity in capacities:
                target["required_rnd"][c][capacity] = concat(target["required_rnd"][c][capacity])
        if include_commercial_requirement:
            for capacity in capacities:
                target["required_commercial"][capacity] = concat(target["required_commercial"][capacity])
    return accum


def e4_rows_for_structure(
    config: dict,
    plan: dict,
    reference: ReferenceStore,
    accum: dict,
    strategy: str,
    topology: str,
    rho: float,
    acceleration: float,
    total_runs: int,
    include_commercial: bool,
) -> tuple[list[dict], list[dict], list[dict]]:
    cost_grid = numeric_grid(plan["e4"]["rnd_synergy_search"])
    commercial_grid = numeric_grid(plan["e4"]["commercial_synergy_search"])
    coarse_cost = [float(x) for x in plan["e4"]["cost_synergy_for_regret"]]
    cannibalization = [float(x) for x in plan["e4"]["cannibalization_for_regret"]]
    primary_rows: list[dict] = []
    interaction_rows: list[dict] = []
    commercial_rows: list[dict] = []
    for loe, data in accum.items():
        reference_npv = reference.array(loe, "npv", total_runs)
        launches = launch_summary(data["launch_time"], loe, 6)
        for synergy in cost_grid:
            npv = data["base_npv"] + synergy * data["rnd_npv_gain"]
            pvdev = data["base_pvdev"] - synergy * data["rnd_pvdev_saving"]
            row = {
                "experiment": "E4",
                "strategy": strategy,
                "topology": topology,
                "rho": rho,
                "loe_year": loe,
                "acceleration_years": acceleration,
                "rnd_cost_synergy": synergy,
                "commercial_cost_synergy": 0.0,
                "cannibalization": 0.0,
                **path_value_summary(npv, pvdev),
                **paired_mean_interval(npv, reference_npv),
                **launches,
            }
            peak = data["peak_rnd_base"] - synergy * data["peak_rnd_gain"]
            row["mean_peak_annual_RnD_linear_proxy_usd_m"] = float(np.mean(peak))
            for capacity in config["enterprise"]["capacity_profiles"]:
                probability = float(np.mean(data["required_rnd"][0.0][capacity] > synergy + 1e-12))
                row[f"{capacity}_capacity_breach_probability"] = probability
                row[f"{capacity}_breach_half_width"] = max_probability_half_width(probability, total_runs)
            primary_rows.append(row)

        for c in cannibalization:
            for synergy in coarse_cost:
                npv = (
                    data["base_npv"]
                    + synergy * data["rnd_npv_gain"]
                    - c * data["overlap_npv_loss"]
                )
                pvdev = data["base_pvdev"] - synergy * data["rnd_pvdev_saving"]
                row = {
                    "experiment": "E4_interaction",
                    "strategy": strategy,
                    "topology": topology,
                    "rho": rho,
                    "loe_year": loe,
                    "acceleration_years": acceleration,
                    "rnd_cost_synergy": synergy,
                    "commercial_cost_synergy": 0.0,
                    "cannibalization": c,
                    **path_value_summary(npv, pvdev),
                    **paired_mean_interval(npv, reference_npv),
                    **launches,
                }
                for capacity in config["enterprise"]["capacity_profiles"]:
                    probability = float(np.mean(data["required_rnd"][c][capacity] > synergy + 1e-12))
                    row[f"{capacity}_capacity_breach_probability"] = probability
                interaction_rows.append(row)

        if include_commercial and abs(float(loe) - 15.0) < 1e-12:
            for synergy in commercial_grid:
                npv = data["base_npv"] + synergy * data["commercial_npv_gain"]
                row = {
                    "experiment": "E4_commercial_synergy",
                    "strategy": strategy,
                    "topology": topology,
                    "rho": rho,
                    "loe_year": loe,
                    "acceleration_years": acceleration,
                    "rnd_cost_synergy": 0.0,
                    "commercial_cost_synergy": synergy,
                    "cannibalization": 0.0,
                    **path_value_summary(npv, data["base_pvdev"]),
                    **paired_mean_interval(npv, reference_npv),
                    **launches,
                }
                for capacity in config["enterprise"]["capacity_profiles"]:
                    probability = float(
                        np.mean(data["required_commercial"][capacity] > synergy + 1e-12)
                    )
                    row[f"{capacity}_capacity_breach_probability"] = probability
                commercial_rows.append(row)
    return primary_rows, interaction_rows, commercial_rows


def build_frontiers(
    cells: pd.DataFrame,
    config: dict,
    synergy_column: str,
) -> pd.DataFrame:
    grouping = [
        "strategy",
        "topology",
        "rho",
        "loe_year",
        "acceleration_years",
        "cannibalization",
    ]
    rows = []
    for keys, group in cells.groupby(grouping, dropna=False):
        group = group.sort_values(synergy_column)
        for capacity in config["enterprise"]["capacity_profiles"]:
            for profile_name, profile in config["risk_policy_profiles"].items():
                feasible = (
                    (group[f"{capacity}_capacity_breach_probability"] <= float(profile["maximum_capacity_breach_probability"]) + 1e-12)
                    & (group["P5_NPV_usd_m"] >= float(profile["minimum_p5_npv_usd_m"]) - 1e-12)
                    & (group["delta_eNPV_usd_m"] >= -1e-12)
                )
                candidates = group.loc[feasible]
                selected = candidates.iloc[0] if not candidates.empty else None
                row = dict(zip(grouping, keys))
                row.update(
                    {
                        "capacity_profile": capacity,
                        "risk_policy_profile": profile_name,
                        "minimum_synergy": float(selected[synergy_column]) if selected is not None else np.nan,
                        "solution_exists": bool(selected is not None),
                        "binding_P5_NPV_usd_m": float(selected.P5_NPV_usd_m) if selected is not None else np.nan,
                        "binding_capacity_breach_probability": float(selected[f"{capacity}_capacity_breach_probability"]) if selected is not None else np.nan,
                        "delta_eNPV_at_frontier_usd_m": float(selected.delta_eNPV_usd_m) if selected is not None else np.nan,
                        "n_runs": int(group.n_runs.max()),
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def run_e4(
    config: dict,
    plan: dict,
    blocks: list[RandomBlock],
    reference: ReferenceStore,
    output: Path,
) -> dict[str, pd.DataFrame]:
    primary_rows: list[dict] = []
    interaction_rows: list[dict] = []
    commercial_rows: list[dict] = []
    loe_years = [float(x) for x in plan["e4"]["loe_years"]]
    cannibalization = [float(x) for x in plan["e4"]["cannibalization_for_regret"]]
    primary_runs = int(plan["precision"]["primary_runs"])
    boundary_runs = int(plan["precision"]["boundary_runs"])
    total_structures = sum(len(config["pilot"]["e1"][grid]) for _, _, grid in STRATEGIES) * len(plan["e4"]["acceleration_years"])
    completed = 0
    for strategy, topology, grid_name in STRATEGIES:
        boundary_rhos = plan["e4"]["boundary_rhos"]["parallel_piap" if strategy == "Parallel PIAP" else "msog"]
        central = 0.415 if strategy == "Parallel PIAP" else 0.25
        for rho in [float(x) for x in config["pilot"]["e1"][grid_name]]:
            for acceleration in [float(x) for x in plan["e4"]["acceleration_years"]]:
                n_runs = boundary_runs if any(abs(rho - float(x)) < 1e-12 for x in boundary_rhos) else primary_runs
                include_commercial = abs(rho - central) < 1e-12
                accum = simulate_e4_structure(
                    config,
                    blocks,
                    n_runs,
                    strategy,
                    topology,
                    rho,
                    acceleration,
                    loe_years,
                    cannibalization,
                    include_commercial,
                )
                p, i, c = e4_rows_for_structure(
                    config,
                    plan,
                    reference,
                    accum,
                    strategy,
                    topology,
                    rho,
                    acceleration,
                    n_runs,
                    include_commercial,
                )
                primary_rows.extend(p)
                interaction_rows.extend(i)
                commercial_rows.extend(c)
                completed += 1
                print(f"E4 structures {completed}/{total_structures}", flush=True)
    primary = pd.DataFrame(primary_rows)
    interaction = pd.DataFrame(interaction_rows)
    commercial = pd.DataFrame(commercial_rows)
    primary.to_csv(output / "e4_cell_metrics.csv", index=False)
    interaction.to_csv(output / "e4_interaction_metrics.csv", index=False)
    commercial.to_csv(output / "e4_commercial_synergy_metrics.csv", index=False)
    frontier = build_frontiers(primary, config, "rnd_cost_synergy")
    frontier.to_csv(output / "e4_rnd_acceleration_frontier.csv", index=False)
    interaction_frontier = build_frontiers(interaction, config, "rnd_cost_synergy")
    interaction_frontier.to_csv(output / "e4_cannibalization_shift_frontier.csv", index=False)
    commercial_frontier = build_frontiers(commercial, config, "commercial_cost_synergy")
    commercial_frontier.to_csv(output / "e4_commercial_synergy_frontier.csv", index=False)
    return {
        "primary": primary,
        "interaction": interaction,
        "commercial": commercial,
        "frontier": frontier,
        "interaction_frontier": interaction_frontier,
        "commercial_frontier": commercial_frontier,
    }


def summarize_e5_cell(
    config: dict,
    npv: np.ndarray,
    pvdev: np.ndarray,
    launch_time: np.ndarray,
    loe: float,
    peak_rnd: np.ndarray,
    funding: dict[str, np.ndarray],
) -> dict[str, Any]:
    row = {
        **path_value_summary(npv, pvdev),
        **launch_summary(launch_time, loe, 6),
        "mean_peak_annual_RnD_usd_m": float(np.mean(peak_rnd)),
    }
    finite = np.isfinite(launch_time)
    first = np.min(np.where(finite, launch_time, np.inf), axis=1)
    first[first == np.inf] = np.nan
    row["mean_first_launch_year"] = float(np.nanmean(first))
    row["probability_two_launches_by_year_9"] = float(np.mean(((launch_time <= 9.0) & finite).sum(axis=1) >= 2))
    for capacity, values in funding.items():
        financing = float(config["enterprise"]["capacity_profiles"][capacity]["financing_capacity_usd_m"])
        row[f"{capacity}_capacity_breach_probability"] = float(np.mean(values > financing + 1e-10))
        row[f"{capacity}_mean_funding_need_usd_m"] = float(np.mean(values))
        row[f"{capacity}_P95_funding_need_usd_m"] = float(np.quantile(values, 0.95))
    return row


def run_e5(config: dict, plan: dict, blocks: list[RandomBlock], output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_runs = int(plan["precision"]["primary_runs"])
    take = n_runs // len(blocks)
    probabilities = np.array(
        [
            config["development"]["phase2"]["pass_probability"],
            config["development"]["phase3"]["pass_probability"],
            config["development"]["regulatory"]["pass_probability"],
        ]
    )
    rows: list[dict] = []
    contrasts: list[dict] = []
    for shock in [float(x) for x in plan["e5"]["common_shock_probabilities"]]:
        for acceleration in [float(x) for x in plan["e5"]["acceleration_years"]]:
            cell_store: dict[tuple[str, float], dict[str, list[np.ndarray]]] = {}
            for policy in ("parallel", "staged"):
                for loe in [float(x) for x in plan["e5"]["loe_years"]]:
                    cell_store[(policy, loe)] = {
                        "npv": [], "pvdev": [], "launch": [], "peak": [],
                        **{f"funding_{capacity}": [] for capacity in config["enterprise"]["capacity_profiles"]},
                    }
            for full_block in blocks:
                block = slice_block(full_block, take)
                outcomes = generate_outcomes(
                    block,
                    probabilities,
                    rho=float(plan["e5"]["same_asset_rho"]),
                    common_shock_probability=shock,
                    common_shock_mode="marginal_parity",
                )
                for policy, scheduler in (("parallel", schedule_parallel), ("staged", schedule_staged)):
                    schedule = scheduler(outcomes, config, acceleration_years=acceleration)
                    basis = build_economic_basis(schedule, config)
                    peak = peak_annual_development(basis.development_base, basis.times)
                    for loe in [float(x) for x in plan["e5"]["loe_years"]]:
                        components = linear_components(basis, config, loe, commercial_clustered=False)
                        target = cell_store[(policy, loe)]
                        target["npv"].append(components.base_npv)
                        target["pvdev"].append(components.base_pv_development)
                        target["launch"].append(schedule.launch_time)
                        target["peak"].append(peak)
                        for capacity in config["enterprise"]["capacity_profiles"]:
                            target[f"funding_{capacity}"].append(
                                direct_funding_metrics(components.base_cash_flow, config, capacity)["funding_need"]
                            )
            finalized: dict[tuple[str, float], dict[str, np.ndarray]] = {}
            for key, values in cell_store.items():
                finalized[key] = {name: concat(parts) for name, parts in values.items()}
                policy, loe = key
                funding = {
                    capacity: finalized[key][f"funding_{capacity}"]
                    for capacity in config["enterprise"]["capacity_profiles"]
                }
                rows.append(
                    {
                        "experiment": "E5",
                        "policy": policy,
                        "rho": float(plan["e5"]["same_asset_rho"]),
                        "common_shock_probability": shock,
                        "acceleration_years": acceleration,
                        "loe_year": loe,
                        **summarize_e5_cell(
                            config,
                            finalized[key]["npv"],
                            finalized[key]["pvdev"],
                            finalized[key]["launch"],
                            loe,
                            finalized[key]["peak"],
                            funding,
                        ),
                    }
                )
            for loe in [float(x) for x in plan["e5"]["loe_years"]]:
                parallel = finalized[("parallel", loe)]
                staged = finalized[("staged", loe)]
                p5_parallel = float(np.quantile(parallel["npv"], 0.05))
                p5_staged = float(np.quantile(staged["npv"], 0.05))
                contrast = {
                    "experiment": "E5",
                    "comparison": "staged minus parallel",
                    "rho": float(plan["e5"]["same_asset_rho"]),
                    "common_shock_probability": shock,
                    "acceleration_years": acceleration,
                    "loe_year": loe,
                    **paired_mean_interval(staged["npv"], parallel["npv"]),
                    "delta_eDevCost_usd_m": float(np.mean(staged["pvdev"] - parallel["pvdev"])),
                    "delta_mean_peak_annual_RnD_usd_m": float(np.mean(staged["peak"] - parallel["peak"])),
                    "delta_P5_NPV_usd_m": p5_staged - p5_parallel,
                    "delta_CVaR5_NPV_usd_m": cvar_lower(staged["npv"]) - cvar_lower(parallel["npv"]),
                    "delta_mean_launches": float(np.mean(np.isfinite(staged["launch"]).sum(axis=1) - np.isfinite(parallel["launch"]).sum(axis=1))),
                    "delta_mean_launches_before_loe": float(
                        np.mean(((staged["launch"] < loe) & np.isfinite(staged["launch"])).sum(axis=1))
                        - np.mean(((parallel["launch"] < loe) & np.isfinite(parallel["launch"])).sum(axis=1))
                    ),
                }
                for capacity in config["enterprise"]["capacity_profiles"]:
                    financing = float(config["enterprise"]["capacity_profiles"][capacity]["financing_capacity_usd_m"])
                    p_breach = parallel[f"funding_{capacity}"] > financing + 1e-10
                    s_breach = staged[f"funding_{capacity}"] > financing + 1e-10
                    contrast[f"delta_{capacity}_capacity_breach_probability"] = float(np.mean(s_breach) - np.mean(p_breach))
                    contrast[f"delta_{capacity}_mean_funding_need_usd_m"] = float(
                        np.mean(staged[f"funding_{capacity}"] - parallel[f"funding_{capacity}"])
                    )
                contrasts.append(contrast)
            print(f"E5 shock={shock:.2f}, acceleration={acceleration:.1f} complete", flush=True)
    summary = pd.DataFrame(rows)
    contrast_frame = pd.DataFrame(contrasts)
    summary.to_csv(output / "e5_policy_metrics.csv", index=False)
    contrast_frame.to_csv(output / "e5_staged_vs_parallel.csv", index=False)
    return summary, contrast_frame


def run_e6(config: dict, plan: dict, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = int(plan["e6"]["runs"])
    max_n = int(plan["e6"]["maximum_nominal_breadth"])
    targets = [int(x) for x in plan["e6"]["launch_targets"]]
    service_targets = [float(x) for x in plan["e6"]["service_targets"]]
    seeds = [int(x) for x in config["simulation"]["seed_blocks"]]
    per_seed = runs // len(seeds)
    probabilities = np.array(
        [
            config["development"]["phase2"]["pass_probability"],
            config["development"]["phase3"]["pass_probability"],
            config["development"]["regulatory"]["pass_probability"],
        ]
    )
    thresholds = ndtri(probabilities)
    rows: list[dict] = []
    for geometry in plan["e6"]["geometries"]:
        name = str(geometry["name"])
        cluster_size = int(geometry["cluster_size"])
        rho = float(geometry["rho"])
        success_sums = np.zeros((len(targets), max_n), dtype=np.int64)
        count_sums = np.zeros(max_n, dtype=float)
        count_square_sums = np.zeros(max_n, dtype=float)
        program_success_sums = np.zeros(max_n, dtype=float)
        n_clusters = int(math.ceil(max_n / cluster_size))
        cluster_ids = np.arange(max_n) // cluster_size
        for seed in seeds:
            rng = np.random.default_rng(seed)
            residuals = rng.standard_normal((per_seed, max_n, 3))
            factors = rng.standard_normal((per_seed, n_clusters, 3))
            latent = math.sqrt(rho) * factors[:, cluster_ids, :] + math.sqrt(1.0 - rho) * residuals
            launches = np.all(latent <= thresholds[None, None, :], axis=2)
            cumulative = np.cumsum(launches, axis=1)
            count_sums += cumulative.sum(axis=0)
            count_square_sums += np.square(cumulative).sum(axis=0)
            program_success_sums += launches.sum(axis=0)
            for t_index, target in enumerate(targets):
                success_sums[t_index] += (cumulative >= target).sum(axis=0)
        cumulative_program_success = np.cumsum(program_success_sums)
        for n in range(1, max_n + 1):
            mean_count = count_sums[n - 1] / runs
            variance_count = (count_square_sums[n - 1] - runs * mean_count**2) / (runs - 1)
            marginal_ps = program_success_sums[:n] / runs
            mean_marginal_variance = float(np.mean(marginal_ps * (1.0 - marginal_ps) * runs / (runs - 1)))
            breadth = n**2 * mean_marginal_variance / variance_count if variance_count > 0 else np.nan
            for t_index, target in enumerate(targets):
                probability = float(success_sums[t_index, n - 1] / runs)
                rows.append(
                    {
                        "experiment": "E6",
                        "geometry": name,
                        "cluster_size": cluster_size,
                        "rho": rho,
                        "nominal_breadth": n,
                        "launch_target_per_3y": target,
                        "service_probability": probability,
                        "service_half_width": max_probability_half_width(probability, runs),
                        "mean_launches_per_3y": mean_count,
                        "variance_launches_per_3y": variance_count,
                        "effective_breadth": breadth,
                        "n_runs": runs,
                    }
                )
        print(f"E6 geometry {name} complete", flush=True)
    cells = pd.DataFrame(rows)
    frontiers = []
    for (geometry, cluster_size, rho, target), group in cells.groupby(
        ["geometry", "cluster_size", "rho", "launch_target_per_3y"]
    ):
        group = group.sort_values("nominal_breadth")
        for service_target in service_targets:
            eligible = group[group.service_probability >= service_target]
            selected = eligible.iloc[0] if not eligible.empty else None
            frontiers.append(
                {
                    "geometry": geometry,
                    "cluster_size": cluster_size,
                    "rho": rho,
                    "launch_target_per_3y": target,
                    "service_target": service_target,
                    "minimum_nominal_breadth": int(selected.nominal_breadth) if selected is not None else np.nan,
                    "effective_breadth_at_frontier": float(selected.effective_breadth) if selected is not None else np.nan,
                    "achieved_service_probability": float(selected.service_probability) if selected is not None else np.nan,
                    "service_half_width": float(selected.service_half_width) if selected is not None else np.nan,
                    "solution_exists": bool(selected is not None),
                    "n_runs": runs,
                }
            )
    frontier_frame = pd.DataFrame(frontiers)
    cells.to_csv(output / "e6_launch_service_cells.csv", index=False)
    frontier_frame.to_csv(output / "e6_launch_service_frontier.csv", index=False)
    independent = cells[cells.geometry == "Diversified"].copy()
    cumulative_ptrs = float(np.prod(probabilities))
    independent["analytic_probability"] = independent.apply(
        lambda row: float(binom.sf(int(row.launch_target_per_3y) - 1, int(row.nominal_breadth), cumulative_ptrs)),
        axis=1,
    )
    independent[["nominal_breadth", "launch_target_per_3y", "service_probability", "analytic_probability"]].to_csv(
        output / "e6_independent_binomial_check.csv", index=False
    )
    return cells, frontier_frame


def run_e7(
    config: dict,
    interaction: pd.DataFrame,
    reference_summary: pd.DataFrame,
    output: Path,
) -> pd.DataFrame:
    rows = []
    decision_tolerance_usd_m = 25.0
    assumed = interaction[(interaction.rho == 0.0) & (interaction.cannibalization == 0.0)]
    true_rows = interaction[interaction.rho > 0.0]
    key_cols = ["strategy", "loe_year", "acceleration_years", "rnd_cost_synergy"]
    assumed_lookup = {tuple(row[col] for col in key_cols): row for _, row in assumed.iterrows()}
    for _, true in true_rows.iterrows():
        key = tuple(true[col] for col in key_cols)
        assumed_row = assumed_lookup[key]
        ref_candidates = reference_summary[
            (reference_summary.loe_year == true.loe_year) & (reference_summary.n_runs == true.n_runs)
        ]
        reference = ref_candidates.iloc[0]
        ref_assumed = reference_summary[
            (reference_summary.loe_year == true.loe_year) & (reference_summary.n_runs == assumed_row.n_runs)
        ].iloc[0]
        assumed_value_choice = (
            "concentrated"
            if assumed_row.eNPV_usd_m - ref_assumed.eNPV_usd_m > decision_tolerance_usd_m
            else "diversified"
        )
        true_value_oracle = "concentrated" if true.eNPV_usd_m >= reference.eNPV_usd_m else "diversified"
        selected_true_enpv = true.eNPV_usd_m if assumed_value_choice == "concentrated" else reference.eNPV_usd_m
        oracle_true_enpv = true.eNPV_usd_m if true_value_oracle == "concentrated" else reference.eNPV_usd_m
        selected_true_p5 = true.P5_NPV_usd_m if assumed_value_choice == "concentrated" else reference.P5_NPV_usd_m
        oracle_true_p5 = true.P5_NPV_usd_m if true_value_oracle == "concentrated" else reference.P5_NPV_usd_m
        for capacity in config["enterprise"]["capacity_profiles"]:
            for profile_name, profile in config["risk_policy_profiles"].items():
                assumed_conc_feasible = (
                    assumed_row[f"{capacity}_capacity_breach_probability"] <= float(profile["maximum_capacity_breach_probability"])
                    and assumed_row.P5_NPV_usd_m >= float(profile["minimum_p5_npv_usd_m"])
                )
                ref_assumed_feasible = (
                    ref_assumed[f"{capacity}_capacity_breach_probability"] <= float(profile["maximum_capacity_breach_probability"])
                    and ref_assumed.P5_NPV_usd_m >= float(profile["minimum_p5_npv_usd_m"])
                )
                true_conc_feasible = (
                    true[f"{capacity}_capacity_breach_probability"] <= float(profile["maximum_capacity_breach_probability"])
                    and true.P5_NPV_usd_m >= float(profile["minimum_p5_npv_usd_m"])
                )
                ref_true_feasible = (
                    reference[f"{capacity}_capacity_breach_probability"] <= float(profile["maximum_capacity_breach_probability"])
                    and reference.P5_NPV_usd_m >= float(profile["minimum_p5_npv_usd_m"])
                )
                if assumed_conc_feasible and (
                    not ref_assumed_feasible
                    or assumed_row.eNPV_usd_m - ref_assumed.eNPV_usd_m > decision_tolerance_usd_m
                ):
                    governance_selected = "concentrated"
                elif ref_assumed_feasible:
                    governance_selected = "diversified"
                else:
                    governance_selected = "no feasible action"
                if true_conc_feasible and (
                    not ref_true_feasible
                    or true.eNPV_usd_m - reference.eNPV_usd_m > decision_tolerance_usd_m
                ):
                    governance_oracle = "concentrated"
                elif ref_true_feasible:
                    governance_oracle = "diversified"
                else:
                    governance_oracle = "no feasible action"
                selected_breach = (
                    true[f"{capacity}_capacity_breach_probability"]
                    if governance_selected == "concentrated"
                    else reference[f"{capacity}_capacity_breach_probability"]
                    if governance_selected == "diversified"
                    else np.nan
                )
                oracle_breach = (
                    true[f"{capacity}_capacity_breach_probability"]
                    if governance_oracle == "concentrated"
                    else reference[f"{capacity}_capacity_breach_probability"]
                    if governance_oracle == "diversified"
                    else np.nan
                )
                value_selected_breach = (
                    true[f"{capacity}_capacity_breach_probability"]
                    if assumed_value_choice == "concentrated"
                    else reference[f"{capacity}_capacity_breach_probability"]
                )
                rows.append(
                    {
                        "experiment": "E7",
                        "strategy": true.strategy,
                        "rho_true": true.rho,
                        "cannibalization_true": true.cannibalization,
                        "loe_year": true.loe_year,
                        "acceleration_years": true.acceleration_years,
                        "rnd_cost_synergy": true.rnd_cost_synergy,
                        "capacity_profile": capacity,
                        "risk_policy_profile": profile_name,
                        "assumed_value_choice": assumed_value_choice,
                        "true_value_oracle": true_value_oracle,
                        "eNPV_regret_usd_m": max(float(oracle_true_enpv - selected_true_enpv), 0.0),
                        "P5_regret_usd_m": max(float(oracle_true_p5 - selected_true_p5), 0.0),
                        "governance_choice_under_independence_additivity": governance_selected,
                        "governance_oracle_under_truth": governance_oracle,
                        "wrong_governance_choice": bool(governance_selected != governance_oracle),
                        "selected_true_capacity_breach_probability": selected_breach,
                        "oracle_capacity_breach_probability": oracle_breach,
                        "incremental_capacity_breach_probability": float(selected_breach - oracle_breach) if np.isfinite(selected_breach) and np.isfinite(oracle_breach) else np.nan,
                        "value_selected_true_capacity_breach_probability": value_selected_breach,
                        "value_selected_incremental_breach_vs_diversified": float(
                            value_selected_breach - reference[f"{capacity}_capacity_breach_probability"]
                        ),
                        "value_selected_true_P5_NPV_usd_m": selected_true_p5,
                        "value_selected_P5_shortfall_vs_diversified_usd_m": max(
                            float(reference.P5_NPV_usd_m - selected_true_p5), 0.0
                        ),
                        "concentrated_true_P5_NPV_usd_m": true.P5_NPV_usd_m,
                        "diversified_P5_NPV_usd_m": reference.P5_NPV_usd_m,
                        "concentrated_true_eNPV_usd_m": true.eNPV_usd_m,
                        "diversified_eNPV_usd_m": reference.eNPV_usd_m,
                        "n_runs": int(true.n_runs),
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(output / "e7_assumption_regret.csv", index=False)
    return result


def validation_report(
    config: dict,
    plan: dict,
    e4: dict[str, pd.DataFrame],
    e5_summary: pd.DataFrame,
    e5_contrasts: pd.DataFrame,
    e6_cells: pd.DataFrame,
    e6_frontier: pd.DataFrame,
    e7: pd.DataFrame,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    primary = e4["primary"]
    boundary = primary[primary.n_runs == int(plan["precision"]["boundary_runs"])]
    checks["V4_synergy_monotonicity"] = {
        "pass": bool(
            all(
                np.all(np.diff(group.sort_values("rnd_cost_synergy").eNPV_usd_m) >= -1e-8)
                and np.all(np.diff(group.sort_values("rnd_cost_synergy").ePI) >= -1e-8)
                for _, group in primary.groupby(["strategy", "rho", "loe_year", "acceleration_years"])
            )
        )
    }
    checks["V6_boundary_precision"] = {
        "pass": bool(
            boundary[[f"{c}_breach_half_width" for c in config["enterprise"]["capacity_profiles"]]].max().max()
            <= float(plan["precision"]["probability_half_width_target"]) + 1e-12
            and (1.96 * boundary.paired_se_delta_eNPV_usd_m.max())
            <= float(plan["precision"]["eNPV_difference_half_width_usd_m"]) + 1e-12
        ),
        "maximum_boundary_probability_half_width": float(
            boundary[[f"{c}_breach_half_width" for c in config["enterprise"]["capacity_profiles"]]].max().max()
        ),
        "maximum_boundary_eNPV_half_width_usd_m": float(1.96 * boundary.paired_se_delta_eNPV_usd_m.max()),
    }
    staged_cost = e5_contrasts.delta_eDevCost_usd_m
    checks["E5_staging_cost_avoidance"] = {
        "pass": bool(np.all(staged_cost <= 1e-8)),
        "maximum_staged_minus_parallel_eDevCost_usd_m": float(staged_cost.max()),
    }
    service_monotone = True
    for _, group in e6_cells.groupby(["geometry", "launch_target_per_3y"]):
        service_monotone &= bool(np.all(np.diff(group.sort_values("nominal_breadth").service_probability) >= -1e-12))
    checks["E6_service_monotonicity"] = {"pass": service_monotone}
    checks["E6_boundary_precision"] = {
        "pass": bool(e6_frontier.service_half_width.dropna().max() <= float(plan["precision"]["probability_half_width_target"]) + 1e-12),
        "maximum_frontier_half_width": float(e6_frontier.service_half_width.dropna().max()),
    }
    checks["E7_nonnegative_value_regret"] = {
        "pass": bool(np.all(e7.eNPV_regret_usd_m >= -1e-12))
    }
    return {"pass": bool(all(item["pass"] for item in checks.values())), "checks": checks}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "benchmark_v0.2.yaml"))
    parser.add_argument("--plan", default=str(PACKAGE_ROOT / "config" / "e4_e7_plan_v0.2.yaml"))
    parser.add_argument("--output", default=str(PACKAGE_ROOT / "outputs_e4_e7"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    plan = load_plan(Path(args.plan))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    boundary_runs = int(plan["precision"]["boundary_runs"])
    per_seed = boundary_runs // len(config["simulation"]["seed_blocks"])
    blocks = [
        generate_random_block(int(seed), index, per_seed, int(config["simulation"]["n_programs"]))
        for index, seed in enumerate(config["simulation"]["seed_blocks"])
    ]
    loe_years = [float(x) for x in plan["e4"]["loe_years"]]
    reference, reference_summary = run_reference(config, blocks, loe_years)
    reference_summary.to_csv(output / "e4_diversified_reference.csv", index=False)
    e4 = run_e4(config, plan, blocks, reference, output)
    e5_summary, e5_contrasts = run_e5(config, plan, blocks, output)
    e6_cells, e6_frontier = run_e6(config, plan, output)
    e7 = run_e7(config, e4["interaction"], reference_summary, output)
    validation = validation_report(
        config, plan, e4, e5_summary, e5_contrasts, e6_cells, e6_frontier, e7
    )
    (output / "validation_report.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8"
    )
    elapsed = time.perf_counter() - start
    manifest = {
        "analysis_version": plan["analysis"]["version"],
        "base_configuration_hash": scenario_hash(config),
        "plan_hash": scenario_hash(plan),
        "primary_runs": int(plan["precision"]["primary_runs"]),
        "boundary_runs": boundary_runs,
        "e6_runs": int(plan["e6"]["runs"]),
        "seed_blocks": config["simulation"]["seed_blocks"],
        "runtime_seconds": elapsed,
        "validation_pass": validation["pass"],
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "PyYAML": yaml.__version__,
        },
        "files": sorted(path.name for path in output.iterdir() if path.is_file()),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"E4-E7 complete: validation_pass={validation['pass']}, runtime={elapsed:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
