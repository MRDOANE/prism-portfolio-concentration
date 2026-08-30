#!/usr/bin/env python3
"""Run E8 matched-heterogeneity robustness and boundary confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import yaml
from scipy.special import ndtri

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from prism_sim.advanced import (
    capacity_requirement_for_gain,
    direct_funding_metrics,
    linear_components,
    mc_mean_se,
)
from prism_sim.config import load_config, scenario_hash
from prism_sim.heterogeneity import (
    ProgramParameters,
    build_heterogeneous_basis,
    enumerate_program_states,
    generate_heterogeneous_outcomes,
    generate_lhs_program_parameters,
    homogeneous_program_parameters,
    joint_state_probabilities,
    schedule_parallel_heterogeneous,
    schedule_staged_heterogeneous,
    states_to_outcomes,
    triangular_ppf,
    weighted_launch_summary,
    weighted_quantile,
    weighted_value_summary,
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def numeric_grid(spec: dict[str, float]) -> np.ndarray:
    start = float(spec["start"])
    stop = float(spec["stop"])
    step = float(spec["step"])
    count = int(round((stop - start) / step))
    return np.round(np.linspace(start, stop, count + 1), 10)


def probability_half_width(probability: float, n: int) -> float:
    return float(1.96 * math.sqrt(max(probability * (1.0 - probability), 0.0) / n))


def weighted_probability(weights: np.ndarray, condition: np.ndarray) -> float:
    return float(np.asarray(weights, dtype=float)[np.asarray(condition, dtype=bool)].sum())


def frontier_for_weights(
    weights: np.ndarray,
    reference_weights: np.ndarray,
    components,
    reference_npv: np.ndarray,
    required_synergy: np.ndarray,
    synergy_grid: np.ndarray,
    cannibalization: float,
    p5_floor: float,
    breach_limit: float,
) -> dict[str, float | bool]:
    reference_mean = float(np.dot(reference_weights, reference_npv))
    last = None
    for synergy in synergy_grid:
        npv = (
            components.base_npv
            + float(synergy) * components.rnd_npv_gain
            - float(cannibalization) * components.cannibalization_npv_loss
        )
        delta = float(np.dot(weights, npv) - reference_mean)
        p5 = weighted_quantile(npv, weights, 0.05)
        breach = weighted_probability(weights, required_synergy > float(synergy) + 1e-12)
        last = {
            "minimum_synergy": float(synergy),
            "delta_eNPV_usd_m": delta,
            "P5_NPV_usd_m": p5,
            "capacity_breach_probability": breach,
            "eNPV_margin_usd_m": delta,
            "P5_margin_usd_m": p5 - float(p5_floor),
            "breach_margin": float(breach_limit) - breach,
        }
        if delta >= -1e-10 and p5 >= float(p5_floor) - 1e-10 and breach <= float(breach_limit) + 1e-12:
            last["solution_exists"] = True
            last["minimum_normalized_margin"] = float(
                min(
                    abs(delta) / 50.0,
                    abs(p5 - float(p5_floor)) / 100.0,
                    abs(float(breach_limit) - breach) / 0.025,
                )
            )
            return last
    assert last is not None
    last["solution_exists"] = False
    last["minimum_synergy"] = float("nan")
    violations = [
        max(-float(last["delta_eNPV_usd_m"]), 0.0) / 50.0,
        max(float(p5_floor) - float(last["P5_NPV_usd_m"]), 0.0) / 100.0,
        max(float(last["capacity_breach_probability"]) - float(breach_limit), 0.0) / 0.025,
    ]
    last["minimum_normalized_margin"] = float(max(violations))
    return last


def cell_descriptors(parameters: ProgramParameters) -> dict[str, float | str]:
    durations = parameters.durations
    return {
        "opportunity_hash": parameters.stable_hash(),
        "mean_cumulative_ptrs": float(np.mean(parameters.cumulative_ptrs)),
        "sd_cumulative_ptrs": float(np.std(parameters.cumulative_ptrs, ddof=1)),
        "minimum_cumulative_ptrs": float(np.min(parameters.cumulative_ptrs)),
        "maximum_cumulative_ptrs": float(np.max(parameters.cumulative_ptrs)),
        "mean_cost_multiplier": float(np.mean(parameters.cost_multiplier)),
        "sd_cost_multiplier": float(np.std(parameters.cost_multiplier, ddof=1)),
        "mean_peak_sales_multiplier": float(np.mean(parameters.peak_sales_multiplier)),
        "sd_peak_sales_multiplier": float(np.std(parameters.peak_sales_multiplier, ddof=1)),
        "mean_phase2_duration_years": float(np.mean(durations[:, 0])),
        "mean_phase3_duration_years": float(np.mean(durations[:, 1])),
        "mean_regulatory_duration_years": float(np.mean(durations[:, 2])),
        "mean_total_duration_years": float(np.mean(durations.sum(axis=1))),
        "opportunity_quality_index": float(
            np.mean(
                parameters.cumulative_ptrs
                * parameters.peak_sales_multiplier
                / parameters.cost_multiplier
            )
        ),
    }


def evaluate_cell(
    cell_id: int,
    parameters: ProgramParameters,
    states: np.ndarray,
    state_outcomes: np.ndarray,
    config: dict,
    plan: dict,
) -> dict[str, Any]:
    decision = plan["decision"]
    loe = float(decision["loe_year"])
    nodes = int(plan["design"]["gaussian_hermite_nodes"])
    synergy_grid = numeric_grid(decision["synergy_grid"])
    weights = {
        "diversified": joint_state_probabilities(
            parameters.stage_probabilities, 0.0, states=states, nodes=nodes
        ),
        "piap": joint_state_probabilities(
            parameters.stage_probabilities, float(decision["piap_rho"]), states=states, nodes=nodes
        ),
        "msog": joint_state_probabilities(
            parameters.stage_probabilities, float(decision["msog_rho"]), states=states, nodes=nodes
        ),
    }
    parallel_schedule = schedule_parallel_heterogeneous(state_outcomes, parameters)
    parallel_basis = build_heterogeneous_basis(parallel_schedule, config, parameters)
    parallel = linear_components(parallel_basis, config, loe, commercial_clustered=True)
    staged_schedule = schedule_staged_heterogeneous(state_outcomes, parameters)
    staged_basis = build_heterogeneous_basis(staged_schedule, config, parameters)
    staged = linear_components(staged_basis, config, loe, commercial_clustered=True)
    reference_npv = parallel.base_npv
    reference_summary = weighted_value_summary(
        reference_npv, parallel.base_pv_development, weights["diversified"]
    )
    row: dict[str, Any] = {
        "experiment": "E8",
        "cell_id": int(cell_id),
        **cell_descriptors(parameters),
        "diversified_eNPV_usd_m": reference_summary["eNPV_usd_m"],
        "diversified_ePI": reference_summary["ePI"],
        "diversified_P5_NPV_usd_m": reference_summary["P5_NPV_usd_m"],
        "diversified_CVaR5_NPV_usd_m": reference_summary["CVaR5_NPV_usd_m"],
    }
    required: dict[tuple[float, str], np.ndarray] = {}
    for cannibalization in [float(x) for x in decision["cannibalization_levels"]]:
        cash = parallel.base_cash_flow - cannibalization * parallel.cannibalization_cash_loss
        for capacity in (
            str(decision["primary_capacity_profile"]),
            str(decision["secondary_capacity_profile"]),
        ):
            required[(cannibalization, capacity)] = capacity_requirement_for_gain(
                cash, parallel.rnd_cash_gain, config, capacity
            )
    for topology in ("piap", "msog"):
        for cannibalization in [float(x) for x in decision["cannibalization_levels"]]:
            suffix = f"{topology}_c{int(round(100 * cannibalization))}"
            for role, capacity_key, policy_key in (
                (
                    "primary",
                    str(decision["primary_capacity_profile"]),
                    str(decision["primary_risk_policy_profile"]),
                ),
                (
                    "secondary",
                    str(decision["secondary_capacity_profile"]),
                    str(decision["secondary_risk_policy_profile"]),
                ),
            ):
                policy = config["risk_policy_profiles"][policy_key]
                frontier = frontier_for_weights(
                    weights[topology],
                    weights["diversified"],
                    parallel,
                    reference_npv,
                    required[(cannibalization, capacity_key)],
                    synergy_grid,
                    cannibalization,
                    float(policy["minimum_p5_npv_usd_m"]),
                    float(policy["maximum_capacity_breach_probability"]),
                )
                for key, value in frontier.items():
                    row[f"{suffix}_{role}_{key}"] = value
    central_synergy = 0.20
    central_npv = parallel.base_npv + central_synergy * parallel.rnd_npv_gain
    central_pvdev = parallel.base_pv_development - central_synergy * parallel.rnd_pv_development_saving
    central_summary = weighted_value_summary(central_npv, central_pvdev, weights["piap"])
    for key, value in central_summary.items():
        row[f"piap_s20_{key}"] = value
    row["piap_s20_high_capacity_breach_probability"] = weighted_probability(
        weights["piap"], required[(0.0, str(decision["primary_capacity_profile"]))] > central_synergy
    )
    row["piap_s20_base_capacity_breach_probability"] = weighted_probability(
        weights["piap"], required[(0.0, str(decision["secondary_capacity_profile"]))] > central_synergy
    )
    parallel_summary = weighted_value_summary(
        parallel.base_npv, parallel.base_pv_development, weights["piap"]
    )
    staged_summary = weighted_value_summary(
        staged.base_npv, staged.base_pv_development, weights["piap"]
    )
    row["staged_minus_parallel_eNPV_usd_m"] = (
        staged_summary["eNPV_usd_m"] - parallel_summary["eNPV_usd_m"]
    )
    row["staged_minus_parallel_eDevCost_usd_m"] = (
        staged_summary["eDevCost_usd_m"] - parallel_summary["eDevCost_usd_m"]
    )
    row["staged_minus_parallel_P5_NPV_usd_m"] = (
        staged_summary["P5_NPV_usd_m"] - parallel_summary["P5_NPV_usd_m"]
    )
    staged_launch = weighted_launch_summary(staged_schedule.launch_time, weights["piap"], loe)
    parallel_launch = weighted_launch_summary(parallel_schedule.launch_time, weights["piap"], loe)
    row["staged_minus_parallel_probability_two_launches_by_year_9"] = (
        staged_launch["probability_two_launches_by_year_9"]
        - parallel_launch["probability_two_launches_by_year_9"]
    )
    row["staged_minus_parallel_protected_runway_years"] = (
        staged_launch["mean_protected_runway_per_launch_years"]
        - parallel_launch["mean_protected_runway_per_launch_years"]
    )
    for label, components in (("parallel", parallel), ("staged", staged)):
        funding = direct_funding_metrics(
            components.base_cash_flow, config, str(decision["secondary_capacity_profile"])
        )["capacity_breach"]
        row[f"{label}_base_capacity_breach_probability"] = weighted_probability(
            weights["piap"], funding
        )
    row["staged_minus_parallel_base_capacity_breach_probability"] = (
        row["staged_base_capacity_breach_probability"]
        - row["parallel_base_capacity_breach_probability"]
    )
    regret_synergy = float(decision["additivity_regret_rnd_synergy"])
    regret_cannibalization = float(decision["additivity_regret_cannibalization"])
    tolerance = float(decision["value_choice_tolerance_usd_m"])
    assumed_concentrated_npv = parallel.base_npv + regret_synergy * parallel.rnd_npv_gain
    true_concentrated_npv = (
        assumed_concentrated_npv
        - regret_cannibalization * parallel.cannibalization_npv_loss
    )
    assumed_delta = float(
        np.dot(weights["diversified"], assumed_concentrated_npv)
        - np.dot(weights["diversified"], reference_npv)
    )
    assumed_choice = "concentrated" if assumed_delta > tolerance else "diversified"
    true_concentrated_mean = float(np.dot(weights["piap"], true_concentrated_npv))
    diversified_mean = float(np.dot(weights["diversified"], reference_npv))
    true_oracle = "concentrated" if true_concentrated_mean >= diversified_mean else "diversified"
    selected_mean = true_concentrated_mean if assumed_choice == "concentrated" else diversified_mean
    oracle_mean = max(true_concentrated_mean, diversified_mean)
    row["additivity_assumed_choice"] = assumed_choice
    row["additivity_true_oracle"] = true_oracle
    row["additivity_wrong_value_choice"] = bool(assumed_choice != true_oracle)
    row["additivity_eNPV_regret_usd_m"] = max(oracle_mean - selected_mean, 0.0)
    row["additivity_assumed_delta_eNPV_usd_m"] = assumed_delta
    row["additivity_true_concentrated_minus_diversified_eNPV_usd_m"] = (
        true_concentrated_mean - diversified_mean
    )
    selected_npv = true_concentrated_npv if assumed_choice == "concentrated" else reference_npv
    selected_weights = weights["piap"] if assumed_choice == "concentrated" else weights["diversified"]
    oracle_npv = true_concentrated_npv if true_oracle == "concentrated" else reference_npv
    oracle_weights = weights["piap"] if true_oracle == "concentrated" else weights["diversified"]
    row["additivity_P5_shortfall_usd_m"] = max(
        weighted_quantile(oracle_npv, oracle_weights, 0.05)
        - weighted_quantile(selected_npv, selected_weights, 0.05),
        0.0,
    )
    return row


def opportunity_frame(parameters: list[ProgramParameters]) -> pd.DataFrame:
    rows = []
    for cell_id, item in enumerate(parameters):
        for program in range(item.n_programs):
            rows.append(
                {
                    "cell_id": cell_id,
                    "program_slot": program,
                    "original_opportunity_index": int(item.permutation[program]),
                    "cumulative_ptrs": float(item.cumulative_ptrs[program]),
                    "phase2_probability": float(item.stage_probabilities[program, 0]),
                    "phase3_probability": float(item.stage_probabilities[program, 1]),
                    "regulatory_probability": float(item.stage_probabilities[program, 2]),
                    "cost_multiplier": float(item.cost_multiplier[program]),
                    "peak_sales_multiplier": float(item.peak_sales_multiplier[program]),
                    "phase2_duration_years": float(item.durations[program, 0]),
                    "phase3_duration_years": float(item.durations[program, 1]),
                    "regulatory_duration_years": float(item.durations[program, 2]),
                    "opportunity_hash": item.stable_hash(),
                }
            )
    return pd.DataFrame(rows)


def select_confirmation_cells(results: pd.DataFrame, count: int) -> list[int]:
    valid = results[results.piap_c0_primary_solution_exists].copy()
    if valid.empty:
        return []
    valid["frontier_bin"] = pd.qcut(
        valid.piap_c0_primary_minimum_synergy,
        q=min(count, valid.piap_c0_primary_minimum_synergy.nunique()),
        duplicates="drop",
    )
    selected = (
        valid.sort_values("piap_c0_primary_minimum_normalized_margin")
        .groupby("frontier_bin", observed=True)
        .head(1)
        .cell_id.astype(int)
        .tolist()
    )
    if len(selected) < count:
        extras = (
            valid[~valid.cell_id.isin(selected)]
            .sort_values("piap_c0_primary_minimum_normalized_margin")
            .head(count - len(selected))
            .cell_id.astype(int)
            .tolist()
        )
        selected.extend(extras)
    return selected[:count]


def monte_carlo_confirmation(
    selected: list[int],
    parameters: list[ProgramParameters],
    exact: pd.DataFrame,
    config: dict,
    plan: dict,
) -> pd.DataFrame:
    rows = []
    runs = int(plan["confirmation"]["runs"])
    seeds = [int(x) for x in config["simulation"]["seed_blocks"]]
    per_seed = runs // len(seeds)
    loe = float(plan["decision"]["loe_year"])
    rho = float(plan["decision"]["piap_rho"])
    offset = int(plan["confirmation"]["seed_offset"])
    capacity = str(plan["decision"]["primary_capacity_profile"])
    for position, cell_id in enumerate(selected, start=1):
        item = parameters[cell_id]
        exact_row = exact.loc[exact.cell_id == cell_id].iloc[0]
        synergy = float(exact_row.piap_c0_primary_minimum_synergy)
        previous = max(synergy - 0.01, 0.0)
        reference_parts = []
        candidate_parts = []
        previous_parts = []
        breach_parts = []
        previous_breach_parts = []
        for block_index, seed in enumerate(seeds):
            rng = np.random.default_rng(seed + offset + cell_id * 10007)
            residuals = rng.standard_normal((per_seed, item.n_programs, 3))
            factors = rng.standard_normal((per_seed, 3))
            diversified_outcomes = generate_heterogeneous_outcomes(
                residuals, factors, item.stage_probabilities, 0.0
            )
            piap_outcomes = generate_heterogeneous_outcomes(
                residuals, factors, item.stage_probabilities, rho
            )
            diversified_schedule = schedule_parallel_heterogeneous(diversified_outcomes, item)
            piap_schedule = schedule_parallel_heterogeneous(piap_outcomes, item)
            diversified_components = linear_components(
                build_heterogeneous_basis(diversified_schedule, config, item),
                config,
                loe,
                commercial_clustered=False,
            )
            piap_components = linear_components(
                build_heterogeneous_basis(piap_schedule, config, item),
                config,
                loe,
                commercial_clustered=True,
            )
            reference_parts.append(diversified_components.base_npv)
            candidate_npv = piap_components.base_npv + synergy * piap_components.rnd_npv_gain
            previous_npv = piap_components.base_npv + previous * piap_components.rnd_npv_gain
            candidate_parts.append(candidate_npv)
            previous_parts.append(previous_npv)
            candidate_cash = piap_components.base_cash_flow + synergy * piap_components.rnd_cash_gain
            previous_cash = piap_components.base_cash_flow + previous * piap_components.rnd_cash_gain
            breach_parts.append(
                direct_funding_metrics(candidate_cash, config, capacity)["capacity_breach"]
            )
            previous_breach_parts.append(
                direct_funding_metrics(previous_cash, config, capacity)["capacity_breach"]
            )
        reference = np.concatenate(reference_parts)
        candidate = np.concatenate(candidate_parts)
        previous_npv = np.concatenate(previous_parts)
        breach = np.concatenate(breach_parts)
        previous_breach = np.concatenate(previous_breach_parts)
        difference = candidate - reference
        delta = float(np.mean(difference))
        delta_half_width = float(1.96 * mc_mean_se(difference))
        breach_probability = float(np.mean(breach))
        previous_breach_probability = float(np.mean(previous_breach))
        p5 = float(np.quantile(candidate, 0.05))
        previous_p5 = float(np.quantile(previous_npv, 0.05))
        exact_delta = float(exact_row.piap_c0_primary_delta_eNPV_usd_m)
        exact_p5 = float(exact_row.piap_c0_primary_P5_NPV_usd_m)
        exact_breach = float(exact_row.piap_c0_primary_capacity_breach_probability)
        rows.append(
            {
                "cell_id": cell_id,
                "opportunity_hash": item.stable_hash(),
                "frontier_synergy": synergy,
                "previous_synergy": previous,
                "n_runs": runs,
                "mc_delta_eNPV_usd_m": delta,
                "mc_delta_eNPV_half_width_usd_m": delta_half_width,
                "exact_delta_eNPV_usd_m": exact_delta,
                "mc_minus_exact_delta_eNPV_usd_m": delta - exact_delta,
                "mc_P5_NPV_usd_m": p5,
                "exact_P5_NPV_usd_m": exact_p5,
                "mc_minus_exact_P5_usd_m": p5 - exact_p5,
                "mc_capacity_breach_probability": breach_probability,
                "mc_capacity_breach_half_width": probability_half_width(breach_probability, runs),
                "exact_capacity_breach_probability": exact_breach,
                "mc_minus_exact_capacity_breach_probability": breach_probability - exact_breach,
                "previous_mc_delta_eNPV_usd_m": float(np.mean(previous_npv - reference)),
                "previous_mc_P5_NPV_usd_m": previous_p5,
                "previous_mc_capacity_breach_probability": previous_breach_probability,
                "eNPV_precision_pass": bool(
                    delta_half_width
                    <= float(plan["confirmation"]["eNPV_difference_half_width_usd_m"])
                ),
                "probability_precision_pass": bool(
                    probability_half_width(breach_probability, runs)
                    <= float(plan["confirmation"]["probability_half_width_target"])
                ),
                "exact_delta_inside_mc_interval": bool(abs(delta - exact_delta) <= delta_half_width),
                "exact_breach_inside_mc_interval": bool(
                    abs(breach_probability - exact_breach)
                    <= probability_half_width(breach_probability, runs)
                ),
            }
        )
        print(f"E8 confirmation {position}/{len(selected)} complete", flush=True)
    return pd.DataFrame(rows)


def launch_service_robustness(config: dict, plan: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    launch_plan = plan["launch_service"]
    runs = int(launch_plan["runs"])
    maximum = int(launch_plan["maximum_nominal_breadth"])
    targets = [int(x) for x in launch_plan["launch_targets"]]
    service_targets = [float(x) for x in launch_plan["service_targets"]]
    seeds = [int(x) for x in config["simulation"]["seed_blocks"]]
    per_seed = runs // len(seeds)
    ptrs_spec = plan["heterogeneity"]["cumulative_ptrs"]
    rows = []
    accumulators = {}
    for geometry in launch_plan["geometries"]:
        accumulators[str(geometry["name"])] = {
            "success": np.zeros((len(targets), maximum), dtype=np.int64),
            "count": np.zeros(maximum, dtype=float),
            "count2": np.zeros(maximum, dtype=float),
            "program_success": np.zeros(maximum, dtype=float),
        }
    for seed in seeds:
        rng = np.random.default_rng(seed + int(plan["confirmation"]["seed_offset"]) + 99173)
        ptrs = triangular_ppf(
            rng.random((per_seed, maximum)),
            float(ptrs_spec["lower"]),
            float(ptrs_spec["mode"]),
            float(ptrs_spec["upper"]),
        )
        thresholds = ndtri(ptrs)
        residuals = rng.standard_normal((per_seed, maximum))
        factor_pool = rng.standard_normal((per_seed, maximum))
        for geometry in launch_plan["geometries"]:
            name = str(geometry["name"])
            cluster_size = int(geometry["cluster_size"])
            rho = float(geometry["rho"])
            cluster_ids = np.arange(maximum) // cluster_size
            latent = (
                math.sqrt(rho) * factor_pool[:, cluster_ids]
                + math.sqrt(1.0 - rho) * residuals
            )
            launches = latent <= thresholds
            cumulative = np.cumsum(launches, axis=1)
            acc = accumulators[name]
            acc["count"] += cumulative.sum(axis=0)
            acc["count2"] += np.square(cumulative).sum(axis=0)
            acc["program_success"] += launches.sum(axis=0)
            for target_index, target in enumerate(targets):
                acc["success"][target_index] += (cumulative >= target).sum(axis=0)
    for geometry in launch_plan["geometries"]:
        name = str(geometry["name"])
        cluster_size = int(geometry["cluster_size"])
        rho = float(geometry["rho"])
        acc = accumulators[name]
        for n in range(1, maximum + 1):
            mean = acc["count"][n - 1] / runs
            variance = (acc["count2"][n - 1] - runs * mean**2) / (runs - 1)
            marginal = acc["program_success"][:n] / runs
            mean_marginal_variance = float(
                np.mean(marginal * (1.0 - marginal) * runs / (runs - 1))
            )
            breadth = n**2 * mean_marginal_variance / variance if variance > 0 else np.nan
            for target_index, target in enumerate(targets):
                probability = float(acc["success"][target_index, n - 1] / runs)
                rows.append(
                    {
                        "experiment": "E8_launch_service",
                        "geometry": name,
                        "cluster_size": cluster_size,
                        "rho": rho,
                        "nominal_breadth": n,
                        "launch_target_per_3y": target,
                        "service_probability": probability,
                        "service_half_width": probability_half_width(probability, runs),
                        "mean_launches_per_3y": mean,
                        "variance_launches_per_3y": variance,
                        "effective_breadth": breadth,
                        "n_runs": runs,
                    }
                )
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
                    "minimum_nominal_breadth": (
                        int(selected.nominal_breadth) if selected is not None else np.nan
                    ),
                    "effective_breadth_at_frontier": (
                        float(selected.effective_breadth) if selected is not None else np.nan
                    ),
                    "achieved_service_probability": (
                        float(selected.service_probability) if selected is not None else np.nan
                    ),
                    "service_half_width": (
                        float(selected.service_half_width) if selected is not None else np.nan
                    ),
                    "solution_exists": bool(selected is not None),
                    "n_runs": runs,
                }
            )
    return cells, pd.DataFrame(frontiers)


def flat_reconciliation(flat: dict[str, Any], config: dict, output: Path) -> dict[str, Any]:
    e4_path = PACKAGE_ROOT / "outputs_e4_e7" / "e4_cell_metrics.csv"
    reference_path = PACKAGE_ROOT / "outputs_e4_e7" / "e4_diversified_reference.csv"
    result: dict[str, Any] = {"available": e4_path.exists() and reference_path.exists()}
    if not result["available"]:
        return result
    e4 = pd.read_csv(e4_path)
    reference = pd.read_csv(reference_path)
    candidate = e4[
        (e4.strategy == "Parallel PIAP")
        & np.isclose(e4.rho, 0.415)
        & np.isclose(e4.loe_year, 15.0)
        & np.isclose(e4.acceleration_years, 0.0)
        & np.isclose(e4.rnd_cost_synergy, 0.20)
        & (e4.n_runs == 250000)
    ].iloc[0]
    ref = reference[(np.isclose(reference.loe_year, 15.0)) & (reference.n_runs == 250000)].iloc[0]
    comparisons = {
        "piap_s20_eNPV_usd_m": (flat["piap_s20_eNPV_usd_m"], float(candidate.eNPV_usd_m), 25.0),
        "piap_s20_P5_NPV_usd_m": (flat["piap_s20_P5_NPV_usd_m"], float(candidate.P5_NPV_usd_m), 35.0),
        "piap_s20_high_capacity_breach_probability": (
            flat["piap_s20_high_capacity_breach_probability"],
            float(candidate.high_capacity_breach_probability),
            0.005,
        ),
        "diversified_eNPV_usd_m": (flat["diversified_eNPV_usd_m"], float(ref.eNPV_usd_m), 25.0),
    }
    result["comparisons"] = {
        key: {
            "exact": float(exact),
            "monte_carlo": float(mc),
            "absolute_difference": abs(float(exact) - float(mc)),
            "tolerance": float(tolerance),
            "pass": bool(abs(float(exact) - float(mc)) <= float(tolerance)),
        }
        for key, (exact, mc, tolerance) in comparisons.items()
    }
    result["pass"] = bool(all(item["pass"] for item in result["comparisons"].values()))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "benchmark_v0.2.yaml"))
    parser.add_argument("--plan", default=str(PACKAGE_ROOT / "config" / "e8_plan_v0.3.yaml"))
    parser.add_argument("--output", default=str(PACKAGE_ROOT / "outputs_e8"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    plan = load_yaml(Path(args.plan))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    parameters = generate_lhs_program_parameters(config, plan)
    opportunities = opportunity_frame(parameters)
    opportunities.to_csv(output / "e8_opportunity_cells.csv", index=False)
    states = enumerate_program_states(int(config["simulation"]["n_programs"]))
    state_outcomes = states_to_outcomes(states)
    rows = []
    for cell_id, item in enumerate(parameters):
        rows.append(evaluate_cell(cell_id, item, states, state_outcomes, config, plan))
        if (cell_id + 1) % 25 == 0 or cell_id + 1 == len(parameters):
            print(f"E8 exact cells {cell_id + 1}/{len(parameters)}", flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(output / "e8_cell_results.csv", index=False)
    flat = evaluate_cell(
        -1,
        homogeneous_program_parameters(config),
        states,
        state_outcomes,
        config,
        plan,
    )
    (output / "e8_flat_exact.json").write_text(
        json.dumps(flat, indent=2, sort_keys=True), encoding="utf-8"
    )
    reconciliation = flat_reconciliation(flat, config, output)
    (output / "e8_flat_reconciliation.json").write_text(
        json.dumps(reconciliation, indent=2, sort_keys=True), encoding="utf-8"
    )
    selected = select_confirmation_cells(results, int(plan["confirmation"]["selected_boundary_cells"]))
    confirmation = monte_carlo_confirmation(selected, parameters, results, config, plan)
    confirmation.to_csv(output / "e8_boundary_confirmation.csv", index=False)
    launch_cells, launch_frontier = launch_service_robustness(config, plan)
    launch_cells.to_csv(output / "e8_launch_service_cells.csv", index=False)
    launch_frontier.to_csv(output / "e8_launch_service_frontier.csv", index=False)
    validation = {
        "pass": bool(
            reconciliation.get("pass", False)
            and confirmation.eNPV_precision_pass.all()
            and confirmation.probability_precision_pass.all()
            and confirmation.exact_delta_inside_mc_interval.mean() >= 0.75
            and confirmation.exact_breach_inside_mc_interval.mean() >= 0.75
            and launch_frontier.service_half_width.dropna().max()
            <= float(plan["confirmation"]["probability_half_width_target"])
        ),
        "checks": {
            "flat_reconciliation": reconciliation,
            "boundary_eNPV_precision": {
                "pass": bool(confirmation.eNPV_precision_pass.all()),
                "maximum_half_width_usd_m": float(confirmation.mc_delta_eNPV_half_width_usd_m.max()),
            },
            "boundary_probability_precision": {
                "pass": bool(confirmation.probability_precision_pass.all()),
                "maximum_half_width": float(confirmation.mc_capacity_breach_half_width.max()),
            },
            "exact_vs_mc_eNPV_coverage": {
                "pass": bool(confirmation.exact_delta_inside_mc_interval.mean() >= 0.75),
                "coverage": float(confirmation.exact_delta_inside_mc_interval.mean()),
            },
            "exact_vs_mc_breach_coverage": {
                "pass": bool(confirmation.exact_breach_inside_mc_interval.mean() >= 0.75),
                "coverage": float(confirmation.exact_breach_inside_mc_interval.mean()),
            },
            "launch_service_precision": {
                "pass": bool(
                    launch_frontier.service_half_width.dropna().max()
                    <= float(plan["confirmation"]["probability_half_width_target"])
                ),
                "maximum_half_width": float(launch_frontier.service_half_width.dropna().max()),
            },
        },
    }
    (output / "validation_report.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8"
    )
    elapsed = time.perf_counter() - start
    manifest = {
        "analysis_version": plan["analysis"]["version"],
        "base_configuration_hash": scenario_hash(config),
        "plan_hash": scenario_hash(plan),
        "lhs_cells": int(plan["design"]["lhs_cells"]),
        "exact_states_per_cell": int(len(states)),
        "confirmation_runs": int(plan["confirmation"]["runs"]),
        "confirmation_cells": selected,
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
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"E8 complete: validation_pass={validation['pass']}, runtime={elapsed:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
