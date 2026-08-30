"""Pilot experiment definitions, execution, validation, and tidy outputs."""

from __future__ import annotations

import json
import math
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import matplotlib
import yaml

from .config import scenario_hash
from .economics import EconomicPaths, calculate_economics, enterprise_metrics
from .engine import Scenario, simulate_from_randomness
from .metrics import effective_breadth, mean_pairwise_binary_correlation, summarize_paths
from .policy import schedule_parallel, schedule_staged
from .randomness import RandomBlock, generate_random_block


def build_scenarios(config: dict) -> list[Scenario]:
    scenarios: list[Scenario] = []

    scenarios.append(Scenario("E1", "Diversified", "diversified"))
    for rho in config["pilot"]["e1"]["piap_rhos"]:
        scenarios.append(Scenario("E1", "Parallel PIAP", "same_asset", rho=float(rho)))
    for rho in config["pilot"]["e1"]["msog_rhos"]:
        scenarios.append(Scenario("E1", "MSOG", "shared_biology", rho=float(rho)))

    scenarios.append(Scenario("E2", "Diversified", "diversified"))
    for strategy, topology, grid_name in (
        ("Parallel PIAP", "same_asset", "piap_rhos"),
        ("MSOG", "shared_biology", "msog_rhos"),
    ):
        for rho in config["pilot"]["e2"][grid_name]:
            for shock in config["pilot"]["e2"]["common_shock_probabilities"]:
                scenarios.append(
                    Scenario(
                        "E2",
                        strategy,
                        topology,
                        rho=float(rho),
                        common_shock_probability=float(shock),
                    )
                )

    scenarios.append(Scenario("E3", "Diversified", "diversified"))
    for strategy, topology, grid_name in (
        ("Parallel PIAP", "same_asset", "piap_rhos"),
        ("MSOG", "same_indication", "msog_rhos"),
    ):
        for rho in config["pilot"]["e3"][grid_name]:
            for cannibalization in config["pilot"]["e3"]["cannibalization"]:
                scenarios.append(
                    Scenario(
                        "E3",
                        strategy,
                        topology,
                        rho=float(rho),
                        cannibalization=float(cannibalization),
                        commercial_clustered=True,
                    )
                )
    return scenarios


def build_random_blocks(config: dict) -> list[RandomBlock]:
    seeds = [int(seed) for seed in config["simulation"]["seed_blocks"]]
    total = int(config["simulation"]["pilot_runs_per_cell"])
    block_runs = total // len(seeds)
    programs = int(config["simulation"]["n_programs"])
    return [generate_random_block(seed, index, block_runs, programs) for index, seed in enumerate(seeds)]


def concatenate_paths(paths: list[EconomicPaths]) -> EconomicPaths:
    if not paths:
        raise ValueError("No paths supplied")
    return EconomicPaths(
        times=paths[0].times,
        development_spend=np.concatenate([p.development_spend for p in paths], axis=0),
        gross_new_contribution=np.concatenate([p.gross_new_contribution for p in paths], axis=0),
        commercial_spend=np.concatenate([p.commercial_spend for p in paths], axis=0),
        portfolio_cash_flow=np.concatenate([p.portfolio_cash_flow for p in paths], axis=0),
        portfolio_npv=np.concatenate([p.portfolio_npv for p in paths], axis=0),
        pv_development_spend=np.concatenate([p.pv_development_spend for p in paths], axis=0),
        realized_pi=np.concatenate([p.realized_pi for p in paths], axis=0),
        launch_count=np.concatenate([p.launch_count for p in paths], axis=0),
        launch_time=np.concatenate([p.launch_time for p in paths], axis=0),
        program_revenue=np.concatenate([p.program_revenue for p in paths], axis=0),
    )


def run_cell(scenario: Scenario, blocks: list[RandomBlock], config: dict) -> tuple[np.ndarray, EconomicPaths, pd.DataFrame]:
    all_outcomes = []
    all_paths = []
    run_rows = []
    offset = 0
    for block in blocks:
        outcomes, paths = simulate_from_randomness(scenario, block, config)
        all_outcomes.append(outcomes)
        all_paths.append(paths)
        for local in range(block.n_runs):
            run_rows.append(
                {
                    "scenario_hash": scenario.hash,
                    "block_index": block.block_index,
                    "seed": block.seed,
                    "run_id": offset + local,
                    "launches": int(paths.launch_count[local]),
                    "portfolio_NPV_usd_m": float(paths.portfolio_npv[local]),
                    "PV_development_spend_usd_m": float(paths.pv_development_spend[local]),
                }
            )
        offset += block.n_runs
    return np.concatenate(all_outcomes, axis=0), concatenate_paths(all_paths), pd.DataFrame(run_rows)


def deterministic_fixtures(config: dict) -> tuple[pd.DataFrame, dict[str, Any]]:
    n_programs = int(config["simulation"]["n_programs"])
    fixtures = []
    details: dict[str, Any] = {}
    forced = {
        "all_pass": np.ones((1, n_programs, 3), dtype=bool),
        "all_fail_phase2": np.zeros((1, n_programs, 3), dtype=bool),
        "lead_only_pass": np.zeros((1, n_programs, 3), dtype=bool),
    }
    forced["lead_only_pass"][0, 0, :] = True

    for fixture_name, outcomes in forced.items():
        for policy_name, scheduler in (("parallel", schedule_parallel), ("staged", schedule_staged)):
            schedule = scheduler(outcomes, config)
            paths = calculate_economics(schedule, config)
            raw_dev = float(paths.development_spend.sum())
            fixtures.append(
                {
                    "fixture": fixture_name,
                    "policy": policy_name,
                    "launch_count": int(paths.launch_count[0]),
                    "launch_times": ";".join(f"{x:.1f}" for x in schedule.launch_time[0] if np.isfinite(x)),
                    "undiscounted_development_spend_usd_m": raw_dev,
                    "portfolio_NPV_usd_m": float(paths.portfolio_npv[0]),
                    "PV_development_spend_usd_m": float(paths.pv_development_spend[0]),
                }
            )
    frame = pd.DataFrame(fixtures)

    all_pass_parallel = frame[(frame.fixture == "all_pass") & (frame.policy == "parallel")].iloc[0]
    all_pass_staged = frame[(frame.fixture == "all_pass") & (frame.policy == "staged")].iloc[0]
    all_fail = frame[(frame.fixture == "all_fail_phase2") & (frame.policy == "parallel")].iloc[0]
    checks = {
        "all_pass_parallel_launches": bool(all_pass_parallel.launch_count == n_programs),
        "all_pass_parallel_launch_time_6": bool(all_pass_parallel.launch_times == ";".join(["6.0"] * n_programs)),
        "all_pass_total_development_spend_3_3B": bool(abs(all_pass_parallel.undiscounted_development_spend_usd_m - 3300.0) < 1e-8),
        "all_fail_phase2_spend_600M": bool(abs(all_fail.undiscounted_development_spend_usd_m - 600.0) < 1e-8),
        "all_fail_phase2_zero_launches": bool(all_fail.launch_count == 0),
        "staged_all_pass_launches_6_and_10": bool(all_pass_staged.launch_times == "6.0;10.0;10.0;10.0;10.0;10.0"),
        "staged_all_pass_lower_npv_than_parallel": bool(all_pass_staged.portfolio_NPV_usd_m < all_pass_parallel.portfolio_NPV_usd_m),
    }
    details["checks"] = checks
    details["pass"] = bool(all(checks.values()))
    return frame, details


def _paired_se(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def paired_contrast_row(
    scenario: Scenario,
    paths: EconomicPaths,
    reference: EconomicPaths,
    config: dict,
) -> dict[str, Any]:
    diff = paths.portfolio_npv - reference.portfolio_npv
    launch_diff = paths.launch_count.astype(float) - reference.launch_count.astype(float)
    unpaired_se = float(
        np.sqrt(
            np.var(paths.portfolio_npv, ddof=1) / len(paths.portfolio_npv)
            + np.var(reference.portfolio_npv, ddof=1) / len(reference.portfolio_npv)
        )
    )
    row: dict[str, Any] = {
        **scenario.payload(),
        "scenario_hash": scenario.hash,
        "delta_eNPV_usd_m": float(np.mean(diff)),
        "paired_se_delta_eNPV_usd_m": _paired_se(diff),
        "unpaired_se_delta_eNPV_usd_m": unpaired_se,
        "paired_to_unpaired_se_ratio": float(_paired_se(diff) / unpaired_se),
        "paired_95ci_low_delta_eNPV_usd_m": float(np.mean(diff) - 1.96 * _paired_se(diff)),
        "paired_95ci_high_delta_eNPV_usd_m": float(np.mean(diff) + 1.96 * _paired_se(diff)),
        "delta_mean_launches": float(np.mean(launch_diff)),
        "paired_se_delta_mean_launches": _paired_se(launch_diff),
        "delta_sd_NPV_usd_m": float(np.std(paths.portfolio_npv, ddof=1) - np.std(reference.portfolio_npv, ddof=1)),
        "delta_P5_NPV_usd_m": float(np.quantile(paths.portfolio_npv, 0.05) - np.quantile(reference.portfolio_npv, 0.05)),
    }
    for capacity in config["enterprise"]["capacity_profiles"]:
        candidate_ent = enterprise_metrics(paths, config, capacity)
        reference_ent = enterprise_metrics(reference, config, capacity)
        row[f"delta_{capacity}_capacity_breach_probability"] = float(
            np.mean(candidate_ent["capacity_breach"]) - np.mean(reference_ent["capacity_breach"])
        )
    return row


def statistical_validation(
    summaries: pd.DataFrame,
    contrasts: pd.DataFrame,
    deterministic: dict[str, Any],
    selected_paths: dict[str, EconomicPaths],
    reproduction_exact: bool,
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    checks["V1_accounting"] = {"pass": bool(deterministic["pass"]), "details": deterministic["checks"]}

    e1_concentrated = summaries[(summaries.experiment == "E1") & (summaries.strategy != "Diversified")]
    target_mean = 6.0 * 0.289 * 0.578 * 0.906
    stage_checks = []
    for rate, se, target in (
        ("phase2_potential_pass_rate", "se_phase2_potential_pass_rate", 0.289),
        ("phase3_potential_pass_rate", "se_phase3_potential_pass_rate", 0.578),
        ("regulatory_potential_pass_rate", "se_regulatory_potential_pass_rate", 0.906),
        ("cumulative_launch_rate", "se_cumulative_launch_rate", target_mean / 6.0),
    ):
        stage_checks.append(np.abs(e1_concentrated[rate] - target) <= 3.0 * e1_concentrated[se])
    marginal_ok = bool(np.all(np.column_stack(stage_checks)))
    standardized = []
    for rate, se, target in (
        ("phase2_potential_pass_rate", "se_phase2_potential_pass_rate", 0.289),
        ("phase3_potential_pass_rate", "se_phase3_potential_pass_rate", 0.578),
        ("regulatory_potential_pass_rate", "se_regulatory_potential_pass_rate", 0.906),
        ("cumulative_launch_rate", "se_cumulative_launch_rate", target_mean / 6.0),
    ):
        standardized.extend((np.abs(e1_concentrated[rate] - target) / e1_concentrated[se]).tolist())
    checks["V2_cumulative_marginals"] = {
        "pass": bool(marginal_ok),
        "target_mean_launches": target_mean,
        "maximum_standardized_deviation": float(np.max(standardized)),
    }

    e1_contrasts = contrasts[contrasts.experiment == "E1"]
    npv_parity_ok = np.all(
        (e1_contrasts.paired_95ci_low_delta_eNPV_usd_m <= 0)
        & (e1_contrasts.paired_95ci_high_delta_eNPV_usd_m >= 0)
    )
    launch_low = e1_contrasts.delta_mean_launches - 1.96 * e1_contrasts.paired_se_delta_mean_launches
    launch_high = e1_contrasts.delta_mean_launches + 1.96 * e1_contrasts.paired_se_delta_mean_launches
    launch_parity_ok = np.all((launch_low <= 0) & (launch_high >= 0))
    checks["V3_mean_parity"] = {
        "pass": bool(npv_parity_ok and launch_parity_ok),
        "cells_with_ci_excluding_zero": int(
            np.sum(
                (e1_contrasts.paired_95ci_low_delta_eNPV_usd_m > 0)
                | (e1_contrasts.paired_95ci_high_delta_eNPV_usd_m < 0)
            )
        ),
        "launch_cells_with_ci_excluding_zero": int(np.sum((launch_low > 0) | (launch_high < 0))),
        "n_cells": int(len(e1_contrasts)),
    }

    e3 = summaries[(summaries.experiment == "E3") & (summaries.strategy != "Diversified")]
    monotone = True
    for _, group in e3.groupby(["strategy", "rho"]):
        ordered = group.sort_values("cannibalization")
        monotone &= bool(np.all(np.diff(ordered.eNPV_usd_m.to_numpy()) <= 1e-8))
        monotone &= bool(np.all(np.diff(ordered.ePI.to_numpy()) <= 1e-8))
    checks["V4_cannibalization_monotonicity"] = {"pass": bool(monotone)}

    crn_ok = bool(np.all(e1_contrasts.paired_to_unpaired_se_ratio <= 1.0 + 1e-12))
    checks["V5_common_random_numbers"] = {
        "pass": crn_ok,
        "maximum_paired_to_unpaired_se_ratio": float(e1_contrasts.paired_to_unpaired_se_ratio.max()),
    }

    finite = all(bool(np.all(np.isfinite(path.portfolio_npv))) for path in selected_paths.values())
    checks["V7_reproduction"] = {
        "pass": bool(finite and reproduction_exact),
        "n_selected_cells": len(selected_paths),
        "selected_outputs_finite": bool(finite),
        "bitwise_rerun_exact": bool(reproduction_exact),
    }
    return {"pass": bool(all(item["pass"] for item in checks.values())), "checks": checks}


def run_pilot(config: dict, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    blocks = build_random_blocks(config)
    scenarios = build_scenarios(config)
    deterministic_frame, deterministic_report = deterministic_fixtures(config)
    deterministic_frame.to_csv(output / "deterministic_fixtures.csv", index=False)

    summary_rows = []
    block_summary_rows = []
    run_frames: dict[str, pd.DataFrame] = {}
    cell_paths: dict[str, EconomicPaths] = {}
    cell_outcomes: dict[str, np.ndarray] = {}
    scenario_by_hash = {scenario.hash: scenario for scenario in scenarios}

    for index, scenario in enumerate(scenarios, start=1):
        outcomes, paths, run_frame = run_cell(scenario, blocks, config)
        cell_paths[scenario.hash] = paths
        cell_outcomes[scenario.hash] = outcomes
        run_frames[scenario.hash] = run_frame
        summary = {**scenario.payload(), "scenario_hash": scenario.hash, **summarize_paths(paths, config)}
        stage_names = ("phase2", "phase3", "regulatory")
        for stage_index, stage_name in enumerate(stage_names):
            per_run_rate = outcomes[:, :, stage_index].mean(axis=1)
            summary[f"{stage_name}_potential_pass_rate"] = float(per_run_rate.mean())
            summary[f"se_{stage_name}_potential_pass_rate"] = _paired_se(per_run_rate)
        cumulative_per_run = np.isfinite(paths.launch_time).mean(axis=1)
        summary["cumulative_launch_rate"] = float(cumulative_per_run.mean())
        summary["se_cumulative_launch_rate"] = _paired_se(cumulative_per_run)
        launch_binary = np.isfinite(paths.launch_time)
        binary_rho = mean_pairwise_binary_correlation(launch_binary)
        summary["mean_pairwise_launch_correlation"] = binary_rho
        marginal_variance = float(np.mean(np.var(launch_binary.astype(float), axis=0, ddof=1)))
        n_programs = int(config["simulation"]["n_programs"])
        summary["effective_breadth_variance_ratio"] = float(
            n_programs**2 * marginal_variance / np.var(paths.launch_count, ddof=1)
        ) if np.var(paths.launch_count, ddof=1) > 0 else float("nan")
        summary["effective_breadth_equal_correlation"] = effective_breadth(
            n_programs, binary_rho
        ) if np.isfinite(binary_rho) else float("nan")
        summary_rows.append(summary)
        for block_index, block_frame in run_frame.groupby("block_index"):
            block_npvs = block_frame.portfolio_NPV_usd_m.to_numpy()
            block_launches = block_frame.launches.to_numpy()
            block_summary_rows.append(
                {
                    **scenario.payload(),
                    "scenario_hash": scenario.hash,
                    "block_index": int(block_index),
                    "seed": int(block_frame.seed.iloc[0]),
                    "n_runs": int(len(block_frame)),
                    "eNPV_usd_m": float(block_npvs.mean()),
                    "P5_NPV_usd_m": float(np.quantile(block_npvs, 0.05)),
                    "CVaR5_NPV_usd_m": float(block_npvs[block_npvs <= np.quantile(block_npvs, 0.05)].mean()),
                    "mean_launches": float(block_launches.mean()),
                    "sd_launches": float(block_launches.std(ddof=1)),
                }
            )
        if index % 10 == 0 or index == len(scenarios):
            print(f"completed {index}/{len(scenarios)} scenario cells", flush=True)

    summaries = pd.DataFrame(summary_rows).sort_values(
        ["experiment", "strategy", "rho", "common_shock_probability", "cannibalization"]
    )
    summaries.to_csv(output / "pilot_summary.csv", index=False)
    pd.DataFrame(block_summary_rows).sort_values(
        ["experiment", "strategy", "rho", "common_shock_probability", "cannibalization", "block_index"]
    ).to_csv(output / "seed_block_summaries.csv", index=False)

    contrast_rows = []
    for experiment in ("E1", "E2", "E3"):
        reference_scenario = next(s for s in scenarios if s.experiment == experiment and s.strategy == "Diversified")
        reference = cell_paths[reference_scenario.hash]
        for scenario in scenarios:
            if scenario.experiment == experiment and scenario.strategy != "Diversified":
                contrast_rows.append(paired_contrast_row(scenario, cell_paths[scenario.hash], reference, config))
    contrasts = pd.DataFrame(contrast_rows).sort_values(
        ["experiment", "strategy", "rho", "common_shock_probability", "cannibalization"]
    )
    contrasts.to_csv(output / "paired_contrasts_vs_diversified.csv", index=False)

    selected_predicates = [
        lambda s: s.experiment == "E1" and s.strategy == "Diversified",
        lambda s: s.experiment == "E1" and s.strategy == "Parallel PIAP" and abs(s.rho - 0.415) < 1e-12,
        lambda s: s.experiment == "E1" and s.strategy == "MSOG" and abs(s.rho - 0.25) < 1e-12,
        lambda s: s.experiment == "E2" and s.strategy == "Parallel PIAP" and abs(s.rho - 0.415) < 1e-12 and abs(s.common_shock_probability - 0.05) < 1e-12,
        lambda s: s.experiment == "E3" and s.strategy == "Parallel PIAP" and abs(s.rho - 0.415) < 1e-12 and abs(s.cannibalization - 0.25) < 1e-12,
        lambda s: s.experiment == "E3" and s.strategy == "MSOG" and abs(s.rho - 0.25) < 1e-12 and abs(s.cannibalization - 0.25) < 1e-12,
    ]
    selected_scenarios = [s for s in scenarios if any(predicate(s) for predicate in selected_predicates)]
    selected_frames = []
    selected_paths = {}
    for scenario in selected_scenarios:
        frame = run_frames[scenario.hash].copy()
        for key, value in scenario.payload().items():
            frame[key] = value
        selected_frames.append(frame)
        selected_paths[scenario.hash] = cell_paths[scenario.hash]
    pd.concat(selected_frames, ignore_index=True).to_csv(output / "selected_run_distributions.csv.gz", index=False, compression="gzip")

    reproduction_scenario = selected_scenarios[0]
    _, reproduction_paths = simulate_from_randomness(reproduction_scenario, blocks[0], config)
    stored_paths = cell_paths[reproduction_scenario.hash]
    reproduction_exact = bool(
        np.array_equal(reproduction_paths.portfolio_npv, stored_paths.portfolio_npv[: blocks[0].n_runs])
        and np.array_equal(
            reproduction_paths.launch_time,
            stored_paths.launch_time[: blocks[0].n_runs],
            equal_nan=True,
        )
    )
    validation = statistical_validation(
        summaries,
        contrasts,
        deterministic_report,
        selected_paths,
        reproduction_exact,
    )
    (output / "validation_report.json").write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")

    scenario_dictionary = pd.DataFrame([{**s.payload(), "scenario_hash": s.hash} for s in scenarios])
    scenario_dictionary.to_csv(output / "scenario_dictionary.csv", index=False)

    elapsed = time.perf_counter() - start
    manifest = {
        "specification_version": config["specification"]["version"],
        "configuration_hash": scenario_hash(config),
        "pilot_runs_per_cell": int(config["simulation"]["pilot_runs_per_cell"]),
        "seed_blocks": config["simulation"]["seed_blocks"],
        "n_scenario_cells": len(scenarios),
        "runtime_seconds": elapsed,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "PyYAML": yaml.__version__,
        },
        "validation_pass": validation["pass"],
        "files": [
            "deterministic_fixtures.csv",
            "pilot_summary.csv",
            "seed_block_summaries.csv",
            "paired_contrasts_vs_diversified.csv",
            "selected_run_distributions.csv.gz",
            "scenario_dictionary.csv",
            "validation_report.json",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "summaries": summaries,
        "contrasts": contrasts,
        "validation": validation,
        "manifest": manifest,
        "paths": cell_paths,
        "scenarios": scenario_by_hash,
    }
