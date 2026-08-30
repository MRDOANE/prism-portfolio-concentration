"""Parallel and staged development-policy scheduling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Schedule:
    phase2_start: np.ndarray
    phase3_start: np.ndarray
    regulatory_start: np.ndarray
    launch_time: np.ndarray


def _duration(config: dict, stage: str, program_index: int, acceleration_years: float) -> float:
    base = float(config["development"][stage]["duration_years"])
    if stage == "phase3" and program_index > 0:
        return max(float(config["simulation"]["time_step_years"]), base - acceleration_years)
    return base


def schedule_parallel(outcomes: np.ndarray, config: dict, acceleration_years: float = 0.0) -> Schedule:
    n_runs, n_programs, _ = outcomes.shape
    p2_start = np.zeros((n_runs, n_programs), dtype=float)
    p3_start = np.full((n_runs, n_programs), np.nan)
    reg_start = np.full((n_runs, n_programs), np.nan)
    launch = np.full((n_runs, n_programs), np.nan)
    p2_duration = float(config["development"]["phase2"]["duration_years"])
    reg_duration = float(config["development"]["regulatory"]["duration_years"])

    for i in range(n_programs):
        p3_start[outcomes[:, i, 0], i] = p2_duration
        p3_duration = _duration(config, "phase3", i, acceleration_years)
        p3_pass = outcomes[:, i, 0] & outcomes[:, i, 1]
        reg_start[p3_pass, i] = p2_duration + p3_duration
        approved = p3_pass & outcomes[:, i, 2]
        launch[approved, i] = p2_duration + p3_duration + reg_duration
    return Schedule(p2_start, p3_start, reg_start, launch)


def schedule_staged(outcomes: np.ndarray, config: dict, acceleration_years: float = 0.0) -> Schedule:
    """Implement the locked staged PIAP rule for each simulated path."""

    n_runs, n_programs, _ = outcomes.shape
    p2_start = np.zeros((n_runs, n_programs), dtype=float)
    p3_start = np.full((n_runs, n_programs), np.nan)
    reg_start = np.full((n_runs, n_programs), np.nan)
    launch = np.full((n_runs, n_programs), np.nan)
    p2_duration = float(config["development"]["phase2"]["duration_years"])
    reg_duration = float(config["development"]["regulatory"]["duration_years"])

    for run in range(n_runs):
        eligible = [i for i in range(n_programs) if outcomes[run, i, 0]]
        decision_time = p2_duration
        approved_lead = False
        remaining: list[int] = []

        for position, i in enumerate(eligible):
            p3_start[run, i] = decision_time
            p3_duration = _duration(config, "phase3", i, acceleration_years)
            p3_complete = decision_time + p3_duration
            if not outcomes[run, i, 1]:
                decision_time = p3_complete
                continue
            reg_start[run, i] = p3_complete
            decision_time = p3_complete + reg_duration
            if not outcomes[run, i, 2]:
                continue
            launch[run, i] = decision_time
            approved_lead = True
            remaining = eligible[position + 1 :]
            break

        if approved_lead:
            for i in remaining:
                p3_start[run, i] = decision_time
                p3_duration = _duration(config, "phase3", i, acceleration_years)
                p3_complete = decision_time + p3_duration
                if outcomes[run, i, 1]:
                    reg_start[run, i] = p3_complete
                    if outcomes[run, i, 2]:
                        launch[run, i] = p3_complete + reg_duration

    return Schedule(p2_start, p3_start, reg_start, launch)

