"""Matched program heterogeneity and exact six-program outcome integration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import brentq
from scipy.special import expit, logit, ndtr, ndtri
from scipy.stats import qmc

from .advanced import EconomicBasis, _stage_spend, time_grid
from .policy import Schedule


@dataclass(frozen=True)
class ProgramParameters:
    """One matched six-opportunity multiset in program-slot order."""

    cumulative_ptrs: np.ndarray
    stage_probabilities: np.ndarray
    cost_multiplier: np.ndarray
    peak_sales_multiplier: np.ndarray
    durations: np.ndarray
    permutation: np.ndarray

    @property
    def n_programs(self) -> int:
        return int(len(self.cumulative_ptrs))

    def stable_hash(self) -> str:
        payload = {
            "cumulative_ptrs": self.cumulative_ptrs.tolist(),
            "stage_probabilities": self.stage_probabilities.tolist(),
            "cost_multiplier": self.cost_multiplier.tolist(),
            "peak_sales_multiplier": self.peak_sales_multiplier.tolist(),
            "durations": self.durations.tolist(),
            "permutation": self.permutation.tolist(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]


def triangular_ppf(u: np.ndarray, lower: float, mode: float, upper: float) -> np.ndarray:
    """Vectorized inverse CDF for a bounded triangular distribution."""

    values = np.asarray(u, dtype=float)
    if not lower <= mode <= upper or not lower < upper:
        raise ValueError("triangular bounds must satisfy lower <= mode <= upper")
    split = (mode - lower) / (upper - lower)
    left = lower + np.sqrt(values * (upper - lower) * (mode - lower))
    right = upper - np.sqrt((1.0 - values) * (upper - lower) * (upper - mode))
    return np.where(values <= split, left, right)


def _rounded_triangular(u: np.ndarray, spec: dict) -> np.ndarray:
    values = triangular_ppf(u, float(spec["lower"]), float(spec["mode"]), float(spec["upper"]))
    increment = float(spec["increment"])
    rounded = np.round(values / increment) * increment
    return np.clip(rounded, float(spec["lower"]), float(spec["upper"]))


def cumulative_to_stage_probabilities(target: float, base: np.ndarray) -> np.ndarray:
    """Apply a common log-odds shift whose stage-probability product is target."""

    base = np.asarray(base, dtype=float)
    target = float(target)
    if not 0.0 < target < 1.0:
        raise ValueError("cumulative pTRS target must lie in (0, 1)")
    base_logits = logit(base)

    def objective(delta: float) -> float:
        return float(np.prod(expit(base_logits + delta)) - target)

    delta = brentq(objective, -20.0, 20.0)
    probabilities = expit(base_logits + delta)
    if not np.isclose(np.prod(probabilities), target, atol=1e-12):
        raise RuntimeError("stage-probability calibration failed")
    return probabilities


def generate_lhs_program_parameters(
    config: dict,
    plan: dict,
) -> list[ProgramParameters]:
    """Generate frozen matched opportunity sets from one 24-dimensional LHS."""

    n_cells = int(plan["design"]["lhs_cells"])
    n_programs = int(config["simulation"]["n_programs"])
    sampler = qmc.LatinHypercube(d=n_programs * 4, seed=int(plan["design"]["lhs_seed"]))
    lhs = sampler.random(n_cells).reshape(n_cells, n_programs, 4)
    h = plan["heterogeneity"]
    ptrs_spec = h["cumulative_ptrs"]
    cost_spec = h["development_cost_multiplier"]
    peak_spec = h["peak_sales_multiplier"]
    base_stage = np.array(
        [
            config["development"]["phase2"]["pass_probability"],
            config["development"]["phase3"]["pass_probability"],
            config["development"]["regulatory"]["pass_probability"],
        ],
        dtype=float,
    )
    permutation_rng = np.random.default_rng(int(plan["design"]["opportunity_permutation_seed"]))
    result: list[ProgramParameters] = []
    for cell in range(n_cells):
        raw = lhs[cell]
        cumulative = triangular_ppf(
            raw[:, 0],
            float(ptrs_spec["lower"]),
            float(ptrs_spec["mode"]),
            float(ptrs_spec["upper"]),
        )
        cost = triangular_ppf(
            raw[:, 1],
            float(cost_spec["lower"]),
            float(cost_spec["mode"]),
            float(cost_spec["upper"]),
        )
        peak = triangular_ppf(
            raw[:, 2],
            float(peak_spec["lower"]),
            float(peak_spec["mode"]),
            float(peak_spec["upper"]),
        )
        duration_rank = raw[:, 3]
        durations = np.column_stack(
            [
                _rounded_triangular(duration_rank, h["duration_years"]["phase2"]),
                _rounded_triangular(duration_rank, h["duration_years"]["phase3"]),
                _rounded_triangular(duration_rank, h["duration_years"]["regulatory"]),
            ]
        )
        stage = np.vstack([cumulative_to_stage_probabilities(x, base_stage) for x in cumulative])
        permutation = permutation_rng.permutation(n_programs)
        result.append(
            ProgramParameters(
                cumulative_ptrs=cumulative[permutation],
                stage_probabilities=stage[permutation],
                cost_multiplier=cost[permutation],
                peak_sales_multiplier=peak[permutation],
                durations=durations[permutation],
                permutation=permutation,
            )
        )
    return result


def homogeneous_program_parameters(config: dict) -> ProgramParameters:
    n = int(config["simulation"]["n_programs"])
    base_stage = np.array(
        [
            config["development"]["phase2"]["pass_probability"],
            config["development"]["phase3"]["pass_probability"],
            config["development"]["regulatory"]["pass_probability"],
        ],
        dtype=float,
    )
    durations = np.tile(
        [
            config["development"]["phase2"]["duration_years"],
            config["development"]["phase3"]["duration_years"],
            config["development"]["regulatory"]["duration_years"],
        ],
        (n, 1),
    ).astype(float)
    return ProgramParameters(
        cumulative_ptrs=np.full(n, np.prod(base_stage)),
        stage_probabilities=np.tile(base_stage, (n, 1)),
        cost_multiplier=np.ones(n),
        peak_sales_multiplier=np.ones(n),
        durations=durations,
        permutation=np.arange(n),
    )


def enumerate_program_states(n_programs: int = 6) -> np.ndarray:
    """Return all realized states: 0=P2 fail, 1=P3 fail, 2=reg fail, 3=launch."""

    codes = np.arange(4**n_programs, dtype=np.int64)
    divisors = np.power(4, np.arange(n_programs, dtype=np.int64))
    return ((codes[:, None] // divisors[None, :]) % 4).astype(np.int8)


def gaussian_bernoulli_joint(probabilities: np.ndarray, rho: float, nodes: int = 32) -> np.ndarray:
    """Joint probabilities of every Bernoulli pattern under a one-factor Gaussian copula."""

    probabilities = np.asarray(probabilities, dtype=float)
    n = len(probabilities)
    patterns = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(bool)
    if abs(float(rho)) < 1e-15:
        values = np.prod(np.where(patterns, probabilities[None, :], 1.0 - probabilities[None, :]), axis=1)
        return values / values.sum()
    x, w = hermgauss(int(nodes))
    factors = np.sqrt(2.0) * x
    conditional = ndtr(
        (ndtri(probabilities)[None, :] - math.sqrt(float(rho)) * factors[:, None])
        / math.sqrt(1.0 - float(rho))
    )
    values = np.zeros(2**n, dtype=float)
    for index, pattern in enumerate(patterns):
        integrand = np.prod(np.where(pattern[None, :], conditional, 1.0 - conditional), axis=1)
        values[index] = float(np.dot(w, integrand) / np.sqrt(np.pi))
    values = np.maximum(values, 0.0)
    return values / values.sum()


def _partial_pattern_table(joint: np.ndarray, n_programs: int) -> np.ndarray:
    patterns = np.arange(2**n_programs, dtype=np.int64)
    table = np.zeros((2**n_programs, 2**n_programs), dtype=float)
    for known in range(2**n_programs):
        for passed in range(2**n_programs):
            if passed & ~known:
                continue
            table[known, passed] = float(joint[(patterns & known) == passed].sum())
    return table


def joint_state_probabilities(
    stage_probabilities: np.ndarray,
    rho: float,
    states: np.ndarray | None = None,
    nodes: int = 32,
) -> np.ndarray:
    """Probability of each realized 0/1/2/3 state vector."""

    stage_probabilities = np.asarray(stage_probabilities, dtype=float)
    n_programs = stage_probabilities.shape[0]
    if stage_probabilities.shape != (n_programs, 3):
        raise ValueError("stage_probabilities must have shape (program, 3)")
    if states is None:
        states = enumerate_program_states(n_programs)
    bit_weights = np.power(2, np.arange(n_programs, dtype=np.int64))
    p2_mask = ((states > 0).astype(np.int64) * bit_weights[None, :]).sum(axis=1)
    p3_known = p2_mask
    p3_pass = ((states >= 2).astype(np.int64) * bit_weights[None, :]).sum(axis=1)
    reg_known = p3_pass
    reg_pass = ((states == 3).astype(np.int64) * bit_weights[None, :]).sum(axis=1)
    stage_joint = [
        gaussian_bernoulli_joint(stage_probabilities[:, stage], rho, nodes)
        for stage in range(3)
    ]
    p3_partial = _partial_pattern_table(stage_joint[1], n_programs)
    reg_partial = _partial_pattern_table(stage_joint[2], n_programs)
    probabilities = (
        stage_joint[0][p2_mask]
        * p3_partial[p3_known, p3_pass]
        * reg_partial[reg_known, reg_pass]
    )
    probabilities = np.maximum(probabilities, 0.0)
    return probabilities / probabilities.sum()


def states_to_outcomes(states: np.ndarray) -> np.ndarray:
    return np.stack([states >= 1, states >= 2, states == 3], axis=2)


def schedule_parallel_heterogeneous(outcomes: np.ndarray, parameters: ProgramParameters) -> Schedule:
    n_runs, n_programs, _ = outcomes.shape
    p2_start = np.zeros((n_runs, n_programs), dtype=float)
    p3_start = np.full((n_runs, n_programs), np.nan)
    reg_start = np.full((n_runs, n_programs), np.nan)
    launch = np.full((n_runs, n_programs), np.nan)
    for program in range(n_programs):
        d2, d3, dr = parameters.durations[program]
        p2_pass = outcomes[:, program, 0]
        p3_pass = p2_pass & outcomes[:, program, 1]
        approved = p3_pass & outcomes[:, program, 2]
        p3_start[p2_pass, program] = d2
        reg_start[p3_pass, program] = d2 + d3
        launch[approved, program] = d2 + d3 + dr
    return Schedule(p2_start, p3_start, reg_start, launch)


def schedule_staged_heterogeneous(outcomes: np.ndarray, parameters: ProgramParameters) -> Schedule:
    n_runs, n_programs, _ = outcomes.shape
    p2_start = np.zeros((n_runs, n_programs), dtype=float)
    p3_start = np.full((n_runs, n_programs), np.nan)
    reg_start = np.full((n_runs, n_programs), np.nan)
    launch = np.full((n_runs, n_programs), np.nan)
    for run in range(n_runs):
        eligible = [program for program in range(n_programs) if outcomes[run, program, 0]]
        decision_time = 0.0
        remaining: list[int] = []
        approved_lead = False
        for position, program in enumerate(eligible):
            d2, d3, dr = parameters.durations[program]
            start = max(decision_time, float(d2))
            p3_start[run, program] = start
            p3_complete = start + float(d3)
            if not outcomes[run, program, 1]:
                decision_time = p3_complete
                continue
            reg_start[run, program] = p3_complete
            decision_time = p3_complete + float(dr)
            if not outcomes[run, program, 2]:
                continue
            launch[run, program] = decision_time
            approved_lead = True
            remaining = eligible[position + 1 :]
            break
        if approved_lead:
            for program in remaining:
                d2, d3, dr = parameters.durations[program]
                start = max(decision_time, float(d2))
                p3_start[run, program] = start
                p3_complete = start + float(d3)
                if outcomes[run, program, 1]:
                    reg_start[run, program] = p3_complete
                    if outcomes[run, program, 2]:
                        launch[run, program] = p3_complete + float(dr)
    return Schedule(p2_start, p3_start, reg_start, launch)


def build_heterogeneous_basis(
    schedule: Schedule,
    config: dict,
    parameters: ProgramParameters,
) -> EconomicBasis:
    times = time_grid(config)
    dt = float(config["simulation"]["time_step_years"])
    n_runs, n_programs = schedule.launch_time.shape
    development_base = np.zeros((n_runs, len(times)), dtype=float)
    follower_development = np.zeros_like(development_base)
    commercial_base = np.zeros_like(development_base)
    follower_commercial = np.zeros_like(development_base)
    pre_loe_revenue = np.zeros((n_runs, n_programs, len(times)), dtype=float)
    ramp = np.asarray(config["commercial"]["revenue_ramp"], dtype=float)
    base_peak = float(config["commercial"]["peak_sales_usd_m"])
    support = float(config["commercial"]["support_cost_usd_m_per_year"])
    stage_names = ("phase2", "phase3", "regulatory")
    starts_by_stage = (schedule.phase2_start, schedule.phase3_start, schedule.regulatory_start)
    for program in range(n_programs):
        program_development = np.zeros_like(development_base)
        for stage_index, (stage, starts) in enumerate(zip(stage_names, starts_by_stage)):
            duration = float(parameters.durations[program, stage_index])
            cost = float(config["development"][stage]["cost_usd_m"]) * float(parameters.cost_multiplier[program])
            program_development += _stage_spend(times, starts[:, program], duration, cost)
        development_base += program_development
        if program > 0:
            follower_development += program_development
        launch = schedule.launch_time[:, program]
        finite = np.isfinite(launch)
        safe_launch = np.where(finite, launch, np.inf)
        active = finite[:, None] & (times[None, :] >= safe_launch[:, None] + 1.0 - 1e-10)
        elapsed = np.maximum(times[None, :] - (safe_launch[:, None] + 1.0), 0.0)
        commercial_year = np.floor(elapsed + 1e-10).astype(int)
        ramp_fraction = ramp[np.minimum(commercial_year, len(ramp) - 1)]
        peak = base_peak * float(parameters.peak_sales_multiplier[program])
        program_revenue = peak * ramp_fraction * dt * active
        program_commercial = support * dt * active
        pre_loe_revenue[:, program, :] = program_revenue
        commercial_base += program_commercial
        if program > 0:
            follower_commercial += program_commercial
    return EconomicBasis(
        times=times,
        launch_time=schedule.launch_time,
        development_base=development_base,
        follower_development=follower_development,
        commercial_base=commercial_base,
        follower_commercial=follower_commercial,
        pre_loe_revenue_by_program=pre_loe_revenue,
    )


def generate_heterogeneous_outcomes(
    residuals: np.ndarray,
    cluster_factors: np.ndarray,
    stage_probabilities: np.ndarray,
    rho: float,
) -> np.ndarray:
    thresholds = ndtri(np.asarray(stage_probabilities, dtype=float))
    latent = (
        math.sqrt(float(rho)) * cluster_factors[:, None, :]
        + math.sqrt(1.0 - float(rho)) * residuals
    )
    return latent <= thresholds[None, :, :]


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = np.asarray(values, dtype=float)[order]
    sorted_weights = np.asarray(weights, dtype=float)[order]
    cumulative = np.cumsum(sorted_weights)
    index = int(np.searchsorted(cumulative, float(probability), side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def weighted_cvar_lower(values: np.ndarray, weights: np.ndarray, probability: float = 0.05) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = np.asarray(values, dtype=float)[order]
    sorted_weights = np.asarray(weights, dtype=float)[order]
    remaining = float(probability)
    total = 0.0
    for value, weight in zip(sorted_values, sorted_weights):
        take = min(float(weight), remaining)
        total += float(value) * take
        remaining -= take
        if remaining <= 1e-15:
            break
    return total / float(probability)


def weighted_value_summary(npv: np.ndarray, pv_development: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    mean = float(np.dot(weights, npv))
    dev = float(np.dot(weights, pv_development))
    variance = float(np.dot(weights, np.square(npv - mean)))
    return {
        "eNPV_usd_m": mean,
        "eDevCost_usd_m": dev,
        "ePI": mean / dev,
        "sd_NPV_usd_m": math.sqrt(max(variance, 0.0)),
        "P5_NPV_usd_m": weighted_quantile(npv, weights, 0.05),
        "CVaR5_NPV_usd_m": weighted_cvar_lower(npv, weights, 0.05),
        "probability_NPV_negative": float(weights[npv < 0.0].sum()),
    }


def weighted_launch_summary(launch_time: np.ndarray, weights: np.ndarray, loe_year: float) -> dict[str, float]:
    launched = np.isfinite(launch_time)
    counts = launched.sum(axis=1).astype(float)
    mean = float(np.dot(weights, counts))
    variance = float(np.dot(weights, np.square(counts - mean)))
    before = launched & (launch_time < float(loe_year) - 1e-10)
    runway = np.where(launched, np.maximum(float(loe_year) - launch_time, 0.0), 0.0)
    total_launches = float(np.dot(weights, counts))
    return {
        "mean_launches": mean,
        "sd_launches": math.sqrt(max(variance, 0.0)),
        "p_zero_launches": float(weights[counts == 0].sum()),
        "mean_launches_before_loe": float(np.dot(weights, before.sum(axis=1))),
        "probability_at_least_one_launch_before_loe": float(weights[before.any(axis=1)].sum()),
        "mean_protected_runway_per_launch_years": (
            float(np.dot(weights, runway.sum(axis=1))) / total_launches if total_launches > 0 else float("nan")
        ),
        "probability_two_launches_by_year_9": float(
            weights[((launch_time <= 9.0) & launched).sum(axis=1) >= 2].sum()
        ),
    }
