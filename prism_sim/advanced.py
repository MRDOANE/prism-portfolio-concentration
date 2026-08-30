"""Efficient basis decomposition for E4-E7 frontier experiments.

The advanced experiments repeatedly value the same technical outcomes under
different economic assumptions.  Once outcomes and a development schedule are
fixed, R&D cost saving, commercial cost saving, and cannibalization enter the
cash-flow path linearly.  This module exposes that decomposition so boundary
searches do not resimulate technical outcomes or rebuild program cash flows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .metrics import cvar_lower, mc_mean_se, mean_pairwise_binary_correlation
from .policy import Schedule


@dataclass
class EconomicBasis:
    times: np.ndarray
    launch_time: np.ndarray
    development_base: np.ndarray
    follower_development: np.ndarray
    commercial_base: np.ndarray
    follower_commercial: np.ndarray
    pre_loe_revenue_by_program: np.ndarray

    @property
    def n_runs(self) -> int:
        return int(self.launch_time.shape[0])


@dataclass
class LinearComponents:
    times: np.ndarray
    launch_time: np.ndarray
    base_cash_flow: np.ndarray
    rnd_cash_gain: np.ndarray
    commercial_cash_gain: np.ndarray
    cannibalization_cash_loss: np.ndarray
    base_npv: np.ndarray
    rnd_npv_gain: np.ndarray
    commercial_npv_gain: np.ndarray
    cannibalization_npv_loss: np.ndarray
    base_pv_development: np.ndarray
    rnd_pv_development_saving: np.ndarray
    development_base: np.ndarray
    follower_development: np.ndarray


def time_grid(config: dict) -> np.ndarray:
    dt = float(config["simulation"]["time_step_years"])
    horizon = float(config["simulation"]["horizon_years"])
    return np.arange(0.0, horizon + dt / 2.0, dt)


def _stage_spend(times: np.ndarray, starts: np.ndarray, duration: float, cost: float) -> np.ndarray:
    dt = float(times[1] - times[0])
    starts = np.asarray(starts, dtype=float)
    active = (
        np.isfinite(starts)[:, None]
        & (times[None, :] >= starts[:, None] - 1e-10)
        & (times[None, :] < starts[:, None] + duration - 1e-10)
    )
    return active * (float(cost) * dt / float(duration))


def _phase3_duration(schedule: Schedule, config: dict, program: int) -> float:
    duration = float(config["development"]["phase3"]["duration_years"])
    if program == 0:
        return duration
    observed = schedule.regulatory_start[:, program] - schedule.phase3_start[:, program]
    finite = observed[np.isfinite(observed)]
    if finite.size:
        return float(np.median(finite))
    return duration


def build_economic_basis(schedule: Schedule, config: dict) -> EconomicBasis:
    """Build cash-flow components that are invariant to LOE and synergy rates."""

    times = time_grid(config)
    dt = float(config["simulation"]["time_step_years"])
    n_runs, n_programs = schedule.launch_time.shape
    n_times = len(times)
    development_base = np.zeros((n_runs, n_times), dtype=float)
    follower_development = np.zeros_like(development_base)
    commercial_base = np.zeros_like(development_base)
    follower_commercial = np.zeros_like(development_base)
    pre_loe_revenue = np.zeros((n_runs, n_programs, n_times), dtype=float)

    peak = float(config["commercial"]["peak_sales_usd_m"])
    ramp = np.asarray(config["commercial"]["revenue_ramp"], dtype=float)
    support = float(config["commercial"]["support_cost_usd_m_per_year"])

    for program in range(n_programs):
        program_development = np.zeros_like(development_base)
        for stage, starts in (
            ("phase2", schedule.phase2_start[:, program]),
            ("phase3", schedule.phase3_start[:, program]),
            ("regulatory", schedule.regulatory_start[:, program]),
        ):
            duration = (
                _phase3_duration(schedule, config, program)
                if stage == "phase3"
                else float(config["development"][stage]["duration_years"])
            )
            cost = float(config["development"][stage]["cost_usd_m"])
            program_development += _stage_spend(times, starts, duration, cost)
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


def linear_components(
    basis: EconomicBasis,
    config: dict,
    loe_year: float,
    commercial_clustered: bool,
) -> LinearComponents:
    """Return exact linear cash-flow bases for synergy and overlap searches."""

    times = basis.times
    post_loe = float(config["commercial"]["post_loe_sales_fraction"])
    loe_factor = np.where(times >= float(loe_year) - 1e-10, post_loe, 1.0)
    revenue = basis.pre_loe_revenue_by_program * loe_factor[None, None, :]
    total_revenue = revenue.sum(axis=1)
    if commercial_clustered:
        overlap_revenue = total_revenue - revenue.max(axis=1)
    else:
        overlap_revenue = np.zeros_like(total_revenue)
    margin = float(config["commercial"]["gross_contribution_margin"])
    gross = total_revenue * margin
    cannibalization_loss = overlap_revenue * margin
    base_cash = gross - basis.development_base - basis.commercial_base
    discount = np.power(1.0 + float(config["commercial"]["discount_rate"]), times)
    return LinearComponents(
        times=times,
        launch_time=basis.launch_time,
        base_cash_flow=base_cash,
        rnd_cash_gain=basis.follower_development,
        commercial_cash_gain=basis.follower_commercial,
        cannibalization_cash_loss=cannibalization_loss,
        base_npv=(base_cash / discount[None, :]).sum(axis=1),
        rnd_npv_gain=(basis.follower_development / discount[None, :]).sum(axis=1),
        commercial_npv_gain=(basis.follower_commercial / discount[None, :]).sum(axis=1),
        cannibalization_npv_loss=(cannibalization_loss / discount[None, :]).sum(axis=1),
        base_pv_development=(basis.development_base / discount[None, :]).sum(axis=1),
        rnd_pv_development_saving=(basis.follower_development / discount[None, :]).sum(axis=1),
        development_base=basis.development_base,
        follower_development=basis.follower_development,
    )


def legacy_net_contribution(config: dict, times: np.ndarray) -> np.ndarray:
    """Legacy contribution less fixed enterprise cost on the simulation grid."""

    dt = float(config["simulation"]["time_step_years"])
    legacy_revenue = float(config["enterprise"]["legacy_revenue_usd_m_per_year"])
    legacy_margin = float(config["enterprise"]["legacy_contribution_margin"])
    legacy_loe = float(config["enterprise"]["legacy_loe_year"])
    erosion = np.asarray(config["enterprise"]["legacy_remaining_fraction_by_year_after_loe"], dtype=float)
    terminal = float(config["enterprise"]["legacy_terminal_fraction"])
    fixed = float(config["enterprise"]["fixed_operating_cost_usd_m_per_year"]) * dt
    remaining = np.ones_like(times)
    after = times >= legacy_loe - 1e-10
    years_after = np.floor(np.maximum(times - legacy_loe, 0.0) + 1e-10).astype(int)
    remaining[after] = np.where(
        years_after[after] < len(erosion),
        erosion[np.minimum(years_after[after], len(erosion) - 1)],
        terminal,
    )
    return legacy_revenue * legacy_margin * remaining * dt - fixed


def capacity_requirement_for_gain(
    base_portfolio_cash: np.ndarray,
    gain_cash: np.ndarray,
    config: dict,
    capacity_name: str,
) -> np.ndarray:
    """Minimum nonnegative synergy fraction required to avoid a capacity breach.

    The result is exact for a nonnegative linear cash-flow gain.  Values above
    the searched synergy range remain useful because breach at synergy ``s`` is
    simply ``required_synergy > s``.
    """

    profile = config["enterprise"]["capacity_profiles"][capacity_name]
    base_oi = legacy_net_contribution(config, time_grid(config))[None, :] + base_portfolio_cash
    base_cumulative = np.cumsum(base_oi, axis=1)
    gain_cumulative = np.cumsum(gain_cash, axis=1)
    threshold = (
        float(profile["required_reserve_usd_m"])
        - float(profile["financing_capacity_usd_m"])
        - float(profile["starting_liquidity_usd_m"])
    )
    positive = gain_cumulative > 1e-12
    ratios = np.full_like(base_cumulative, -np.inf)
    np.divide(
        threshold - base_cumulative,
        gain_cumulative,
        out=ratios,
        where=positive,
    )
    impossible = np.any((~positive) & (base_cumulative < threshold - 1e-10), axis=1)
    required = np.maximum(np.max(ratios, axis=1), 0.0)
    required[impossible] = np.inf
    return required


def direct_funding_metrics(
    portfolio_cash: np.ndarray,
    config: dict,
    capacity_name: str,
) -> dict[str, np.ndarray]:
    profile = config["enterprise"]["capacity_profiles"][capacity_name]
    oi = legacy_net_contribution(config, time_grid(config))[None, :] + portfolio_cash
    starting = float(profile["starting_liquidity_usd_m"])
    cash = starting + np.cumsum(oi, axis=1)
    reserve = float(profile["required_reserve_usd_m"])
    financing = float(profile["financing_capacity_usd_m"])
    funding_need = np.maximum(reserve - cash.min(axis=1), 0.0)
    return {
        "funding_need": funding_need,
        "capacity_breach": funding_need > financing + 1e-10,
        "minimum_cash": cash.min(axis=1),
    }


def path_value_summary(npv: np.ndarray, pv_development: np.ndarray) -> dict[str, float]:
    p5 = float(np.quantile(npv, 0.05))
    return {
        "n_runs": int(len(npv)),
        "eNPV_usd_m": float(np.mean(npv)),
        "se_eNPV_usd_m": mc_mean_se(npv),
        "median_NPV_usd_m": float(np.median(npv)),
        "sd_NPV_usd_m": float(np.std(npv, ddof=1)),
        "P5_NPV_usd_m": p5,
        "P10_NPV_usd_m": float(np.quantile(npv, 0.10)),
        "CVaR5_NPV_usd_m": cvar_lower(npv, 0.05),
        "probability_NPV_negative": float(np.mean(npv < 0.0)),
        "eDevCost_usd_m": float(np.mean(pv_development)),
        "ePI": float(np.mean(npv) / np.mean(pv_development)),
    }


def launch_summary(launch_time: np.ndarray, loe_year: float, n_programs: int) -> dict[str, float]:
    launched = np.isfinite(launch_time)
    counts = launched.sum(axis=1)
    runway = np.where(launched, np.maximum(float(loe_year) - launch_time, 0.0), np.nan)
    before_loe = launched & (launch_time < float(loe_year) - 1e-10)
    binary_rho = mean_pairwise_binary_correlation(launched)
    marginal_variance = float(np.mean(np.var(launched.astype(float), axis=0, ddof=1)))
    launch_variance = float(np.var(counts, ddof=1))
    breadth = n_programs**2 * marginal_variance / launch_variance if launch_variance > 0 else float("nan")
    return {
        "mean_launches": float(np.mean(counts)),
        "sd_launches": float(np.std(counts, ddof=1)),
        "p_zero_launches": float(np.mean(counts == 0)),
        "mean_pairwise_launch_correlation": float(binary_rho),
        "effective_breadth_variance_ratio": float(breadth),
        "mean_protected_runway_per_launch_years": float(np.nanmean(runway)) if np.any(launched) else float("nan"),
        "mean_launches_before_loe": float(np.mean(before_loe.sum(axis=1))),
        "probability_at_least_one_launch_before_loe": float(np.mean(before_loe.any(axis=1))),
    }


def peak_annual_development(development_spend: np.ndarray, times: np.ndarray) -> np.ndarray:
    calendar_year = np.floor(times + 1e-10).astype(int)
    annual = np.stack(
        [development_spend[:, calendar_year == year].sum(axis=1) for year in np.unique(calendar_year)],
        axis=1,
    )
    return annual.max(axis=1)


def paired_mean_interval(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    diff = np.asarray(candidate, dtype=float) - np.asarray(reference, dtype=float)
    se = mc_mean_se(diff)
    mean = float(np.mean(diff))
    return {
        "delta_eNPV_usd_m": mean,
        "paired_se_delta_eNPV_usd_m": se,
        "paired_95ci_low_delta_eNPV_usd_m": mean - 1.96 * se,
        "paired_95ci_high_delta_eNPV_usd_m": mean + 1.96 * se,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
