"""Program and enterprise cash-flow calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .policy import Schedule


@dataclass
class EconomicPaths:
    times: np.ndarray
    development_spend: np.ndarray
    gross_new_contribution: np.ndarray
    commercial_spend: np.ndarray
    portfolio_cash_flow: np.ndarray
    portfolio_npv: np.ndarray
    pv_development_spend: np.ndarray
    realized_pi: np.ndarray
    launch_count: np.ndarray
    launch_time: np.ndarray
    program_revenue: np.ndarray


def _time_grid(config: dict) -> np.ndarray:
    dt = float(config["simulation"]["time_step_years"])
    horizon = float(config["simulation"]["horizon_years"])
    return np.arange(0.0, horizon + dt / 2.0, dt)


def _add_stage_cost(
    spend: np.ndarray,
    times: np.ndarray,
    start: np.ndarray,
    duration: float,
    cost: float,
    program: int,
) -> None:
    dt = float(times[1] - times[0])
    stage_start = np.asarray(start, dtype=float)
    active = (
        np.isfinite(stage_start)[:, None]
        & (times[None, :] >= stage_start[:, None] - 1e-10)
        & (times[None, :] < stage_start[:, None] + duration - 1e-10)
    )
    spend[:, program, :] += active * (cost * dt / duration)


def calculate_economics(
    schedule: Schedule,
    config: dict,
    rnd_cost_synergy: float = 0.0,
    commercial_cost_synergy: float = 0.0,
    cannibalization: float = 0.0,
    commercial_clustered: bool = False,
) -> EconomicPaths:
    times = _time_grid(config)
    dt = float(config["simulation"]["time_step_years"])
    n_runs, n_programs = schedule.launch_time.shape
    development_by_program = np.zeros((n_runs, n_programs, len(times)), dtype=float)
    revenue_by_program = np.zeros_like(development_by_program)
    commercial_by_program = np.zeros_like(development_by_program)

    for i in range(n_programs):
        follower_factor = 1.0 - float(rnd_cost_synergy) if i > 0 else 1.0
        for stage, starts in (
            ("phase2", schedule.phase2_start[:, i]),
            ("phase3", schedule.phase3_start[:, i]),
            ("regulatory", schedule.regulatory_start[:, i]),
        ):
            duration = float(config["development"][stage]["duration_years"])
            if stage == "phase3" and i > 0:
                nominal_start = float(config["development"]["phase2"]["duration_years"])
                observed = schedule.regulatory_start[:, i] - schedule.phase3_start[:, i]
                finite = observed[np.isfinite(observed)]
                if finite.size:
                    duration = float(np.median(finite))
                elif np.any(np.isfinite(starts)):
                    duration = max(dt, duration)
            stage_cost = float(config["development"][stage]["cost_usd_m"]) * follower_factor
            _add_stage_cost(development_by_program, times, starts, duration, stage_cost, i)

        launch = schedule.launch_time[:, i]
        peak = float(config["commercial"]["peak_sales_usd_m"])
        ramp = np.asarray(config["commercial"]["revenue_ramp"], dtype=float)
        support = float(config["commercial"]["support_cost_usd_m_per_year"])
        support_factor = 1.0 - float(commercial_cost_synergy) if i > 0 else 1.0
        loe = float(config["commercial"]["loe_year"])
        post_loe = float(config["commercial"]["post_loe_sales_fraction"])

        finite_launch = np.isfinite(launch)
        safe_launch = np.where(finite_launch, launch, np.inf)
        active = finite_launch[:, None] & (times[None, :] >= safe_launch[:, None] + 1.0 - 1e-10)
        elapsed = np.maximum(times[None, :] - (safe_launch[:, None] + 1.0), 0.0)
        commercial_year = np.floor(elapsed + 1e-10).astype(int)
        ramp_fraction = ramp[np.minimum(commercial_year, len(ramp) - 1)]
        loe_factor = np.where(times >= loe - 1e-10, post_loe, 1.0)
        revenue_by_program[:, i, :] = (
            peak * ramp_fraction * loe_factor[None, :] * dt * active
        )
        commercial_by_program[:, i, :] = support * support_factor * dt * active

    if commercial_clustered:
        total_revenue = revenue_by_program.sum(axis=1)
        max_revenue = revenue_by_program.max(axis=1)
        market_revenue = max_revenue + (1.0 - float(cannibalization)) * (total_revenue - max_revenue)
    else:
        market_revenue = revenue_by_program.sum(axis=1)

    gross_margin = float(config["commercial"]["gross_contribution_margin"])
    gross_contribution = market_revenue * gross_margin
    development_spend = development_by_program.sum(axis=1)
    commercial_spend = commercial_by_program.sum(axis=1)
    portfolio_cash = gross_contribution - development_spend - commercial_spend
    discount_rate = float(config["commercial"]["discount_rate"])
    discount = np.power(1.0 + discount_rate, times)
    portfolio_npv = (portfolio_cash / discount[None, :]).sum(axis=1)
    pv_dev = (development_spend / discount[None, :]).sum(axis=1)
    realized_pi = np.divide(portfolio_npv, pv_dev, out=np.full_like(portfolio_npv, np.nan), where=pv_dev > 0)

    return EconomicPaths(
        times=times,
        development_spend=development_spend,
        gross_new_contribution=gross_contribution,
        commercial_spend=commercial_spend,
        portfolio_cash_flow=portfolio_cash,
        portfolio_npv=portfolio_npv,
        pv_development_spend=pv_dev,
        realized_pi=realized_pi,
        launch_count=np.isfinite(schedule.launch_time).sum(axis=1),
        launch_time=schedule.launch_time,
        program_revenue=revenue_by_program,
    )


def enterprise_metrics(paths: EconomicPaths, config: dict, capacity_name: str) -> dict[str, np.ndarray]:
    profile = config["enterprise"]["capacity_profiles"][capacity_name]
    times = paths.times
    dt = float(config["simulation"]["time_step_years"])
    legacy_revenue = float(config["enterprise"]["legacy_revenue_usd_m_per_year"])
    legacy_margin = float(config["enterprise"]["legacy_contribution_margin"])
    legacy_loe = float(config["enterprise"]["legacy_loe_year"])
    erosion = np.asarray(config["enterprise"]["legacy_remaining_fraction_by_year_after_loe"], dtype=float)
    terminal = float(config["enterprise"]["legacy_terminal_fraction"])
    fixed_cost = float(config["enterprise"]["fixed_operating_cost_usd_m_per_year"])

    remaining = np.ones_like(times)
    after = times >= legacy_loe - 1e-10
    years_after = np.floor(np.maximum(times - legacy_loe, 0.0) + 1e-10).astype(int)
    remaining[after] = np.where(
        years_after[after] < len(erosion),
        erosion[np.minimum(years_after[after], len(erosion) - 1)],
        terminal,
    )
    legacy_contribution = legacy_revenue * legacy_margin * remaining * dt
    fixed = fixed_cost * dt
    oi = legacy_contribution[None, :] + paths.gross_new_contribution - paths.development_spend - paths.commercial_spend - fixed
    starting = float(profile["starting_liquidity_usd_m"])
    cash_pre = starting + np.cumsum(oi, axis=1)
    reserve = float(profile["required_reserve_usd_m"])
    funding_need = np.maximum(reserve - cash_pre.min(axis=1), 0.0)
    financing = float(profile["financing_capacity_usd_m"])
    breach = funding_need > financing + 1e-10
    discount = np.power(1.0 + float(config["commercial"]["discount_rate"]), times)
    enterprise_npv = (oi / discount[None, :]).sum(axis=1)
    negative_oi_years = (oi < 0).sum(axis=1) * dt
    first_breach_time = np.full(len(funding_need), np.nan)
    breach_path = cash_pre < (reserve - financing)
    for run in np.flatnonzero(breach_path.any(axis=1)):
        first_breach_time[run] = times[np.argmax(breach_path[run])]
    return {
        "enterprise_npv": enterprise_npv,
        "funding_need": funding_need,
        "capacity_breach": breach,
        "minimum_cash": cash_pre.min(axis=1),
        "negative_oi_years": negative_oi_years,
        "time_to_breach": first_breach_time,
    }
