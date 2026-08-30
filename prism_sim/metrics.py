"""Portfolio, tail, launch, and enterprise summary metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .economics import EconomicPaths, enterprise_metrics


def cvar_lower(values: np.ndarray, alpha: float = 0.05) -> float:
    threshold = float(np.quantile(values, alpha))
    return float(np.mean(values[values <= threshold]))


def mc_mean_se(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def _launch_metrics(paths: EconomicPaths, config: dict, target: int = 2) -> dict[str, float]:
    horizon = int(config["simulation"]["horizon_years"])
    windows = [(start, start + 3.0) for start in range(0, horizon - 2)]
    counts = np.zeros((len(paths.launch_count), len(windows)), dtype=float)
    for w, (start, end) in enumerate(windows):
        counts[:, w] = ((paths.launch_time >= start) & (paths.launch_time < end)).sum(axis=1)
    service = (counts >= target).mean(axis=0)
    shortfall = np.maximum(target - counts, 0.0)
    return {
        "launch_service_n2_descriptive": float(service.mean()),
        "mean_launch_shortfall_n2_descriptive": float(shortfall.mean()),
        "persistent_shortfall_probability_descriptive": float(
            np.mean(np.any((counts[:, :-1] < target) & (counts[:, 1:] < target), axis=1))
        ),
    }


def summarize_paths(paths: EconomicPaths, config: dict) -> dict[str, Any]:
    npv = paths.portfolio_npv
    launches = paths.launch_count.astype(float)
    p5 = float(np.quantile(npv, 0.05))
    downside = np.minimum(npv - (-1000.0), 0.0)
    calendar_year = np.floor(paths.times + 1e-10).astype(int)
    annual_rnd = np.stack(
        [paths.development_spend[:, calendar_year == year].sum(axis=1) for year in np.unique(calendar_year)],
        axis=1,
    )
    peak_annual_rnd = annual_rnd.max(axis=1)
    result: dict[str, Any] = {
        "n_runs": int(len(npv)),
        "mean_launches": float(np.mean(launches)),
        "se_mean_launches": mc_mean_se(launches),
        "sd_launches": float(np.std(launches, ddof=1)),
        "p_zero_launches": float(np.mean(launches == 0)),
        "eNPV_usd_m": float(np.mean(npv)),
        "se_eNPV_usd_m": mc_mean_se(npv),
        "median_NPV_usd_m": float(np.median(npv)),
        "sd_NPV_usd_m": float(np.std(npv, ddof=1)),
        "P5_NPV_usd_m": p5,
        "P10_NPV_usd_m": float(np.quantile(npv, 0.10)),
        "CVaR5_NPV_usd_m": cvar_lower(npv, 0.05),
        "probability_NPV_negative": float(np.mean(npv < 0)),
        "downside_semideviation_vs_minus_1B_usd_m": float(np.sqrt(np.mean(downside**2))),
        "eDevCost_usd_m": float(np.mean(paths.pv_development_spend)),
        "ePI": float(np.mean(npv) / np.mean(paths.pv_development_spend)),
        "median_realized_PI": float(np.nanmedian(paths.realized_pi)),
        "mean_peak_annual_RnD_usd_m": float(np.mean(peak_annual_rnd)),
    }
    result.update(_launch_metrics(paths, config))
    for capacity_name in config["enterprise"]["capacity_profiles"]:
        enterprise = enterprise_metrics(paths, config, capacity_name)
        result[f"{capacity_name}_capacity_breach_probability"] = float(np.mean(enterprise["capacity_breach"]))
        result[f"{capacity_name}_mean_funding_need_usd_m"] = float(np.mean(enterprise["funding_need"]))
        result[f"{capacity_name}_P95_funding_need_usd_m"] = float(np.quantile(enterprise["funding_need"], 0.95))
        result[f"{capacity_name}_mean_minimum_cash_usd_m"] = float(np.mean(enterprise["minimum_cash"]))
        result[f"{capacity_name}_mean_negative_OI_years"] = float(np.mean(enterprise["negative_oi_years"]))
        result[f"{capacity_name}_mean_enterprise_NPV_usd_m"] = float(np.mean(enterprise["enterprise_npv"]))
    return result


def implied_binary_correlation(probability: float, latent_rho: float, n_draws: int = 500000) -> float:
    """Deterministic numerical approximation used only for reporting effective breadth."""
    from scipy.stats import multivariate_normal, norm

    threshold = norm.ppf(probability)
    joint = multivariate_normal.cdf([threshold, threshold], mean=[0.0, 0.0], cov=[[1.0, latent_rho], [latent_rho, 1.0]])
    return float((joint - probability**2) / (probability * (1.0 - probability)))


def effective_breadth(n_programs: int, binary_correlation: float) -> float:
    return float(n_programs / (1.0 + (n_programs - 1.0) * binary_correlation))


def mean_pairwise_binary_correlation(binary_outcomes: np.ndarray) -> float:
    binary = np.asarray(binary_outcomes, dtype=float)
    if binary.ndim != 2 or binary.shape[1] < 2:
        return float("nan")
    correlations = []
    for i in range(binary.shape[1]):
        for j in range(i + 1, binary.shape[1]):
            if np.std(binary[:, i]) == 0 or np.std(binary[:, j]) == 0:
                continue
            correlations.append(float(np.corrcoef(binary[:, i], binary[:, j])[0, 1]))
    return float(np.mean(correlations)) if correlations else float("nan")
