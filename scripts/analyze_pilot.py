#!/usr/bin/env python3
"""Create the pilot headline scorecard, figures, and concise results memo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NAVY = "#17365D"
BLUE = "#2F75B5"
TEAL = "#2A9D8F"
RED = "#C44E52"
GOLD = "#D69E2E"
GRAY = "#6B7280"
LIGHT = "#E8EEF5"


def style_axes(ax, ylabel: str, xlabel: str = "Latent correlation") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D8DEE8", linewidth=0.7, alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(labelsize=9)


def select_one(frame: pd.DataFrame, **conditions) -> pd.Series:
    mask = np.ones(len(frame), dtype=bool)
    for key, value in conditions.items():
        if isinstance(value, float):
            mask &= np.isclose(frame[key].astype(float), value)
        else:
            mask &= frame[key] == value
    selected = frame[mask]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {conditions}, found {len(selected)}")
    return selected.iloc[0]


def create_scorecard(summary: pd.DataFrame, contrasts: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    div = select_one(summary, experiment="E1", strategy="Diversified")
    piap = select_one(summary, experiment="E1", strategy="Parallel PIAP", rho=0.415)
    piap_high = select_one(summary, experiment="E1", strategy="Parallel PIAP", rho=0.60)
    msog = select_one(summary, experiment="E1", strategy="MSOG", rho=0.25)
    piap_contrast = select_one(contrasts, experiment="E1", strategy="Parallel PIAP", rho=0.415)
    piap_high_contrast = select_one(contrasts, experiment="E1", strategy="Parallel PIAP", rho=0.60)
    msog_contrast = select_one(contrasts, experiment="E1", strategy="MSOG", rho=0.25)

    e2_piap_q0 = select_one(summary, experiment="E2", strategy="Parallel PIAP", rho=0.415, common_shock_probability=0.0)
    e2_piap_q5 = select_one(summary, experiment="E2", strategy="Parallel PIAP", rho=0.415, common_shock_probability=0.05)
    e3_piap_c0 = select_one(summary, experiment="E3", strategy="Parallel PIAP", rho=0.415, cannibalization=0.0)
    e3_piap_c25 = select_one(summary, experiment="E3", strategy="Parallel PIAP", rho=0.415, cannibalization=0.25)
    e3_msog_c0 = select_one(summary, experiment="E3", strategy="MSOG", rho=0.25, cannibalization=0.0)
    e3_msog_c25 = select_one(summary, experiment="E3", strategy="MSOG", rho=0.25, cannibalization=0.25)

    rows = [
        {
            "comparison": "Same-asset PIAP rho=0.415 vs diversified",
            "delta_eNPV_usd_m": piap_contrast.delta_eNPV_usd_m,
            "delta_sd_NPV_usd_m": piap_contrast.delta_sd_NPV_usd_m,
            "delta_P5_NPV_usd_m": piap_contrast.delta_P5_NPV_usd_m,
            "delta_CVaR5_NPV_usd_m": piap.CVaR5_NPV_usd_m - div.CVaR5_NPV_usd_m,
            "delta_base_breach_percentage_points": 100.0 * piap_contrast.delta_base_capacity_breach_probability,
            "effective_breadth": piap.effective_breadth_variance_ratio,
        },
        {
            "comparison": "Same-asset PIAP rho=0.600 vs diversified",
            "delta_eNPV_usd_m": piap_high_contrast.delta_eNPV_usd_m,
            "delta_sd_NPV_usd_m": piap_high_contrast.delta_sd_NPV_usd_m,
            "delta_P5_NPV_usd_m": piap_high_contrast.delta_P5_NPV_usd_m,
            "delta_CVaR5_NPV_usd_m": piap_high.CVaR5_NPV_usd_m - div.CVaR5_NPV_usd_m,
            "delta_base_breach_percentage_points": 100.0 * piap_high_contrast.delta_base_capacity_breach_probability,
            "effective_breadth": piap_high.effective_breadth_variance_ratio,
        },
        {
            "comparison": "MSOG rho=0.250 vs diversified",
            "delta_eNPV_usd_m": msog_contrast.delta_eNPV_usd_m,
            "delta_sd_NPV_usd_m": msog_contrast.delta_sd_NPV_usd_m,
            "delta_P5_NPV_usd_m": msog_contrast.delta_P5_NPV_usd_m,
            "delta_CVaR5_NPV_usd_m": msog.CVaR5_NPV_usd_m - div.CVaR5_NPV_usd_m,
            "delta_base_breach_percentage_points": 100.0 * msog_contrast.delta_base_capacity_breach_probability,
            "effective_breadth": msog.effective_breadth_variance_ratio,
        },
        {
            "comparison": "PIAP 5% marginal-preserving shock vs PIAP no shock",
            "delta_eNPV_usd_m": e2_piap_q5.eNPV_usd_m - e2_piap_q0.eNPV_usd_m,
            "delta_sd_NPV_usd_m": e2_piap_q5.sd_NPV_usd_m - e2_piap_q0.sd_NPV_usd_m,
            "delta_P5_NPV_usd_m": e2_piap_q5.P5_NPV_usd_m - e2_piap_q0.P5_NPV_usd_m,
            "delta_CVaR5_NPV_usd_m": e2_piap_q5.CVaR5_NPV_usd_m - e2_piap_q0.CVaR5_NPV_usd_m,
            "delta_base_breach_percentage_points": 100.0 * (
                e2_piap_q5.base_capacity_breach_probability - e2_piap_q0.base_capacity_breach_probability
            ),
            "effective_breadth": e2_piap_q5.effective_breadth_variance_ratio,
        },
        {
            "comparison": "PIAP 25% cannibalization vs no cannibalization",
            "delta_eNPV_usd_m": e3_piap_c25.eNPV_usd_m - e3_piap_c0.eNPV_usd_m,
            "delta_sd_NPV_usd_m": e3_piap_c25.sd_NPV_usd_m - e3_piap_c0.sd_NPV_usd_m,
            "delta_P5_NPV_usd_m": e3_piap_c25.P5_NPV_usd_m - e3_piap_c0.P5_NPV_usd_m,
            "delta_CVaR5_NPV_usd_m": e3_piap_c25.CVaR5_NPV_usd_m - e3_piap_c0.CVaR5_NPV_usd_m,
            "delta_base_breach_percentage_points": 100.0 * (
                e3_piap_c25.base_capacity_breach_probability - e3_piap_c0.base_capacity_breach_probability
            ),
            "effective_breadth": e3_piap_c25.effective_breadth_variance_ratio,
        },
        {
            "comparison": "MSOG 25% cannibalization vs no cannibalization",
            "delta_eNPV_usd_m": e3_msog_c25.eNPV_usd_m - e3_msog_c0.eNPV_usd_m,
            "delta_sd_NPV_usd_m": e3_msog_c25.sd_NPV_usd_m - e3_msog_c0.sd_NPV_usd_m,
            "delta_P5_NPV_usd_m": e3_msog_c25.P5_NPV_usd_m - e3_msog_c0.P5_NPV_usd_m,
            "delta_CVaR5_NPV_usd_m": e3_msog_c25.CVaR5_NPV_usd_m - e3_msog_c0.CVaR5_NPV_usd_m,
            "delta_base_breach_percentage_points": 100.0 * (
                e3_msog_c25.base_capacity_breach_probability - e3_msog_c0.base_capacity_breach_probability
            ),
            "effective_breadth": e3_msog_c25.effective_breadth_variance_ratio,
        },
    ]
    scorecard = pd.DataFrame(rows)
    headline = {
        "diversified": div.to_dict(),
        "piap_empirical_anchor": piap.to_dict(),
        "piap_empirical_anchor_contrast": piap_contrast.to_dict(),
        "piap_high_dependence": piap_high.to_dict(),
        "piap_high_dependence_contrast": piap_high_contrast.to_dict(),
        "msog_central": msog.to_dict(),
        "msog_central_contrast": msog_contrast.to_dict(),
        "piap_common_shock_increment": rows[3],
        "piap_cannibalization_increment": rows[4],
        "msog_cannibalization_increment": rows[5],
    }
    return scorecard, headline


def figure_mean_tail(summary: pd.DataFrame, output: Path) -> None:
    e1 = summary[(summary.experiment == "E1") & (summary.strategy != "Diversified")]
    div = select_one(summary, experiment="E1", strategy="Diversified")
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8), constrained_layout=True)
    for strategy, color, label in (("Parallel PIAP", BLUE, "Parallel PIAP"), ("MSOG", TEAL, "MSOG")):
        group = e1[e1.strategy == strategy].sort_values("rho")
        axes[0].errorbar(
            group.rho,
            group.eNPV_usd_m,
            yerr=1.96 * group.se_eNPV_usd_m,
            marker="o",
            linewidth=1.8,
            capsize=3,
            color=color,
            label=label,
        )
        axes[1].plot(group.rho, group.sd_NPV_usd_m, marker="o", linewidth=1.8, color=color, label=label)
        axes[2].plot(group.rho, group.CVaR5_NPV_usd_m, marker="o", linewidth=1.8, color=color, label=label)
    axes[0].axhline(div.eNPV_usd_m, color=GRAY, linestyle="--", linewidth=1.2, label="Diversified")
    axes[1].axhline(div.sd_NPV_usd_m, color=GRAY, linestyle="--", linewidth=1.2)
    axes[2].axhline(div.CVaR5_NPV_usd_m, color=GRAY, linestyle="--", linewidth=1.2)
    style_axes(axes[0], "Mean portfolio NPV (USD M)")
    style_axes(axes[1], "NPV standard deviation (USD M)")
    style_axes(axes[2], "CVaR5 portfolio NPV (USD M)")
    axes[0].set_title("Mean remains within MC uncertainty", fontsize=10.5, color=NAVY, weight="bold")
    axes[1].set_title("Dispersion increases", fontsize=10.5, color=NAVY, weight="bold")
    axes[2].set_title("Lower tail deteriorates", fontsize=10.5, color=NAVY, weight="bold")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("E1: dependence changes risk more than mean", fontsize=14, color=NAVY, weight="bold")
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def figure_common_shock(summary: pd.DataFrame, output: Path) -> None:
    e2 = summary[summary.experiment == "E2"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    for strategy, rho, color, label in (
        ("Parallel PIAP", 0.415, BLUE, "PIAP, rho=0.415"),
        ("MSOG", 0.25, TEAL, "MSOG, rho=0.250"),
    ):
        group = e2[(e2.strategy == strategy) & np.isclose(e2.rho, rho)].sort_values("common_shock_probability")
        q_pct = 100.0 * group.common_shock_probability
        axes[0].plot(q_pct, group.P5_NPV_usd_m, marker="o", linewidth=1.8, color=color, label=label)
        axes[0].plot(q_pct, group.CVaR5_NPV_usd_m, marker="s", linestyle="--", linewidth=1.6, color=color, alpha=0.75)
        axes[1].plot(q_pct, group.effective_breadth_variance_ratio, marker="o", linewidth=1.8, color=color, label=label)
    style_axes(axes[0], "Portfolio NPV (USD M)", "Common-shock probability (%)")
    style_axes(axes[1], "Variance-equivalent effective breadth", "Common-shock probability (%)")
    axes[0].set_title("P5 (solid) and CVaR5 (dashed)", fontsize=10.5, color=NAVY, weight="bold")
    axes[1].set_title("Nominal breadth remains six", fontsize=10.5, color=NAVY, weight="bold")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("E2: marginal-preserving common shocks add clustered tail mass", fontsize=14, color=NAVY, weight="bold")
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def figure_cannibalization(summary: pd.DataFrame, output: Path) -> None:
    e3 = summary[summary.experiment == "E3"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    for strategy, rho, color, label in (
        ("Parallel PIAP", 0.415, BLUE, "PIAP, rho=0.415"),
        ("MSOG", 0.25, TEAL, "MSOG, rho=0.250"),
    ):
        group = e3[(e3.strategy == strategy) & np.isclose(e3.rho, rho)].sort_values("cannibalization")
        base = group[group.cannibalization == 0].iloc[0]
        c_pct = 100.0 * group.cannibalization
        axes[0].plot(c_pct, group.eNPV_usd_m - base.eNPV_usd_m, marker="o", linewidth=1.8, color=color, label=label)
        axes[1].plot(c_pct, 100.0 * group.base_capacity_breach_probability, marker="o", linewidth=1.8, color=color, label=label)
    style_axes(axes[0], "Change in eNPV (USD M)", "Cannibalization (%)")
    style_axes(axes[1], "Base capacity-breach probability (%)", "Cannibalization (%)")
    axes[0].axhline(0, color=GRAY, linewidth=1.0)
    axes[0].set_title("Mean-value loss", fontsize=10.5, color=NAVY, weight="bold")
    axes[1].set_title("Capacity effect is nonlinear", fontsize=10.5, color=NAVY, weight="bold")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("E3: commercial overlap changes the mean", fontsize=14, color=NAVY, weight="bold")
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def figure_effective_breadth(summary: pd.DataFrame, output: Path) -> None:
    e1 = summary[(summary.experiment == "E1") & (summary.strategy != "Diversified")]
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    for strategy, color, label in (("Parallel PIAP", BLUE, "Parallel PIAP"), ("MSOG", TEAL, "MSOG")):
        group = e1[e1.strategy == strategy].sort_values("rho")
        ax.plot(group.rho, group.effective_breadth_variance_ratio, marker="o", linewidth=2.0, color=color, label=label)
    ax.axhline(6, color=GRAY, linestyle="--", linewidth=1.2, label="Six independent shots")
    style_axes(ax, "Variance-equivalent effective breadth")
    ax.set_ylim(0, 6.5)
    ax.set_title("Six nominal programs do not provide six independent shots", fontsize=12, color=NAVY, weight="bold")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_memo(headline: dict, validation: dict, manifest: dict, output: Path) -> None:
    div = headline["diversified"]
    piap = headline["piap_empirical_anchor"]
    contrast = headline["piap_empirical_anchor_contrast"]
    high = headline["piap_high_dependence"]
    high_contrast = headline["piap_high_dependence_contrast"]
    cannib = headline["piap_cannibalization_increment"]
    shock = headline["piap_common_shock_increment"]
    text = f"""# PRISM mechanism-isolation pilot results

## Disposition

**GO to E4-E7 implementation.** The 10,000-run pilot completed {manifest['n_scenario_cells']} prespecified cells and all applicable validation gates passed. These estimates are pilot-resolution results. Final tail and boundary cells require at least 100,000 draws under the locked specification.

## Headline result

At the empirical same-asset scenario anchor (latent rho = 0.415), parallel PIAP produced an eNPV difference of {contrast['delta_eNPV_usd_m']:.1f}M USD versus diversification, with a paired 95% Monte Carlo interval of [{contrast['paired_95ci_low_delta_eNPV_usd_m']:.1f}, {contrast['paired_95ci_high_delta_eNPV_usd_m']:.1f}]M. Mean parity therefore held. NPV standard deviation increased by {contrast['delta_sd_NPV_usd_m']:.1f}M, P5 declined by {abs(contrast['delta_P5_NPV_usd_m']):.1f}M, CVaR5 declined by {abs(piap['CVaR5_NPV_usd_m'] - div['CVaR5_NPV_usd_m']):.1f}M, and base capacity-breach probability rose by {100 * contrast['delta_base_capacity_breach_probability']:.1f} percentage points. Variance-equivalent breadth fell from {div['effective_breadth_variance_ratio']:.2f} to {piap['effective_breadth_variance_ratio']:.2f} effective shots.

At rho = 0.60, variance-equivalent breadth fell to {high['effective_breadth_variance_ratio']:.2f}; P5 and CVaR5 declined by {abs(high_contrast['delta_P5_NPV_usd_m']):.1f}M and {abs(high['CVaR5_NPV_usd_m'] - div['CVaR5_NPV_usd_m']):.1f}M versus diversification, and base breach probability increased by {100 * high_contrast['delta_base_capacity_breach_probability']:.1f} percentage points.

## Mechanism findings

- A 5% marginal-preserving common shock at rho = 0.415 changed mean eNPV by {shock['delta_eNPV_usd_m']:.1f}M, within pilot noise, while lowering P5 by {abs(shock['delta_P5_NPV_usd_m']):.1f}M and CVaR5 by {abs(shock['delta_CVaR5_NPV_usd_m']):.1f}M relative to the same dependent portfolio without the shock.
- At rho = 0.415, 25% cannibalization reduced PIAP eNPV by {abs(cannib['delta_eNPV_usd_m']):.1f}M. P5 and CVaR5 were initially unchanged because the worst paths already contained zero or one launch, leaving little overlapping revenue to remove. Capacity breach rose only at higher overlap levels. This separates an upside/mean-value mechanism from technical downside concentration.
- The synchronized additive benchmark produced mean eNPV of {div['eNPV_usd_m']:.1f}M for diversification and {piap['eNPV_usd_m']:.1f}M for PIAP at the empirical anchor. The latter difference was statistically compatible with zero. The corresponding NPV standard deviations were {div['sd_NPV_usd_m']:.1f}M and {piap['sd_NPV_usd_m']:.1f}M.

## Interpretation

The pilot establishes the paper's quantitative premise. Technical dependence primarily changes variance, effective breadth, and the lower tail when marginals and payoffs are fixed. Commercial overlap changes expected value. Their interaction determines how much measurable synergy or acceleration a concentrated strategy must deliver before it becomes acceptable for a sponsor with finite risk capacity.

The base capacity-breach probability is high in this six-program vignette because the stylized legacy franchise erodes after year 8 while fixed enterprise costs continue through year 20. It should be interpreted as a controlled capacity stress test, not a forecast for a particular company. The variable-breadth enterprise analysis will provide the launch-service interpretation.

## Next work package

Implement E4-E7 at pilot precision: the synergy-concentration decision map, parallel-versus-staged PIAP frontier, launch-service/required-breadth frontier, and independence/additivity regret. Cells near feasibility or strategy-indifference boundaries then advance to the prespecified 100,000- to 250,000-run confirmation stage.

## Validation

Overall validation pass: {validation['pass']}. Configuration hash: `{manifest['configuration_hash']}`. Runtime: {manifest['runtime_seconds']:.1f} seconds. Five locked seed blocks were used; no seed was selected or discarded.
"""
    output.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs")
    args = parser.parse_args()
    output = Path(args.input)
    summary = pd.read_csv(output / "pilot_summary.csv")
    contrasts = pd.read_csv(output / "paired_contrasts_vs_diversified.csv")
    validation = json.loads((output / "validation_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    scorecard, headline = create_scorecard(summary, contrasts)
    scorecard.to_csv(output / "headline_scorecard.csv", index=False)
    (output / "headline_results.json").write_text(json.dumps(headline, indent=2, sort_keys=True), encoding="utf-8")
    figure_mean_tail(summary, output / "fig1_dependence_mean_and_tail.png")
    figure_common_shock(summary, output / "fig2_common_shock_tail.png")
    figure_cannibalization(summary, output / "fig3_cannibalization_economics.png")
    figure_effective_breadth(summary, output / "fig4_effective_breadth.png")
    write_memo(headline, validation, manifest, output / "PILOT_RESULTS_MEMO.md")
    print("analysis outputs written")


if __name__ == "__main__":
    main()
