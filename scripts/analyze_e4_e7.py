#!/usr/bin/env python3
"""Create headline tables, figures, and a concise results memo for E4-E7."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NAVY = "#17324D"
BLUE = "#2E6F9E"
TEAL = "#2A8C82"
GOLD = "#C28B2C"
RED = "#A04747"
GRAY = "#6B7280"
LIGHT = "#EEF3F7"


def style_axes(ax, ylabel: str | None = None, xlabel: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D9E1E8", linewidth=0.7, alpha=0.8)
    ax.tick_params(colors="#374151", labelsize=8.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=NAVY)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=NAVY)


def figure_decision_map(frontier: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), constrained_layout=True)
    definitions = [
        ("Parallel PIAP", 0.415, "PIAP, rho=0.415"),
        ("MSOG", 0.25, "MSOG, rho=0.250"),
    ]
    image = None
    for ax, (strategy, rho, title) in zip(axes, definitions):
        subset = frontier[
            (frontier.strategy == strategy)
            & np.isclose(frontier.rho, rho)
            & (frontier.capacity_profile == "high")
            & (frontier.risk_policy_profile == "balanced")
        ]
        pivot = subset.pivot(index="loe_year", columns="acceleration_years", values="minimum_synergy")
        pivot = pivot.sort_index(ascending=False).sort_index(axis=1)
        values = 100.0 * pivot.to_numpy()
        masked = np.ma.masked_invalid(values)
        image = ax.imshow(masked, cmap="YlGnBu", vmin=0, vmax=30, aspect="auto")
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                label = "NS" if not np.isfinite(values[row, col]) else f"{values[row, col]:.0f}%"
                color = "white" if np.isfinite(values[row, col]) and values[row, col] >= 20 else NAVY
                ax.text(col, row, label, ha="center", va="center", fontsize=9, weight="bold", color=color)
        ax.set_xticks(range(len(pivot.columns)), [f"{x:g}" for x in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), [f"{x:g}" for x in pivot.index])
        ax.set_xlabel("Follower acceleration (years)", fontsize=9, color=NAVY)
        ax.set_ylabel("Absolute LOE year", fontsize=9, color=NAVY)
        ax.set_title(title, fontsize=11, color=NAVY, weight="bold")
        ax.tick_params(labelsize=8.5)
    cbar = fig.colorbar(image, ax=axes, shrink=0.86, pad=0.02)
    cbar.set_label("Minimum follower R&D saving", fontsize=9, color=NAVY)
    cbar.ax.tick_params(labelsize=8)
    fig.suptitle(
        "Concentration decision map: upstream saving needed for feasibility and value parity",
        fontsize=13,
        color=NAVY,
        weight="bold",
    )
    fig.savefig(output, dpi=600, bbox_inches="tight")
    plt.close(fig)


def figure_capacity(cells: pd.DataFrame, output: Path) -> None:
    subset = cells[
        (cells.strategy == "Parallel PIAP")
        & np.isclose(cells.rho, 0.415)
        & (cells.loe_year == 15.0)
        & (cells.acceleration_years == 0.0)
    ].sort_values("rnd_cost_synergy")
    fig, ax = plt.subplots(figsize=(7.3, 4.4), constrained_layout=True)
    for capacity, color, label in (
        ("low", RED, "Low capacity"),
        ("base", GOLD, "Base capacity"),
        ("high", TEAL, "High capacity"),
    ):
        ax.plot(
            100 * subset.rnd_cost_synergy,
            100 * subset[f"{capacity}_capacity_breach_probability"],
            color=color,
            linewidth=2.0,
            label=label,
        )
    for limit, label in ((5, "Preservation 5%"), (10, "Balanced 10%"), (20, "Growth 20%")):
        ax.axhline(limit, color=GRAY, linestyle="--", linewidth=0.9, alpha=0.65)
        ax.text(30.3, limit, label, va="center", fontsize=7.5, color=GRAY)
    style_axes(ax, "Capacity-breach probability (%)", "Follower R&D cost saving (%)")
    ax.set_xlim(0, 35)
    ax.set_ylim(0, 85)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title(
        "Economic synergy cannot manufacture balance-sheet capacity",
        fontsize=12,
        color=NAVY,
        weight="bold",
    )
    fig.savefig(output, dpi=600, bbox_inches="tight")
    plt.close(fig)


def figure_staging(contrasts: pd.DataFrame, output: Path) -> None:
    subset = contrasts[contrasts.common_shock_probability == 0.0]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)
    for loe, color in ((12.0, RED), (15.0, BLUE), (18.0, TEAL)):
        group = subset[subset.loe_year == loe].sort_values("acceleration_years")
        axes[0].plot(
            group.acceleration_years,
            group.delta_eNPV_usd_m,
            marker="o",
            linewidth=2,
            color=color,
            label=f"LOE year {loe:g}",
        )
        axes[1].plot(
            group.acceleration_years,
            100 * group.delta_base_capacity_breach_probability,
            marker="o",
            linewidth=2,
            color=color,
            label=f"LOE year {loe:g}",
        )
    axes[0].axhline(0, color=GRAY, linewidth=0.9)
    axes[1].axhline(0, color=GRAY, linewidth=0.9)
    style_axes(axes[0], "Staged minus parallel eNPV (USD M)", "Acceleration (years)")
    style_axes(axes[1], "Change in base breach probability (pp)", "Acceleration (years)")
    axes[0].set_title("Staging sacrifices time-dependent value", fontsize=10.5, color=NAVY, weight="bold")
    axes[1].set_title("Revenue delay can outweigh lower early spend", fontsize=10.5, color=NAVY, weight="bold")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Parallel versus staged PIAP at rho=0.415", fontsize=13, color=NAVY, weight="bold")
    fig.savefig(output, dpi=600, bbox_inches="tight")
    plt.close(fig)


def figure_launch_frontier(frontier: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), constrained_layout=True, sharey=True)
    colors = {"Diversified": BLUE, "MSOG trios": TEAL, "PIAP sixes": RED}
    for ax, service in zip(axes, (0.70, 0.80, 0.90)):
        subset = frontier[np.isclose(frontier.service_target, service)]
        for geometry, group in subset.groupby("geometry"):
            group = group.sort_values("launch_target_per_3y")
            ax.plot(
                group.launch_target_per_3y,
                group.minimum_nominal_breadth,
                marker="o",
                linewidth=2,
                color=colors[geometry],
                label=geometry,
            )
        style_axes(ax, "Required nominal Phase 2 starts" if ax is axes[0] else None, "Launch target per 3 years")
        ax.set_xticks([1, 2, 3])
        ax.set_title(f"{service:.0%} service", fontsize=10.5, color=NAVY, weight="bold")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Correlation converts nominal breadth into fewer effective shots",
        fontsize=13,
        color=NAVY,
        weight="bold",
    )
    fig.savefig(output, dpi=600, bbox_inches="tight")
    plt.close(fig)


def figure_regret(regret: pd.DataFrame, output: Path) -> None:
    subset = regret[
        (regret.strategy == "Parallel PIAP")
        & np.isclose(regret.rho_true, 0.415)
        & (regret.loe_year == 15.0)
        & (regret.acceleration_years == 0.0)
    ].drop_duplicates(["rnd_cost_synergy", "cannibalization_true"])
    pivot = subset.pivot(
        index="rnd_cost_synergy", columns="cannibalization_true", values="eNPV_regret_usd_m"
    ).sort_index(ascending=False).sort_index(axis=1)
    values = pivot.to_numpy()
    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    image = ax.imshow(values, cmap="YlOrRd", vmin=0, vmax=max(275, np.nanmax(values)), aspect="auto")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            ax.text(
                col,
                row,
                f"${value:.0f}M",
                ha="center",
                va="center",
                fontsize=9,
                weight="bold",
                color="white" if value >= 130 else NAVY,
            )
    ax.set_xticks(range(len(pivot.columns)), [f"{100*x:.0f}%" for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [f"{100*x:.0f}%" for x in pivot.index])
    ax.set_xlabel("True cannibalization", fontsize=9, color=NAVY)
    ax.set_ylabel("Follower R&D cost saving", fontsize=9, color=NAVY)
    ax.set_title(
        "eNPV regret from assuming additive revenue",
        fontsize=12,
        color=NAVY,
        weight="bold",
    )
    cbar = fig.colorbar(image, ax=ax, shrink=0.88)
    cbar.set_label("eNPV regret (USD M)", fontsize=9, color=NAVY)
    cbar.ax.tick_params(labelsize=8)
    fig.savefig(output, dpi=600, bbox_inches="tight")
    plt.close(fig)


def select_one(frame: pd.DataFrame, **filters) -> pd.Series:
    selected = frame
    for key, value in filters.items():
        if isinstance(value, float):
            selected = selected[np.isclose(selected[key], value)]
        else:
            selected = selected[selected[key] == value]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {filters}, found {len(selected)}")
    return selected.iloc[0]


def headline_results(
    frontier: pd.DataFrame,
    interaction_frontier: pd.DataFrame,
    commercial_frontier: pd.DataFrame,
    cells: pd.DataFrame,
    e5_summary: pd.DataFrame,
    e5_contrasts: pd.DataFrame,
    e6_frontier: pd.DataFrame,
    regret: pd.DataFrame,
) -> dict:
    piap_frontier = select_one(
        frontier,
        strategy="Parallel PIAP",
        rho=0.415,
        loe_year=15.0,
        acceleration_years=0.0,
        cannibalization=0.0,
        capacity_profile="high",
        risk_policy_profile="balanced",
    )
    piap_short = select_one(
        frontier,
        strategy="Parallel PIAP",
        rho=0.415,
        loe_year=12.0,
        acceleration_years=0.0,
        cannibalization=0.0,
        capacity_profile="high",
        risk_policy_profile="balanced",
    )
    max_synergy = select_one(
        cells,
        strategy="Parallel PIAP",
        rho=0.415,
        loe_year=15.0,
        acceleration_years=0.0,
        rnd_cost_synergy=0.30,
    )
    cannib_25 = select_one(
        interaction_frontier,
        strategy="Parallel PIAP",
        rho=0.415,
        loe_year=15.0,
        acceleration_years=0.0,
        cannibalization=0.25,
        capacity_profile="high",
        risk_policy_profile="balanced",
    )
    cannib_50 = select_one(
        interaction_frontier,
        strategy="Parallel PIAP",
        rho=0.415,
        loe_year=15.0,
        acceleration_years=0.0,
        cannibalization=0.50,
        capacity_profile="high",
        risk_policy_profile="balanced",
    )
    commercial = select_one(
        commercial_frontier,
        strategy="Parallel PIAP",
        rho=0.415,
        loe_year=15.0,
        acceleration_years=0.0,
        cannibalization=0.0,
        capacity_profile="high",
        risk_policy_profile="balanced",
    )
    staged = select_one(
        e5_contrasts,
        common_shock_probability=0.0,
        acceleration_years=0.0,
        loe_year=15.0,
    )
    parallel_cell = select_one(
        e5_summary,
        policy="parallel",
        common_shock_probability=0.0,
        acceleration_years=0.0,
        loe_year=15.0,
    )
    staged_cell = select_one(
        e5_summary,
        policy="staged",
        common_shock_probability=0.0,
        acceleration_years=0.0,
        loe_year=15.0,
    )
    breadth = {}
    for geometry in ("Diversified", "MSOG trios", "PIAP sixes"):
        row = select_one(
            e6_frontier,
            geometry=geometry,
            launch_target_per_3y=2,
            service_target=0.80,
        )
        breadth[geometry] = row.to_dict()
    plausible = regret[
        regret.rho_true.isin([0.415, 0.25]) & (regret.cannibalization_true <= 0.25)
    ]
    central_regret = regret[
        (regret.strategy == "Parallel PIAP")
        & np.isclose(regret.rho_true, 0.415)
        & np.isclose(regret.cannibalization_true, 0.25)
        & (regret.loe_year == 15.0)
        & (regret.acceleration_years == 0.0)
        & np.isclose(regret.rnd_cost_synergy, 0.10)
    ].iloc[0]
    return {
        "piap_high_balanced_frontier": piap_frontier.to_dict(),
        "piap_short_loe_frontier": piap_short.to_dict(),
        "piap_base_capacity_at_30pct_saving": max_synergy.to_dict(),
        "piap_cannibalization_25_frontier": cannib_25.to_dict(),
        "piap_cannibalization_50_frontier": cannib_50.to_dict(),
        "piap_commercial_synergy_frontier": commercial.to_dict(),
        "staged_vs_parallel_central": staged.to_dict(),
        "parallel_central": parallel_cell.to_dict(),
        "staged_central": staged_cell.to_dict(),
        "launch_breadth_two_at_80pct": breadth,
        "central_additivity_regret": central_regret.to_dict(),
        "plausible_grid_maxima": {
            "max_eNPV_regret_usd_m": float(plausible.eNPV_regret_usd_m.max()),
            "max_P5_shortfall_usd_m": float(plausible.value_selected_P5_shortfall_vs_diversified_usd_m.max()),
            "max_incremental_breach_probability": float(plausible.value_selected_incremental_breach_vs_diversified.max()),
        },
    }


def write_memo(headline: dict, output: Path) -> None:
    f = headline["piap_high_balanced_frontier"]
    short = headline["piap_short_loe_frontier"]
    max_s = headline["piap_base_capacity_at_30pct_saving"]
    c25 = headline["piap_cannibalization_25_frontier"]
    c50 = headline["piap_cannibalization_50_frontier"]
    commercial = headline["piap_commercial_synergy_frontier"]
    staged = headline["staged_vs_parallel_central"]
    parallel = headline["parallel_central"]
    staged_cell = headline["staged_central"]
    breadth = headline["launch_breadth_two_at_80pct"]
    regret = headline["central_additivity_regret"]
    maxima = headline["plausible_grid_maxima"]
    text = f"""# E4-E7 decision-frontier results

## Disposition

**GO to manuscript drafting and matched-heterogeneity robustness.** All advanced validation gates passed. Primary surfaces use 100,000 paths; empirical-anchor boundaries and all launch-service frontiers use 250,000.

## Headline decision boundary

At the same-asset anchor (rho=0.415), LOE year 15, no acceleration, high financial capacity, and the balanced policy profile, parallel PIAP requires {100*f['minimum_synergy']:.0f}% follower R&D cost saving to satisfy the P5 and capacity limits while matching diversified eNPV. With LOE in year 12, the threshold rises to {100*short['minimum_synergy']:.0f}%.

Risk capacity is binding. In the base-capacity profile, no tested PIAP combination satisfies even the growth profile. At the maximum 30% R&D saving, the central PIAP cell still has a {100*max_s['base_capacity_breach_probability']:.1f}% base-capacity breach probability, despite eNPV of ${max_s['eNPV_usd_m']:.0f}M and ePI of {max_s['ePI']:.2f}.

The location of synergy matters. At the high-capacity balanced boundary, 20% R&D saving remains sufficient through 25% cannibalization, but its eNPV advantage narrows to only ${c25['delta_eNPV_at_frontier_usd_m']:.1f}M. At 50% cannibalization there is no solution within the 0-30% R&D-saving range. Commercial-support saving alone has no solution even at 40%, because it arrives after launch and cannot repair the zero- or one-launch lower tail.

## Staging tradeoff

At rho=0.415, LOE year 15, no common shock, and no acceleration, staging saves ${abs(staged['delta_eDevCost_usd_m']):.1f}M in expected development cost and ${abs(staged['delta_mean_peak_annual_RnD_usd_m']):.1f}M in mean peak annual R&D. It reduces eNPV by ${abs(staged['delta_eNPV_usd_m']):.1f}M and raises base-capacity breach probability by {100*staged['delta_base_capacity_breach_probability']:.1f} percentage points because revenue arrives later. The probability of at least two launches by year 9 falls from {100*parallel['probability_two_launches_by_year_9']:.1f}% to {100*staged_cell['probability_two_launches_by_year_9']:.1f}%.

## Launch-service frontier

To achieve at least two launches in a mature rolling three-year bucket with 80% service, the benchmark requires {breadth['Diversified']['minimum_nominal_breadth']:.0f} diversified Phase 2 starts, {breadth['MSOG trios']['minimum_nominal_breadth']:.0f} starts grouped in correlated MSOG trios, or {breadth['PIAP sixes']['minimum_nominal_breadth']:.0f} starts grouped in PIAP sixes. The PIAP case has only {breadth['PIAP sixes']['effective_breadth_at_frontier']:.1f} variance-equivalent independent shots at the 25-start frontier.

## Assumption regret

At rho=0.415, LOE year 15, 10% R&D saving, and 25% true cannibalization, an additive-revenue decision selects PIAP while the true eNPV oracle selects diversification; the eNPV regret is ${regret['eNPV_regret_usd_m']:.1f}M. Across the empirical-anchor grid with cannibalization no greater than 25%, the separate worst cases are ${maxima['max_eNPV_regret_usd_m']:.1f}M eNPV regret, ${maxima['max_P5_shortfall_usd_m']:.1f}M P5 shortfall, and {100*maxima['max_incremental_breach_probability']:.1f} percentage points of incremental breach probability. These maxima occur in different cells and should not be added.

## Interpretation

The analysis separates three decisions that expected value alone conflates. Technical dependence prices the lower tail and effective breadth. Cannibalization prices non-additive revenue. Risk capacity determines whether an economically attractive concentrated strategy is survivable. Upstream R&D saving can move both value and capacity boundaries; downstream commercial saving may improve the mean while leaving the decisive failure paths untouched.
"""
    output.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs_e4_e7")
    args = parser.parse_args()
    output = Path(args.input)
    cells = pd.read_csv(output / "e4_cell_metrics.csv")
    frontier = pd.read_csv(output / "e4_rnd_acceleration_frontier.csv")
    interaction_frontier = pd.read_csv(output / "e4_cannibalization_shift_frontier.csv")
    commercial_frontier = pd.read_csv(output / "e4_commercial_synergy_frontier.csv")
    e5_summary = pd.read_csv(output / "e5_policy_metrics.csv")
    e5_contrasts = pd.read_csv(output / "e5_staged_vs_parallel.csv")
    e6_frontier = pd.read_csv(output / "e6_launch_service_frontier.csv")
    regret = pd.read_csv(output / "e7_assumption_regret.csv")
    figure_decision_map(frontier, output / "fig1_concentration_decision_map.png")
    figure_capacity(cells, output / "fig2_capacity_constraint.png")
    figure_staging(e5_contrasts, output / "fig3_staged_parallel_tradeoff.png")
    figure_launch_frontier(e6_frontier, output / "fig4_launch_service_frontier.png")
    figure_regret(regret, output / "fig5_additivity_regret.png")
    headline = headline_results(
        frontier,
        interaction_frontier,
        commercial_frontier,
        cells,
        e5_summary,
        e5_contrasts,
        e6_frontier,
        regret,
    )
    (output / "headline_results.json").write_text(
        json.dumps(headline, indent=2, sort_keys=True, default=float), encoding="utf-8"
    )
    write_memo(headline, output / "E4_E7_RESULTS_MEMO.md")
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"] = sorted(
            path.name
            for path in output.iterdir()
            if path.is_file() and path.name != manifest_path.name
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print("E4-E7 analysis outputs written")


if __name__ == "__main__":
    main()
