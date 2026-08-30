"""Scenario assembly and end-to-end simulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .config import scenario_hash
from .economics import EconomicPaths, calculate_economics
from .outcomes import generate_outcomes
from .policy import schedule_parallel, schedule_staged
from .randomness import RandomBlock


@dataclass(frozen=True)
class Scenario:
    experiment: str
    strategy: str
    topology: str
    policy: str = "parallel"
    rho: float = 0.0
    common_shock_probability: float = 0.0
    common_shock_mode: str = "marginal_parity"
    rnd_cost_synergy: float = 0.0
    acceleration_years: float = 0.0
    commercial_cost_synergy: float = 0.0
    cannibalization: float = 0.0
    commercial_clustered: bool = False

    def payload(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def hash(self) -> str:
        return scenario_hash(self.payload())


def simulate_from_randomness(scenario: Scenario, block: RandomBlock, config: dict) -> tuple[np.ndarray, EconomicPaths]:
    probabilities = np.array(
        [
            config["development"]["phase2"]["pass_probability"],
            config["development"]["phase3"]["pass_probability"],
            config["development"]["regulatory"]["pass_probability"],
        ],
        dtype=float,
    )
    rho = 0.0 if scenario.topology == "diversified" else float(scenario.rho)
    shock = 0.0 if scenario.topology == "diversified" else float(scenario.common_shock_probability)
    outcomes = generate_outcomes(
        block,
        probabilities,
        rho=rho,
        common_shock_probability=shock,
        common_shock_mode=scenario.common_shock_mode,
    )
    if scenario.policy == "parallel":
        schedule = schedule_parallel(outcomes, config, scenario.acceleration_years)
    elif scenario.policy == "staged":
        schedule = schedule_staged(outcomes, config, scenario.acceleration_years)
    else:
        raise ValueError(f"Unknown policy: {scenario.policy}")
    paths = calculate_economics(
        schedule,
        config,
        rnd_cost_synergy=scenario.rnd_cost_synergy,
        commercial_cost_synergy=scenario.commercial_cost_synergy,
        cannibalization=scenario.cannibalization,
        commercial_clustered=scenario.commercial_clustered,
    )
    return outcomes, paths

