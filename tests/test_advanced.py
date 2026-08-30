from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prism_sim.advanced import build_economic_basis, linear_components
from prism_sim.config import load_config
from prism_sim.economics import calculate_economics
from prism_sim.engine import Scenario, simulate_from_randomness
from prism_sim.policy import schedule_parallel
from prism_sim.randomness import generate_random_block


class TestAdvancedBasis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / "config" / "benchmark_v0.2.yaml")
        block = generate_random_block(991, 0, 2000, 6)
        outcomes, _ = simulate_from_randomness(
            Scenario("test", "PIAP", "same_asset", rho=0.415), block, cls.config
        )
        cls.schedule = schedule_parallel(outcomes, cls.config, acceleration_years=0.5)

    def test_linear_decomposition_matches_full_economics(self):
        basis = build_economic_basis(self.schedule, self.config)
        components = linear_components(basis, self.config, loe_year=15.0, commercial_clustered=True)
        rnd = 0.20
        commercial = 0.25
        cannibalization = 0.10
        decomposed_npv = (
            components.base_npv
            + rnd * components.rnd_npv_gain
            + commercial * components.commercial_npv_gain
            - cannibalization * components.cannibalization_npv_loss
        )
        direct = calculate_economics(
            self.schedule,
            self.config,
            rnd_cost_synergy=rnd,
            commercial_cost_synergy=commercial,
            cannibalization=cannibalization,
            commercial_clustered=True,
        )
        np.testing.assert_allclose(decomposed_npv, direct.portfolio_npv, rtol=0.0, atol=1e-10)

    def test_development_saving_is_nonnegative(self):
        basis = build_economic_basis(self.schedule, self.config)
        components = linear_components(basis, self.config, loe_year=15.0, commercial_clustered=False)
        self.assertTrue(np.all(components.rnd_npv_gain >= -1e-12))
        self.assertTrue(np.all(components.rnd_pv_development_saving >= -1e-12))


if __name__ == "__main__":
    unittest.main()
