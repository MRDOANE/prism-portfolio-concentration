from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prism_sim.config import load_config, scenario_hash
from prism_sim.economics import calculate_economics, enterprise_metrics
from prism_sim.engine import Scenario, simulate_from_randomness
from prism_sim.outcomes import generate_outcomes
from prism_sim.policy import schedule_parallel, schedule_staged
from prism_sim.randomness import generate_random_block
from prism_sim.runner import deterministic_fixtures


class TestPrismEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / "config" / "benchmark_v0.2.yaml")
        cls.n_programs = int(cls.config["simulation"]["n_programs"])

    def test_locked_cumulative_ptrs(self):
        development = self.config["development"]
        ptrs = np.prod(
            [
                development["phase2"]["pass_probability"],
                development["phase3"]["pass_probability"],
                development["regulatory"]["pass_probability"],
            ]
        )
        self.assertAlmostEqual(ptrs, 0.151, places=3)

    def test_scenario_hash_is_stable_and_sensitive(self):
        payload = {"rho": 0.415, "strategy": "PIAP"}
        self.assertEqual(scenario_hash(payload), scenario_hash(dict(reversed(list(payload.items())))))
        self.assertNotEqual(scenario_hash(payload), scenario_hash({"rho": 0.25, "strategy": "PIAP"}))

    def test_deterministic_accounting_fixtures(self):
        _, report = deterministic_fixtures(self.config)
        self.assertTrue(report["pass"], report)

    def test_rho_zero_piap_equals_diversified_exactly(self):
        block = generate_random_block(17, 0, 2000, self.n_programs)
        piap = Scenario("test", "Parallel PIAP", "same_asset", rho=0.0)
        diversified = Scenario("test", "Diversified", "diversified", rho=0.0)
        outcomes_piap, paths_piap = simulate_from_randomness(piap, block, self.config)
        outcomes_div, paths_div = simulate_from_randomness(diversified, block, self.config)
        np.testing.assert_array_equal(outcomes_piap, outcomes_div)
        np.testing.assert_array_equal(paths_piap.portfolio_npv, paths_div.portfolio_npv)

    def test_latent_factor_preserves_program_marginals(self):
        n_runs = 100000
        block = generate_random_block(31, 0, n_runs, self.n_programs)
        probabilities = np.array([0.289, 0.578, 0.906])
        outcomes = generate_outcomes(block, probabilities, rho=0.60)
        observed = outcomes.mean(axis=0)
        se = np.sqrt(probabilities * (1.0 - probabilities) / n_runs)
        for program in range(self.n_programs):
            np.testing.assert_array_less(np.abs(observed[program] - probabilities), 4.0 * se)

    def test_common_shock_marginal_parity(self):
        n_runs = 120000
        block = generate_random_block(47, 0, n_runs, self.n_programs)
        probabilities = np.array([0.289, 0.578, 0.906])
        outcomes = generate_outcomes(
            block,
            probabilities,
            rho=0.415,
            common_shock_probability=0.10,
            common_shock_mode="marginal_parity",
        )
        observed = outcomes[:, :, 1].mean(axis=0)
        se = np.sqrt(probabilities[1] * (1.0 - probabilities[1]) / n_runs)
        np.testing.assert_array_less(np.abs(observed - probabilities[1]), np.repeat(4.0 * se, self.n_programs))

    def test_cannibalization_is_nonincreasing(self):
        outcomes = np.ones((10, self.n_programs, 3), dtype=bool)
        schedule = schedule_parallel(outcomes, self.config)
        additive = calculate_economics(schedule, self.config, cannibalization=0.0, commercial_clustered=True)
        partial = calculate_economics(schedule, self.config, cannibalization=0.5, commercial_clustered=True)
        full = calculate_economics(schedule, self.config, cannibalization=1.0, commercial_clustered=True)
        self.assertTrue(np.all(additive.portfolio_npv >= partial.portfolio_npv))
        self.assertTrue(np.all(partial.portfolio_npv >= full.portfolio_npv))

    def test_cost_synergy_increases_npv_and_lowers_spend(self):
        outcomes = np.ones((10, self.n_programs, 3), dtype=bool)
        schedule = schedule_parallel(outcomes, self.config)
        base = calculate_economics(schedule, self.config)
        synergy = calculate_economics(schedule, self.config, rnd_cost_synergy=0.30)
        self.assertTrue(np.all(synergy.portfolio_npv > base.portfolio_npv))
        self.assertTrue(np.all(synergy.pv_development_spend < base.pv_development_spend))

    def test_capacity_breach_is_nonincreasing_with_capacity(self):
        block = generate_random_block(59, 0, 3000, self.n_programs)
        scenario = Scenario("test", "Parallel PIAP", "same_asset", rho=0.60)
        _, paths = simulate_from_randomness(scenario, block, self.config)
        low = enterprise_metrics(paths, self.config, "low")["capacity_breach"].mean()
        base = enterprise_metrics(paths, self.config, "base")["capacity_breach"].mean()
        high = enterprise_metrics(paths, self.config, "high")["capacity_breach"].mean()
        self.assertGreaterEqual(low, base)
        self.assertGreaterEqual(base, high)

    def test_staged_all_pass_timing(self):
        outcomes = np.ones((1, self.n_programs, 3), dtype=bool)
        schedule = schedule_staged(outcomes, self.config)
        np.testing.assert_allclose(schedule.launch_time[0], [6.0, 10.0, 10.0, 10.0, 10.0, 10.0])

    def test_reproduction_is_bitwise_exact(self):
        scenario = Scenario("test", "Parallel PIAP", "same_asset", rho=0.415, common_shock_probability=0.05)
        block_a = generate_random_block(73, 0, 1000, self.n_programs)
        block_b = generate_random_block(73, 0, 1000, self.n_programs)
        outcomes_a, paths_a = simulate_from_randomness(scenario, block_a, self.config)
        outcomes_b, paths_b = simulate_from_randomness(scenario, block_b, self.config)
        np.testing.assert_array_equal(outcomes_a, outcomes_b)
        np.testing.assert_array_equal(paths_a.portfolio_npv, paths_b.portfolio_npv)


if __name__ == "__main__":
    unittest.main()

