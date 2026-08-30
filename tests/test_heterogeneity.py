import unittest

import numpy as np

from prism_sim.heterogeneity import (
    cumulative_to_stage_probabilities,
    enumerate_program_states,
    gaussian_bernoulli_joint,
    joint_state_probabilities,
    triangular_ppf,
    weighted_cvar_lower,
    weighted_quantile,
)


class HeterogeneityTests(unittest.TestCase):
    def test_triangular_bounds_and_center(self):
        values = triangular_ppf(np.array([0.0, 0.5, 1.0]), 0.5, 1.0, 1.5)
        np.testing.assert_allclose(values, [0.5, 1.0, 1.5])

    def test_cumulative_stage_calibration(self):
        base = np.array([0.289, 0.578, 0.906])
        for target in (0.075, 0.151, 0.227):
            calibrated = cumulative_to_stage_probabilities(target, base)
            self.assertAlmostEqual(float(np.prod(calibrated)), target, places=12)

    def test_gaussian_joint_preserves_marginals(self):
        probabilities = np.array([0.2, 0.35, 0.6])
        joint = gaussian_bernoulli_joint(probabilities, rho=0.4, nodes=40)
        patterns = ((np.arange(8)[:, None] >> np.arange(3)) & 1).astype(bool)
        self.assertAlmostEqual(float(joint.sum()), 1.0, places=12)
        np.testing.assert_allclose((joint[:, None] * patterns).sum(axis=0), probabilities, atol=2e-6)

    def test_realized_state_probabilities_preserve_launch_marginals(self):
        stage = np.array(
            [
                [0.25, 0.55, 0.90],
                [0.30, 0.60, 0.88],
                [0.35, 0.50, 0.92],
            ]
        )
        states = enumerate_program_states(3)
        weights = joint_state_probabilities(stage, rho=0.3, states=states, nodes=40)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=12)
        launches = states == 3
        np.testing.assert_allclose(
            (weights[:, None] * launches).sum(axis=0),
            np.prod(stage, axis=1),
            atol=3e-6,
        )

    def test_weighted_tail_metrics(self):
        values = np.array([-10.0, 0.0, 10.0])
        weights = np.array([0.1, 0.8, 0.1])
        self.assertEqual(weighted_quantile(values, weights, 0.05), -10.0)
        self.assertEqual(weighted_cvar_lower(values, weights, 0.05), -10.0)


if __name__ == "__main__":
    unittest.main()
