"""Locked common-random-number streams."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RandomBlock:
    seed: int
    block_index: int
    residuals: np.ndarray
    cluster_factors: np.ndarray
    shock_uniform: np.ndarray

    @property
    def n_runs(self) -> int:
        return int(self.residuals.shape[0])


def generate_random_block(seed: int, block_index: int, n_runs: int, n_programs: int) -> RandomBlock:
    rng = np.random.default_rng(seed)
    residuals = rng.standard_normal((n_runs, n_programs, 3))
    cluster_factors = rng.standard_normal((n_runs, 3))
    shock_uniform = rng.random(n_runs)
    return RandomBlock(
        seed=seed,
        block_index=block_index,
        residuals=residuals,
        cluster_factors=cluster_factors,
        shock_uniform=shock_uniform,
    )

