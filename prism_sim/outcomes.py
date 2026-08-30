"""Marginal-preserving correlated stage-gate outcomes."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtri

from .randomness import RandomBlock


STAGES = ("phase2", "phase3", "regulatory")


def generate_outcomes(
    block: RandomBlock,
    probabilities: np.ndarray,
    rho: float,
    common_shock_probability: float = 0.0,
    common_shock_mode: str = "marginal_parity",
) -> np.ndarray:
    """Return potential pass outcomes with shape (run, program, stage).

    The same total latent correlation is used at each gate. A common shock acts
    at Phase 3. Under marginal_parity, the non-shock Phase 3 probability is
    increased so the unconditional program-level Phase 3 probability remains
    at the benchmark.
    """

    rho = float(rho)
    q = float(common_shock_probability)
    if not 0 <= rho < 1:
        raise ValueError("rho must lie in [0, 1)")
    if not 0 <= q < 1:
        raise ValueError("common_shock_probability must lie in [0, 1)")

    probs = np.asarray(probabilities, dtype=float).copy()
    if probs.shape != (3,):
        raise ValueError("probabilities must contain Phase 2, Phase 3, and regulatory values")
    if q > 0 and common_shock_mode == "marginal_parity":
        probs[1] = probs[1] / (1.0 - q)
        if probs[1] > 1.0:
            raise ValueError("shock probability is too large for marginal parity")
    elif common_shock_mode not in {"marginal_parity", "adverse_information"}:
        raise ValueError(f"Unknown common_shock_mode: {common_shock_mode}")

    latent = np.sqrt(rho) * block.cluster_factors[:, None, :] + np.sqrt(1.0 - rho) * block.residuals
    outcomes = latent <= ndtri(probs)[None, None, :]

    if q > 0:
        shocked = block.shock_uniform < q
        outcomes[shocked, :, 1] = False
    return outcomes

