"""Configuration loading, validation, and stable hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = ["simulation", "development", "commercial", "enterprise", "dependence", "pilot"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing configuration sections: {missing}")

    sim = config["simulation"]
    if int(sim["n_programs"]) < 1:
        raise ValueError("n_programs must be positive")
    if int(sim["pilot_runs_per_cell"]) % len(sim["seed_blocks"]) != 0:
        raise ValueError("pilot_runs_per_cell must divide evenly across seed blocks")
    dt = float(sim["time_step_years"])
    if dt <= 0 or float(sim["horizon_years"]) <= 0:
        raise ValueError("time step and horizon must be positive")

    for stage in ("phase2", "phase3", "regulatory"):
        values = config["development"][stage]
        probability = float(values["pass_probability"])
        if not 0 < probability < 1:
            raise ValueError(f"{stage} pass probability must lie in (0, 1)")
        steps = float(values["duration_years"]) / dt
        if abs(steps - round(steps)) > 1e-9:
            raise ValueError(f"{stage} duration must be a multiple of time_step_years")

    for rho in (
        config["dependence"]["same_asset_rho_grid"]
        + config["dependence"]["shared_biology_rho_grid"]
        + config["dependence"]["same_indication_rho_grid"]
    ):
        if not 0 <= float(rho) < 1:
            raise ValueError("latent correlations must lie in [0, 1)")

    p3 = float(config["development"]["phase3"]["pass_probability"])
    for q in config["dependence"]["common_shock_probability_grid"]:
        if float(q) < 0 or p3 / (1.0 - float(q)) > 1.0:
            raise ValueError("common-shock grid is incompatible with marginal parity")


def scenario_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

