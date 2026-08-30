#!/usr/bin/env python3
"""Run the locked 10,000-trial mechanism-isolation pilot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from prism_sim.config import load_config
from prism_sim.runner import run_pilot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "benchmark_v0.2.yaml"))
    parser.add_argument("--output", default=str(PACKAGE_ROOT / "outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    result = run_pilot(config, args.output)
    print(
        f"pilot complete: {result['manifest']['n_scenario_cells']} cells, "
        f"validation_pass={result['manifest']['validation_pass']}, "
        f"runtime={result['manifest']['runtime_seconds']:.1f}s"
    )


if __name__ == "__main__":
    main()

