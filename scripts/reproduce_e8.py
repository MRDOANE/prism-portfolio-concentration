#!/usr/bin/env python3
"""One-command clean reproduction of E8 tests, results, and figures."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "outputs_e8_reproduced"))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "tests"), "-v"],
        check=True,
        cwd=ROOT,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_e8.py"),
            "--config",
            str(ROOT / "config" / "benchmark_v0.2.yaml"),
            "--plan",
            str(ROOT / "config" / "e8_plan_v0.3.yaml"),
            "--output",
            str(output),
        ],
        check=True,
        cwd=ROOT,
    )
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "analyze_e8.py"), "--input", str(output)],
        check=True,
        cwd=ROOT,
    )
    print(f"E8 reproduction complete: {output}")


if __name__ == "__main__":
    main()
