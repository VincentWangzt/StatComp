from __future__ import annotations

import _bootstrap  # noqa: F401

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    python = sys.executable
    _run([python, "scripts/fetch_grid_benchmark_artifacts.py"])
    _run([python, "scripts/show_grid_status.py", "--phase", "official"])
    _run([python, "scripts/summarize_grid_benchmark.py", "--phase", "official"])
    print()
    print("Manual next steps:")
    print("1. Review grid_benchmark_2026-03-30.md and add the latest notes/anomalies.")
    print("2. Commit local progress if this is a scheduled 2-hour check or a failure milestone.")


if __name__ == "__main__":
    main()
