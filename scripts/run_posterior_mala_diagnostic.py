"""Run the representative x_shaped posterior-epsilon MALA diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finalization.posterior_mala_diagnostic import (  # noqa: E402
    load_posterior_mala_config,
    run_posterior_mala_diagnostic,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run parallel MALA chains for q_phi(epsilon | z) from one "
            "checkpointed x_shaped model."
        )
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="OmegaConf dotlist override.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_posterior_mala_config(args.config, args.overrides)
    result = run_posterior_mala_diagnostic(cfg)
    summary = {
        "acceptance_rate": result["acceptance_rate"],
        "post_burn_acceptance_rate": result[
            "post_burn_acceptance_rate"
        ],
        "split_rhat_max": result["split_rhat_max"],
        "ess_min": result["ess_min"],
        "early_late_standardized_drift_max": result[
            "early_late_standardized_drift_max"
        ],
        "convergence_pass": result["convergence_pass"],
        "runtime_sec": result["runtime_sec"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
