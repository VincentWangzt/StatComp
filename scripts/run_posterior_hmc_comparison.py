"""Run retained-chain HMC for the representative posterior epsilon."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finalization.posterior_hmc_comparison import (  # noqa: E402
    load_posterior_hmc_config,
    run_posterior_hmc_comparison,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare terminal MALA samples with retained HMC chain samples "
            "for one checkpointed x_shaped posterior."
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
    cfg = load_posterior_hmc_config(args.config, args.overrides)
    result = run_posterior_hmc_comparison(cfg)
    summary = {
        "acceptance_rate": result["acceptance_rate"],
        "post_burn_acceptance_rate": result[
            "post_burn_acceptance_rate"
        ],
        "split_rhat_max": result["split_rhat_max"],
        "ess_min": result["ess_min"],
        "divergence_fraction": result["divergence_fraction"],
        "final_step_size_median": result["final_step_size_median"],
        "convergence_pass": result["convergence_pass"],
        "runtime_sec": result["runtime_sec"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
