"""Run the posterior-HMC initialization-jitter ablation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finalization.score_approximation import (  # noqa: E402
    build_cell_specs,
    config_fingerprint,
)
from finalization.score_jitter_ablation import (  # noqa: E402
    load_jitter_config,
    run_ablation,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure posterior-HMC score sensitivity to chain "
            "initialization jitter."
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate checkpoints and print the ablation budget.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only this many pending seeds.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_jitter_config(args.config, args.overrides)
    specs = build_cell_specs(cfg)
    fingerprint = config_fingerprint(cfg)
    if args.dry_run:
        reference = cfg.evaluation.reference
        samples_per_chain = (
            int(reference.total_samples)
            // int(reference.num_chains)
        )
        transitions = int(reference.burn_in_steps) + (
            samples_per_chain * int(reference.thinning)
        )
        jitter_count = len(cfg.ablation.jitter_scales)
        print(f"analysis_fingerprint={fingerprint}")
        print(f"seeds={len(specs)}")
        print(f"jitter_scales={list(cfg.ablation.jitter_scales)}")
        print(f"reference_chains={int(reference.num_chains)}")
        print(f"reference_samples_per_chain={samples_per_chain}")
        print(f"reference_transitions_per_chain={transitions}")
        print(f"reference_runs={len(specs) * jitter_count}")
        print(
            "hmc_gradient_point_evaluations="
            f"{len(specs) * jitter_count * int(cfg.evaluation.forward_batch_size) * int(reference.num_chains) * transitions * (int(reference.leapfrog_steps) + 1)}"
        )
        return 0
    completed, pairwise = run_ablation(
        cfg,
        limit=args.limit,
        resume=bool(args.resume),
        aggregate_only=bool(args.aggregate_only),
    )
    if args.aggregate_only or args.limit is None:
        print(f"jitter_summary_rows={completed}")
        print(f"pairwise_summary_rows={pairwise}")
    else:
        print(f"completed_seeds={completed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
