"""Run the checkpoint-based score-approximation analysis."""

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
    load_score_config,
    run_analysis,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare method-native checkpoint scores with a high-budget "
            "marginal-score reference."
        )
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="OmegaConf dotlist override, e.g. evaluation.device=cpu",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate selection/checkpoints and print the work budget.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only this many pending cells (for benchmarking).",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip matching completed cell records (default: true).",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Regenerate reports from a complete set of cell records.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_score_config(args.config, args.overrides)
    specs = build_cell_specs(cfg)
    fingerprint = config_fingerprint(cfg)
    if args.dry_run:
        reference = cfg.evaluation.reference
        per_z = int(reference.reverse_batch_size) * int(
            reference.num_batches
        )
        pair_evaluations = (
            len(specs)
            * int(cfg.evaluation.forward_batch_size)
            * per_z
            * int(reference.repeats)
        )
        print(f"analysis_fingerprint={fingerprint}")
        print(f"runs={len({cell.record.run_id for cell in specs})}")
        print(f"cells={len(specs)}")
        print(f"reference_auxiliaries_per_z_repeat={per_z}")
        print(f"reference_conditional_pair_evaluations={pair_evaluations}")
        return 0

    completed, summaries = run_analysis(
        cfg,
        limit=args.limit,
        resume=bool(args.resume),
        aggregate_only=bool(args.aggregate_only),
    )
    print(f"completed_cells={completed}")
    if summaries:
        print(f"summary_rows={summaries}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
