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
    select_cell_specs,
    shard_cell_specs,
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
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Number of deterministic run-level worker shards.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based worker shard index.",
    )
    parser.add_argument(
        "--worker-only",
        action="store_true",
        help=(
            "Evaluate this shard without aggregating; run --aggregate-only "
            "after all shards finish."
        ),
    )
    parser.add_argument(
        "--cell-key",
        dest="cell_keys",
        action="append",
        default=[],
        help=(
            "Evaluate only an exact cell key; repeat for multiple cells. "
            "This runtime filter does not alter the analysis fingerprint."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.aggregate_only and args.worker_only:
        raise ValueError("--aggregate-only and --worker-only are incompatible.")
    cfg = load_score_config(args.config, args.overrides)
    all_specs = build_cell_specs(cfg)
    specs = shard_cell_specs(
        all_specs,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )
    cell_keys = set(args.cell_keys) or None
    specs = select_cell_specs(specs, cell_keys)
    fingerprint = config_fingerprint(cfg)
    if args.dry_run:
        reference = cfg.evaluation.reference
        total_samples = int(reference.total_samples)
        num_chains = int(reference.num_chains)
        if total_samples % num_chains != 0:
            raise ValueError(
                "reference.total_samples must be divisible by "
                "reference.num_chains."
            )
        draws_per_chain = total_samples // num_chains
        transitions = int(reference.burn_in_steps) + (
            draws_per_chain * int(reference.thinning)
        )
        state_transitions = (
            len(specs)
            * int(cfg.evaluation.forward_batch_size)
            * num_chains
            * transitions
        )
        gradient_point_evaluations = state_transitions * (
            int(reference.leapfrog_steps) + 1
        )
        print(f"analysis_fingerprint={fingerprint}")
        print(f"total_runs={len({cell.record.run_id for cell in all_specs})}")
        print(f"total_cells={len(all_specs)}")
        print(f"shard_count={args.shard_count}")
        print(f"shard_index={args.shard_index}")
        print(f"shard_runs={len({cell.record.run_id for cell in specs})}")
        print(f"shard_cells={len(specs)}")
        print(f"reference_estimator={reference.estimator}")
        print(f"reference_total_samples_per_z={total_samples}")
        print(f"reference_chains={num_chains}")
        print(f"reference_samples_per_chain={draws_per_chain}")
        print(f"reference_transitions_per_chain={transitions}")
        print(f"hmc_state_transitions={state_transitions}")
        print(
            "hmc_gradient_point_evaluations="
            f"{gradient_point_evaluations}"
        )
        return 0

    completed, summaries = run_analysis(
        cfg,
        limit=args.limit,
        resume=bool(args.resume),
        aggregate_only=bool(args.aggregate_only),
        shard_count=args.shard_count,
        shard_index=args.shard_index,
        aggregate_after_run=not bool(args.worker_only),
        cell_keys=cell_keys,
    )
    print(f"completed_cells={completed}")
    if summaries:
        print(f"summary_rows={summaries}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
