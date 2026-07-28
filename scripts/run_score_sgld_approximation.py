"""Run the checkpoint-based DSIVI versus posterior-SGLD score analysis."""

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
    select_cell_specs,
    shard_cell_specs,
)
from finalization.score_sgld_approximation import (  # noqa: E402
    load_sgld_score_config,
    run_sgld_analysis,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare checkpointed DSIVI scores with a grouped terminal-"
            "particle posterior-SGLD reference."
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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--worker-only", action="store_true")
    parser.add_argument(
        "--cell-key",
        dest="cell_keys",
        action="append",
        default=[],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.aggregate_only and args.worker_only:
        raise ValueError("--aggregate-only and --worker-only are incompatible.")
    cfg = load_sgld_score_config(args.config, args.overrides)
    all_specs = build_cell_specs(cfg)
    specs = shard_cell_specs(
        all_specs,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )
    specs = select_cell_specs(specs, set(args.cell_keys) or None)
    fingerprint = config_fingerprint(cfg)

    if args.dry_run:
        reference = cfg.evaluation.reference
        groups = int(reference.num_groups)
        chains_per_group = int(reference.chains_per_group)
        steps = int(reference.num_steps)
        forward = int(cfg.evaluation.forward_batch_size)
        transitions_per_cell = (
            forward * groups * chains_per_group * steps
        )
        print(f"analysis_fingerprint={fingerprint}")
        print(f"total_runs={len({cell.record.run_id for cell in all_specs})}")
        print(f"total_cells={len(all_specs)}")
        print(f"shard_count={args.shard_count}")
        print(f"shard_index={args.shard_index}")
        print(f"shard_cells={len(specs)}")
        print(f"reference_estimator={reference.estimator}")
        print(f"forward_z_per_cell={forward}")
        print(f"sgld_groups={groups}")
        print(f"sgld_chains_per_group={chains_per_group}")
        print(f"sgld_terminal_particles_per_z={groups * chains_per_group}")
        print(f"sgld_steps={steps}")
        print(f"sgld_step_size={float(reference.step_size):.10f}")
        print(f"sgld_langevin_time={steps * float(reference.step_size):.6f}")
        print(f"sgld_particle_transitions_per_cell={transitions_per_cell}")
        print(
            "sgld_particle_transitions_selected="
            f"{transitions_per_cell * len(specs)}"
        )
        return 0

    completed, summaries = run_sgld_analysis(
        cfg,
        limit=args.limit,
        resume=bool(args.resume),
        aggregate_only=bool(args.aggregate_only),
        shard_count=args.shard_count,
        shard_index=args.shard_index,
        aggregate_after_run=not bool(args.worker_only),
        cell_keys=set(args.cell_keys) or None,
    )
    print(f"completed_cells={completed}")
    if summaries:
        print(f"summary_rows={summaries}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
