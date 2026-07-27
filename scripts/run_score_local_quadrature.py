"""Run the checkpoint-based local-quadrature score analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finalization.score_local_quadrature import (  # noqa: E402
    build_cell_specs,
    config_fingerprint,
    filter_cell_specs,
    load_local_quadrature_config,
    run_local_quadrature_analysis,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare checkpoint-native scores with a curvature-standardized "
            "local Gauss-Legendre quadrature reference."
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
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--targets", nargs="+")
    parser.add_argument("--epochs", nargs="+", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--report-label")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_local_quadrature_config(args.config, args.overrides)
    full_specs = build_cell_specs(cfg)
    selected = filter_cell_specs(
        full_specs,
        seeds=args.seeds,
        methods=args.methods,
        targets=args.targets,
        epochs=args.epochs,
    )
    fingerprint = config_fingerprint(cfg)

    if args.dry_run:
        quad = cfg.evaluation.quadrature
        order = int(quad.order)
        forward_count = int(cfg.evaluation.forward_batch_size)
        print(f"analysis_fingerprint={fingerprint}")
        print(f"full_cells={len(full_specs)}")
        print(f"selected_cells={len(selected)}")
        print(f"quadrature_order={order}")
        print(f"quadrature_epsilon_dim={quad.epsilon_dim}")
        if str(quad.epsilon_dim).lower() in {
            "auto",
            "checkpoint",
            "auto_from_checkpoint",
        }:
            print("quadrature_nodes_per_z=checkpoint_dependent")
            print("conditional_evaluations_per_cell=checkpoint_dependent")
            print("selected_conditional_evaluations=checkpoint_dependent")
        else:
            nodes_per_z = order ** int(quad.epsilon_dim)
            print(f"quadrature_nodes_per_z={nodes_per_z}")
            print(
                "conditional_evaluations_per_cell="
                f"{nodes_per_z * forward_count}"
            )
            print(
                "selected_conditional_evaluations="
                f"{nodes_per_z * forward_count * len(selected)}"
            )
        return 0

    completed, summaries = run_local_quadrature_analysis(
        cfg,
        seeds=args.seeds,
        methods=args.methods,
        targets=args.targets,
        epochs=args.epochs,
        limit=args.limit,
        resume=bool(args.resume),
        aggregate_only=bool(args.aggregate_only),
        report_label=args.report_label,
    )
    print(f"completed_cells={completed}")
    if summaries:
        print(f"summary_rows={summaries}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
