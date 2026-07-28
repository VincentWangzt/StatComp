"""Run a common-checkpoint score comparison in independently schedulable tasks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finalization.shared_checkpoint_score import (  # noqa: E402
    aggregate_shared_results,
    analysis_fingerprint,
    artifact_paths,
    build_shared_checkpoint_specs,
    load_shared_score_config,
    prepare_forward_bank,
    run_hmc_reference,
    run_method_score,
    validate_production_budget,
)


TASKS = (
    "prepare",
    "reference",
    "method",
    "aggregate",
    "all-serial",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare method-native scores on a shared DSIVI checkpoint "
            "against one persisted posterior-HMC reference."
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
        "--task",
        choices=TASKS,
        required=True,
        help=(
            "Use separate reference/method tasks for multi-GPU production. "
            "all-serial is a one-device diagnostic convenience."
        ),
    )
    parser.add_argument(
        "--method",
        help="Required for --task method.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.task == "method" and not args.method:
        raise ValueError("--task method requires --method.")
    if args.task != "method" and args.method:
        raise ValueError("--method is only valid with --task method.")

    cfg = load_shared_score_config(args.config, args.overrides)
    specs = build_shared_checkpoint_specs(cfg)
    validate_production_budget(cfg)
    if not specs:
        raise RuntimeError("No shared checkpoint cells were selected.")

    print(f"analysis_fingerprint={analysis_fingerprint(cfg)}")
    print(f"source_cells={len(specs)}")
    print(
        "methods="
        + ",".join(str(value).upper() for value in cfg.selection.methods)
    )
    for spec in specs:
        paths = artifact_paths(cfg, spec)
        print(
            f"source={spec.source_record.run_id} "
            f"target={spec.source_record.target} "
            f"seed={spec.source_record.seed} epoch={spec.epoch}"
        )
        print(f"forward_bank={paths.forward_bank}")
        print(f"hmc_reference={paths.hmc_reference}")
    if args.dry_run:
        return 0

    if args.task in {"prepare", "all-serial"}:
        for spec in specs:
            prepare_forward_bank(
                cfg,
                spec,
                resume=bool(args.resume),
            )
    if args.task in {"reference", "all-serial"}:
        for spec in specs:
            run_hmc_reference(
                cfg,
                spec,
                resume=bool(args.resume),
            )
    if args.task == "method":
        assert args.method is not None
        for spec in specs:
            run_method_score(
                cfg,
                spec,
                args.method,
                resume=bool(args.resume),
            )
    elif args.task == "all-serial":
        for method in cfg.selection.methods:
            for spec in specs:
                run_method_score(
                    cfg,
                    spec,
                    str(method),
                    resume=bool(args.resume),
                )
    if args.task in {"aggregate", "all-serial"}:
        rows, report_dir = aggregate_shared_results(cfg, specs)
        print(f"metric_rows={len(rows)}")
        print(f"report_dir={report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
