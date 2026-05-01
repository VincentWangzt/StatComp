from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_default_config_grid_sweep as sweep  # noqa: E402


CAMPAIGN_SLUG = "toy_no_anneal_ablation"
CONFIG_HASH_VERSION = "toy-no-anneal-ablation-effective-v1"
TARGETS = ("banana", "multimodal", "x_shaped", "student_uc", "8_gaussians")
SEEDS = (42, 43, 44)
BASE_EPOCHS = (3000, 5000, 10000)
RESULTS_DIR = f"results/{CAMPAIGN_SLUG}"
TB_DIR = f"tb_logs/{CAMPAIGN_SLUG}"

VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "slug": "sivi",
        "method": "SIVI",
        "runner_type": "SIVI",
        "source_method": "sivi",
        "epoch_multiplier": 2,
        "overrides": (),
    },
    {
        "slug": "uivi",
        "method": "UIVI",
        "runner_type": "UIVI",
        "source_method": "uivi",
        "epoch_multiplier": 1,
        "overrides": (),
    },
    {
        "slug": "aisivi",
        "method": "AISIVI",
        "runner_type": "AISIVI",
        "source_method": "aisivi",
        "epoch_multiplier": 1,
        "overrides": (),
    },
    {
        "slug": "dsivi",
        "method": "DSIVI",
        "runner_type": "DSIVI",
        "source_method": "dsivi",
        "epoch_multiplier": 1,
        "overrides": (),
    },
    {
        "slug": "dsivi_rev5",
        "method": "DSIVI-rev5",
        "runner_type": "DSIVI",
        "source_method": "dsivi",
        "epoch_multiplier": 1,
        "overrides": ("train.reverse.epochs=5",),
    },
    {
        "slug": "ksivi",
        "method": "KSIVI",
        "runner_type": "KSIVI",
        "source_method": "ksivi",
        "epoch_multiplier": 5,
        "overrides": (),
    },
)


def run_id_for(seed: int, base_epochs: int, variant_slug: str, target: str) -> str:
    return f"seed{seed}_base{base_epochs // 1000}k_{variant_slug}_{target.lower()}"


def base_config_path(source_method: str, target: str) -> Path:
    return sweep.REPO_ROOT / "configs" / f"{source_method}_{target}.yaml"


def build_variant_overrides(
    variant: dict[str, Any],
    base_epochs: int,
    user_overrides: list[str],
) -> list[str]:
    effective_epochs = int(base_epochs) * int(variant["epoch_multiplier"])
    overrides = [
        "train.annealing.enabled=false",
        f"train.epochs={effective_epochs}",
    ]
    overrides.extend(str(item) for item in variant.get("overrides", ()))
    overrides.extend(user_overrides)
    return overrides


def build_manifest_entries(args: argparse.Namespace) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for seed in args.seeds:
        for base_epochs in BASE_EPOCHS:
            for target in TARGETS:
                for variant in VARIANTS:
                    config_path = base_config_path(str(variant["source_method"]), target)
                    if not config_path.exists():
                        raise FileNotFoundError(config_path)

                    effective_epochs = int(base_epochs) * int(variant["epoch_multiplier"])
                    extra_overrides = build_variant_overrides(
                        variant,
                        base_epochs=base_epochs,
                        user_overrides=list(args.extra_override),
                    )
                    run_id = run_id_for(seed, base_epochs, str(variant["slug"]), target)
                    config_path_rel = sweep.relpath(config_path)
                    entry = {
                        "run_id": run_id,
                        "campaign_slug": args.campaign_slug,
                        "seed": seed,
                        "base_epochs": base_epochs,
                        "effective_epochs": effective_epochs,
                        "epoch_multiplier": variant["epoch_multiplier"],
                        "method": variant["method"],
                        "method_slug": variant["slug"],
                        "variant": variant["slug"],
                        "variant_label": variant["method"],
                        "source_method": variant["source_method"],
                        "target": target,
                        "target_slug": target,
                        "runner_type": variant["runner_type"],
                        "config_path": config_path_rel,
                        "expected_epochs": effective_epochs,
                        "batch_size": sweep.load_config(config_path).get("train", {}).get("batch_size", ""),
                        "results_dir": args.results_dir,
                        "tb_dir": args.tb_dir,
                        "status": "pending",
                        "runtime_gpu": "",
                        "annealing_enabled": False,
                        "extra_overrides": extra_overrides,
                        "config_hash_version": sweep.CONFIG_HASH_VERSION,
                        "config_hash": sweep.effective_config_hash(
                            config_path_rel or config_path,
                            seed=seed,
                            extra_overrides=extra_overrides,
                        ),
                        "config_hash_basis": (
                            "resolved main config plus seed, no-anneal override, epoch override, "
                            "variant overrides, and user extra overrides; target/vi/reverse config "
                            "files expanded; scheduler/output/device paths ignored"
                        ),
                    }
                    entry["command_template"] = sweep.build_command(
                        entry,
                        gpu=0,
                        results_dir=args.results_dir,
                        tb_dir=args.tb_dir,
                        extra_overrides=extra_overrides,
                    )
                    entries.append(entry)

    if args.limit is not None:
        entries = entries[:args.limit]
    return entries


def print_dry_run(entries: list[dict[str, Any]], gpus: list[int], args: argparse.Namespace) -> None:
    print(f"campaign_slug: {args.campaign_slug}")
    print(f"discovered_gpus: {gpus if gpus else 'none'}")
    print(f"runs: {len(entries)}")
    for entry in entries:
        command = sweep.build_command(
            entry,
            gpu=gpus[0] if gpus else 0,
            results_dir=args.results_dir,
            tb_dir=args.tb_dir,
            extra_overrides=entry.get("extra_overrides", []),
        )
        print(f"{entry['run_id']}: {' '.join(command)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 270-run toy no-anneal ablation sweep with dynamic GPU scheduling."
    )
    parser.add_argument("--campaign-slug", default=CAMPAIGN_SLUG)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--tb-dir", default=TB_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--gpus", nargs="+", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--rerun-stale",
        action="store_true",
        help="Rerun completed runs whose saved config hash differs from the current effective config hash.",
    )
    parser.add_argument(
        "--hash-existing-artifacts",
        action="store_true",
        help="Hash completed runs' result_path/full_config.yaml files and write an inventory under campaign runtime.",
    )
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--extra-override", action="append", default=[])
    parser.add_argument("--finalize-mode", choices=["async", "sync"], default="async")
    parser.add_argument("--finalize-workers", type=int, default=1)
    parser.add_argument("--summary-interval-sec", type=float, default=120.0)
    parser.add_argument("--finalize-retries", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    sweep.DEFAULT_CAMPAIGN_SLUG = CAMPAIGN_SLUG
    sweep.CONFIG_HASH_VERSION = CONFIG_HASH_VERSION
    sweep.build_manifest_entries = build_manifest_entries
    sweep.print_dry_run = print_dry_run
    sweep.parse_args = parse_args
    sweep.main()


if __name__ == "__main__":
    main()
