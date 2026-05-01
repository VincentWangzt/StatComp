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


CAMPAIGN_SLUG = "toy_sivi_dsivi_epoch_anneal_grid_20260427"
CONFIG_HASH_VERSION = "toy-sivi-dsivi-epoch-anneal-grid-effective-v1"
TARGETS = ("banana", "multimodal", "x_shaped", "8_gaussians")
SEEDS = (42, 43, 44)
RESULTS_DIR = f"results/{CAMPAIGN_SLUG}"
TB_DIR = f"tb_logs/{CAMPAIGN_SLUG}"

REGIMES: tuple[dict[str, int | str], ...] = (
    {"slug": "ep2000_a1000", "epochs": 2000, "anneal_steps": 1000},
    {"slug": "ep4000_a2000", "epochs": 4000, "anneal_steps": 2000},
)

VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "slug": "dsivi",
        "method": "DSIVI",
        "runner_type": "DSIVI",
        "source_method": "dsivi",
        "reverse_sample_num": None,
        "overrides": (),
    },
    {
        "slug": "sivi_rsn4096",
        "method": "SIVI-rsn4096",
        "runner_type": "SIVI",
        "source_method": "sivi",
        "reverse_sample_num": 4096,
        "overrides": ("train.reverse_sample_num=4096",),
    },
    {
        "slug": "sivi_rsn1024",
        "method": "SIVI-rsn1024",
        "runner_type": "SIVI",
        "source_method": "sivi",
        "reverse_sample_num": 1024,
        "overrides": ("train.reverse_sample_num=1024",),
    },
)


def run_id_for(seed: int, regime_slug: str, variant_slug: str, target: str) -> str:
    return f"seed{seed}_{regime_slug}_{variant_slug}_{target.lower()}"


def base_config_path(source_method: str, target: str) -> Path:
    return sweep.REPO_ROOT / "configs" / f"{source_method}_{target}.yaml"


def build_overrides(
    regime: dict[str, int | str],
    variant: dict[str, Any],
    user_overrides: list[str],
) -> list[str]:
    overrides = [
        "train.annealing.enabled=true",
        f"train.epochs={regime['epochs']}",
        f"train.annealing.steps={regime['anneal_steps']}",
    ]
    overrides.extend(str(item) for item in variant.get("overrides", ()))
    overrides.extend(user_overrides)
    return overrides


def build_manifest_entries(args: argparse.Namespace) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for seed in args.seeds:
        for regime in REGIMES:
            for target in TARGETS:
                for variant in VARIANTS:
                    config_path = base_config_path(str(variant["source_method"]), target)
                    if not config_path.exists():
                        raise FileNotFoundError(config_path)

                    cfg = sweep.load_config(config_path)
                    train_cfg = cfg.get("train", {})
                    reverse_cfg = train_cfg.get("reverse", {})
                    extra_overrides = build_overrides(regime, variant, list(args.extra_override))
                    run_id = run_id_for(seed, str(regime["slug"]), str(variant["slug"]), target)
                    run_results_dir = (
                        f"{args.results_dir}/{regime['slug']}/{target}/{variant['slug']}/{run_id}"
                    )
                    run_tb_dir = (
                        f"{args.tb_dir}/{regime['slug']}/{target}/{variant['slug']}/{run_id}"
                    )
                    config_path_rel = sweep.relpath(config_path)
                    entry = {
                        "run_id": run_id,
                        "campaign_slug": args.campaign_slug,
                        "seed": seed,
                        "method": variant["method"],
                        "method_slug": variant["slug"],
                        "variant": variant["slug"],
                        "variant_label": variant["method"],
                        "source_method": variant["source_method"],
                        "target": target,
                        "target_slug": target,
                        "runner_type": variant["runner_type"],
                        "config_path": config_path_rel,
                        "expected_epochs": regime["epochs"],
                        "epochs": regime["epochs"],
                        "base_config_epochs": train_cfg.get("epochs", ""),
                        "batch_size": train_cfg.get("batch_size", ""),
                        "reverse_batch_size": reverse_cfg.get("batch_size", ""),
                        "reverse_sample_num": variant["reverse_sample_num"] or "",
                        "regime": regime["slug"],
                        "regime_label": f"{regime['epochs']} epochs / {regime['anneal_steps']} anneal steps",
                        "annealing_steps": regime["anneal_steps"],
                        "annealing_mode": "anneal_on",
                        "annealing_label": "Annealing on",
                        "annealing_enabled": True,
                        "campaign_results_dir": args.results_dir,
                        "campaign_tb_dir": args.tb_dir,
                        "results_dir": run_results_dir,
                        "tb_dir": run_tb_dir,
                        "status": "pending",
                        "runtime_gpu": "",
                        "extra_overrides": extra_overrides,
                        "config_hash_version": sweep.CONFIG_HASH_VERSION,
                        "config_hash": sweep.effective_config_hash(
                            config_path_rel or config_path,
                            seed=seed,
                            extra_overrides=extra_overrides,
                        ),
                        "config_hash_basis": (
                            "resolved default SIVI/DSIVI toy main config plus seed, "
                            "annealing-on override, train.epochs override, "
                            "train.annealing.steps override, and SIVI reverse_sample_num "
                            "override when applicable; target/vi/reverse config files expanded; "
                            "scheduler/output/device paths ignored"
                        ),
                    }
                    entry["command_template"] = sweep.build_command(
                        entry,
                        gpu=0,
                        results_dir=run_results_dir,
                        tb_dir=run_tb_dir,
                        extra_overrides=extra_overrides,
                    )
                    entries.append(entry)

    if args.limit is not None:
        entries = entries[: args.limit]
    return entries


def print_dry_run(entries: list[dict[str, Any]], gpus: list[int], args: argparse.Namespace) -> None:
    print(f"campaign_slug: {args.campaign_slug}")
    print(f"discovered_gpus: {gpus if gpus else 'none'}")
    print(f"runs: {len(entries)}")
    for entry in entries:
        command = sweep.build_command(
            entry,
            gpu=gpus[0] if gpus else 0,
            results_dir=entry.get("results_dir", args.results_dir),
            tb_dir=entry.get("tb_dir", args.tb_dir),
            extra_overrides=entry.get("extra_overrides", []),
        )
        print(f"{entry['run_id']}: {' '.join(command)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run DSIVI, SIVI reverse_sample_num=4096, and SIVI reverse_sample_num=1024 "
            "on toy targets under 2k/1k and 4k/2k epoch/annealing regimes."
        )
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
