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
from run_8gaussians_vi_scheduler_grid import METHODS, SEEDS, TARGET, base_config_path  # noqa: E402


CAMPAIGN_SLUG = "8gaussians_vi_lr_anneal_step1000_gamma09_20260428"
CONFIG_HASH_VERSION = "8gaussians-vi-lr-anneal-step1000-gamma09-v1"
RESULTS_DIR = f"results/{CAMPAIGN_SLUG}"
TB_DIR = f"tb_logs/{CAMPAIGN_SLUG}"

LR_REGIMES: tuple[dict[str, Any], ...] = (
    {"slug": "lr2e3", "label": "2e-3", "value": 2.0e-3},
    {"slug": "lr5e3", "label": "5e-3", "value": 5.0e-3},
)

BASE_OVERRIDES = (
    "train.vi.scheduler.step_size=1000",
    "train.vi.scheduler.gamma=0.9",
    "train.annealing.enabled=true",
)


def _float_override(value: float) -> str:
    return f"{value:.1e}"


def run_id_for(seed: int, method_slug: str, lr_slug: str) -> str:
    return f"seed{seed}_{method_slug}_{TARGET}_{lr_slug}_anneal_step1000_gamma09"


def build_manifest_entries(args: argparse.Namespace) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for seed in args.seeds:
        for method in METHODS:
            config_path = base_config_path(str(method["source_method"]))
            if not config_path.exists():
                raise FileNotFoundError(config_path)

            cfg = sweep.load_config(config_path)
            train_cfg = cfg.get("train", {})
            reverse_cfg = train_cfg.get("reverse", {})

            for lr_regime in LR_REGIMES:
                lr_value = float(lr_regime["value"])
                extra_overrides = list(BASE_OVERRIDES)
                extra_overrides.append(f"train.vi.lr={_float_override(lr_value)}")
                extra_overrides.extend(args.extra_override)

                run_id = run_id_for(seed, str(method["slug"]), str(lr_regime["slug"]))
                run_results_dir = f"{args.results_dir}/{run_id}"
                run_tb_dir = f"{args.tb_dir}/{run_id}"
                config_path_rel = sweep.relpath(config_path)
                entry = {
                    "run_id": run_id,
                    "campaign_slug": args.campaign_slug,
                    "seed": seed,
                    "method": method["method"],
                    "method_slug": method["slug"],
                    "variant": f"{method['slug']}_{lr_regime['slug']}",
                    "variant_label": f"{method['method']} {lr_regime['label']}",
                    "source_method": method["source_method"],
                    "target": TARGET,
                    "target_slug": TARGET,
                    "runner_type": method["runner_type"],
                    "config_path": config_path_rel,
                    "expected_epochs": train_cfg.get("epochs", ""),
                    "epochs": train_cfg.get("epochs", ""),
                    "batch_size": train_cfg.get("batch_size", ""),
                    "reverse_batch_size": reverse_cfg.get("batch_size", ""),
                    "annealing_mode": "on",
                    "annealing_enabled": True,
                    "scheduler_step_size": 1000,
                    "scheduler_gamma": 0.9,
                    "vi_lr": lr_value,
                    "vi_lr_label": lr_regime["label"],
                    "vi_lr_slug": lr_regime["slug"],
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
                        "resolved main 8_gaussians config plus seed, VI scheduler "
                        "step_size=1000/gamma=0.9, annealing enabled, VI lr regime, "
                        "and user extra overrides; KSIVI excluded; target/vi/reverse "
                        "config files expanded; scheduler output/device paths ignored"
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
            results_dir=entry.get("results_dir", args.results_dir),
            tb_dir=entry.get("tb_dir", args.tb_dir),
            extra_overrides=entry.get("extra_overrides", []),
        )
        print(f"{entry['run_id']}: {' '.join(command)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run 8-Gaussians default non-KSIVI configs with VI lr in {2e-3, 5e-3}, "
            "annealing on, and VI StepLR step_size=1000 gamma=0.9."
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
    parser.add_argument("--rerun-stale", action="store_true")
    parser.add_argument("--hash-existing-artifacts", action="store_true")
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
