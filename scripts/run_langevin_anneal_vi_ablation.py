from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_default_config_grid_sweep as sweep  # noqa: E402


CAMPAIGN_SLUG = "langevin_anneal_vi_ablation_20260427"
CONFIG_HASH_VERSION = "langevin-anneal-vi-ablation-effective-v1"
TARGET = "Langevin_post"
SEEDS = (42, 43, 44)
BASE_EPOCHS = 10000
KSIVI_EPOCH_MULTIPLIER = 5
RESULTS_DIR = f"results/{CAMPAIGN_SLUG}"
TB_DIR = f"tb_logs/{CAMPAIGN_SLUG}"

METHODS: tuple[dict[str, Any], ...] = (
    {
        "slug": "sivi",
        "method": "SIVI",
        "runner_type": "SIVI",
        "source_method": "sivi",
        "epoch_multiplier": 1,
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
        "slug": "rsivi",
        "method": "RSIVI",
        "runner_type": "RSIVI",
        "source_method": "rsivi",
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
        "slug": "ksivi",
        "method": "KSIVI",
        "runner_type": "KSIVI",
        "source_method": "ksivi",
        "epoch_multiplier": KSIVI_EPOCH_MULTIPLIER,
        "overrides": (),
    },
)

ANNEALING_REGIMES: tuple[dict[str, Any], ...] = (
    {"slug": "anneal_on", "label": "anneal_on", "enabled": True},
    {"slug": "anneal_off", "label": "anneal_off", "enabled": False},
)

VI_REGIMES: tuple[dict[str, str], ...] = (
    {
        "slug": "uniform_aisivi",
        "label": "ConditionalGaussianGlobalUniform-AISIVI",
        "vi_model_type": "ConditionalGaussianGlobalUniform",
        "vi_model_config_path": "configs/vi_models/ConditionalGaussianGlobalUniform-AISIVI.yaml",
    },
    {
        "slug": "cgglobal_langevin",
        "label": "ConditionalGaussianGlobal-Langevin",
        "vi_model_type": "ConditionalGaussianGlobal",
        "vi_model_config_path": "configs/vi_models/ConditionalGaussianGlobal-Langevin.yaml",
    },
)


def run_id_for(
    seed: int,
    method_slug: str,
    annealing_slug: str,
    vi_slug: str,
) -> str:
    return f"seed{seed}_{annealing_slug}_{vi_slug}_{method_slug}_{TARGET.lower()}"


def base_config_path(source_method: str) -> Path:
    return sweep.REPO_ROOT / "configs" / f"{source_method}_{TARGET}.yaml"


def _bool_override(value: bool) -> str:
    return "true" if value else "false"


def build_regime_overrides(
    method: dict[str, Any],
    annealing: dict[str, Any],
    vi_regime: dict[str, str],
    user_overrides: list[str],
) -> tuple[list[str], int]:
    effective_epochs = BASE_EPOCHS * int(method.get("epoch_multiplier", 1))
    overrides = [
        f"train.annealing.enabled={_bool_override(bool(annealing['enabled']))}",
        f"train.annealing.steps={max(1, effective_epochs // 2)}",
        f"train.epochs={effective_epochs}",
        f"vi_model_type={vi_regime['vi_model_type']}",
        f"vi_model_config_path={vi_regime['vi_model_config_path']}",
    ]
    overrides.extend(str(item) for item in method.get("overrides", ()))
    overrides.extend(user_overrides)
    return overrides, effective_epochs


def build_manifest_entries(args: argparse.Namespace) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for seed in args.seeds:
        for method in METHODS:
            config_path = base_config_path(str(method["source_method"]))
            if not config_path.exists():
                raise FileNotFoundError(config_path)

            base_cfg = sweep.load_config(config_path)
            train_cfg = base_cfg.get("train", {})

            for annealing in ANNEALING_REGIMES:
                for vi_regime in VI_REGIMES:
                    extra_overrides, effective_epochs = build_regime_overrides(
                        method,
                        annealing,
                        vi_regime,
                        user_overrides=list(args.extra_override),
                    )
                    run_id = run_id_for(
                        seed,
                        str(method["slug"]),
                        str(annealing["slug"]),
                        str(vi_regime["slug"]),
                    )
                    config_path_rel = sweep.relpath(config_path)
                    entry = {
                        "run_id": run_id,
                        "campaign_slug": args.campaign_slug,
                        "seed": seed,
                        "base_epochs": BASE_EPOCHS,
                        "effective_epochs": effective_epochs,
                        "epochs": effective_epochs,
                        "method": method["method"],
                        "method_slug": method["slug"],
                        "variant": method["slug"],
                        "variant_label": method["method"],
                        "source_method": method["source_method"],
                        "target": TARGET,
                        "target_slug": TARGET,
                        "runner_type": method["runner_type"],
                        "config_path": config_path_rel,
                        "expected_epochs": effective_epochs,
                        "batch_size": train_cfg.get("batch_size", ""),
                        "reverse_batch_size": train_cfg.get("reverse", {}).get("batch_size", ""),
                        "results_dir": args.results_dir,
                        "tb_dir": args.tb_dir,
                        "status": "pending",
                        "runtime_gpu": "",
                        "annealing_mode": annealing["label"],
                        "annealing_enabled": bool(annealing["enabled"]),
                        "annealing_steps": max(1, effective_epochs // 2),
                        "vi_regime": vi_regime["slug"],
                        "vi_regime_label": vi_regime["label"],
                        "vi_model_type_override": vi_regime["vi_model_type"],
                        "vi_model_config_path_override": vi_regime["vi_model_config_path"],
                        "extra_overrides": extra_overrides,
                        "config_hash_version": sweep.CONFIG_HASH_VERSION,
                        "config_hash": sweep.effective_config_hash(
                            config_path_rel or config_path,
                            seed=seed,
                            extra_overrides=extra_overrides,
                        ),
                        "config_hash_basis": (
                            "resolved main config plus seed, annealing on/off override, "
                            "10k base epoch override with 5x KSIVI multiplier, VI model type/path "
                            "override, method overrides, and user extra overrides; target/vi/reverse "
                            "config files expanded; scheduler/output/device paths ignored"
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
        description="Run the 72-run Langevin annealing/VI-model ablation sweep with dynamic GPU scheduling."
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
