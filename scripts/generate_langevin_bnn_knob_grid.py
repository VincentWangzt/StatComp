from __future__ import annotations

import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from omegaconf import OmegaConf

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from grid_benchmark_common import ensure_dir, load_yaml, metric_budgets, save_json, save_yaml, to_relpath  # noqa: E402


REPO_ROOT = SCRIPT_DIR.parent
CAMPAIGN_SLUG = "langevin_bnn_knob_grid_20260422"
CAMPAIGN_TITLE = "Langevin/BNN Knob Grid 2026-04-22"
CAMPAIGN_DIR = REPO_ROOT / "campaigns" / CAMPAIGN_SLUG
GENERATED_CONFIG_DIR = REPO_ROOT / "configs" / "generated" / CAMPAIGN_SLUG
MARKDOWN_PATH = REPO_ROOT / f"{CAMPAIGN_SLUG}.md"

AUTODL_ARTIFACT_ROOT = "/root/autodl-tmp"
OFFICIAL_RESULTS_DIR = f"{AUTODL_ARTIFACT_ROOT}/results/{CAMPAIGN_SLUG}/official"
OFFICIAL_TB_DIR = f"{AUTODL_ARTIFACT_ROOT}/tb_logs/{CAMPAIGN_SLUG}/official"
SMOKE_RESULTS_DIR = f"{AUTODL_ARTIFACT_ROOT}/results/{CAMPAIGN_SLUG}/smoke"
SMOKE_TB_DIR = f"{AUTODL_ARTIFACT_ROOT}/tb_logs/{CAMPAIGN_SLUG}/smoke"

MANIFEST_PATH = CAMPAIGN_DIR / "manifest.json"
MANIFEST_CSV_PATH = CAMPAIGN_DIR / "manifest.csv"
SMOKE_MANIFEST_PATH = CAMPAIGN_DIR / "smoke_manifest.json"
QUEUE_PATH = CAMPAIGN_DIR / "queue_gpu0.txt"
README_PATH = CAMPAIGN_DIR / "README.md"

BNN_TARGETS = [
    "Bnn_boston",
    "Bnn_concrete",
    "Bnn_power",
    "Bnn_protein",
    "Bnn_winered",
]


def _source_config_path(method: str, target: str) -> Path:
    return REPO_ROOT / "configs" / f"{method}_{target}.yaml"


def _metadata(config: dict[str, Any], run_id: str, question: str, variant: str, smoke: bool) -> None:
    config["campaign"] = {
        "slug": CAMPAIGN_SLUG,
        "run_id": run_id,
        "question": question,
        "variant": variant,
        "target": config["target_type"],
        "method": config["runner_type"],
        "smoke": smoke,
    }
    config["resume"] = {"enabled": False}
    config["output"] = {
        "results_dir": SMOKE_RESULTS_DIR if smoke else OFFICIAL_RESULTS_DIR,
        "tb_dir": SMOKE_TB_DIR if smoke else OFFICIAL_TB_DIR,
    }


def _annealing_mode(config: dict[str, Any]) -> str:
    enabled = bool(config.get("train", {}).get("annealing", {}).get("enabled", False))
    return "on" if enabled else "off"


def _expected_metrics(target: str) -> dict[str, bool]:
    is_bnn = target in BNN_TARGETS
    is_langevin = target == "Langevin_post"
    return {
        "elbo": is_langevin,
        "expected_log_marginal": is_langevin,
        "w2": is_langevin,
        "bnn": is_bnn,
        "rmse": is_bnn,
        "test_llk": is_bnn,
        "nll": is_bnn,
    }


def _entry(
    config: dict[str, Any],
    config_path: Path,
    source_config: Path,
    run_id: str,
    question: str,
    variant: str,
    smoke: bool,
) -> dict[str, Any]:
    train = config["train"]
    target = config["target_type"]
    return {
        "run_id": run_id,
        "phase": "smoke" if smoke else "official",
        "smoke": smoke,
        "question": question,
        "variant": variant,
        "variant_label": variant,
        "target": target,
        "target_label": target,
        "method": config["runner_type"],
        "method_slug": str(config["runner_type"]).lower(),
        "runner_type": config["runner_type"],
        "source_config": to_relpath(source_config),
        "config_path": to_relpath(config_path),
        "annealing_mode": _annealing_mode(config),
        "annealing_enabled": bool(train.get("annealing", {}).get("enabled", False)),
        "queue_name": "gpu0",
        "queue_gpu": 0,
        "output_roots": deepcopy(config["output"]),
        "expected_metrics": _expected_metrics(target),
        "epochs": train["epochs"],
        "batch_size": train.get("batch_size", ""),
        "reverse_batch_size": train.get("reverse", {}).get("batch_size", ""),
        "vi_lr": train.get("vi", {}).get("lr", ""),
        "vi_var_lr": train.get("vi", {}).get("var_lr", ""),
        "reverse_lr": train.get("reverse", {}).get("lr", ""),
    }


def _write_config(config: dict[str, Any], run_id: str) -> Path:
    config_path = GENERATED_CONFIG_DIR / f"{run_id}.yaml"
    save_yaml(config, config_path)
    return config_path


def _load_base(method: str, target: str) -> tuple[dict[str, Any], Path]:
    source_config = _source_config_path(method, target)
    if not source_config.exists():
        raise FileNotFoundError(source_config)
    return load_yaml(source_config), source_config


def _smoke_metrics(config: dict[str, Any], target: str) -> None:
    train = config.setdefault("train", {})
    train["epochs"] = 2
    train.setdefault("log", {})
    train["log"]["metric_log_freq"] = 1
    train["log"]["loss_log_freq"] = 1
    train["log"]["reverse_log_freq"] = 1
    train["checkpoint"] = {"enabled": False, "freq": 999999}
    train["sample"] = {"freq": 999999, "num": 64}
    train["plot"] = {"freq": 999999, "num": 64}
    if "reverse" in train:
        train["reverse"]["epochs"] = 1
        train["reverse"]["batch_size"] = min(int(train["reverse"].get("batch_size", 128)), 128)
    if "reverse_sample_num" in train:
        train["reverse_sample_num"] = min(int(train["reverse_sample_num"]), 64)
    if "pretrain" in train:
        train["pretrain"]["enabled"] = False
    if "reverse_model" in config:
        config["reverse_model"].setdefault("warmup", {})
        config["reverse_model"]["warmup"]["enabled"] = False

    metric = config.setdefault("metric", {})
    if target == "Langevin_post":
        budgets = deepcopy(metric_budgets(target))
        budgets.update(
            {
                "w2_num_samples": 128,
                "w2_num_projections": 16,
                "elbo_batch_size": 32,
                "elbo_num_batches": 1,
                "elbo_num_z_samples": 64,
                "elm_num_ref_samples": 64,
                "elm_num_model_samples": 128,
                "elm_sample_batch_size": 128,
                "elm_dim_chunk": 25,
                "elm_ref_chunk": 64,
                "elm_model_chunk": 128,
            }
        )
        metric.setdefault("w2", {})
        metric["w2"]["enabled"] = True
        metric["w2"]["num_samples"] = budgets["w2_num_samples"]
        metric["w2"]["num_projections"] = budgets["w2_num_projections"]
        metric.setdefault("elbo", {})
        metric["elbo"]["enabled"] = True
        metric["elbo"]["batch_size"] = budgets["elbo_batch_size"]
        metric["elbo"]["num_batches"] = budgets["elbo_num_batches"]
        metric["elbo"]["num_z_samples"] = budgets["elbo_num_z_samples"]
        metric.setdefault("expected_log_marginal", {})
        metric["expected_log_marginal"]["enabled"] = True
        metric["expected_log_marginal"]["num_ref_samples"] = budgets["elm_num_ref_samples"]
        metric["expected_log_marginal"]["num_model_samples"] = budgets["elm_num_model_samples"]
        metric["expected_log_marginal"]["sample_batch_size"] = budgets["elm_sample_batch_size"]
        metric["expected_log_marginal"]["dim_chunk"] = budgets["elm_dim_chunk"]
        metric["expected_log_marginal"]["ref_chunk"] = budgets["elm_ref_chunk"]
        metric["expected_log_marginal"]["model_chunk"] = budgets["elm_model_chunk"]
        metric["expected_log_marginal"]["min_bandwidth"] = 1.0e-6
        metric["expected_log_marginal"]["dtype"] = "float32"
    else:
        metric.setdefault("bnn", {})
        metric["bnn"]["enabled"] = True
        metric["bnn"]["num_samples"] = 16

    for name in ("kl_ite", "fisher", "ksd", "mmd"):
        metric.setdefault(name, {})
        metric[name]["enabled"] = False


def _q1_runs() -> list[tuple[str, str, str, str, Callable[[dict[str, Any]], None]]]:
    def vi_default(_: dict[str, Any]) -> None:
        return None

    def vi_cgglobal(config: dict[str, Any]) -> None:
        config["vi_model_type"] = "ConditionalGaussianGlobal"
        config["vi_model_config_path"] = "configs/vi_models/ConditionalGaussianGlobal-Langevin.yaml"

    return [
        ("official_q1_aisivi_langevin_vi_default", "q1_aisivi_vi_model", "aisivi_vi_default", "aisivi", vi_default),
        ("official_q1_aisivi_langevin_vi_cgglobal", "q1_aisivi_vi_model", "aisivi_vi_cgglobal", "aisivi", vi_cgglobal),
    ]


def _q2_runs() -> list[tuple[str, str, str, str, Callable[[dict[str, Any]], None]]]:
    def anneal_off(_: dict[str, Any]) -> None:
        return None

    def anneal_on25k(config: dict[str, Any]) -> None:
        config["train"].setdefault("annealing", {})
        config["train"]["annealing"]["enabled"] = True
        config["train"]["annealing"]["steps"] = 25000

    return [
        ("official_q2_ksivi_langevin_anneal_off", "q2_ksivi_anneal", "ksivi_anneal_off_default", "ksivi", anneal_off),
        ("official_q2_ksivi_langevin_anneal_on25k", "q2_ksivi_anneal", "ksivi_anneal_on25k", "ksivi", anneal_on25k),
    ]


def _q3_runs() -> list[tuple[str, str, str, str, Callable[[dict[str, Any]], None]]]:
    runs = []
    for method in ("dsivi", "ksivi"):
        for lr_label, lr in (("lr2e4", 2.0e-4), ("lr1e3", 1.0e-3)):
            def apply_lr(config: dict[str, Any], lr_value: float = lr, method_slug: str = method) -> None:
                config["train"].setdefault("vi", {})
                config["train"]["vi"]["lr"] = lr_value
                if method_slug == "ksivi":
                    config["train"]["vi"]["var_lr"] = lr_value

            runs.append(
                (
                    f"official_q3_{method}_langevin_{lr_label}",
                    "q3_langevin_lr",
                    f"{method}_{lr_label}",
                    method,
                    apply_lr,
                )
            )
    return runs


def _q4_runs() -> list[tuple[str, str, str, str, str, Callable[[dict[str, Any]], None]]]:
    runs = []
    for target in BNN_TARGETS:
        short = target.removeprefix("Bnn_").lower()

        def keep_default(_: dict[str, Any]) -> None:
            return None

        def bs128(config: dict[str, Any]) -> None:
            config["train"]["batch_size"] = 128

        runs.append(
            (
                f"official_q4_dsivi_{short}_bs1024",
                "q4_dsivi_bnn_batch",
                "dsivi_bs1024_default",
                "dsivi",
                target,
                keep_default,
            )
        )
        runs.append(
            (
                f"official_q4_dsivi_{short}_bs128",
                "q4_dsivi_bnn_batch",
                "dsivi_bs128",
                "dsivi",
                target,
                bs128,
            )
        )
    return runs


def _official_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for run_id, question, variant, method, apply in [*_q1_runs(), *_q2_runs(), *_q3_runs()]:
        target = "Langevin_post"
        config, source_config = _load_base(method, target)
        apply(config)
        _metadata(config, run_id, question, variant, smoke=False)
        config_path = _write_config(config, run_id)
        entries.append(_entry(config, config_path, source_config, run_id, question, variant, smoke=False))

    for run_id, question, variant, method, target, apply in _q4_runs():
        config, source_config = _load_base(method, target)
        apply(config)
        _metadata(config, run_id, question, variant, smoke=False)
        config_path = _write_config(config, run_id)
        entries.append(_entry(config, config_path, source_config, run_id, question, variant, smoke=False))
    return entries


def _smoke_entries() -> list[dict[str, Any]]:
    specs = [
        ("smoke_aisivi_langevin_vi_default", "q1_aisivi_vi_model", "aisivi_vi_default", "aisivi", "Langevin_post", lambda _cfg: None),
        ("smoke_ksivi_langevin_anneal_on25k", "q2_ksivi_anneal", "ksivi_anneal_on25k", "ksivi", "Langevin_post", lambda cfg: (cfg["train"]["annealing"].update({"enabled": True, "steps": 25000}))),
        ("smoke_dsivi_langevin_lr2e4", "q3_langevin_lr", "dsivi_lr2e4", "dsivi", "Langevin_post", lambda cfg: cfg["train"]["vi"].update({"lr": 2.0e-4})),
        ("smoke_dsivi_boston_bs128", "q4_dsivi_bnn_batch", "dsivi_bs128", "dsivi", "Bnn_boston", lambda cfg: cfg["train"].update({"batch_size": 128})),
    ]
    entries: list[dict[str, Any]] = []
    for run_id, question, variant, method, target, apply in specs:
        config, source_config = _load_base(method, target)
        apply(config)
        _smoke_metrics(config, target)
        _metadata(config, run_id, question, variant, smoke=True)
        config_path = _write_config(config, run_id)
        entries.append(_entry(config, config_path, source_config, run_id, question, variant, smoke=True))
    return entries


def _write_manifest_csv(entries: list[dict[str, Any]]) -> None:
    fieldnames = [
        "run_id",
        "phase",
        "question",
        "target",
        "method",
        "variant",
        "annealing_mode",
        "queue_name",
        "config_path",
        "epochs",
        "batch_size",
        "reverse_batch_size",
        "vi_lr",
        "vi_var_lr",
        "reverse_lr",
    ]
    with MANIFEST_CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow({key: entry.get(key) for key in fieldnames})


def _write_readme(entries: list[dict[str, Any]], smoke_entries: list[dict[str, Any]]) -> None:
    lines = [
        f"# {CAMPAIGN_TITLE}",
        "",
        f"- Official runs: {len(entries)}",
        f"- Smoke runs: {len(smoke_entries)}",
        f"- Artifact root: `{AUTODL_ARTIFACT_ROOT}`",
        f"- Generated configs: `{to_relpath(GENERATED_CONFIG_DIR)}`",
        f"- Manifest: `{to_relpath(MANIFEST_PATH)}`",
        f"- Smoke manifest: `{to_relpath(SMOKE_MANIFEST_PATH)}`",
        f"- Queue: `{to_relpath(QUEUE_PATH)}`",
        "",
        "## Local Commands",
        "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe scripts\\generate_langevin_bnn_knob_grid.py",
        ".\\.venv\\Scripts\\python.exe scripts\\show_grid_status.py --phase smoke --manifest campaigns\\langevin_bnn_knob_grid_20260422\\smoke_manifest.json --campaign-dir campaigns\\langevin_bnn_knob_grid_20260422",
        ".\\.venv\\Scripts\\python.exe scripts\\show_grid_status.py --phase official --manifest campaigns\\langevin_bnn_knob_grid_20260422\\manifest.json --campaign-dir campaigns\\langevin_bnn_knob_grid_20260422",
        ".\\.venv\\Scripts\\python.exe scripts\\summarize_grid_benchmark.py --phase official --manifest campaigns\\langevin_bnn_knob_grid_20260422\\manifest.json --campaign-dir campaigns\\langevin_bnn_knob_grid_20260422",
        ".\\.venv\\Scripts\\python.exe scripts\\render_langevin_bnn_knob_grid_report.py",
        ".\\.venv\\Scripts\\python.exe scripts\\fetch_grid_benchmark_artifacts.py --remote-repo ~/ruivi --campaign-slug langevin_bnn_knob_grid_20260422 --remote-artifact-root /root/autodl-tmp",
        "```",
        "",
        "## Remote Queue Commands",
        "",
        "```bash",
        "source /root/miniconda3/etc/profile.d/conda.sh",
        "conda activate ruivi",
        "mkdir -p /root/autodl-tmp/results/langevin_bnn_knob_grid_20260422 /root/autodl-tmp/tb_logs/langevin_bnn_knob_grid_20260422",
        "python scripts/run_grid_queue.py --phase smoke --queue gpu0 --gpu 0 --manifest campaigns/langevin_bnn_knob_grid_20260422/smoke_manifest.json --campaign-dir campaigns/langevin_bnn_knob_grid_20260422",
        "python scripts/run_grid_queue.py --phase official --queue gpu0 --gpu 0 --manifest campaigns/langevin_bnn_knob_grid_20260422/manifest.json --campaign-dir campaigns/langevin_bnn_knob_grid_20260422",
        "```",
    ]
    README_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_markdown_template(entries: list[dict[str, Any]], smoke_entries: list[dict[str, Any]]) -> None:
    content = f"""# {CAMPAIGN_TITLE}

## Campaign Header

- Commit SHA:
- Remote worktree: `~/ruivi`
- Remote artifact root: `{AUTODL_ARTIFACT_ROOT}`
- Remote GPU: RTX 3080 queue `gpu0`
- Official run count: {len(entries)}
- Smoke run count: {len(smoke_entries)}

## Progress Table

| Status | Count |
|--------|-------|
| Pending | {len(entries)} |
| Running | 0 |
| Completed | 0 |
| Failed | 0 |

## Monitoring Log

| Time | Check Type | Notes |
|------|------------|-------|

## Failure Log

| Time | Run ID | Issue | Resolution |
|------|--------|-------|------------|

## Final Report

Pending.
"""
    MARKDOWN_PATH.write_text(content, encoding="utf-8")


def _validate_configs(entries: list[dict[str, Any]], expected_count: int) -> None:
    if len(entries) != expected_count:
        raise RuntimeError(f"Expected {expected_count} entries, got {len(entries)}")
    seen = set()
    for entry in entries:
        if entry["run_id"] in seen:
            raise RuntimeError(f"Duplicate run_id: {entry['run_id']}")
        seen.add(entry["run_id"])
        cfg_path = REPO_ROOT / entry["config_path"]
        loaded = OmegaConf.load(cfg_path)
        if loaded.campaign.run_id != entry["run_id"]:
            raise RuntimeError(f"{entry['config_path']} campaign.run_id mismatch")
        if str(loaded.output.results_dir) != entry["output_roots"]["results_dir"]:
            raise RuntimeError(f"{entry['config_path']} output.results_dir mismatch")


def main() -> None:
    ensure_dir(CAMPAIGN_DIR)
    ensure_dir(GENERATED_CONFIG_DIR)

    for old_config in GENERATED_CONFIG_DIR.glob("*.yaml"):
        old_config.unlink()

    entries = _official_entries()
    smoke_entries = _smoke_entries()

    _validate_configs(entries, 18)
    _validate_configs(smoke_entries, 4)

    save_json(entries, MANIFEST_PATH)
    save_json(smoke_entries, SMOKE_MANIFEST_PATH)
    _write_manifest_csv(entries)
    QUEUE_PATH.write_text("\n".join(entry["run_id"] for entry in entries) + "\n", encoding="utf-8")
    _write_readme(entries, smoke_entries)
    _write_markdown_template(entries, smoke_entries)

    generated_configs = sorted(GENERATED_CONFIG_DIR.glob("*.yaml"))
    if len(generated_configs) != 22:
        raise RuntimeError(f"Expected 22 generated configs, got {len(generated_configs)}")

    print(f"Generated {len(entries)} official configs and {len(smoke_entries)} smoke configs.")
    print(f"Manifest: {to_relpath(MANIFEST_PATH)}")
    print(f"Smoke manifest: {to_relpath(SMOKE_MANIFEST_PATH)}")
    print(f"Queue: {to_relpath(QUEUE_PATH)}")


if __name__ == "__main__":
    main()
