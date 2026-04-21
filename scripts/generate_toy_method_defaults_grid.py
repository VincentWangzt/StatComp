from __future__ import annotations

import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from grid_benchmark_common import ensure_dir, load_yaml, metric_budgets, save_json, save_yaml, to_relpath  # noqa: E402


REPO_ROOT = SCRIPT_DIR.parent
CAMPAIGN_SLUG = "toy_method_defaults_20260420"
CAMPAIGN_TITLE = "Toy Method Defaults Grid 2026-04-20"
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

TARGETS = [
    "banana",
    "multimodal",
    "x_shaped",
    "student_uc",
    "8_gaussians",
]

METHODS = [
    {"slug": "sivi", "label": "SIVI", "runner_type": "SIVI", "cost_factor": 1.0},
    {"slug": "uivi", "label": "UIVI", "runner_type": "UIVI", "cost_factor": 5.5},
    {"slug": "aisivi", "label": "AISIVI", "runner_type": "AISIVI", "cost_factor": 3.0},
    {"slug": "dsivi", "label": "DSIVI", "runner_type": "DSIVI", "cost_factor": 2.0},
    {"slug": "ksivi", "label": "KSIVI", "runner_type": "KSIVI", "cost_factor": 1.8},
]

TARGET_COST_FACTORS = {
    "banana": 1.0,
    "multimodal": 1.0,
    "x_shaped": 1.0,
    "student_uc": 1.1,
    "8_gaussians": 1.1,
}


def _run_id(method_slug: str, target: str, smoke: bool = False) -> str:
    prefix = "smoke" if smoke else "official"
    return f"{prefix}_{method_slug}_{target.lower()}"


def _base_config_path(method_slug: str, target: str) -> Path:
    return REPO_ROOT / "configs" / f"{method_slug}_{target}.yaml"


def _expected_annealing(target: str) -> bool:
    return target != "student_uc"


def _annealing_mode(enabled: bool) -> str:
    return "on" if enabled else "off"


def _enable_toy_metrics(config: dict[str, Any], target: str, smoke: bool = False) -> None:
    budgets = deepcopy(metric_budgets(target))
    if smoke:
        budgets.update(
            {
                "kl_num_samples": 128,
                "w2_num_samples": 128,
                "w2_num_projections": 32,
                "mmd_num_samples": 128,
                "ksd_num_samples": 128,
                "fisher_num_samples": 128,
                "fisher_num_is_samples": 32,
                "elbo_batch_size": 64,
                "elbo_num_batches": 1,
                "elbo_num_z_samples": 128,
                "elm_num_ref_samples": 128,
                "elm_num_model_samples": 256,
                "elm_sample_batch_size": 256,
                "elm_dim_chunk": 25,
                "elm_ref_chunk": 128,
                "elm_model_chunk": 1024,
            }
        )

    metric = config.setdefault("metric", {})

    metric.setdefault("kl_ite", {})
    metric["kl_ite"]["enabled"] = True
    metric["kl_ite"]["num_samples"] = budgets["kl_num_samples"]

    metric.setdefault("w2", {})
    metric["w2"]["enabled"] = True
    metric["w2"]["num_samples"] = budgets["w2_num_samples"]
    metric["w2"]["num_projections"] = budgets["w2_num_projections"]

    metric.setdefault("mmd", {})
    metric["mmd"]["enabled"] = True
    metric["mmd"]["num_samples"] = budgets["mmd_num_samples"]

    metric.setdefault("ksd", {})
    metric["ksd"]["enabled"] = True
    metric["ksd"]["num_samples"] = budgets["ksd_num_samples"]

    metric.setdefault("fisher", {})
    metric["fisher"]["enabled"] = True
    metric["fisher"]["num_samples"] = budgets["fisher_num_samples"]
    metric["fisher"]["num_is_samples"] = budgets["fisher_num_is_samples"]

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

    metric.setdefault("bnn", {})
    metric["bnn"]["enabled"] = False


def _validate_default_policy(config: dict[str, Any], source_config: Path, target: str) -> None:
    vi_model_type = config.get("vi_model_type")
    if vi_model_type != "ConditionalGaussian":
        raise RuntimeError(f"{to_relpath(source_config)} has vi_model_type={vi_model_type!r}, expected ConditionalGaussian")

    annealing = config.get("train", {}).get("annealing", {})
    annealing_enabled = bool(annealing.get("enabled", False))
    expected = _expected_annealing(target)
    if annealing_enabled != expected:
        raise RuntimeError(
            f"{to_relpath(source_config)} has annealing.enabled={annealing_enabled!r}, expected {expected!r}"
        )


def _standardize_config(
    config: dict[str, Any],
    method: dict[str, Any],
    target: str,
    smoke: bool = False,
) -> None:
    config["runner_type"] = method["runner_type"]
    config["target_type"] = target
    config["use_cuda"] = True

    train = config.setdefault("train", {})
    if smoke:
        train["epochs"] = 20
    train.setdefault("annealing", {})
    train["annealing"]["enabled"] = _expected_annealing(target)
    train["annealing"]["scheme"] = "linear"
    train["annealing"]["steps"] = 5000

    train.setdefault("log", {})
    train["log"]["metric_log_freq"] = 10 if smoke else train["log"].get("metric_log_freq", 100)
    train["log"]["loss_log_freq"] = train["log"].get("loss_log_freq", 100)
    train["log"]["reverse_log_freq"] = train["log"].get("reverse_log_freq", 500)
    train["log"]["time_avg_window"] = train["log"].get("time_avg_window", 500)

    train["checkpoint"] = {
        "enabled": not smoke,
        "freq": train.get("checkpoint", {}).get("freq", 1000),
    }
    train["sample"] = {
        "freq": train.get("sample", {}).get("freq", 500) if not smoke else 999999,
        "num": train.get("sample", {}).get("num", 10000) if not smoke else 128,
    }
    train["plot"] = {
        "freq": train.get("plot", {}).get("freq", 500) if not smoke else 999999,
        "num": train.get("plot", {}).get("num", 1000) if not smoke else 128,
    }

    config["resume"] = {"enabled": False}
    config["output"] = {
        "results_dir": SMOKE_RESULTS_DIR if smoke else OFFICIAL_RESULTS_DIR,
        "tb_dir": SMOKE_TB_DIR if smoke else OFFICIAL_TB_DIR,
    }
    if smoke:
        config.setdefault("reverse_model", {})
        config["reverse_model"].setdefault("warmup", {})
        config["reverse_model"]["warmup"]["enabled"] = False

    _enable_toy_metrics(config, target, smoke=smoke)

    annealing_enabled = bool(train["annealing"]["enabled"])
    run_id = _run_id(method["slug"], target, smoke=smoke)
    config["campaign"] = {
        "slug": CAMPAIGN_SLUG,
        "run_id": run_id,
        "target": target,
        "method": method["label"],
        "method_slug": method["slug"],
        "annealing_mode": _annealing_mode(annealing_enabled),
        "smoke": smoke,
    }


def _entry(
    config: dict[str, Any],
    config_path: Path,
    source_config: Path,
    method: dict[str, Any],
    target: str,
    smoke: bool = False,
) -> dict[str, Any]:
    train = config["train"]
    annealing_enabled = bool(train["annealing"]["enabled"])
    estimated_cost = float(train["epochs"]) * float(method["cost_factor"]) * TARGET_COST_FACTORS[target]

    return {
        "run_id": config["campaign"]["run_id"],
        "phase": "smoke" if smoke else "official",
        "smoke": smoke,
        "target": target,
        "target_label": target,
        "variant": method["slug"],
        "variant_label": method["label"],
        "method": method["label"],
        "method_slug": method["slug"],
        "runner_type": method["runner_type"],
        "source_config": to_relpath(source_config),
        "config_path": to_relpath(config_path),
        "annealing_mode": _annealing_mode(annealing_enabled),
        "annealing_enabled": annealing_enabled,
        "queue_name": "gpu0",
        "queue_gpu": 0,
        "output_roots": deepcopy(config["output"]),
        "expected_metrics": {
            "elbo": True,
            "expected_log_marginal": True,
            "kl": True,
            "w2": True,
            "ksd": True,
            "mmd": True,
            "fisher": True,
            "bnn": False,
        },
        "epochs": train["epochs"],
        "batch_size": train.get("batch_size", ""),
        "reverse_batch_size": train.get("reverse", {}).get("batch_size", ""),
        "estimated_cost": estimated_cost,
    }


def _write_manifest_csv(entries: list[dict[str, Any]]) -> None:
    fieldnames = [
        "run_id",
        "phase",
        "target",
        "variant",
        "variant_label",
        "method",
        "runner_type",
        "annealing_mode",
        "queue_name",
        "config_path",
        "epochs",
        "batch_size",
        "reverse_batch_size",
        "estimated_cost",
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
        ".\\.venv\\Scripts\\python.exe scripts\\generate_toy_method_defaults_grid.py",
        ".\\.venv\\Scripts\\python.exe scripts\\show_grid_status.py --phase smoke --manifest campaigns\\toy_method_defaults_20260420\\smoke_manifest.json --campaign-dir campaigns\\toy_method_defaults_20260420",
        ".\\.venv\\Scripts\\python.exe scripts\\summarize_grid_benchmark.py --phase official --manifest campaigns\\toy_method_defaults_20260420\\manifest.json --campaign-dir campaigns\\toy_method_defaults_20260420",
        ".\\.venv\\Scripts\\python.exe scripts\\summarize_toy_method_defaults_grid.py",
        ".\\.venv\\Scripts\\python.exe scripts\\fetch_grid_benchmark_artifacts.py --remote-repo ~/ruivi-toy-method-defaults --campaign-slug toy_method_defaults_20260420 --remote-artifact-root /root/autodl-tmp",
        "```",
        "",
        "## Remote Queue Commands",
        "",
        "```bash",
        "source /root/miniconda3/etc/profile.d/conda.sh",
        "conda activate ruivi",
        "mkdir -p /root/autodl-tmp/results/toy_method_defaults_20260420 /root/autodl-tmp/tb_logs/toy_method_defaults_20260420",
        "python scripts/generate_toy_method_defaults_grid.py",
        "python scripts/run_grid_queue.py --phase smoke --queue gpu0 --gpu 0 --manifest campaigns/toy_method_defaults_20260420/smoke_manifest.json --campaign-dir campaigns/toy_method_defaults_20260420",
        "python scripts/run_grid_queue.py --phase official --queue gpu0 --gpu 0 --manifest campaigns/toy_method_defaults_20260420/manifest.json --campaign-dir campaigns/toy_method_defaults_20260420",
        "```",
    ]
    README_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_markdown_template(entries: list[dict[str, Any]], smoke_entries: list[dict[str, Any]]) -> None:
    content = f"""# {CAMPAIGN_TITLE}

## Campaign Header

- Commit SHA:
- Remote worktree: `~/ruivi-toy-method-defaults`
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

## Method Notes

Pending.
"""
    MARKDOWN_PATH.write_text(content, encoding="utf-8")


def main() -> None:
    ensure_dir(CAMPAIGN_DIR)
    ensure_dir(GENERATED_CONFIG_DIR)

    entries: list[dict[str, Any]] = []
    smoke_entries: list[dict[str, Any]] = []

    for target in TARGETS:
        for method in METHODS:
            source_config = _base_config_path(method["slug"], target)
            if not source_config.exists():
                raise FileNotFoundError(source_config)
            base_config = load_yaml(source_config)
            _validate_default_policy(base_config, source_config, target)

            config = deepcopy(base_config)
            _standardize_config(config, method, target)
            config_path = GENERATED_CONFIG_DIR / f"{config['campaign']['run_id']}.yaml"
            save_yaml(config, config_path)
            entries.append(_entry(config, config_path, source_config, method, target))

            if target == "banana":
                smoke_config = deepcopy(base_config)
                _standardize_config(smoke_config, method, target, smoke=True)
                smoke_path = GENERATED_CONFIG_DIR / f"{smoke_config['campaign']['run_id']}.yaml"
                save_yaml(smoke_config, smoke_path)
                smoke_entries.append(_entry(smoke_config, smoke_path, source_config, method, target, smoke=True))

    save_json(entries, MANIFEST_PATH)
    save_json(smoke_entries, SMOKE_MANIFEST_PATH)
    _write_manifest_csv(entries)
    QUEUE_PATH.write_text("\n".join(entry["run_id"] for entry in entries) + "\n", encoding="utf-8")
    _write_readme(entries, smoke_entries)
    _write_markdown_template(entries, smoke_entries)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    smoke_manifest = json.loads(SMOKE_MANIFEST_PATH.read_text(encoding="utf-8"))
    generated_configs = sorted(GENERATED_CONFIG_DIR.glob("*.yaml"))
    if len(manifest) != 25:
        raise RuntimeError(f"Expected 25 official runs, got {len(manifest)}")
    if len(smoke_manifest) != 5:
        raise RuntimeError(f"Expected 5 smoke runs, got {len(smoke_manifest)}")
    if len(generated_configs) != 30:
        raise RuntimeError(f"Expected 30 generated configs, got {len(generated_configs)}")

    print(f"Generated {len(manifest)} official configs and {len(smoke_manifest)} smoke configs.")
    print(f"Manifest: {to_relpath(MANIFEST_PATH)}")
    print(f"Smoke manifest: {to_relpath(SMOKE_MANIFEST_PATH)}")
    print(f"Queue: {to_relpath(QUEUE_PATH)}")


if __name__ == "__main__":
    main()
