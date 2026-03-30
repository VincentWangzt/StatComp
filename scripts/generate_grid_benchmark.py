from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path

from grid_benchmark_common import (
    ANNEALING_MODES,
    BASELINE_TARGETS,
    BNN_TARGETS,
    CAMPAIGN_DIR,
    CAMPAIGN_SLUG,
    CAMPAIGN_TITLE,
    GENERATED_CONFIG_DIR,
    MANIFEST_CSV_PATH,
    MANIFEST_PATH,
    MARKDOWN_PATH,
    METHOD_VARIANTS,
    OFFICIAL_RESULTS_DIR,
    OFFICIAL_TB_DIR,
    QUEUE_GPU0_PATH,
    QUEUE_GPU1_PATH,
    README_PATH,
    REPO_ROOT,
    SMOKE_MANIFEST_PATH,
    SMOKE_RUNS,
    SMOKE_RESULTS_DIR,
    SMOKE_TB_DIR,
    TARGETS,
    TARGET_COST_FACTORS,
    VARIANT_SPECS,
    display_target,
    ensure_dir,
    load_yaml,
    metric_support,
    run_id_for,
    save_json,
    save_yaml,
    target_schedule,
    to_relpath,
)


def _enable_metrics(config: dict, target: str) -> None:
    support = metric_support(target)
    metric = config.setdefault("metric", {})

    metric.setdefault("kl_ite", {})
    metric["kl_ite"]["enabled"] = support["kl"]
    metric["kl_ite"]["num_samples"] = 10000

    metric.setdefault("w2", {})
    metric["w2"]["enabled"] = support["w2"]
    metric["w2"]["num_samples"] = 10000
    metric["w2"]["num_projections"] = 1000

    metric.setdefault("mmd", {})
    metric["mmd"]["enabled"] = support["mmd"]
    metric["mmd"]["num_samples"] = 1000

    metric.setdefault("ksd", {})
    metric["ksd"]["enabled"] = True
    metric["ksd"]["num_samples"] = 2000

    metric.setdefault("fisher", {})
    metric["fisher"]["enabled"] = True
    metric["fisher"]["num_samples"] = 1000
    metric["fisher"]["num_is_samples"] = 512

    metric.setdefault("elbo", {})
    metric["elbo"]["enabled"] = True
    metric["elbo"]["batch_size"] = 512
    metric["elbo"]["num_batches"] = 2
    metric["elbo"]["num_z_samples"] = 1024

    metric.setdefault("bnn", {})
    metric["bnn"]["enabled"] = target in BNN_TARGETS
    metric["bnn"]["num_samples"] = 500


def _standardize_common(config: dict, target: str, variant: str, anneal_enabled: bool) -> None:
    train = config.setdefault("train", {})
    log_cfg = train.setdefault("log", {})
    log_cfg["metric_log_freq"] = 100

    train.setdefault("annealing", {})
    train["annealing"]["enabled"] = anneal_enabled
    train["annealing"]["scheme"] = "linear"
    train["annealing"]["steps"] = 5000

    lr, step_size, gamma = target_schedule(target)
    train.setdefault("vi", {})
    train["vi"]["lr"] = lr
    train["vi"].setdefault("scheduler", {})
    train["vi"]["scheduler"]["type"] = "StepLR"
    train["vi"]["scheduler"]["step_size"] = step_size
    train["vi"]["scheduler"]["gamma"] = gamma

    if variant == "ksivi_custom":
        train["vi"]["var_lr"] = lr

    config["resume"] = {"enabled": False}
    config["output"] = {
        "results_dir": OFFICIAL_RESULTS_DIR,
        "tb_dir": OFFICIAL_TB_DIR,
    }

    _enable_metrics(config, target)


def _apply_variant_overrides(config: dict, target: str, variant: str) -> None:
    if variant in {"sivi", "uivi", "rsivi", "aisivi", "dsivi_default"}:
        config["vi_model_type"] = "ConditionalGaussian"
        config.pop("vi_model", None)
        config.pop("vi_model_config_path", None)

    if variant == "ksivi_custom":
        pass
    elif variant == "ksivi_standard_cg":
        config["vi_model_type"] = "ConditionalGaussian"
        config.pop("vi_model", None)
        config.pop("vi_model_config_path", None)
        train = config.setdefault("train", {})
        train["pretrain"] = {"enabled": False}
        train["ema"] = {"enabled": False}
        train.setdefault("vi", {})
        train["vi"].pop("var_lr", None)
    elif variant == "dsivi_default":
        config["vi_model_type"] = "ConditionalGaussian"
        config.pop("vi_model_config_path", None)
    elif variant == "dsivi_bs4096_rbs2048":
        config["vi_model_type"] = "ConditionalGaussian"
        config.pop("vi_model_config_path", None)
        config["train"]["batch_size"] = 4096
        config["train"]["reverse"]["batch_size"] = 2048
    elif variant == "dsivi_bs4096_rbs4096":
        config["vi_model_type"] = "ConditionalGaussian"
        config.pop("vi_model_config_path", None)
        config["train"]["batch_size"] = 4096
        config["train"]["reverse"]["batch_size"] = 4096

    if variant == "rsivi":
        lr, _, _ = target_schedule(target)
        config["train"]["vi"]["lr"] = lr


def _annotate_config(config: dict, run_id: str, target: str, variant: str, annealing_mode: str) -> None:
    config["campaign"] = {
        "slug": CAMPAIGN_SLUG,
        "run_id": run_id,
        "target": target,
        "variant": variant,
        "annealing_mode": annealing_mode,
    }


def _estimated_cost(config: dict, target: str, variant: str) -> float:
    epochs = float(config["train"]["epochs"])
    return epochs * VARIANT_SPECS[variant]["cost_factor"] * TARGET_COST_FACTORS[target]


def _assign_gpu_queues(entries: list[dict]) -> None:
    queue_costs = {0: 0.0, 1: 0.0}
    for entry in sorted(entries, key=lambda item: item["estimated_cost"], reverse=True):
        gpu = 0 if queue_costs[0] <= queue_costs[1] else 1
        entry["queue_gpu"] = gpu
        entry["queue_name"] = f"gpu{gpu}"
        queue_costs[gpu] += entry["estimated_cost"]


def _smoke_manifest(entries: list[dict]) -> list[dict]:
    smoke_queue_overrides = {
        "official_on_banana_sivi": ("gpu0", 0),
        "official_on_bnn_yacht_uivi": ("gpu0", 0),
        "official_on_banana_ksivi_custom": ("gpu0", 0),
        "official_on_banana_ksivi_standard_cg": ("gpu1", 1),
        "official_on_bnn_yacht_dsivi_bs4096_rbs2048": ("gpu1", 1),
    }
    smoke_entries = []
    by_id = {entry["run_id"]: entry for entry in entries}
    for run_id in SMOKE_RUNS:
        base = deepcopy(by_id[run_id])
        base["phase"] = "smoke"
        base["smoke"] = True
        queue_name, queue_gpu = smoke_queue_overrides[run_id]
        base["queue_name"] = queue_name
        base["queue_gpu"] = queue_gpu
        base["output_overrides"] = {
            "results_dir": SMOKE_RESULTS_DIR,
            "tb_dir": SMOKE_TB_DIR,
        }
        smoke_entries.append(base)
    return smoke_entries


def _write_markdown_template(entries: list[dict]) -> None:
    content = f"""# {CAMPAIGN_TITLE}

## Campaign Header

- Commit SHA:
- Remote environment: 2x RTX 3080, Python 3.14.2, PyTorch 2.9.0+cu126
- Official run count: {len(entries)}
- Queue plan: GPU0 + GPU1 independent single-GPU queues

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

| Time | Run ID | GPU | Issue | Resolution |
|------|--------|-----|-------|------------|

## Per-Target Summary Tables

Update manually using script-generated summaries at each 2-hour manual check.

## End-of-Campaign Summary

Pending.
"""
    MARKDOWN_PATH.write_text(content, encoding="utf-8")


def _write_readme(entries: list[dict], smoke_entries: list[dict]) -> None:
    lines = [
        f"# {CAMPAIGN_TITLE}",
        "",
        f"- Official runs: {len(entries)}",
        f"- Smoke runs: {len(smoke_entries)}",
        f"- Generated configs: `{to_relpath(GENERATED_CONFIG_DIR)}`",
        f"- Manifest: `{to_relpath(MANIFEST_PATH)}`",
        f"- Smoke manifest: `{to_relpath(SMOKE_MANIFEST_PATH)}`",
        f"- Markdown log: `{to_relpath(MARKDOWN_PATH)}`",
        "",
        "## Local Commands",
        "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe scripts\\generate_grid_benchmark.py",
        ".\\.venv\\Scripts\\python.exe scripts\\fetch_grid_benchmark_artifacts.py",
        ".\\.venv\\Scripts\\python.exe scripts\\show_grid_status.py --phase official",
        ".\\.venv\\Scripts\\python.exe scripts\\summarize_grid_benchmark.py --phase official",
        ".\\.venv\\Scripts\\python.exe scripts\\manual_check_grid_benchmark.py",
        "```",
        "",
        "## Remote Queue Commands",
        "",
        "```bash",
        "source /root/miniconda3/etc/profile.d/conda.sh",
        "conda activate ruivi",
        "python scripts/run_grid_queue.py --phase smoke --queue gpu0 --gpu 0",
        "python scripts/run_grid_queue.py --phase official --queue gpu0 --gpu 0",
        "python scripts/run_grid_queue.py --phase official --queue gpu1 --gpu 1",
        "```",
    ]
    README_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dir(CAMPAIGN_DIR)
    ensure_dir(GENERATED_CONFIG_DIR)

    manifest_entries: list[dict] = []

    for target in TARGETS:
        for annealing_mode, annealing_enabled in ANNEALING_MODES.items():
            for variant in METHOD_VARIANTS:
                spec = VARIANT_SPECS[variant]
                base_config_path = REPO_ROOT / "configs" / f"{spec['source_method']}_{target}.yaml"
                config = load_yaml(base_config_path)

                _standardize_common(config, target, variant, annealing_enabled)
                _apply_variant_overrides(config, target, variant)

                run_id = run_id_for(target, variant, annealing_mode)
                _annotate_config(config, run_id, target, variant, annealing_mode)

                config_path = GENERATED_CONFIG_DIR / f"{run_id}.yaml"
                save_yaml(config, config_path)

                support = metric_support(target)
                estimated_cost = _estimated_cost(config, target, variant)
                entry = {
                    "run_id": run_id,
                    "phase": "official",
                    "smoke": False,
                    "target": target,
                    "target_label": display_target(target),
                    "variant": variant,
                    "variant_label": spec["label"],
                    "runner_type": spec["runner_type"],
                    "source_config": to_relpath(base_config_path),
                    "config_path": to_relpath(config_path),
                    "annealing_mode": annealing_mode,
                    "annealing_enabled": annealing_enabled,
                    "expected_metrics": {
                        "elbo": True,
                        "kl": support["kl"],
                        "w2": support["w2"],
                        "ksd": True,
                        "mmd": support["mmd"],
                        "fisher": True,
                        "bnn": support["bnn"],
                    },
                    "epochs": config["train"]["epochs"],
                    "batch_size": config["train"]["batch_size"],
                    "reverse_batch_size": config.get("train", {}).get("reverse", {}).get("batch_size"),
                    "estimated_cost": estimated_cost,
                }
                manifest_entries.append(entry)

    _assign_gpu_queues(manifest_entries)
    smoke_entries = _smoke_manifest(manifest_entries)

    save_json(manifest_entries, MANIFEST_PATH)
    save_json(smoke_entries, SMOKE_MANIFEST_PATH)

    with MANIFEST_CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "run_id",
            "phase",
            "target",
            "variant",
            "variant_label",
            "runner_type",
            "annealing_mode",
            "queue_name",
            "config_path",
            "epochs",
            "batch_size",
            "reverse_batch_size",
            "estimated_cost",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for entry in manifest_entries:
            writer.writerow({key: entry.get(key) for key in fieldnames})

    gpu0_ids = [entry["run_id"] for entry in manifest_entries if entry["queue_gpu"] == 0]
    gpu1_ids = [entry["run_id"] for entry in manifest_entries if entry["queue_gpu"] == 1]
    QUEUE_GPU0_PATH.write_text("\n".join(gpu0_ids) + "\n", encoding="utf-8")
    QUEUE_GPU1_PATH.write_text("\n".join(gpu1_ids) + "\n", encoding="utf-8")

    _write_markdown_template(manifest_entries)
    _write_readme(manifest_entries, smoke_entries)

    print(f"Generated {len(manifest_entries)} official configs in {to_relpath(GENERATED_CONFIG_DIR)}")
    print(f"Manifest: {to_relpath(MANIFEST_PATH)}")
    print(f"Smoke manifest: {to_relpath(SMOKE_MANIFEST_PATH)}")
    print(f"Markdown template: {to_relpath(MARKDOWN_PATH)}")


if __name__ == "__main__":
    main()
