from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
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
    DEFAULT_QUEUE_COUNT,
    GENERATED_CONFIG_DIR,
    MANIFEST_CSV_PATH,
    MANIFEST_PATH,
    MARKDOWN_PATH,
    METHOD_VARIANTS,
    OFFICIAL_RESULTS_DIR,
    OFFICIAL_TB_DIR,
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
    metric_budgets,
    metric_support,
    queue_name_for,
    queue_names,
    queue_path_for,
    run_id_for,
    save_json,
    save_yaml,
    target_schedule,
    to_relpath,
)


def _enable_metrics(config: dict, target: str) -> None:
    support = metric_support(target)
    budgets = metric_budgets(target)
    metric = config.setdefault("metric", {})

    metric.setdefault("kl_ite", {})
    metric["kl_ite"]["enabled"] = support["kl"]
    metric["kl_ite"]["num_samples"] = budgets["kl_num_samples"]

    metric.setdefault("w2", {})
    metric["w2"]["enabled"] = support["w2"]
    metric["w2"]["num_samples"] = budgets["w2_num_samples"]
    metric["w2"]["num_projections"] = budgets["w2_num_projections"]

    metric.setdefault("mmd", {})
    metric["mmd"]["enabled"] = support["mmd"]
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
    metric["expected_log_marginal"]["enabled"] = support["mmd"]
    metric["expected_log_marginal"]["num_ref_samples"] = budgets["elm_num_ref_samples"]
    metric["expected_log_marginal"]["num_model_samples"] = budgets["elm_num_model_samples"]
    metric["expected_log_marginal"]["sample_batch_size"] = budgets["elm_sample_batch_size"]
    metric["expected_log_marginal"]["dim_chunk"] = budgets["elm_dim_chunk"]
    metric["expected_log_marginal"]["ref_chunk"] = budgets["elm_ref_chunk"]
    metric["expected_log_marginal"]["model_chunk"] = budgets["elm_model_chunk"]
    metric["expected_log_marginal"]["min_bandwidth"] = 1.0e-6
    metric["expected_log_marginal"]["dtype"] = "float32"

    metric.setdefault("bnn", {})
    metric["bnn"]["enabled"] = target in BNN_TARGETS
    metric["bnn"]["num_samples"] = budgets["bnn_num_samples"]


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
    if variant in {"sivi", "uivi", "rsivi", "dsivi_default"}:
        config["vi_model_type"] = "ConditionalGaussian"
        config.pop("vi_model", None)
        config.pop("vi_model_config_path", None)

    if variant == "aisivi":
        train = config.setdefault("train", {})
        train["grad_clip"] = 10.0
        if target == "Langevin_post":
            config["vi_model_type"] = "ConditionalGaussianGlobalUniform"
            config[
                "vi_model_config_path"
            ] = "configs/vi_models/ConditionalGaussianGlobalUniform-AISIVI.yaml"
            config[
                "reverse_model_config_path"
            ] = "configs/reverse_models/ConditionalRealNVP-AISIVI-Langevin.yaml"
            train["batch_size"] = 128
            train["reverse_sample_num"] = 256
            train.setdefault("vi", {})["lr"] = 2.0e-4
            reverse = train.setdefault("reverse", {})
            reverse["lr"] = 2.0e-4
            reverse["weight_decay"] = 0.0
            reverse["batch_size"] = 128
        elif target in BNN_TARGETS:
            config["vi_model_type"] = "ConditionalGaussianGlobalUniform"
            config[
                "vi_model_config_path"
            ] = "configs/vi_models/ConditionalGaussianGlobalUniform-Bnn-aisivi.yaml"
            config[
                "reverse_model_config_path"
            ] = "configs/reverse_models/ConditionalRealNVP-AISIVI-Bnn.yaml"
            train["batch_size"] = 100
            train["reverse_sample_num"] = 256
            reverse = train.setdefault("reverse", {})
            reverse["lr"] = 5.0e-5
            reverse["weight_decay"] = 0.0
            reverse["batch_size"] = 100
            reverse["scheduler"] = {
                "type": "StepLR",
                "step_size": 5000,
                "gamma": 0.01,
            }

    if variant == "sivi" and target in BNN_TARGETS:
        config.setdefault("train", {})
        config["train"]["reverse_sample_num"] = 2048

    if variant == "ksivi_custom":
        if target in BNN_TARGETS:
            config["vi_model_type"] = "ConditionalGaussianGlobal"
            config[
                "vi_model_config_path"
            ] = "configs/vi_models/ConditionalGaussianGlobal-Bnn-ksivi.yaml"
            config.pop("vi_model", None)
            train = config.setdefault("train", {})
            target_cfg = config.setdefault("target", {})
            target_data = target_cfg.setdefault("data", {})
            metric = config.setdefault("metric", {})
            metric_bnn = metric.setdefault("bnn", {})

            target_data["batch_mode"] = "cyclic"
            target_data["dev_fraction"] = 0.1
            target_data["dev_max_size"] = 500

            train["pretrain"] = {
                "enabled": True,
                "steps": 100,
                "lr": 1.0e-2,
                "batch_size": 100,
            }
            train["ema"] = {"enabled": True, "beta": 0.999}
            train.setdefault("vi", {})
            train["vi"]["lr"] = 1.0e-3
            train["vi"]["var_lr"] = 1.0e-3
            train["vi"]["betas"] = [0.9, 0.999]
            train["vi"].setdefault("scheduler", {})
            train["vi"]["scheduler"]["type"] = "StepLR"
            train["vi"]["scheduler"]["step_size"] = 3000
            train["vi"]["scheduler"]["gamma"] = 0.9
            train.setdefault("annealing", {})
            train["annealing"]["enabled"] = False
            train.setdefault("ksivi", {})
            train["ksivi"]["statistic"] = "v"
            train["ksivi"]["kernel"] = "gaussian"
            train["ksivi"]["detach_kernel"] = False
            train["ksivi"]["log_p_reg"] = 1.0
            train["ksivi"]["log_p_reg_mode"] = "always"
            train.setdefault("log", {})
            train["log"]["metric_log_freq"] = 1000
            train["log"]["loss_log_freq"] = 100
            train["log"]["reverse_log_freq"] = 500
            train.setdefault("checkpoint", {})
            train["checkpoint"]["enabled"] = True
            train["checkpoint"]["freq"] = 5000
            train.setdefault("sample", {})
            train["sample"]["freq"] = 1000
            train["sample"]["num"] = 100
            train.setdefault("plot", {})
            train["plot"]["freq"] = 999999
            train["plot"]["num"] = 100
            metric_bnn["enabled"] = True
            metric_bnn["num_samples"] = 100
            metric.setdefault("fisher", {})
            metric["fisher"]["enabled"] = False
            metric.setdefault("expected_log_marginal", {})
            metric["expected_log_marginal"]["enabled"] = False
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


def _assign_gpu_queues(entries: list[dict], num_gpus: int) -> None:
    if num_gpus < 1:
        raise ValueError(f"num_gpus must be at least 1, got {num_gpus}.")

    queue_costs = {gpu: 0.0 for gpu in range(num_gpus)}
    for entry in sorted(entries, key=lambda item: item["estimated_cost"], reverse=True):
        gpu = min(queue_costs, key=queue_costs.get)
        entry["queue_gpu"] = gpu
        entry["queue_name"] = queue_name_for(gpu)
        queue_costs[gpu] += entry["estimated_cost"]


def _smoke_manifest(entries: list[dict], num_gpus: int) -> list[dict]:
    smoke_queue_overrides = {
        "official_on_banana_sivi": 0,
        "official_on_bnn_yacht_uivi": 0,
        "official_on_banana_ksivi_custom": 0,
        "official_on_banana_ksivi_standard_cg": 1,
        "official_on_bnn_yacht_dsivi_bs4096_rbs2048": 1,
    }
    smoke_entries = []
    by_id = {entry["run_id"]: entry for entry in entries}
    for run_id in SMOKE_RUNS:
        base = deepcopy(by_id[run_id])
        base["phase"] = "smoke"
        base["smoke"] = True
        preferred_gpu = smoke_queue_overrides[run_id]
        queue_gpu = preferred_gpu if preferred_gpu < num_gpus else preferred_gpu % num_gpus
        queue_name = queue_name_for(queue_gpu)
        base["queue_name"] = queue_name
        base["queue_gpu"] = queue_gpu
        base["output_overrides"] = {
            "results_dir": SMOKE_RESULTS_DIR,
            "tb_dir": SMOKE_TB_DIR,
        }
        smoke_entries.append(base)
    return smoke_entries


def _write_markdown_template(entries: list[dict], num_gpus: int) -> None:
    content = f"""# {CAMPAIGN_TITLE}

## Campaign Header

- Commit SHA:
- Remote environment: fill in the actual host GPU inventory, Python version, and PyTorch build.
- Official run count: {len(entries)}
- Queue plan: {num_gpus} independent single-GPU queue(s)

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


def _write_readme(entries: list[dict], smoke_entries: list[dict], num_gpus: int) -> None:
    remote_cmds = [
        "source /root/miniconda3/etc/profile.d/conda.sh",
        "conda activate ruivi",
    ]
    smoke_queues = sorted({entry["queue_name"] for entry in smoke_entries})
    official_queues = queue_names(num_gpus)
    for queue_name in smoke_queues:
        queue_gpu = queue_name.removeprefix("gpu")
        remote_cmds.append(
            f"python scripts/run_grid_queue.py --phase smoke --queue {queue_name} --gpu {queue_gpu}"
        )
    for queue_name in official_queues:
        queue_gpu = queue_name.removeprefix("gpu")
        remote_cmds.append(
            f"python scripts/run_grid_queue.py --phase official --queue {queue_name} --gpu {queue_gpu}"
        )

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
        *remote_cmds,
        "```",
    ]
    README_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the grid benchmark campaign files.")
    parser.add_argument("--num-gpus", type=int, default=DEFAULT_QUEUE_COUNT)
    args = parser.parse_args()

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
                        "expected_log_marginal": support["mmd"],
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

    _assign_gpu_queues(manifest_entries, args.num_gpus)
    smoke_entries = _smoke_manifest(manifest_entries, args.num_gpus)

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

    expected_queue_files = {queue_path_for(queue_name).name for queue_name in queue_names(args.num_gpus)}
    for queue_file in CAMPAIGN_DIR.glob("queue_gpu*.txt"):
        if queue_file.name not in expected_queue_files:
            queue_file.unlink()
    for queue_name in queue_names(args.num_gpus):
        queue_ids = [entry["run_id"] for entry in manifest_entries if entry["queue_name"] == queue_name]
        queue_text = "\n".join(queue_ids)
        if queue_text:
            queue_text += "\n"
        queue_path_for(queue_name).write_text(queue_text, encoding="utf-8")

    _write_markdown_template(manifest_entries, args.num_gpus)
    _write_readme(manifest_entries, smoke_entries, args.num_gpus)

    print(f"Generated {len(manifest_entries)} official configs in {to_relpath(GENERATED_CONFIG_DIR)}")
    print(f"Manifest: {to_relpath(MANIFEST_PATH)}")
    print(f"Smoke manifest: {to_relpath(SMOKE_MANIFEST_PATH)}")
    print(f"Markdown template: {to_relpath(MARKDOWN_PATH)}")
    print(f"Queues: {', '.join(queue_names(args.num_gpus))}")


if __name__ == "__main__":
    main()
