from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from grid_benchmark_common import BNN_TARGETS, MANIFEST_CSV_PATH, MANIFEST_PATH, load_yaml, metric_budgets, metric_support, save_json, save_yaml  # noqa: E402


def _refresh_config(config_path: Path, target: str, variant: str | None) -> None:
    config = load_yaml(config_path)
    metric = config.setdefault("metric", {})
    support = metric_support(target)
    budgets = metric_budgets(target)

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
    metric["bnn"]["enabled"] = support["bnn"]
    metric["bnn"]["num_samples"] = budgets["bnn_num_samples"]

    if variant == "sivi" and target in BNN_TARGETS:
        train = config.setdefault("train", {})
        train["reverse_sample_num"] = 2048

    save_yaml(config, config_path)


def _refresh_manifest(target: str, variant: str | None) -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    support = metric_support(target)
    count = 0
    for entry in manifest:
        if entry.get("target") != target:
            continue
        if variant is not None and entry.get("variant") != variant:
            continue
        count += 1
        expected = entry.setdefault("expected_metrics", {})
        expected["kl"] = support["kl"]
        expected["w2"] = support["w2"]
        expected["mmd"] = support["mmd"]
        config_path = Path(entry["config_path"])
        _refresh_config(config_path, target, entry.get("variant"))
    save_json(manifest, MANIFEST_PATH)
    return count


def _refresh_manifest_csv(target: str, variant: str | None) -> None:
    if not MANIFEST_CSV_PATH.exists():
        return
    rows = list(csv.DictReader(MANIFEST_CSV_PATH.open("r", encoding="utf-8", newline="")))
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with MANIFEST_CSV_PATH.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh generated grid configs and manifest metadata for one target.")
    parser.add_argument("--target", required=True)
    parser.add_argument("--variant")
    args = parser.parse_args()

    refreshed = _refresh_manifest(args.target, args.variant)
    _refresh_manifest_csv(args.target, args.variant)
    if args.variant:
        print(f"Refreshed {refreshed} manifest entries and generated configs for target {args.target}, variant {args.variant}.")
    else:
        print(f"Refreshed {refreshed} manifest entries and generated configs for target {args.target}.")


if __name__ == "__main__":
    main()
