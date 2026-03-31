from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from grid_benchmark_common import MANIFEST_CSV_PATH, MANIFEST_PATH, load_yaml, metric_budgets, metric_support, save_json, save_yaml  # noqa: E402


def _refresh_config(config_path: Path, target: str) -> None:
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

    metric.setdefault("bnn", {})
    metric["bnn"]["enabled"] = support["bnn"]
    metric["bnn"]["num_samples"] = budgets["bnn_num_samples"]

    save_yaml(config, config_path)


def _refresh_manifest(target: str) -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    support = metric_support(target)
    count = 0
    for entry in manifest:
        if entry.get("target") != target:
            continue
        count += 1
        expected = entry.setdefault("expected_metrics", {})
        expected["kl"] = support["kl"]
        expected["w2"] = support["w2"]
        expected["mmd"] = support["mmd"]
        config_path = Path(entry["config_path"])
        _refresh_config(config_path, target)
    save_json(manifest, MANIFEST_PATH)
    return count


def _refresh_manifest_csv(target: str) -> None:
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
    args = parser.parse_args()

    refreshed = _refresh_manifest(args.target)
    _refresh_manifest_csv(args.target)
    print(f"Refreshed {refreshed} manifest entries and generated configs for target {args.target}.")


if __name__ == "__main__":
    main()
