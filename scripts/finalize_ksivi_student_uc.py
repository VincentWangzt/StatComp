"""Evaluate the five canonical KSIVI Student-t seeds and refresh the paper metrics."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf
import torch

from finalization.artifacts import (
    completed_runs, find_final_checkpoint, find_final_samples, load_manifest, load_sample_z,
)
from finalization.config import load_config, repo_path
from finalization.runner_eval import (
    evaluate_runs, write_csv, write_jsonl, write_warning_outputs,
)
from finalization.tables import render_tables
from scripts.compare_ksivi_student_detached_bandwidth import matched_settings
from scripts.run_default_config_grid_sweep import effective_config_hash

MANIFEST = "campaigns/toy_scatter_ksivi_detached_annealing/manifest.json"
REPORT_DIR = "campaigns/ksivi_student_uc/generated_reports/finalization"
SEEDS = [42, 43, 44, 45, 46]
OVERRIDES = [
    "train.annealing.enabled=true", "train.ksivi.log_p_reg_mode=warmup_only",
    "train.ksivi.log_p_reg=0.05", "train.ksivi.detach_kernel=false",
    "train.ksivi.detach_bandwidth=true", "train.log.metric_log_freq=0",
    "train.plot.freq=1000000000",
]


def is_student_ksivi(row: dict) -> bool:
    return (row.get("target"), row.get("method")) == ("student_uc", "KSIVI")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def canonical_records() -> list:
    manifest = load_manifest(MANIFEST)
    records = sorted(completed_runs(manifest), key=lambda record: record.seed)
    assert len(manifest) == len(records) == 5
    assert [record.seed for record in records] == SEEDS
    reference_settings = None
    snapshots = []
    for record in records:
        assert is_student_ksivi(record.entry)
        assert record.entry["config_hash"] == effective_config_hash(record.config_path, record.seed, OVERRIDES)
        snapshot = record.result_path / "full_config.yaml"
        config = OmegaConf.load(snapshot)
        assert config.seed == record.seed and config.train.epochs == 50000
        assert config.train.batch_size == 128 and config.train.vi.lr == 0.001
        assert config.train.annealing.enabled and config.train.annealing.steps == 25000
        assert config.train.ksivi.kernel == "riesz" and config.train.ksivi.affine_invariant
        assert config.train.ksivi.detach_bandwidth and not config.train.ksivi.detach_kernel
        assert config.train.ksivi.log_p_reg == 0.05 and config.train.ksivi.log_p_reg_mode == "warmup_only"
        assert config.train.log.metric_log_freq == 0 and config.train.plot.freq == 1000000000
        settings = matched_settings(config)
        if reference_settings is None:
            reference_settings = settings
        assert settings == reference_settings, record.seed
        checkpoint, epoch = find_final_checkpoint(record.result_path)
        state = torch.load(checkpoint / "vi_model.pt", map_location="cpu", weights_only=True)
        assert epoch == 50000 and all(torch.isfinite(value).all() for value in state.values())
        sample_path, sample_epoch = find_final_samples(record.result_path)
        samples = load_sample_z(sample_path)
        assert sample_epoch == epoch and len(samples) == 10000 and torch.isfinite(samples).all()
        # Restore the saved architecture and target settings for evaluation.
        snapshots.append(replace(record, config_path=snapshot))
    return snapshots


def evaluation_config():
    config = load_config(None)
    return OmegaConf.merge(config, {
        "campaign": {
            "slug": "ksivi_student_uc", "manifest_path": MANIFEST, "output_dir": REPORT_DIR,
            "scratch_results_dir": "results/ksivi_student_uc/finalization_scratch",
            "scratch_tb_dir": "tb_logs/ksivi_student_uc/finalization_scratch",
        },
        "evaluation": {"device": "cuda", "overwrite": True,
                       "langevin_kde_elm": {"sgld": {"enabled": False}}},
    })


def replace_group(original: list[dict], updated: list[dict]) -> list[dict]:
    assert updated and all(is_student_ksivi(row) for row in updated)
    return [row for row in original if not is_student_ksivi(row)] + updated


def publish_metrics(run_rows: list[dict], summary_rows: list[dict]) -> None:
    assert len(run_rows) == 5 and sorted(int(row["seed"]) for row in run_rows) == SEEDS
    assert len(summary_rows) == 1 and int(summary_rows[0]["seed_count"]) == 5
    for row in run_rows:
        assert is_student_ksivi(row) and json.loads(row["errors"]) == {}
        assert json.loads(row["warnings"]) == {}, row["warnings"]
        for metric in ("elbo", "w2", "w2_trunc_abs_8", "w2_edge_5", "w2_edge_8", "w2_edge_10"):
            assert math.isfinite(float(row[metric])), (row["seed"], metric)
        row["source_manifest"] = MANIFEST
    config_path = repo_path(REPORT_DIR)
    assert config_path is not None
    write_csv(config_path / "reevaluation_runs.csv", run_rows)
    raw_path = config_path / "reevaluation_raw.jsonl"
    raw_rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line]

    paper_config = load_config(None)
    paper_out = repo_path(str(paper_config.campaign.output_dir))
    assert paper_out is not None
    old_runs = read_csv(paper_out / "reevaluation_runs.csv")
    old_summaries = read_csv(paper_out / "reevaluation_summary.csv")
    old_raw = [json.loads(line) for line in (paper_out / "reevaluation_raw.jsonl").read_text().splitlines() if line]
    merged_runs = replace_group(old_runs, run_rows)
    merged_summaries = replace_group(old_summaries, summary_rows)
    merged_raw = replace_group(old_raw, raw_rows)
    assert len(merged_summaries) == len(old_summaries)
    write_csv(paper_out / "reevaluation_runs.csv", merged_runs)
    write_csv(paper_out / "reevaluation_summary.csv", merged_summaries)
    write_jsonl(paper_out / "reevaluation_raw.jsonl", merged_raw)
    write_warning_outputs(paper_out, merged_runs)
    for name in paper_config.modules:
        paper_config.modules[name] = name in {"toy_tables", "toy_method_grid"}
    render_tables(merged_summaries, paper_config)
    warnings = read_csv(paper_out / "reevaluation_warning_summary.csv") if (paper_out / "reevaluation_warning_summary.csv").stat().st_size else []
    paper_report = paper_out / "finalization_report.md"
    report_lines = paper_report.read_text(encoding="utf-8").splitlines()
    report_lines = [f"Per-run rows: {len(merged_runs)}" if line.startswith("Per-run rows:") else line
                    for line in report_lines]
    if "## Warnings" in report_lines:
        report_lines = report_lines[:report_lines.index("## Warnings")]
    if warnings:
        report_lines.extend(["## Warnings", "", "Constrained W2 sampling fallbacks remain in the cached results."])
        report_lines.extend(f"- {row['target']}/{row['method']}/{row['metric']}: {row['count']}" for row in warnings)
    note = "Canonical KSIVI Student-t metrics use seeds 42--46 from `campaigns/ksivi_student_uc/generated_reports/finalization/finalization_report.md`."
    if note not in report_lines:
        report_lines.extend(["", note])
    paper_report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    result = summary_rows[0]
    lines = [
        "# Canonical KSIVI Student-t evaluation", "", f"Source manifest: `{MANIFEST}`.",
        "Seeds: 42, 43, 44, 45, 46. Each checkpoint is at 50,000 iterations.",
        "Annealing: linear over 25,000 iterations. Warmup log-density coefficient: 0.05.",
        "Riesz median bandwidth is detached; sample-input gradients are enabled.", "",
        "Evaluation follows the existing paper workflow: 5,000 VI samples with 20 batches",
        "of 2,048 auxiliary samples for the ELBO; 10,000 accepted samples and 5,000",
        "projections for truncated W2 with coordinate threshold 8.", "",
        "| Seed | KL-style (-ELBO) | Truncated W2 |", "| --- | ---: | ---: |",
    ]
    lines.extend(f"| {row['seed']} | {-float(row['elbo']):.6f} | {float(row['w2_trunc_abs_8']):.6f} |" for row in run_rows)
    lines.extend([
        "", f"Five-seed KL-style mean ± SE: {-float(result['elbo_mean']):.6f} ± {float(result['elbo_se']):.6f}.",
        f"Five-seed truncated W2 mean ± SE: {float(result['w2_trunc_abs_8_mean']):.6f} ± {float(result['w2_trunc_abs_8_se']):.6f}.",
        "", "All evaluation metrics are finite, with no errors or constrained-sampling fallbacks.",
        "The paper metric rows for KSIVI on Student-t are replaced by this five-seed set.",
        "The canonical scatter grid uses its seed 42 checkpoint.", "",
    ])
    (config_path / "finalization_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Validate the five training artifacts without evaluating or publishing.")
    args = parser.parse_args()
    records = canonical_records()
    print("Validated five matching, finite, completed canonical runs.")
    if args.check_only:
        return
    config = evaluation_config()
    run_rows, summary_rows = evaluate_runs(records, config)
    publish_metrics(run_rows, summary_rows)


if __name__ == "__main__":
    main()
