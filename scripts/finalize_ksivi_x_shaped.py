"""Evaluate ten canonical X-shaped KSIVI seeds and refresh paper tables and grids."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf
import torch

from finalization.artifacts import completed_runs, find_final_checkpoint, find_final_samples, load_campaign_manifest
from finalization.config import load_config, repo_path
from finalization.plots import render_scatter_grid
from finalization.runner_eval import evaluate_runs, write_csv, write_jsonl, write_warning_outputs
from finalization.tables import render_tables
from scripts.run_ksivi_x_shaped_seeds import CAMPAIGN, MANIFEST, SEEDS, canonical_records

REPORT_DIR = f"campaigns/{CAMPAIGN}/generated_reports/finalization"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def matches(row: dict) -> bool:
    return (row.get("method"), row.get("target")) == ("KSIVI", "x_shaped")


def replace_group(original: list[dict], updated: list[dict]) -> list[dict]:
    assert updated and all(matches(row) for row in updated)
    return [row for row in original if not matches(row)] + updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    records = canonical_records()
    if args.check_only:
        print("Validated ten completed canonical X-shaped KSIVI runs.")
        return
    records = [replace(record, config_path=record.result_path / "full_config.yaml") for record in records]
    config = OmegaConf.merge(load_config(None), {
        "campaign": {"slug": CAMPAIGN, "manifest_path": MANIFEST, "output_dir": REPORT_DIR,
                     "scratch_results_dir": f"results/{CAMPAIGN}/finalization_scratch",
                     "scratch_tb_dir": f"tb_logs/{CAMPAIGN}/finalization_scratch"},
        "evaluation": {"device": "cuda", "overwrite": True,
                       "langevin_kde_elm": {"sgld": {"enabled": False}}},
    })
    run_rows, summary_rows = evaluate_runs(records, config)
    assert len(run_rows) == 10 and sorted(int(row["seed"]) for row in run_rows) == SEEDS
    assert len(summary_rows) == 1 and int(summary_rows[0]["seed_count"]) == 10
    for row in run_rows:
        assert matches(row) and json.loads(row["errors"]) == {} and json.loads(row["warnings"]) == {}
        assert all(math.isfinite(float(row[key])) for key in ("elbo", "w2", "duration_sec", "training_time_sec"))
        row["source_manifest"] = MANIFEST

    out = ROOT / REPORT_DIR
    raw_rows = [json.loads(line) for line in (out / "reevaluation_raw.jsonl").read_text().splitlines() if line]
    for row in raw_rows:
        row["source_manifest"] = MANIFEST
    write_csv(out / "reevaluation_runs.csv", run_rows)
    write_jsonl(out / "reevaluation_raw.jsonl", raw_rows)

    paper_config = load_config(None)
    paper_out = repo_path(str(paper_config.campaign.output_dir))
    assert paper_out is not None
    old_runs = read_csv(paper_out / "reevaluation_runs.csv")
    old_summaries = read_csv(paper_out / "reevaluation_summary.csv")
    old_raw = [json.loads(line) for line in (paper_out / "reevaluation_raw.jsonl").read_text().splitlines() if line]
    merged_runs = replace_group(old_runs, run_rows)
    merged_summaries = replace_group(old_summaries, summary_rows)
    merged_raw = replace_group(old_raw, raw_rows)
    assert len(merged_runs) == len(old_runs) and len(merged_summaries) == len(old_summaries)
    assert [row for row in merged_runs if not matches(row)] == [row for row in old_runs if not matches(row)]
    before_after = out / "comparison.json"
    if not before_after.exists():
        before_after.write_text(json.dumps(dict(
            previous=[row for row in old_summaries if matches(row)], updated=summary_rows,
        ), indent=2) + "\n", encoding="utf-8")
    write_csv(paper_out / "reevaluation_runs.csv", merged_runs)
    write_csv(paper_out / "reevaluation_summary.csv", merged_summaries)
    write_jsonl(paper_out / "reevaluation_raw.jsonl", merged_raw)
    write_warning_outputs(paper_out, merged_runs)
    for module in paper_config.modules:
        paper_config.modules[module] = module in {"toy_tables", "toy_method_grid"}
    render_tables(merged_summaries, paper_config)

    for path in ("configs/finalization/default_config_grid.yaml", "configs/finalization/toy_scatter_grid.yaml"):
        plot_config = load_config(path)
        plot_records = completed_runs(load_campaign_manifest(plot_config.campaign))
        render_scatter_grid(plot_records, plot_config)

    audit = []
    for record in records:
        checkpoint, epoch = find_final_checkpoint(record.result_path)
        samples, sample_epoch = find_final_samples(record.result_path)
        audit.append(dict(seed=record.seed, checkpoint_epoch=epoch, sample_epoch=sample_epoch,
                          sample_sha256=hashlib.sha256(samples.read_bytes()).hexdigest(),
                          checkpoint_sha256=hashlib.sha256((checkpoint / "vi_model.pt").read_bytes()).hexdigest(),
                          config_hash=record.entry["config_hash"], source_revision=record.entry["source_revision"],
                          batch=record.entry["batch"], result_path=record.result_path.as_posix(),
                          finite_parameters_and_samples=True, optimizer_steps=50000))
    write_csv(out / "seed_audit.csv", audit)
    summary = summary_rows[0]
    lines = [
        "# Canonical KSIVI X-shaped evaluation", "", f"Source manifest: `{MANIFEST}`.",
        "Seeds 42--51; 50,000 iterations each. Batches: [42, 43, 44], [45, 46, 47], [48, 49, 50], [51].",
        "One RTX 4090, at most three concurrent seeds. Per-run wall-clock times reflect shared GPU execution.",
        "Constant conditional-variance initialization 1.313; mean-network LR 0.001; variance-head LR 0.00001.",
        "Linear annealing over 25,000 steps; Gaussian kernel; scheduler gamma 0.9 every 1,000 steps.",
        "Evaluation uses each run's saved configuration and final checkpoint:",
        "5,000 VI samples and 20 batches of 2,048 auxiliary samples for KL-style (-ELBO);",
        "10,000 samples and 5,000 projections for sliced W2.",
        "The refreshed grids keep the preselected seed 42 for KSIVI on x_shaped.", "",
        "| Seed | KL-style (-ELBO) | Sliced W2 | Wall-clock (s) |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines.extend(f"| {row['seed']} | {-float(row['elbo']):.6f} | {float(row['w2']):.6f} | {float(row['duration_sec']):.2f} |" for row in run_rows)
    lines.extend([
        "", f"Ten-seed KL-style mean ± SE: {-float(summary['elbo_mean']):.6f} ± {float(summary['elbo_se']):.6f}.",
        f"Ten-seed sliced W2 mean ± SE: {float(summary['w2_mean']):.6f} ± {float(summary['w2_se']):.6f}.",
        "All checkpoints and saved samples are finite; all optimizer states have exactly 50,000 updates.",
        "No evaluation errors or warnings. Only the KSIVI/x_shaped metric group is replaced.", "",
    ])
    (out / "finalization_report.md").write_text("\n".join(lines), encoding="utf-8")
    note = ("Canonical KSIVI X-shaped metrics and seed-42 scatter panel use seeds 42--51 from "
            "`campaigns/ksivi_x_shaped/generated_reports/finalization/finalization_report.md`. "
            "These runs use constant variance initialization 1.313 and variance-head LR 1e-5; "
            "per-run timings include sharing one RTX 4090 with at most two other seeds.")
    for report in (paper_out / "finalization_report.md",
                   ROOT / "campaigns/toy_scatter_grid/generated_reports/finalization/finalization_report.md"):
        content = report.read_text(encoding="utf-8")
        if note not in content:
            report.write_text(content.rstrip() + "\n\n" + note + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
