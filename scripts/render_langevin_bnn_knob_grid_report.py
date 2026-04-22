from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
CAMPAIGN_SLUG = "langevin_bnn_knob_grid_20260422"
CAMPAIGN_DIR = REPO_ROOT / "campaigns" / CAMPAIGN_SLUG
SUMMARY_CSV = CAMPAIGN_DIR / "generated_reports" / "official_completed_runs.csv"
REPORT_MD = CAMPAIGN_DIR / "generated_reports" / "knob_grid_report.md"


METRIC_COLUMNS = {
    "elbo_final": "metric/vi_model/elbo__final",
    "elbo_best": "metric/vi_model/elbo__best",
    "elm_final": "metric/vi_model/kde_expected_log_marginal__final",
    "elm_best": "metric/vi_model/kde_expected_log_marginal__best",
    "rmse_final": "metric/vi_model/rmse__final",
    "rmse_best": "metric/vi_model/rmse__best",
    "test_llk_final": "metric/vi_model/test_llk__final",
    "test_llk_best": "metric/vi_model/test_llk__best",
    "nll_final": "metric/vi_model/nll__final",
    "nll_best": "metric/vi_model/nll__best",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _fmt(row: dict[str, str], metric_key: str) -> str:
    value = _float(row, METRIC_COLUMNS[metric_key])
    if value is None:
        return ""
    return f"{value:.6g}"


def _runtime(row: dict[str, str]) -> str:
    duration = _float(row, "duration_sec")
    if duration is None:
        return ""
    return f"{duration / 60.0:.1f}"


def _manifest_meta() -> dict[str, dict[str, Any]]:
    import json

    manifest = json.loads((CAMPAIGN_DIR / "manifest.json").read_text(encoding="utf-8"))
    return {entry["run_id"]: entry for entry in manifest}


def _enrich(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    meta = _manifest_meta()
    enriched = []
    for row in rows:
        entry = meta.get(row["run_id"], {})
        merged = dict(row)
        for key in ("question", "variant", "method", "vi_lr", "vi_var_lr", "reverse_lr"):
            merged[key] = str(entry.get(key, merged.get(key, "")))
        enriched.append(merged)
    return enriched


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _langevin_rows(rows: list[dict[str, str]], question: str) -> list[list[str]]:
    selected = [row for row in rows if row.get("question") == question]
    return [
        [
            row["run_id"],
            row.get("method", row.get("runner_type", "")),
            row.get("variant", ""),
            row.get("annealing_mode", ""),
            row.get("vi_lr", ""),
            _fmt(row, "elbo_final"),
            _fmt(row, "elbo_best"),
            _fmt(row, "elm_final"),
            _fmt(row, "elm_best"),
            _runtime(row),
        ]
        for row in selected
    ]


def _bnn_rows(rows: list[dict[str, str]]) -> list[list[str]]:
    selected = [row for row in rows if row.get("question") == "q4_dsivi_bnn_batch"]
    selected.sort(key=lambda row: (row["target"], row.get("batch_size", "")))
    return [
        [
            row["target"],
            row.get("variant", ""),
            row.get("batch_size", ""),
            _fmt(row, "test_llk_final"),
            _fmt(row, "test_llk_best"),
            _fmt(row, "rmse_final"),
            _fmt(row, "rmse_best"),
            _fmt(row, "nll_final"),
            _fmt(row, "nll_best"),
            _runtime(row),
        ]
        for row in selected
    ]


def _winner_note(rows: list[dict[str, str]], question: str, metric: str, larger_is_better: bool = True) -> str:
    selected = [row for row in rows if row.get("question") == question]
    metric_col = METRIC_COLUMNS[metric]
    scored = [(row, _float(row, metric_col)) for row in selected]
    scored = [(row, value) for row, value in scored if value is not None]
    if not scored:
        return "- No winner computed; requested metric was unavailable."
    winner, value = max(scored, key=lambda item: item[1]) if larger_is_better else min(scored, key=lambda item: item[1])
    return f"- `{question}` best by {metric.replace('_', ' ')}: `{winner['run_id']}` ({value:.6g})."


def render(rows: list[dict[str, str]]) -> str:
    lines: list[str] = [
        "# Langevin/BNN Knob Grid Report",
        "",
        f"Completed runs summarized: {len(rows)}",
        "",
        "## Headline Notes",
        "",
        _winner_note(rows, "q1_aisivi_vi_model", "elm_best"),
        _winner_note(rows, "q2_ksivi_anneal", "elm_best"),
        _winner_note(rows, "q3_langevin_lr", "elm_best"),
        "",
        "## Q1 AISIVI Langevin VI Model",
        "",
    ]
    lines.extend(
        _table(
            ["Run ID", "Method", "Variant", "Anneal", "VI LR", "ELBO final", "ELBO best", "KDE ELM final", "KDE ELM best", "Minutes"],
            _langevin_rows(rows, "q1_aisivi_vi_model"),
        )
    )
    lines.extend(["", "## Q2 KSIVI Langevin Annealing", ""])
    lines.extend(
        _table(
            ["Run ID", "Method", "Variant", "Anneal", "VI LR", "ELBO final", "ELBO best", "KDE ELM final", "KDE ELM best", "Minutes"],
            _langevin_rows(rows, "q2_ksivi_anneal"),
        )
    )
    lines.extend(["", "## Q3 Langevin Learning Rate", ""])
    lines.extend(
        _table(
            ["Run ID", "Method", "Variant", "Anneal", "VI LR", "ELBO final", "ELBO best", "KDE ELM final", "KDE ELM best", "Minutes"],
            _langevin_rows(rows, "q3_langevin_lr"),
        )
    )
    lines.extend(["", "## Q4 DSIVI BNN Batch Size", ""])
    lines.extend(
        _table(
            ["Target", "Variant", "Batch", "Test LLK final", "Test LLK best", "RMSE final", "RMSE best", "NLL final", "NLL best", "Minutes"],
            _bnn_rows(rows),
        )
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the Langevin/BNN knob grid report.")
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--out", type=Path, default=REPORT_MD)
    args = parser.parse_args()

    rows = _enrich(_read_csv(args.summary_csv))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(rows), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
