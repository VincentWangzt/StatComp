from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_langevin_anneal_vi_ablation import CAMPAIGN_SLUG, KSIVI_EPOCH_MULTIPLIER  # noqa: E402


REPO_ROOT = SCRIPT_DIR.parent
CAMPAIGN_DIR = REPO_ROOT / "campaigns" / CAMPAIGN_SLUG
MANIFEST_PATH = CAMPAIGN_DIR / "manifest.json"
EVENTS_PATH = CAMPAIGN_DIR / "runtime" / "events.jsonl"
OUT_DIR = CAMPAIGN_DIR / "generated_reports"
REPORT_MD = OUT_DIR / "anneal_vi_ablation_report.md"
SUMMARY_CSV = OUT_DIR / "anneal_vi_ablation_group_summary.csv"
RUN_CSV = OUT_DIR / "anneal_vi_ablation_runs.csv"

ELBO_TAG = "metric/vi_model/elbo"
KDE_TAG = "metric/vi_model/kde_expected_log_marginal"
CHECKPOINT_BASE_EPOCHS = (3000, 5000)


def _repo_path(path: str | Path | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _completed_events(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(event["run_id"]): event
        for event in events
        if event.get("status") == "completed" and event.get("run_id")
    }


def _read_metric_points(metrics_csv: Path) -> dict[str, list[tuple[int, float]]]:
    points: dict[str, list[tuple[int, float]]] = defaultdict(list)
    if not metrics_csv.exists():
        return points
    with metrics_csv.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                step = int(row["step"])
                value = float(row["value"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                points[row["tag"]].append((step, value))
    for values in points.values():
        values.sort(key=lambda item: item[0])
    return points


def _final_value(points: list[tuple[int, float]]) -> tuple[int | None, float | None]:
    if not points:
        return None, None
    step, value = points[-1]
    return step, value


def _value_at_or_before(points: list[tuple[int, float]], step: int) -> float | None:
    selected = [value for point_step, value in points if point_step <= step]
    if not selected:
        return None
    return selected[-1]


def _mean(values: list[float]) -> float | None:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return None
    return statistics.fmean(clean)


def _stdev(values: list[float]) -> float | None:
    clean = [value for value in values if math.isfinite(value)]
    if len(clean) < 2:
        return None
    return statistics.stdev(clean)


def _fmt(value: float | None, digits: int = 6) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.{digits}g}"


def _fmt_minutes(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value / 60.0:.2f}"


def _group_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["method"]),
        str(row["annealing_mode"]),
        str(row["vi_regime"]),
    )


def build_run_rows(manifest: list[dict[str, Any]], completed: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in manifest:
        event = completed.get(entry["run_id"])
        if event is None:
            continue

        tb_path = _repo_path(event.get("tb_path"))
        metrics = {}
        if tb_path is not None:
            metrics = _read_metric_points(tb_path / "extracted" / "metrics.csv")

        elbo_points = metrics.get(ELBO_TAG, [])
        kde_points = metrics.get(KDE_TAG, [])
        elbo_final_epoch, elbo_final = _final_value(elbo_points)
        kde_final_epoch, kde_final = _final_value(kde_points)

        effective_epochs = int(entry["effective_epochs"])
        method_slug = str(entry["method_slug"])
        fold = KSIVI_EPOCH_MULTIPLIER if method_slug == "ksivi" else 1
        duration_sec = float(event.get("duration_sec") or 0.0)

        row: dict[str, Any] = {
            "run_id": entry["run_id"],
            "seed": entry["seed"],
            "method": entry["method"],
            "method_slug": method_slug,
            "annealing_mode": entry["annealing_mode"],
            "vi_regime": entry["vi_regime"],
            "vi_regime_label": entry["vi_regime_label"],
            "base_epochs": entry["base_epochs"],
            "effective_epochs": effective_epochs,
            "duration_sec": duration_sec,
            "duration_min": duration_sec / 60.0,
            "per_epoch_sec": duration_sec / effective_epochs,
            "elbo_final": elbo_final,
            "elbo_final_epoch": elbo_final_epoch,
            "kde_final": kde_final,
            "kde_final_epoch": kde_final_epoch,
            "result_path": event.get("result_path") or "",
            "tb_path": event.get("tb_path") or "",
        }
        for base_epoch in CHECKPOINT_BASE_EPOCHS:
            actual_epoch = base_epoch * fold
            row[f"checkpoint_{base_epoch}_actual_epoch"] = actual_epoch
            row[f"elbo_at_{base_epoch}"] = _value_at_or_before(elbo_points, actual_epoch)
            row[f"kde_at_{base_epoch}"] = _value_at_or_before(kde_points, actual_epoch)
        rows.append(row)
    return rows


def build_group_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[_group_key(row)].append(row)

    group_rows: list[dict[str, Any]] = []
    for (method, annealing_mode, vi_regime), rows in sorted(grouped.items()):
        first = rows[0]
        out: dict[str, Any] = {
            "method": method,
            "annealing_mode": annealing_mode,
            "vi_regime": vi_regime,
            "vi_regime_label": first["vi_regime_label"],
            "runs": len(rows),
            "effective_epochs": first["effective_epochs"],
            "duration_total_sec": sum(float(row["duration_sec"]) for row in rows),
            "duration_mean_sec": _mean([float(row["duration_sec"]) for row in rows]),
            "per_epoch_mean_sec": _mean([float(row["per_epoch_sec"]) for row in rows]),
            "elbo_final_mean": _mean([float(row["elbo_final"]) for row in rows if row["elbo_final"] is not None]),
            "elbo_final_sd": _stdev([float(row["elbo_final"]) for row in rows if row["elbo_final"] is not None]),
            "kde_final_mean": _mean([float(row["kde_final"]) for row in rows if row["kde_final"] is not None]),
            "kde_final_sd": _stdev([float(row["kde_final"]) for row in rows if row["kde_final"] is not None]),
        }
        for base_epoch in CHECKPOINT_BASE_EPOCHS:
            out[f"elbo_at_{base_epoch}_mean"] = _mean(
                [float(row[f"elbo_at_{base_epoch}"]) for row in rows if row[f"elbo_at_{base_epoch}"] is not None]
            )
            out[f"kde_at_{base_epoch}_mean"] = _mean(
                [float(row[f"kde_at_{base_epoch}"]) for row in rows if row[f"kde_at_{base_epoch}"] is not None]
            )
        group_rows.append(out)
    return group_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _comparison_lines(group_rows: list[dict[str, Any]], metric: str) -> list[str]:
    by_group = {
        (
            row["method"],
            row["annealing_mode"],
            row["vi_regime"],
        ): row
        for row in group_rows
    }
    methods = sorted({row["method"] for row in group_rows})
    lines = [f"### {metric.upper()} Deltas", ""]
    rows: list[list[str]] = []
    for method in methods:
        for annealing_mode in ("anneal_on", "anneal_off"):
            uniform = by_group.get((method, annealing_mode, "uniform_aisivi"))
            cgglobal = by_group.get((method, annealing_mode, "cgglobal_langevin"))
            if uniform is None or cgglobal is None:
                continue
            uniform_val = uniform.get(f"{metric}_final_mean")
            cgglobal_val = cgglobal.get(f"{metric}_final_mean")
            delta = None if uniform_val is None or cgglobal_val is None else float(uniform_val) - float(cgglobal_val)
            rows.append([method, annealing_mode, "uniform - cgglobal", _fmt(delta)])

        for vi_regime in ("uniform_aisivi", "cgglobal_langevin"):
            anneal_on = by_group.get((method, "anneal_on", vi_regime))
            anneal_off = by_group.get((method, "anneal_off", vi_regime))
            if anneal_on is None or anneal_off is None:
                continue
            on_val = anneal_on.get(f"{metric}_final_mean")
            off_val = anneal_off.get(f"{metric}_final_mean")
            delta = None if on_val is None or off_val is None else float(on_val) - float(off_val)
            rows.append([method, vi_regime, "anneal_on - anneal_off", _fmt(delta)])
    lines.extend(_table(["Method", "Regime", "Comparison", "Delta"], rows))
    return lines


def render_markdown(run_rows: list[dict[str, Any]], group_rows: list[dict[str, Any]]) -> str:
    completed = len(run_rows)
    lines: list[str] = [
        "# Langevin Annealing/VI Ablation Report",
        "",
        f"Completed runs summarized: {completed}",
        "",
        "Checkpoint columns use 3K and 5K base epochs; KSIVI is read at 15K and 25K actual epochs.",
        "",
        "## Final Means",
        "",
    ]
    table_rows = [
        [
            row["method"],
            row["annealing_mode"],
            row["vi_regime"],
            str(row["runs"]),
            _fmt(row.get("elbo_final_mean")),
            _fmt(row.get("elbo_final_sd")),
            _fmt(row.get("kde_final_mean")),
            _fmt(row.get("kde_final_sd")),
            _fmt_minutes(row.get("duration_total_sec")),
            _fmt_minutes(row.get("duration_mean_sec")),
            _fmt(row.get("per_epoch_mean_sec")),
        ]
        for row in group_rows
    ]
    lines.extend(
        _table(
            [
                "Method",
                "Anneal",
                "VI",
                "Runs",
                "ELBO mean",
                "ELBO sd",
                "KDE mean",
                "KDE sd",
                "Total min",
                "Mean min",
                "Sec/epoch",
            ],
            table_rows,
        )
    )
    lines.extend(["", "## 3K and 5K Base-Epoch Means", ""])
    checkpoint_rows = [
        [
            row["method"],
            row["annealing_mode"],
            row["vi_regime"],
            _fmt(row.get("elbo_at_3000_mean")),
            _fmt(row.get("kde_at_3000_mean")),
            _fmt(row.get("elbo_at_5000_mean")),
            _fmt(row.get("kde_at_5000_mean")),
        ]
        for row in group_rows
    ]
    lines.extend(
        _table(
            ["Method", "Anneal", "VI", "ELBO@3K", "KDE@3K", "ELBO@5K", "KDE@5K"],
            checkpoint_rows,
        )
    )
    lines.extend(["", "## Comparisons", ""])
    lines.extend(_comparison_lines(group_rows, "kde"))
    lines.extend([""])
    lines.extend(_comparison_lines(group_rows, "elbo"))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the Langevin annealing/VI ablation report.")
    parser.add_argument("--campaign-dir", type=Path, default=CAMPAIGN_DIR)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--events", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    campaign_dir = _repo_path(args.campaign_dir) or CAMPAIGN_DIR
    manifest_path = _repo_path(args.manifest) if args.manifest else campaign_dir / "manifest.json"
    events_path = _repo_path(args.events) if args.events else campaign_dir / "runtime" / "events.jsonl"
    out_path = _repo_path(args.out) if args.out else campaign_dir / "generated_reports" / REPORT_MD.name

    manifest = _load_json(manifest_path)
    completed = _completed_events(_load_events(events_path))
    run_rows = build_run_rows(manifest, completed)
    group_rows = build_group_rows(run_rows)

    out_dir = out_path.parent
    _write_csv(out_dir / RUN_CSV.name, run_rows)
    _write_csv(out_dir / SUMMARY_CSV.name, group_rows)
    out_path.write_text(render_markdown(run_rows, group_rows), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Wrote {out_dir / RUN_CSV.name}")
    print(f"Wrote {out_dir / SUMMARY_CSV.name}")


if __name__ == "__main__":
    main()
