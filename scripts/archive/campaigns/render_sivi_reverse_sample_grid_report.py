from __future__ import annotations

import _bootstrap  # noqa: F401

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

from run_sivi_reverse_sample_grid import CAMPAIGN_SLUG  # noqa: E402


REPO_ROOT = SCRIPT_DIR.parents[2]
CAMPAIGN_DIR = REPO_ROOT / "campaigns" / CAMPAIGN_SLUG
REPORT_NAME = "sivi_reverse_sample_grid_report.md"
RUN_CSV_NAME = "sivi_reverse_sample_grid_runs.csv"
SUMMARY_CSV_NAME = "sivi_reverse_sample_grid_summary.csv"

ELBO_TAG = "metric/vi_model/elbo"
W2_TAG = "metric/vi_model/w2"
TOTAL_TRAINING_TIME_TAG = "summary/total_training_time"
AVG_EPOCH_TIME_TAG = "summary/avg_epoch_time"


def _repo_path(path: str | Path | None) -> Path | None:
    if not path:
        return None
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _completed_events(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(event["run_id"]): event
        for event in events
        if (
            event.get("status") == "completed"
            or (event.get("status") == "process_finished" and event.get("run_status") == "completed")
        )
        and event.get("run_id")
    }


def _read_metric_points(metrics_csv: Path) -> dict[str, list[tuple[int, float]]]:
    points: dict[str, list[tuple[int, float]]] = defaultdict(list)
    if not metrics_csv.exists():
        return points
    with metrics_csv.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                step = int(float(row["step"]))
                value = float(row["value"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                points[row["tag"]].append((step, value))
    for values in points.values():
        values.sort(key=lambda item: item[0])
    return points


def _last_value(points: list[tuple[int, float]]) -> float | None:
    return points[-1][1] if points else None


def _mean(values: list[float]) -> float | None:
    clean = [value for value in values if math.isfinite(value)]
    return statistics.fmean(clean) if clean else None


def _stdev(values: list[float]) -> float | None:
    clean = [value for value in values if math.isfinite(value)]
    return statistics.stdev(clean) if len(clean) > 1 else None


def _fmt(value: Any, digits: int = 6) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}g}"


def build_run_rows(manifest: list[dict[str, Any]], completed: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in manifest:
        event = completed.get(str(entry["run_id"]))
        if event is None:
            continue

        tb_path = _repo_path(event.get("tb_path"))
        metrics = _read_metric_points(tb_path / "extracted" / "metrics.csv") if tb_path else {}
        epochs = int(entry.get("epochs") or entry.get("expected_epochs") or 0)
        wall_clock_sec = float(event.get("duration_sec") or 0.0)
        training_time = _last_value(metrics.get(TOTAL_TRAINING_TIME_TAG, []))
        avg_epoch_time = _last_value(metrics.get(AVG_EPOCH_TIME_TAG, []))
        if training_time is None:
            training_time = wall_clock_sec
        if avg_epoch_time is None and epochs:
            avg_epoch_time = training_time / epochs

        rows.append(
            {
                "run_id": entry["run_id"],
                "target": entry["target"],
                "seed": entry["seed"],
                "epochs": epochs,
                "reverse_sample_num": entry["reverse_sample_num"],
                "annealing_enabled": entry.get("annealing_enabled", True),
                "wall_clock_sec": wall_clock_sec,
                "wall_clock_min": wall_clock_sec / 60.0,
                "training_time_sec": training_time,
                "training_time_min": training_time / 60.0 if training_time is not None else None,
                "avg_epoch_time_sec": avg_epoch_time,
                "elbo_final": _last_value(metrics.get(ELBO_TAG, [])),
                "w2_final": _last_value(metrics.get(W2_TAG, [])),
                "result_path": event.get("result_path") or "",
                "tb_path": event.get("tb_path") or "",
            }
        )
    return rows


def build_summary_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[(str(row["target"]), int(row["reverse_sample_num"]))].append(row)

    summary: list[dict[str, Any]] = []
    for (target, reverse_sample_num), rows in sorted(grouped.items()):
        out: dict[str, Any] = {
            "target": target,
            "method": "SIVI",
            "reverse_sample_num": reverse_sample_num,
            "runs": len(rows),
            "epochs": rows[0].get("epochs", ""),
            "wall_clock_total_sec": sum(float(row["wall_clock_sec"]) for row in rows),
            "wall_clock_mean_sec": _mean([float(row["wall_clock_sec"]) for row in rows]),
            "training_time_total_sec": sum(float(row["training_time_sec"]) for row in rows if row.get("training_time_sec") is not None),
            "training_time_mean_sec": _mean([float(row["training_time_sec"]) for row in rows if row.get("training_time_sec") is not None]),
            "avg_epoch_time_mean_sec": _mean([float(row["avg_epoch_time_sec"]) for row in rows if row.get("avg_epoch_time_sec") is not None]),
            "avg_epoch_time_sd_sec": _stdev([float(row["avg_epoch_time_sec"]) for row in rows if row.get("avg_epoch_time_sec") is not None]),
            "elbo_final_mean": _mean([float(row["elbo_final"]) for row in rows if row.get("elbo_final") is not None]),
            "elbo_final_sd": _stdev([float(row["elbo_final"]) for row in rows if row.get("elbo_final") is not None]),
            "w2_final_mean": _mean([float(row["w2_final"]) for row in rows if row.get("w2_final") is not None]),
            "w2_final_sd": _stdev([float(row["w2_final"]) for row in rows if row.get("w2_final") is not None]),
        }
        summary.append(out)
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def render_markdown(summary_rows: list[dict[str, Any]], run_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# SIVI Reverse Sample Grid",
        "",
        f"Completed runs summarized: {len(run_rows)}",
        "",
        "Annealing is enabled for all runs. Metrics are final scalar values averaged over "
        "seeds. Training time uses `summary/total_training_time` when available; "
        "wall-clock time is the scheduler process duration.",
        "",
    ]
    target_order = ["banana", "multimodal", "x_shaped", "8_gaussians"]
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        by_target[str(row["target"])].append(row)

    for target in target_order:
        rows = sorted(by_target.get(target, []), key=lambda item: int(item["reverse_sample_num"]))
        if not rows:
            continue
        lines.extend([f"## {target}", ""])
        lines.extend(
            _table(
                [
                    "reverse_sample_num",
                    "Runs",
                    "Epochs",
                    "Avg epoch s",
                    "Train mean min",
                    "Wall total min",
                    "ELBO final mean",
                    "W2 final mean",
                ],
                [
                    [
                        str(row["reverse_sample_num"]),
                        str(row["runs"]),
                        str(row["epochs"]),
                        _fmt(row.get("avg_epoch_time_mean_sec")),
                        _fmt(float(row["training_time_mean_sec"]) / 60.0 if row.get("training_time_mean_sec") is not None else None),
                        _fmt(float(row["wall_clock_total_sec"]) / 60.0),
                        _fmt(row.get("elbo_final_mean")),
                        _fmt(row.get("w2_final_mean")),
                    ]
                    for row in rows
                ],
            )
        )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the SIVI reverse sample grid report.")
    parser.add_argument("--campaign-dir", type=Path, default=CAMPAIGN_DIR)
    args = parser.parse_args()

    campaign_dir = _repo_path(args.campaign_dir) or CAMPAIGN_DIR
    manifest = _load_json(campaign_dir / "manifest.json")
    completed = _completed_events(_load_events(campaign_dir / "runtime" / "events.jsonl"))
    run_rows = build_run_rows(manifest, completed)
    summary_rows = build_summary_rows(run_rows)

    out_dir = campaign_dir / "generated_reports"
    _write_csv(out_dir / RUN_CSV_NAME, run_rows)
    _write_csv(out_dir / SUMMARY_CSV_NAME, summary_rows)
    report_path = out_dir / REPORT_NAME
    report_path.write_text(render_markdown(summary_rows, run_rows), encoding="utf-8")
    print(f"Wrote {report_path}")
    print(f"Wrote {out_dir / RUN_CSV_NAME}")
    print(f"Wrote {out_dir / SUMMARY_CSV_NAME}")


if __name__ == "__main__":
    main()
