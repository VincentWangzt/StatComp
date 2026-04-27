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

from run_8gaussians_vi_scheduler_grid import CAMPAIGN_SLUG  # noqa: E402


REPO_ROOT = SCRIPT_DIR.parent
CAMPAIGN_DIR = REPO_ROOT / "campaigns" / CAMPAIGN_SLUG
OUT_DIR = CAMPAIGN_DIR / "generated_reports"
REPORT_MD = OUT_DIR / "8gaussians_vi_scheduler_report.md"
RUN_CSV = OUT_DIR / "8gaussians_vi_scheduler_runs.csv"
SUMMARY_CSV = OUT_DIR / "8gaussians_vi_scheduler_group_summary.csv"

METRICS = {
    "elbo": ("metric/vi_model/elbo", "max"),
    "kl": ("metric/vi_model/kl_ite", "min"),
    "w2": ("metric/vi_model/w2", "min"),
}


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
    for metric_points in points.values():
        metric_points.sort(key=lambda item: item[0])
    return points


def _best(points: list[tuple[int, float]], mode: str) -> tuple[int | None, float | None]:
    if not points:
        return None, None
    key = (lambda item: item[1])
    step, value = max(points, key=key) if mode == "max" else min(points, key=key)
    return step, value


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
        event = completed.get(entry["run_id"])
        if event is None:
            continue
        tb_path = _repo_path(event.get("tb_path"))
        metric_points = _read_metric_points(tb_path / "extracted" / "metrics.csv") if tb_path else {}
        duration_sec = float(event.get("duration_sec") or 0.0)
        epochs = int(entry["epochs"])
        row: dict[str, Any] = {
            "run_id": entry["run_id"],
            "seed": entry["seed"],
            "method": entry["method"],
            "method_slug": entry["method_slug"],
            "epochs": epochs,
            "duration_sec": duration_sec,
            "duration_min": duration_sec / 60.0,
            "per_epoch_sec": duration_sec / epochs,
            "result_path": event.get("result_path") or "",
            "tb_path": event.get("tb_path") or "",
        }
        for name, (tag, mode) in METRICS.items():
            points = metric_points.get(tag, [])
            if points:
                row[f"{name}_final_epoch"] = points[-1][0]
                row[f"{name}_final"] = points[-1][1]
            best_epoch, best_value = _best(points, mode)
            row[f"{name}_best_epoch"] = best_epoch
            row[f"{name}_best"] = best_value
        rows.append(row)
    return rows


def build_group_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[str(row["method"])].append(row)

    rows: list[dict[str, Any]] = []
    for method, method_rows in sorted(grouped.items()):
        out: dict[str, Any] = {
            "method": method,
            "runs": len(method_rows),
            "epochs": method_rows[0]["epochs"],
            "duration_total_sec": sum(float(row["duration_sec"]) for row in method_rows),
            "duration_mean_sec": _mean([float(row["duration_sec"]) for row in method_rows]),
            "per_epoch_mean_sec": _mean([float(row["per_epoch_sec"]) for row in method_rows]),
        }
        for metric in METRICS:
            finals = [float(row[f"{metric}_final"]) for row in method_rows if row.get(f"{metric}_final") is not None]
            bests = [float(row[f"{metric}_best"]) for row in method_rows if row.get(f"{metric}_best") is not None]
            out[f"{metric}_final_mean"] = _mean(finals)
            out[f"{metric}_final_sd"] = _stdev(finals)
            out[f"{metric}_best_mean"] = _mean(bests)
            out[f"{metric}_best_sd"] = _stdev(bests)
        rows.append(out)
    return rows


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


def _manifest_overrides(manifest: list[dict[str, Any]]) -> list[str]:
    overrides: list[str] = []
    seen: set[str] = set()
    for entry in manifest:
        for override in entry.get("extra_overrides", []):
            if override not in seen:
                seen.add(override)
                overrides.append(str(override))
    return overrides


def render_markdown(
    group_rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    overrides: list[str],
) -> str:
    override_text = ", ".join(f"`{override}`" for override in overrides)
    lines = [
        "# 8-Gaussians VI Scheduler Grid",
        "",
        f"Completed runs summarized: {len(run_rows)}",
        "",
        f"Overrides: {override_text}. KSIVI excluded.",
        "",
        "## Group Means",
        "",
    ]
    lines.extend(
        _table(
            [
                "Method",
                "Runs",
                "Epochs",
                "ELBO final",
                "ELBO best",
                "KL final",
                "KL best",
                "W2 final",
                "W2 best",
                "Total min",
                "Mean min",
                "Sec/epoch",
            ],
            [
                [
                    row["method"],
                    str(row["runs"]),
                    str(row["epochs"]),
                    _fmt(row.get("elbo_final_mean")),
                    _fmt(row.get("elbo_best_mean")),
                    _fmt(row.get("kl_final_mean")),
                    _fmt(row.get("kl_best_mean")),
                    _fmt(row.get("w2_final_mean")),
                    _fmt(row.get("w2_best_mean")),
                    _fmt(float(row["duration_total_sec"]) / 60.0),
                    _fmt(float(row["duration_mean_sec"]) / 60.0),
                    _fmt(row.get("per_epoch_mean_sec")),
                ]
                for row in group_rows
            ],
        )
    )
    lines.extend(["", "## Per-Seed Finals", ""])
    lines.extend(
        _table(
            ["Run", "Method", "Seed", "ELBO final", "KL final", "W2 final", "Minutes"],
            [
                [
                    row["run_id"],
                    row["method"],
                    str(row["seed"]),
                    _fmt(row.get("elbo_final")),
                    _fmt(row.get("kl_final")),
                    _fmt(row.get("w2_final")),
                    _fmt(row.get("duration_min")),
                ]
                for row in sorted(run_rows, key=lambda item: (str(item["method"]), int(item["seed"])))
            ],
        )
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the 8-Gaussians VI scheduler grid report.")
    parser.add_argument("--campaign-dir", type=Path, default=CAMPAIGN_DIR)
    args = parser.parse_args()

    campaign_dir = _repo_path(args.campaign_dir) or CAMPAIGN_DIR
    manifest = _load_json(campaign_dir / "manifest.json")
    completed = _completed_events(_load_events(campaign_dir / "runtime" / "events.jsonl"))
    run_rows = build_run_rows(manifest, completed)
    group_rows = build_group_rows(run_rows)
    overrides = _manifest_overrides(manifest)

    out_dir = campaign_dir / "generated_reports"
    _write_csv(out_dir / RUN_CSV.name, run_rows)
    _write_csv(out_dir / SUMMARY_CSV.name, group_rows)
    report_path = out_dir / REPORT_MD.name
    report_path.write_text(render_markdown(group_rows, run_rows, overrides), encoding="utf-8")
    print(f"Wrote {report_path}")
    print(f"Wrote {out_dir / RUN_CSV.name}")
    print(f"Wrote {out_dir / SUMMARY_CSV.name}")


if __name__ == "__main__":
    main()
