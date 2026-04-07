from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from grid_benchmark_common import (  # noqa: E402
    BEST_METRIC_MODES,
    CAMPAIGN_DIR,
    discover_queue_names,
    MANIFEST_PATH,
    REPO_ROOT,
    SMOKE_MANIFEST_PATH,
    runtime_dir,
)


def _resolve_repo_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    if path.exists():
        return path
    for anchor in ("tb_logs", "results", "configs", "campaigns"):
        if anchor in path.parts:
            idx = path.parts.index(anchor)
            candidate = REPO_ROOT.joinpath(*path.parts[idx:])
            return candidate
    return REPO_ROOT / path


def _load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_metrics_csv(path: Path) -> dict[str, list[dict[str, float]]]:
    rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows[row["tag"]].append(
                {
                    "step": int(row["step"]),
                    "value": float(row["value"]),
                }
            )
    return rows


def _best_point(points: list[dict[str, float]], mode: str) -> dict[str, float] | None:
    if not points:
        return None
    if mode == "max":
        return max(points, key=lambda item: item["value"])
    return min(points, key=lambda item: item["value"])


def _completed_map(phase: str, queue_names: list[str]) -> dict[str, dict]:
    completed: dict[str, dict] = {}
    for queue in queue_names:
        for event in _load_events(runtime_dir() / f"{phase}_{queue}_events.jsonl"):
            if event.get("status") == "completed":
                completed[event["run_id"]] = event
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize completed benchmark runs.")
    parser.add_argument("--phase", choices=["official", "smoke"], default="official")
    args = parser.parse_args()

    manifest = _load_json(SMOKE_MANIFEST_PATH if args.phase == "smoke" else MANIFEST_PATH)
    completed = _completed_map(args.phase, discover_queue_names(manifest, args.phase))

    out_dir = CAMPAIGN_DIR / "generated_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / f"{args.phase}_completed_runs.csv"
    summary_md = out_dir / f"{args.phase}_completed_runs.md"

    rows: list[dict[str, str | float | int]] = []

    for entry in manifest:
        event = completed.get(entry["run_id"])
        if event is None:
            continue
        tb_path = event.get("tb_path")
        if not tb_path:
            continue
        local_tb_path = _resolve_repo_path(tb_path)
        if local_tb_path is None:
            continue
        metrics_csv = local_tb_path / "extracted" / "metrics.csv"
        metrics = _read_metrics_csv(metrics_csv)

        row: dict[str, str | float | int] = {
            "run_id": entry["run_id"],
            "target": entry["target"],
            "variant": entry["variant_label"],
            "annealing_mode": entry["annealing_mode"],
            "status": event["status"],
            "duration_sec": round(float(event["duration_sec"]), 3),
            "avg_epoch_time_sec": round(float(event["duration_sec"]) / float(entry["epochs"]), 6),
            "epochs": entry["epochs"],
            "batch_size": entry["batch_size"],
            "reverse_batch_size": entry.get("reverse_batch_size") or "",
            "config_path": entry["config_path"],
            "result_path": event.get("result_path") or "",
            "tb_path": local_tb_path.as_posix(),
        }

        for tag, mode in BEST_METRIC_MODES.items():
            points = metrics.get(tag, [])
            if not points:
                continue
            final_point = points[-1]
            best_point = _best_point(points, mode)
            metric_slug = tag.replace("/", "__")
            row[f"{metric_slug}__final"] = round(final_point["value"], 6)
            row[f"{metric_slug}__final_epoch"] = final_point["step"]
            if best_point is not None and math.isfinite(best_point["value"]):
                row[f"{metric_slug}__best"] = round(best_point["value"], 6)
                row[f"{metric_slug}__best_epoch"] = best_point["step"]
        rows.append(row)

    if not rows:
        print("No completed runs found.")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_lines = [
        f"# {args.phase.title()} Completed Runs",
        "",
        f"Completed runs: {len(rows)}",
        "",
        "| Run ID | Target | Variant | Anneal | Duration (s) | Avg epoch (s) |",
        "|--------|--------|---------|--------|--------------|---------------|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['run_id']} | {row['target']} | {row['variant']} | {row['annealing_mode']} | {row['duration_sec']} | {row['avg_epoch_time_sec']} |"
        )
    summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
