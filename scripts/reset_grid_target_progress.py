from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from grid_benchmark_common import (  # noqa: E402
    discover_queue_names,
    MANIFEST_PATH,
    OFFICIAL_RESULTS_DIR,
    OFFICIAL_TB_DIR,
    REPO_ROOT,
    SMOKE_MANIFEST_PATH,
    SMOKE_RESULTS_DIR,
    SMOKE_TB_DIR,
    runtime_dir,
)


def _load_manifest(phase: str) -> list[dict]:
    manifest_path = SMOKE_MANIFEST_PATH if phase == "smoke" else MANIFEST_PATH
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _matching_entries(target: str, phase: str, variant: str | None) -> list[dict]:
    entries = []
    for entry in _load_manifest(phase):
        if entry.get("target") != target:
            continue
        if entry.get("phase", "official") != phase:
            continue
        if variant is not None and entry.get("variant") != variant:
            continue
        entries.append(entry)
    return entries


def _target_run_ids(entries: list[dict]) -> set[str]:
    return {
        entry["run_id"]
        for entry in entries
    }


def _rewrite_events(path: Path, target_run_ids: set[str]) -> int:
    if not path.exists():
        return 0
    kept_lines: list[str] = []
    removed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("run_id") in target_run_ids:
            removed += 1
            continue
        kept_lines.append(line)
    text = "\n".join(kept_lines)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")
    return removed


def _reset_current(path: Path, target_run_ids: set[str]) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("run_id") not in target_run_ids:
        return False
    path.unlink()
    return True


def _remove_console_logs(console_root: Path, target_run_ids: set[str]) -> int:
    removed = 0
    for run_id in target_run_ids:
        log_path = console_root / f"{run_id}.log"
        if log_path.exists():
            log_path.unlink()
            removed += 1
    return removed


def _remove_target_dirs(root: Path, target: str, runner_types: set[str]) -> int:
    if not root.exists():
        return 0
    removed = 0
    for runner_dir in root.iterdir():
        if runner_types and runner_dir.name not in runner_types:
            continue
        target_dir = runner_dir / target
        if target_dir.exists():
            shutil.rmtree(target_dir)
            removed += 1
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset grid campaign progress and artifacts for one target.")
    parser.add_argument("--target", required=True)
    parser.add_argument("--phase", choices=["official", "smoke"], default="official")
    parser.add_argument("--variant")
    args = parser.parse_args()

    manifest = _load_manifest(args.phase)
    entries = _matching_entries(args.target, args.phase, args.variant)
    target_run_ids = _target_run_ids(entries)
    runner_types = {entry.get("runner_type") for entry in entries if entry.get("runner_type")}
    rt_dir = runtime_dir()
    console_root = rt_dir / "console_logs"
    queue_names = discover_queue_names(manifest, args.phase)

    removed_events = 0
    removed_current = 0
    for queue in queue_names:
        removed_events += _rewrite_events(rt_dir / f"{args.phase}_{queue}_events.jsonl", target_run_ids)
        removed_current += int(_reset_current(rt_dir / f"{args.phase}_{queue}_current.json", target_run_ids))

    removed_logs = _remove_console_logs(console_root, target_run_ids)

    if args.phase == "official":
        results_root = REPO_ROOT / OFFICIAL_RESULTS_DIR
        tb_root = REPO_ROOT / OFFICIAL_TB_DIR
    else:
        results_root = REPO_ROOT / SMOKE_RESULTS_DIR
        tb_root = REPO_ROOT / SMOKE_TB_DIR

    removed_result_dirs = _remove_target_dirs(results_root, args.target, runner_types)
    removed_tb_dirs = _remove_target_dirs(tb_root, args.target, runner_types)

    print(
        json.dumps(
            {
                "target": args.target,
                "phase": args.phase,
                "variant": args.variant,
                "run_ids": sorted(target_run_ids),
                "removed_event_records": removed_events,
                "removed_current_files": removed_current,
                "removed_console_logs": removed_logs,
                "removed_result_dirs": removed_result_dirs,
                "removed_tb_dirs": removed_tb_dirs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
