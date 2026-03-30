from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from grid_benchmark_common import MANIFEST_PATH, SMOKE_MANIFEST_PATH, runtime_dir  # noqa: E402


def _load_manifest(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Show grid benchmark queue status.")
    parser.add_argument("--phase", choices=["official", "smoke"], default="official")
    args = parser.parse_args()

    manifest = _load_manifest(SMOKE_MANIFEST_PATH if args.phase == "smoke" else MANIFEST_PATH)
    total = len(manifest)
    rt_dir = runtime_dir()

    for queue in ("gpu0", "gpu1"):
        events = _load_events(rt_dir / f"{args.phase}_{queue}_events.jsonl")
        completed = [event for event in events if event.get("status") == "completed"]
        failed = [event for event in events if event.get("status") == "failed"]
        worker_errors = [event for event in events if event.get("status") == "worker_error"]
        current_path = rt_dir / f"{args.phase}_{queue}_current.json"
        current = json.loads(current_path.read_text(encoding="utf-8")) if current_path.exists() else None

        print(f"[{args.phase}::{queue}]")
        print(f"  completed: {len(completed)}")
        print(f"  failed: {len(failed)}")
        print(f"  worker_errors: {len(worker_errors)}")
        if current is not None:
            print(f"  current_status: {current.get('status')}")
            print(f"  current_run: {current.get('run_id')}")
            print(f"  current_gpu: {current.get('gpu')}")
            print(f"  last_result_path: {current.get('result_path')}")
        if completed:
            last = completed[-1]
            print(f"  last_completed: {last.get('run_id')} ({last.get('duration_sec', 0):.1f}s)")
        if failed:
            last_fail = failed[-1]
            print(f"  last_failed: {last_fail.get('run_id')} -> {last_fail.get('failure_reason')}")
        if worker_errors:
            last_err = worker_errors[-1]
            print(f"  last_worker_error: {last_err.get('run_id')} -> {last_err.get('failure_reason')}")
        print()

    total_completed = 0
    total_failed = 0
    total_worker_errors = 0
    for queue in ("gpu0", "gpu1"):
        events = _load_events(rt_dir / f"{args.phase}_{queue}_events.jsonl")
        total_completed += sum(1 for event in events if event.get("status") == "completed")
        total_failed += sum(1 for event in events if event.get("status") == "failed")
        total_worker_errors += sum(1 for event in events if event.get("status") == "worker_error")
    print(f"total_runs: {total}")
    print(f"completed: {total_completed}")
    print(f"failed: {total_failed}")
    print(f"worker_errors: {total_worker_errors}")
    print(f"pending_or_running: {max(total - total_completed - total_failed - total_worker_errors, 0)}")


if __name__ == "__main__":
    main()
