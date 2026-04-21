from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from grid_benchmark_common import (  # noqa: E402
    CAMPAIGN_DIR,
    MANIFEST_PATH,
    OFFICIAL_RESULTS_DIR,
    OFFICIAL_TB_DIR,
    REPO_ROOT,
    SMOKE_MANIFEST_PATH,
    SMOKE_RESULTS_DIR,
    SMOKE_TB_DIR,
    discover_queue_names,
    ensure_dir,
    queue_index_from_name,
    queue_names_from_manifest,
    runtime_dir,
    to_relpath,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_manifest(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_done_ids(events_path: Path) -> tuple[set[str], set[str]]:
    completed: set[str] = set()
    failed: set[str] = set()
    if not events_path.exists():
        return completed, failed
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        status = event.get("status")
        run_id = event.get("run_id")
        if status == "completed":
            completed.add(run_id)
        elif status == "failed":
            failed.add(run_id)
    return completed, failed


def append_event(events_path: Path, payload: dict) -> None:
    ensure_dir(events_path.parent)
    with events_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def write_current_status(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _campaign_runtime_dir(args: argparse.Namespace) -> Path:
    if args.campaign_dir is None:
        return runtime_dir()
    return _repo_path(args.campaign_dir) / "runtime"


def _build_output_roots(entry: dict, args: argparse.Namespace) -> tuple[str, str]:
    default_results_dir = SMOKE_RESULTS_DIR if args.phase == "smoke" else OFFICIAL_RESULTS_DIR
    default_tb_dir = SMOKE_TB_DIR if args.phase == "smoke" else OFFICIAL_TB_DIR
    output_roots = entry.get("output_roots") or entry.get("output_overrides") or {}
    if not isinstance(output_roots, dict):
        output_roots = {}
    return (
        str(output_roots.get("results_dir") or default_results_dir),
        str(output_roots.get("tb_dir") or default_tb_dir),
    )


def _manifest_path(args: argparse.Namespace) -> Path:
    if args.manifest is not None:
        return _repo_path(args.manifest)
    if args.phase == "smoke":
        return SMOKE_MANIFEST_PATH
    return MANIFEST_PATH


def _known_queue_names(manifest: list[dict], args: argparse.Namespace) -> list[str]:
    if args.manifest is not None or args.campaign_dir is not None:
        return queue_names_from_manifest(manifest)
    return discover_queue_names(manifest, args.phase)


def _launch_command(entry: dict, gpu: int, args: argparse.Namespace) -> list[str]:
    results_dir, tb_dir = _build_output_roots(entry, args)
    cmd = [
        sys.executable,
        "src.py",
        "--config",
        entry["config_path"],
        f"cuda_visible_devices={gpu}",
        f"output.results_dir={results_dir}",
        f"output.tb_dir={tb_dir}",
    ]
    return cmd


def _expected_tb_path(result_path: Path, entry: dict, args: argparse.Namespace) -> Path:
    timestamp = result_path.name
    _, tb_dir = _build_output_roots(entry, args)
    return REPO_ROOT / tb_dir / result_path.parent.parent.name / result_path.parent.name / timestamp


def _run_extractor(tb_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "utils/extract_tensorboard_run.py",
            str(tb_path),
            "--out-dir",
            str(tb_path / "extracted"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _safe_relpath(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return to_relpath(path)
    except Exception:
        return str(path)


def _tail_log(log_path: Path, num_lines: int = 20) -> list[str]:
    if not log_path.exists():
        return []
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-num_lines:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one GPU queue for the grid benchmark.")
    parser.add_argument("--phase", choices=["official", "smoke"], default="official")
    parser.add_argument("--queue", required=True)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--campaign-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--continue-past-failed",
        action="store_true",
        help="Skip runs already recorded as failed and continue with later queue entries.",
    )
    args = parser.parse_args()

    manifest_path = _manifest_path(args)
    manifest = load_manifest(manifest_path)
    known_queues = _known_queue_names(manifest, args)
    if args.queue not in known_queues:
        parser.error(f"Unknown queue {args.queue!r}. Known queues: {', '.join(known_queues)}")
    if args.gpu is None:
        inferred_gpu = queue_index_from_name(args.queue)
        if inferred_gpu is None:
            parser.error("--gpu is required when --queue does not follow the gpuN naming convention.")
        args.gpu = inferred_gpu

    queue_entries = [entry for entry in manifest if entry.get("queue_name", args.queue) == args.queue]
    if args.limit is not None:
        queue_entries = queue_entries[:args.limit]

    runtime = _campaign_runtime_dir(args)
    ensure_dir(runtime)
    events_path = runtime / f"{args.phase}_{args.queue}_events.jsonl"
    current_path = runtime / f"{args.phase}_{args.queue}_current.json"
    console_root = runtime / "console_logs"
    ensure_dir(console_root)

    completed, failed = load_done_ids(events_path)
    queue_finished_cleanly = True
    last_event: dict | None = None

    for entry in queue_entries:
        run_id = entry["run_id"]
        if run_id in completed:
            continue
        if run_id in failed:
            if args.continue_past_failed:
                continue
            queue_finished_cleanly = False
            break

        console_log = console_root / f"{run_id}.log"
        cmd = _launch_command(entry, args.gpu, args)
        start_wall = time.time()
        start_perf = time.perf_counter()

        current_status = {
            "status": "running",
            "run_id": run_id,
            "queue": args.queue,
            "gpu": args.gpu,
            "phase": args.phase,
            "started_at": utc_now(),
            "command": cmd,
            "config_path": entry["config_path"],
            "result_path": None,
            "tb_path": None,
        }
        write_current_status(current_path, current_status)
        append_event(
            events_path,
            {
                "status": "started",
                "run_id": run_id,
                "queue": args.queue,
                "gpu": args.gpu,
                "phase": args.phase,
                "started_at": current_status["started_at"],
            },
        )

        result_path: Path | None = None
        process: subprocess.Popen[str] | None = None
        try:
            with console_log.open("w", encoding="utf-8") as log_fh:
                process = subprocess.Popen(
                    cmd,
                    cwd=REPO_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    log_fh.write(line)
                    log_fh.flush()
                    marker = "Artifacts will be saved to: "
                    if marker in line:
                        result_path = Path(line.split(marker, 1)[1].strip())
                        current_status["result_path"] = str(result_path)
                        current_status["tb_path"] = str(_expected_tb_path(result_path, entry, args))
                        write_current_status(current_path, current_status)

                process.wait()

            end_wall = time.time()
            duration_sec = time.perf_counter() - start_perf
            exit_code = process.returncode

            failure_reason = None
            extractor_stdout = ""
            extractor_stderr = ""
            tb_path: Path | None = None
            run_log_path: Path | None = None

            if exit_code != 0:
                failure_reason = f"training exit code {exit_code}"
            elif result_path is None:
                failure_reason = "missing result path marker in console log"
            else:
                tb_path = _expected_tb_path(result_path, entry, args)
                run_log_path = result_path / "run.log"
                if not result_path.exists():
                    failure_reason = f"result path not found: {result_path}"
                elif not tb_path.exists():
                    failure_reason = f"tb path not found: {tb_path}"
                elif not run_log_path.exists():
                    failure_reason = f"run log not found: {run_log_path}"
                else:
                    extractor = _run_extractor(tb_path)
                    extractor_stdout = extractor.stdout
                    extractor_stderr = extractor.stderr
                    if extractor.returncode != 0:
                        failure_reason = f"extractor failed with exit code {extractor.returncode}"

            status = "completed" if failure_reason is None else "failed"
            event = {
                "status": status,
                "run_id": run_id,
                "queue": args.queue,
                "gpu": args.gpu,
                "phase": args.phase,
                "started_at": current_status["started_at"],
                "finished_at": utc_now(),
                "duration_sec": duration_sec,
                "start_wall_time": start_wall,
                "end_wall_time": end_wall,
                "exit_code": exit_code,
                "config_path": entry["config_path"],
                "result_path": _safe_relpath(result_path),
                "tb_path": _safe_relpath(tb_path),
                "console_log": _safe_relpath(console_log),
                "run_log_tail": [] if run_log_path is None else _tail_log(run_log_path),
                "extractor_stdout": extractor_stdout,
                "extractor_stderr": extractor_stderr,
                "failure_reason": failure_reason,
            }
            append_event(events_path, event)
            last_event = event

            current_status.update(
                {
                    "status": status,
                    "finished_at": event["finished_at"],
                    "duration_sec": duration_sec,
                    "exit_code": exit_code,
                    "failure_reason": failure_reason,
                }
            )
            write_current_status(current_path, current_status)

            if failure_reason is not None:
                queue_finished_cleanly = False
                break
        except Exception as exc:
            queue_finished_cleanly = False
            error_path = runtime / f"{args.phase}_{args.queue}_worker_error.json"
            error_payload = {
                "run_id": run_id,
                "queue": args.queue,
                "gpu": args.gpu,
                "phase": args.phase,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "console_log": _safe_relpath(console_log),
                "result_path": _safe_relpath(result_path),
            }
            write_current_status(current_path, {"status": "worker_error", **current_status, **error_payload})
            error_path.write_text(json.dumps(error_payload, indent=2), encoding="utf-8")
            append_event(
                events_path,
                {
                    "status": "worker_error",
                    "run_id": run_id,
                    "queue": args.queue,
                    "gpu": args.gpu,
                    "phase": args.phase,
                    "started_at": current_status["started_at"],
                    "finished_at": utc_now(),
                    "failure_reason": repr(exc),
                },
            )
            if process is not None and process.poll() is None:
                process.kill()
            break

    if queue_finished_cleanly:
        completed, failed = load_done_ids(events_path)
        assigned_run_ids = {
            entry["run_id"]
            for entry in queue_entries
        }
        completed_count = len(completed & assigned_run_ids)
        failed_count = len(failed & assigned_run_ids)
        final_status = {
            "status": "queue_completed",
            "queue": args.queue,
            "gpu": args.gpu,
            "phase": args.phase,
            "finished_at": utc_now(),
            "completed_runs": completed_count,
            "failed_runs": failed_count,
            "run_id": None,
            "result_path": None,
            "tb_path": None,
        }
        if last_event is not None:
            final_status["last_completed_run"] = last_event.get("run_id")
            final_status["last_result_path"] = last_event.get("result_path")
            final_status["last_tb_path"] = last_event.get("tb_path")
            final_status["last_duration_sec"] = last_event.get("duration_sec")
        write_current_status(current_path, final_status)


if __name__ == "__main__":
    main()
