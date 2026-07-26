"""Run a command after a GPU has remained idle for a sustained interval.

This is intended for queuing remote experiments behind an existing scheduler.
An optional blocker PID prevents the queued command from starting during short
GPU-idle gaps between jobs owned by that scheduler.

Example:

    python -u scripts/run_when_gpu_free.py \
        --gpu 0 \
        --wait-pid 3544 \
        --wait-pid-command run_default_config_grid_sweep.py \
        --idle-seconds 120 \
        --log-file results/nfvi_rebuttal_8_gaussians/benchmark.log \
        --working-directory /root/ruivi \
        -- /root/miniconda3/envs/ruivi/bin/python -u \
        scripts/run_nfvi_rebuttal.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence


ProcessQuery = Callable[[], list[str]]
BlockerQuery = Callable[[], bool]
TelemetryQuery = Callable[[], tuple[float, float]]
Clock = Callable[[], float]
Sleeper = Callable[[float], None]
Reporter = Callable[[str], None]


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for a GPU to remain idle, then run a command. Use "
            "--wait-pid when another scheduler owns the GPU."
        )
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--poll-seconds", type=positive_float, default=30.0)
    parser.add_argument("--idle-seconds", type=nonnegative_float, default=120.0)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument(
        "--wait-pid-command",
        help="Only treat --wait-pid as active while its command contains this text.",
    )
    parser.add_argument(
        "--wait-manifest",
        type=Path,
        help=(
            "Also wait while this campaign manifest contains nonterminal "
            "statuses. Relative paths are resolved under --working-directory."
        ),
    )
    parser.add_argument(
        "--wait-manifest-status",
        action="append",
        default=[],
        help=(
            "Manifest status that blocks launch. May be repeated. Defaults "
            "to pending/running/launched/process_running."
        ),
    )
    parser.add_argument(
        "--max-utilization",
        type=nonnegative_float,
        help="Require GPU utilization at or below this percentage.",
    )
    parser.add_argument(
        "--max-used-memory-mib",
        type=nonnegative_float,
        help="Require GPU used memory at or below this many MiB.",
    )
    parser.add_argument(
        "--working-directory",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Append queue messages and child stdout/stderr to this file.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run, preceded by --.",
    )
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if args.wait_pid_command and args.wait_pid is None:
        parser.error("--wait-pid-command requires --wait-pid")
    return args


def gpu_compute_processes(gpu: int) -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def gpu_telemetry(gpu: int) -> tuple[float, float]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = [field.strip() for field in result.stdout.strip().split(",")]
    if len(fields) != 2:
        raise RuntimeError(
            f"Unexpected nvidia-smi telemetry output: {result.stdout!r}"
        )
    return float(fields[0]), float(fields[1])


def pid_is_active(pid: int | None, command_substring: str | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    if command_substring is None:
        return True

    command_path = Path(f"/proc/{pid}/cmdline")
    try:
        command = (
            command_path.read_bytes()
            .replace(b"\0", b" ")
            .decode(errors="replace")
        )
    except OSError:
        return False
    return command_substring in command


DEFAULT_BLOCKING_MANIFEST_STATUSES = {
    "pending",
    "running",
    "launched",
    "process_running",
}


def manifest_has_nonterminal_status(
    path: Path | None,
    statuses: set[str],
) -> bool:
    if path is None:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # A missing or transiently rewritten manifest is not a safe launch
        # signal. Keep waiting and try again on the next poll.
        return True
    if isinstance(payload, dict):
        rows = payload.get("runs", payload.get("records", []))
    else:
        rows = payload
    if not isinstance(rows, list):
        return True
    normalized = {status.lower() for status in statuses}
    return any(
        isinstance(row, dict)
        and str(row.get("status", "")).lower() in normalized
        for row in rows
    )


def wait_until_gpu_is_free(
    *,
    process_query: ProcessQuery,
    blocker_query: BlockerQuery,
    poll_seconds: float,
    idle_seconds: float,
    report: Reporter,
    telemetry_query: TelemetryQuery | None = None,
    max_utilization: float | None = None,
    max_used_memory_mib: float | None = None,
    clock: Clock = time.monotonic,
    sleep: Sleeper = time.sleep,
) -> None:
    idle_since: float | None = None
    previous_state: str | None = None

    while True:
        if blocker_query():
            state = "blocker-active"
            idle_since = None
            detail = "waiting for blocker process"
        else:
            processes = process_query()
            if processes:
                state = "gpu-busy"
                idle_since = None
                detail = f"GPU compute processes: {', '.join(processes)}"
            else:
                telemetry_detail = ""
                telemetry_is_idle = True
                if telemetry_query is not None:
                    utilization, used_memory = telemetry_query()
                    telemetry_detail = (
                        f"utilization={utilization:.1f}%, "
                        f"used_memory={used_memory:.1f} MiB"
                    )
                    if (
                        max_utilization is not None
                        and utilization > max_utilization
                    ):
                        telemetry_is_idle = False
                    if (
                        max_used_memory_mib is not None
                        and used_memory > max_used_memory_mib
                    ):
                        telemetry_is_idle = False
                if not telemetry_is_idle:
                    state = "gpu-telemetry-busy"
                    idle_since = None
                    detail = f"GPU telemetry not idle: {telemetry_detail}"
                    if state != previous_state:
                        report(detail)
                        previous_state = state
                    sleep(poll_seconds)
                    continue

                now = clock()
                if idle_since is None:
                    idle_since = now
                idle_elapsed = now - idle_since
                if idle_elapsed >= idle_seconds:
                    report(
                        f"GPU idle for {idle_elapsed:.1f}s; starting command"
                    )
                    return
                state = "gpu-idle-grace"
                detail = (
                    f"GPU idle for {idle_elapsed:.1f}/{idle_seconds:.1f}s"
                )
                if telemetry_detail:
                    detail += f" ({telemetry_detail})"

        if state != previous_state:
            report(detail)
            previous_state = state
        sleep(poll_seconds)


def make_reporter(log_path: Path | None) -> tuple[Reporter, object | None]:
    log_handle = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8", buffering=1)

    def report(message: str) -> None:
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        if log_handle is not None:
            print(line, file=log_handle, flush=True)

    return report, log_handle


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    working_directory = args.working_directory.resolve()
    if not working_directory.is_dir():
        raise NotADirectoryError(working_directory)
    manifest_path = args.wait_manifest
    if manifest_path is not None and not manifest_path.is_absolute():
        manifest_path = working_directory / manifest_path
    blocking_statuses = {
        str(status).lower()
        for status in (
            args.wait_manifest_status
            or sorted(DEFAULT_BLOCKING_MANIFEST_STATUSES)
        )
    }
    log_path = args.log_file.resolve() if args.log_file is not None else None
    report, log_handle = make_reporter(log_path)

    try:
        report(
            f"queued for GPU {args.gpu}: "
            f"{subprocess.list2cmdline(args.command)}"
        )
        wait_until_gpu_is_free(
            process_query=lambda: gpu_compute_processes(args.gpu),
            blocker_query=lambda: (
                pid_is_active(
                    args.wait_pid,
                    args.wait_pid_command,
                )
                or manifest_has_nonterminal_status(
                    manifest_path,
                    blocking_statuses,
                )
            ),
            poll_seconds=args.poll_seconds,
            idle_seconds=args.idle_seconds,
            report=report,
            telemetry_query=(
                (lambda: gpu_telemetry(args.gpu))
                if (
                    args.max_utilization is not None
                    or args.max_used_memory_mib is not None
                )
                else None
            ),
            max_utilization=args.max_utilization,
            max_used_memory_mib=args.max_used_memory_mib,
        )
        report(f"working directory: {working_directory}")
        output = log_handle if log_handle is not None else None
        result = subprocess.run(
            args.command,
            cwd=working_directory,
            stdout=output,
            stderr=subprocess.STDOUT if output is not None else None,
        )
        report(f"command exited with status {result.returncode}")
        return result.returncode
    finally:
        if log_handle is not None:
            log_handle.close()


if __name__ == "__main__":
    sys.exit(main())
