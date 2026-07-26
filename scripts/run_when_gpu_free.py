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
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence


ProcessQuery = Callable[[], list[str]]
BlockerQuery = Callable[[], bool]
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


def wait_until_gpu_is_free(
    *,
    process_query: ProcessQuery,
    blocker_query: BlockerQuery,
    poll_seconds: float,
    idle_seconds: float,
    report: Reporter,
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
    log_path = args.log_file.resolve() if args.log_file is not None else None
    report, log_handle = make_reporter(log_path)

    try:
        report(
            f"queued for GPU {args.gpu}: "
            f"{subprocess.list2cmdline(args.command)}"
        )
        wait_until_gpu_is_free(
            process_query=lambda: gpu_compute_processes(args.gpu),
            blocker_query=lambda: pid_is_active(
                args.wait_pid,
                args.wait_pid_command,
            ),
            poll_seconds=args.poll_seconds,
            idle_seconds=args.idle_seconds,
            report=report,
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
