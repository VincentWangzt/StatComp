#!/usr/bin/env python
"""Run and summarize the 76-run IVI/KDVI 8-Gaussian sweep.

The campaign contains:

* IVI and default KDVI on large/small 8-Gaussians, seeds 0..9 (40 runs).
* KDVI on large 8-Gaussians for every combination of MCMC type
  (MALA/SGLD), MCMC steps (1/2/5), step size (0.50/0.10), seeds 0..2
  (36 runs).

Runs are sequential by default, with optional bounded parallel execution via
``--jobs``. State is append-only and resumable. ``summary.md`` is regenerated
after every run and reports collapse counts (final KL ITE > 1) plus the mean
over non-collapsed runs for every method slug/target setup.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import mmap
import os
import re
import shlex
import statistics
import struct
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
CAMPAIGN_ROOT = REPO_ROOT / "campaigns" / "ivi_kdvi_8gauss_sweep"
RUNTIME_ROOT = CAMPAIGN_ROOT / "runtime"
MANIFEST_JSON = CAMPAIGN_ROOT / "manifest.json"
MANIFEST_CSV = CAMPAIGN_ROOT / "manifest.csv"
RECORDS_JSONL = CAMPAIGN_ROOT / "records.jsonl"
SUMMARY_MD = CAMPAIGN_ROOT / "summary.md"
RESULTS_ROOT = REPO_ROOT / "results" / "ivi_kdvi_8gauss_sweep"
TB_ROOT = REPO_ROOT / "tb_logs" / "ivi_kdvi_8gauss_sweep"

TEN_SEEDS = tuple(range(10))
THREE_SEEDS = tuple(range(3))
TARGETS = ("8_gaussians", "8_gaussians_small")
MCMC_TYPES = ("mala", "sgld")
MCMC_STEPS = (1, 2, 5)
MCMC_STEP_SIZES = (0.50, 0.10)
COLLAPSE_THRESHOLD = 1.0
IVI_LATENT_DIM = 32
KDVI_EPSILON_DIM = 32

IVI_SCRIPT = "IVI-via-mcmc-distillation/run_ivi.py"
KDVI_CONFIGS = {
    "8_gaussians": "configs/kdvi_8_gaussians.yaml",
    "8_gaussians_small": "configs/kdvi_8_gaussians_small.yaml",
}
REFERENCE_PATHS = {
    target: f"baselines/exact/{target}_exact_100k.pt" for target in TARGETS
}

ARTIFACT_RE = re.compile(
    r"Artifacts will be saved to:\s*(.+?)\s*$", re.MULTILINE
)
IVI_FINAL_RE = re.compile(
    r"KL ITE \(BDKL_KnnK\).*?=\s*([-+0-9.eE]+)"
)
IVI_EVAL_RE = re.compile(r"\[KL_ITE\]\s+step=(\d+)\s+kl=([-+0-9.eE]+)")
KDVI_KL_RE = re.compile(
    r"Epoch\s+(\d+),\s+VI KL to baseline:\s*([-+0-9.eE]+)"
)


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    method: str
    method_slug: str
    target: str
    seed: int
    config_path: str | None
    mcmc_type: str | None = None
    mcmc_steps: int | None = None
    mcmc_step_size: float | None = None


def _step_size_slug(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def build_manifest() -> list[RunSpec]:
    specs: list[RunSpec] = []

    for method in ("IVI", "KDVI"):
        method_slug = f"{method.lower()}_default"
        for target in TARGETS:
            for seed in TEN_SEEDS:
                run_id = f"{method_slug}__{target}__seed{seed:02d}"
                specs.append(
                    RunSpec(
                        run_id=run_id,
                        method=method,
                        method_slug=method_slug,
                        target=target,
                        seed=seed,
                        config_path=(KDVI_CONFIGS[target]
                                     if method == "KDVI" else None),
                    )
                )

    for mcmc_type in MCMC_TYPES:
        for mcmc_steps in MCMC_STEPS:
            for mcmc_step_size in MCMC_STEP_SIZES:
                size_slug = _step_size_slug(mcmc_step_size)
                method_slug = (
                    f"kdvi_{mcmc_type}_mcmcsteps{mcmc_steps}_"
                    f"stepsize{size_slug}"
                )
                for seed in THREE_SEEDS:
                    run_id = f"{method_slug}__8_gaussians__seed{seed:02d}"
                    specs.append(
                        RunSpec(
                            run_id=run_id,
                            method="KDVI",
                            method_slug=method_slug,
                            target="8_gaussians",
                            seed=seed,
                            config_path=KDVI_CONFIGS["8_gaussians"],
                            mcmc_type=mcmc_type,
                            mcmc_steps=mcmc_steps,
                            mcmc_step_size=mcmc_step_size,
                        )
                    )

    if len(specs) != 76:
        raise AssertionError(f"Expected 76 runs, built {len(specs)}")
    if len({spec.run_id for spec in specs}) != len(specs):
        raise AssertionError("Manifest contains duplicate run IDs")
    return specs


def build_command(spec: RunSpec) -> list[str]:
    run_results_root = RESULTS_ROOT / spec.run_id
    if spec.method == "IVI":
        return [
            sys.executable,
            IVI_SCRIPT,
            "--target",
            spec.target,
            "--seed",
            str(spec.seed),
            "--latent-dim",
            str(IVI_LATENT_DIM),
            "--rng-isolation",
            "--ref-samples-path",
            REFERENCE_PATHS[spec.target],
            "--ref-num",
            "100000",
            "--results-dir",
            str(run_results_root),
        ]

    command = [
        sys.executable,
        "src.py",
        "--config",
        str(spec.config_path),
        f"seed={spec.seed}",
        f"vi_model.epsilon_dim={KDVI_EPSILON_DIM}",
        f"output.results_dir={run_results_root}",
        "tracking.campaign=ivi_kdvi_8gauss_sweep",
    ]
    if spec.mcmc_type is not None:
        command.extend(
            [
                f"train.kdvi.mcmc_type={spec.mcmc_type}",
                f"train.kdvi.mcmc_steps={spec.mcmc_steps}",
                f"train.kdvi.mcmc_step_size={spec.mcmc_step_size:.2f}",
            ]
        )
    return command


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_manifest(specs: list[RunSpec]) -> None:
    CAMPAIGN_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "campaign": "ivi_kdvi_8gauss_sweep",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "collapse_rule": f"final_kl_ite > {COLLAPSE_THRESHOLD}",
        "run_count": len(specs),
        "runs": [asdict(spec) for spec in specs],
    }
    MANIFEST_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    fieldnames = list(asdict(specs[0]).keys())
    with MANIFEST_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(spec) for spec in specs)


def load_latest_records() -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not RECORDS_JSONL.exists():
        return latest
    for line in RECORDS_JSONL.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        run_id = row.get("run_id")
        if run_id:
            latest[run_id] = row
    return latest


def _path_from_log(text: str) -> Path | None:
    matches = ARTIFACT_RE.findall(text)
    if not matches:
        return None
    path = Path(matches[-1].strip())
    return path if path.is_absolute() else REPO_ROOT / path


def _last_csv_kl(path: Path) -> tuple[int | None, float | None]:
    if not path.exists():
        return None, None
    rows = list(csv.DictReader(path.open(newline="")))
    if not rows:
        return None, None
    return int(rows[-1]["step"]), float(rows[-1]["kl"])


def _last_metrics_kl(path: Path) -> tuple[int | None, float | None]:
    if not path.is_file():
        return None, None
    rows = [
        row for row in csv.DictReader(path.open(newline="", encoding="utf-8"))
        if row.get("tag") == "metric/vi_model/kl_ite"
    ]
    if not rows:
        return None, None
    return int(rows[-1]["step"]), float(rows[-1]["value"])


def _last_tensorboard_kl(spec: RunSpec) -> tuple[int | None, float | None]:
    """Read KDVI's final full-precision scalar from its TensorBoard log."""
    run_root = TB_ROOT / spec.run_id / "KDVI" / spec.target
    if not run_root.exists():
        return None, None
    timestamp_dirs = sorted(path for path in run_root.iterdir() if path.is_dir())
    if not timestamp_dirs:
        return None, None
    tag = "metric/vi_model/kl_ite"
    tag_bytes = tag.encode()

    # Each KDVI run writes several diagnostic scalars per epoch, so loading the
    # full event history can take minutes. Locate the final KL tag in the
    # TFRecord file and decode just its enclosing Event protobuf instead.
    try:
        from tensorboard.compat.proto import event_pb2

        event_files = sorted(
            timestamp_dirs[-1].glob("events.out.tfevents.*"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        for event_file in event_files:
            with event_file.open("rb") as handle, mmap.mmap(
                handle.fileno(),
                0,
                access=mmap.ACCESS_READ,
            ) as mapped:
                tag_pos = mapped.rfind(tag_bytes)
                if tag_pos < 0:
                    continue
                for record_start in range(max(0, tag_pos - 2048), tag_pos):
                    if record_start + 12 > len(mapped):
                        continue
                    record_len = struct.unpack_from(
                        "<Q",
                        mapped,
                        record_start,
                    )[0]
                    record_end = record_start + 12 + record_len
                    if (
                        record_len > 1_000_000
                        or not record_start + 12 <= tag_pos < record_end
                        or record_end + 4 > len(mapped)
                    ):
                        continue
                    try:
                        event = event_pb2.Event.FromString(
                            mapped[record_start + 12:record_end]
                        )
                    except Exception:
                        continue
                    for value in event.summary.value:
                        if value.tag == tag:
                            return int(event.step), float(value.simple_value)
    except Exception:
        pass

    # Compatibility fallback for unusual event layouts.
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        accumulator = EventAccumulator(
            str(timestamp_dirs[-1]), size_guidance={"scalars": 0}
        )
        accumulator.Reload()
        if tag not in accumulator.Tags().get("scalars", []):
            return None, None
        values = accumulator.Scalars(tag)
        if not values:
            return None, None
        return int(values[-1].step), float(values[-1].value)
    except Exception:
        return None, None


def extract_final_kl(
    spec: RunSpec,
    stdout_text: str,
) -> tuple[int | None, float | None, str | None, str | None]:
    artifact_path = _path_from_log(stdout_text)

    if spec.method == "IVI":
        if artifact_path is not None:
            step, value = _last_csv_kl(artifact_path / "kl_ite.csv")
            if value is not None:
                return step, value, str(artifact_path), "kl_ite.csv"
        final_matches = IVI_FINAL_RE.findall(stdout_text)
        eval_matches = IVI_EVAL_RE.findall(stdout_text)
        if final_matches:
            step = int(eval_matches[-1][0]) if eval_matches else None
            return step, float(final_matches[-1]), (
                str(artifact_path) if artifact_path else None
            ), "stdout.log"
        if eval_matches:
            step, value = eval_matches[-1]
            return int(step), float(value), (
                str(artifact_path) if artifact_path else None
            ), "stdout.log"
        return None, None, (
            str(artifact_path) if artifact_path else None
        ), None

    if artifact_path is not None:
        step, value = _last_metrics_kl(artifact_path / "metrics.csv")
        if value is not None:
            return step, value, str(artifact_path), "metrics.csv"

    step, value = _last_tensorboard_kl(spec)
    if value is not None:
        return step, value, (
            str(artifact_path) if artifact_path else None
        ), "tensorboard:metric/vi_model/kl_ite"
    if artifact_path is not None:
        run_log = artifact_path / "run.log"
        if run_log.exists():
            matches = KDVI_KL_RE.findall(run_log.read_text(errors="replace"))
            if matches:
                step, value = matches[-1]
                return int(step), float(value), str(artifact_path), "run.log"
    matches = KDVI_KL_RE.findall(stdout_text)
    if matches:
        step, value = matches[-1]
        return int(step), float(value), (
            str(artifact_path) if artifact_path else None
        ), "stdout.log"
    return None, None, (
        str(artifact_path) if artifact_path else None
    ), None


def append_record(record: dict[str, Any]) -> None:
    CAMPAIGN_ROOT.mkdir(parents=True, exist_ok=True)
    with RECORDS_JSONL.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def run_one(spec: RunSpec) -> dict[str, Any]:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    TB_ROOT.mkdir(parents=True, exist_ok=True)
    stdout_path = RUNTIME_ROOT / f"{spec.run_id}.log"
    command = build_command(spec)
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()

    print(f"\n[run] {spec.run_id}", flush=True)
    print(f"[cmd] {shlex.join(command)}", flush=True)
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    with stdout_path.open("w") as handle:
        handle.write(f"# command: {shlex.join(command)}\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    wallclock_s = time.perf_counter() - start
    stdout_text = stdout_path.read_text(errors="replace")
    final_step, final_kl, artifact_path, metric_source = extract_final_kl(
        spec, stdout_text
    )
    status = "completed" if completed.returncode == 0 and final_kl is not None else "failed"
    failure_reason = None
    if completed.returncode != 0:
        failure_reason = f"process exited with code {completed.returncode}"
    elif final_kl is None:
        failure_reason = "final KL ITE not found in run log"

    record: dict[str, Any] = {
        **asdict(spec),
        "latent_dim": IVI_LATENT_DIM if spec.method == "IVI" else None,
        "epsilon_dim": KDVI_EPSILON_DIM if spec.method == "KDVI" else None,
        "status": status,
        "returncode": completed.returncode,
        "final_step": final_step,
        "final_kl_ite": final_kl,
        "collapsed": (
            bool(final_kl > COLLAPSE_THRESHOLD)
            if final_kl is not None and math.isfinite(final_kl)
            else None
        ),
        "artifact_path": artifact_path,
        "metric_source": metric_source,
        "stdout_path": str(stdout_path),
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "wallclock_s": wallclock_s,
        "failure_reason": failure_reason,
        "git_commit": _git_commit(),
    }
    value_text = "missing" if final_kl is None else f"{final_kl:.6f}"
    print(
        f"[done] {spec.run_id}: status={status} final_kl={value_text} "
        f"wallclock={wallclock_s:.1f}s",
        flush=True,
    )
    return record


def _fmt_float(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:.6f}"


def write_summary(specs: list[RunSpec], records: dict[str, dict[str, Any]]) -> None:
    groups: dict[tuple[str, str], list[RunSpec]] = {}
    for spec in specs:
        groups.setdefault((spec.method_slug, spec.target), []).append(spec)

    completed = [
        records[spec.run_id]
        for spec in specs
        if spec.run_id in records
        and records[spec.run_id].get("status") == "completed"
        and records[spec.run_id].get("final_kl_ite") is not None
    ]
    lines = [
        "# IVI/KDVI 8-Gaussian Sweep",
        "",
        f"Updated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Progress: **{len(completed)}/{len(specs)}** runs with a final KL ITE. "
        f"Collapse is defined as **final KL ITE > {COLLAPSE_THRESHOLD:g}**.",
        "All setups use the checked-in 100k exact reference sample for their "
        "target. IVI uses a 32-dimensional latent input and KDVI uses a "
        "32-dimensional epsilon input; all other training hyperparameters "
        "remain at the current IVI/KDVI defaults.",
        "",
        "## Aggregate by setup",
        "",
        "| Method slug | Target | Expected | Observed | Collapsed | "
        "Non-collapsed | Mean KL (non-collapsed) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for (method_slug, target), group_specs in groups.items():
        rows = [
            records[spec.run_id]
            for spec in group_specs
            if spec.run_id in records
            and records[spec.run_id].get("status") == "completed"
            and records[spec.run_id].get("final_kl_ite") is not None
        ]
        values = [float(row["final_kl_ite"]) for row in rows]
        collapsed = sum(value > COLLAPSE_THRESHOLD for value in values)
        kept = [value for value in values if value <= COLLAPSE_THRESHOLD]
        kept_mean = statistics.fmean(kept) if kept else None
        lines.append(
            f"| `{method_slug}` | `{target}` | {len(group_specs)} | "
            f"{len(values)} | {collapsed} | {len(kept)} | "
            f"{_fmt_float(kept_mean)} |"
        )

    lines.extend(
        [
            "",
            "## Per-run final KL ITE",
            "",
            "| Run ID | Status | Final step | Final KL ITE | Collapsed | Metric source |",
            "|---|---|---:|---:|:---:|---|",
        ]
    )
    for spec in specs:
        row = records.get(spec.run_id)
        if row is None:
            lines.append(f"| `{spec.run_id}` | pending | — | — | — | — |")
            continue
        kl = row.get("final_kl_ite")
        collapsed = (
            "yes" if kl is not None and float(kl) > COLLAPSE_THRESHOLD
            else "no" if kl is not None
            else "—"
        )
        lines.append(
            f"| `{spec.run_id}` | {row.get('status', 'unknown')} | "
            f"{row.get('final_step') if row.get('final_step') is not None else '—'} | "
            f"{_fmt_float(float(kl)) if kl is not None else '—'} | {collapsed} | "
            f"{row.get('metric_source') or '—'} |"
        )

    failures = [
        records[spec.run_id]
        for spec in specs
        if spec.run_id in records and records[spec.run_id].get("status") == "failed"
    ]
    if failures:
        lines.extend(["", "## Failures", ""])
        for row in failures:
            lines.append(
                f"- `{row['run_id']}`: {row.get('failure_reason') or 'unknown failure'}"
            )

    SUMMARY_MD.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write/validate the manifest and print commands without running them.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Regenerate summary.md from records.jsonl without running jobs.",
    )
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Run entries again even when their latest record is completed.",
    )
    parser.add_argument(
        "--rerun-stale",
        action="store_true",
        help=(
            "Rerun completed entries whose latest record was produced by a "
            "different git commit."
        ),
    )
    parser.add_argument(
        "--fresh-commit",
        type=str,
        default=None,
        help=(
            "Commit considered fresh by --rerun-stale (default: current "
            "HEAD). Useful when resuming a sweep after a runner-only fix."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Maximum number of runs to execute concurrently (default: 1).",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    return args


def main() -> int:
    args = parse_args()
    specs = build_manifest()
    write_manifest(specs)
    records = load_latest_records()
    write_summary(specs, records)

    if args.summarize_only:
        print(f"Wrote {SUMMARY_MD}")
        return 0
    if args.dry_run:
        for spec in specs:
            print(f"{spec.run_id}: {shlex.join(build_command(spec))}")
        print(f"Validated {len(specs)} unique runs")
        return 0

    pending: list[tuple[int, RunSpec]] = []
    current_commit = _git_commit()
    fresh_commit = args.fresh_commit or current_commit
    for index, spec in enumerate(specs, start=1):
        previous = records.get(spec.run_id)
        rerun_stale = (
            args.rerun_stale
            and previous is not None
            and previous.get("git_commit") != fresh_commit
        )
        if (
            not args.rerun_completed
            and not rerun_stale
            and previous is not None
            and previous.get("status") == "completed"
            and previous.get("final_kl_ite") is not None
        ):
            print(
                f"[skip {index:02d}/{len(specs)}] "
                f"{spec.run_id} already completed",
                flush=True,
            )
            continue
        pending.append((index, spec))

    if args.jobs == 1:
        for index, spec in pending:
            print(f"[progress {index:02d}/{len(specs)}]", flush=True)
            record = run_one(spec)
            append_record(record)
            records[spec.run_id] = record
            write_summary(specs, records)
    elif pending:
        print(
            f"Running {len(pending)} entries with up to {args.jobs} "
            "concurrent jobs",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(run_one, spec): (index, spec)
                for index, spec in pending
            }
            for completed_count, future in enumerate(
                as_completed(futures),
                start=1,
            ):
                index, spec = futures[future]
                record = future.result()
                append_record(record)
                records[spec.run_id] = record
                write_summary(specs, records)
                print(
                    f"[collected {completed_count:02d}/{len(pending)}] "
                    f"manifest={index:02d}/{len(specs)} {spec.run_id}",
                    flush=True,
                )

    failures = sum(
        1
        for spec in specs
        if records.get(spec.run_id, {}).get("status") != "completed"
    )
    write_summary(specs, records)
    print(f"Campaign finished: {len(specs) - failures}/{len(specs)} completed")
    print(f"Summary: {SUMMARY_MD}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
