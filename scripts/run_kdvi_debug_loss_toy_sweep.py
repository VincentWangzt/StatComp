from __future__ import annotations

import argparse
import csv
import math
import os
import queue
import re
import statistics
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_SLUG = "kdvi_debug_loss_toy_sweep"
CAMPAIGN_ROOT = REPO_ROOT / "campaigns" / CAMPAIGN_SLUG
RUNTIME_ROOT = CAMPAIGN_ROOT / "runtime"
LOG_DIR = RUNTIME_ROOT / "logs"
RESULT_MAP_DIR = RUNTIME_ROOT / "result_paths"
MANIFEST_PATH = RUNTIME_ROOT / "manifest.csv"
SUMMARY_CSV = CAMPAIGN_ROOT / "summary.csv"
SUMMARY_MD = CAMPAIGN_ROOT / "summary.md"
ARTIFACT_MARKER = "Artifacts will be saved to: "

DEFAULT_TARGETS = (
    "banana",
    "x_shaped",
    "multimodal",
    "8_gaussians",
    "8_gaussians_small",
    "student_uc",
    "Langevin_post",
)
LOSS_TYPES = ("mmd", "paired_l2", "mmd_per_dim")
DEFAULT_SEEDS = (0, 1, 7)

LOSS_TAG = "train/vi_model/loss"
KL_TAG = "metric/vi_model/kl_ite"
W2_TAG = "metric/vi_model/w2"
ELM_TAG = "metric/vi_model/kde_expected_log_marginal"


@dataclass(frozen=True)
class Job:
    run_id: str
    recipe_id: str
    target: str
    loss_type: str
    seed: int
    config_path: str


def parse_csv_arg(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_int_csv_arg(value: str) -> list[int]:
    return [int(part) for part in parse_csv_arg(value)]


def ensure_dirs() -> None:
    for path in (CAMPAIGN_ROOT, RUNTIME_ROOT, LOG_DIR, RESULT_MAP_DIR):
        path.mkdir(parents=True, exist_ok=True)


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def build_jobs(targets: list[str], seeds: list[int]) -> list[Job]:
    jobs: list[Job] = []
    for target in targets:
        config_path = f"configs/kdvi_{target}.yaml"
        if not (REPO_ROOT / config_path).is_file():
            raise FileNotFoundError(f"Missing config for target {target}: {config_path}")
        for loss_type in LOSS_TYPES:
            recipe_id = f"KDVI-{target}-{loss_type}"
            for seed in seeds:
                run_id = f"{recipe_id}-seed{seed}"
                jobs.append(
                    Job(
                        run_id=run_id,
                        recipe_id=recipe_id,
                        target=target,
                        loss_type=loss_type,
                        seed=seed,
                        config_path=config_path,
                    )
                )
    return jobs


def write_manifest(jobs: list[Job]) -> None:
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "recipe_id",
                "target",
                "loss_type",
                "seed",
                "config_path",
            ],
        )
        writer.writeheader()
        for job in jobs:
            writer.writerow(job.__dict__)


def build_command(
    job: Job,
    gpu_id: int,
    extra_overrides: list[str],
    python_executable: str,
) -> list[str]:
    run_name = slug(job.run_id)
    cmd = [
        python_executable,
        "src.py",
        "--config",
        job.config_path,
        f"cuda_visible_devices={gpu_id}",
        f"seed={job.seed}",
        f"output.results_dir=results/{CAMPAIGN_SLUG}/{run_name}",
        f"train.kdvi.loss_type={job.loss_type}",
        f"tracking.campaign={CAMPAIGN_SLUG}",
        f"tracking.group={job.recipe_id}",
        f"tracking.run_name={run_name}",
    ]
    cmd.extend(extra_overrides)
    return cmd


def discover_gpus(gpu_ids_arg: str | None, max_workers: int) -> list[int]:
    if gpu_ids_arg:
        gpu_ids = parse_int_csv_arg(gpu_ids_arg)
        return gpu_ids[:max_workers]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        gpu_ids = parse_int_csv_arg(visible)
        return gpu_ids[:max_workers]
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        gpu_ids = [
            int(line.strip())
            for line in result.stdout.splitlines()
            if line.strip().isdigit()
        ]
        if gpu_ids:
            return gpu_ids[:max_workers]
    except OSError:
        pass
    return [0]


def result_map_path(job: Job) -> Path:
    return RESULT_MAP_DIR / f"{slug(job.run_id)}.path"


def extract_result_path(log_path: Path) -> Path | None:
    if not log_path.is_file():
        return None
    found = ""
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ARTIFACT_MARKER in line:
            found = line.split(ARTIFACT_MARKER, 1)[1].strip()
    if not found:
        return None
    path = Path(found)
    return path if path.is_absolute() else REPO_ROOT / path


def run_job(
    job: Job,
    gpu_queue: "queue.Queue[int]",
    extra_overrides: list[str],
    python_executable: str,
    force: bool,
) -> tuple[Job, int]:
    map_path = result_map_path(job)
    if map_path.is_file() and not force:
        mapped = Path(map_path.read_text(encoding="utf-8").strip())
        result_path = mapped if mapped.is_absolute() else REPO_ROOT / mapped
        if (result_path / "metrics.csv").is_file():
            return job, 0

    gpu_id = gpu_queue.get()
    try:
        cmd = build_command(job, gpu_id, extra_overrides, python_executable)
        log_path = LOG_DIR / f"{slug(job.run_id)}.log"
        with log_path.open("w", encoding="utf-8", newline="") as log_fh:
            log_fh.write("$ " + " ".join(cmd) + "\n")
            log_fh.flush()
            completed = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        result_path = extract_result_path(log_path)
        if completed.returncode == 0 and result_path is not None:
            try:
                rel = result_path.resolve().relative_to(REPO_ROOT.resolve())
                map_path.write_text(rel.as_posix(), encoding="utf-8")
            except ValueError:
                map_path.write_text(str(result_path), encoding="utf-8")
        return job, completed.returncode
    finally:
        gpu_queue.put(gpu_id)


def final_value(metrics_path: Path, tag: str) -> tuple[int, float] | None:
    values: list[tuple[int, float]] = []
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("tag") != tag:
                continue
            try:
                step = int(float(row["step"]))
                value = float(row["value"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append((step, value))
    if not values:
        return None
    return max(values, key=lambda item: item[0])


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def fmt(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.6g}"


def summarize(jobs: list[Job]) -> None:
    rows_by_recipe: dict[str, dict[str, Any]] = {}
    for job in jobs:
        meta = rows_by_recipe.setdefault(
            job.recipe_id,
            {
                "recipe_id": job.recipe_id,
                "target": job.target,
                "loss_type": job.loss_type,
                "seeds_complete": [],
                "loss": [],
                "kl_ite": [],
                "w2": [],
                "kde_expected_log_marginal": [],
            },
        )
        map_path = result_map_path(job)
        if not map_path.is_file():
            continue
        mapped = Path(map_path.read_text(encoding="utf-8").strip())
        result_path = mapped if mapped.is_absolute() else REPO_ROOT / mapped
        metrics_path = result_path / "metrics.csv"
        if not metrics_path.is_file():
            continue

        loss = final_value(metrics_path, LOSS_TAG)
        w2 = final_value(metrics_path, W2_TAG)
        kl = final_value(metrics_path, KL_TAG)
        elm = final_value(metrics_path, ELM_TAG)

        if loss is not None:
            meta["loss"].append(loss[1])
        if w2 is not None:
            meta["w2"].append(w2[1])
        if kl is not None:
            meta["kl_ite"].append(kl[1])
        if elm is not None:
            meta["kde_expected_log_marginal"].append(elm[1])
        if any(item is not None for item in (loss, w2, kl, elm)):
            meta["seeds_complete"].append(job.seed)

    summary_rows: list[dict[str, Any]] = []
    for recipe, meta in sorted(
        rows_by_recipe.items(),
        key=lambda item: (
            DEFAULT_TARGETS.index(item[1]["target"]),
            LOSS_TYPES.index(item[1]["loss_type"]),
        ),
    ):
        row: dict[str, Any] = {
            "recipe_id": recipe,
            "target": meta["target"],
            "loss_type": meta["loss_type"],
            "seeds_complete": ",".join(str(seed) for seed in sorted(set(meta["seeds_complete"]))),
            "n_seeds": len(set(meta["seeds_complete"])),
        }
        for key in ("loss", "kl_ite", "w2", "kde_expected_log_marginal"):
            mean, std = mean_std(meta[key])
            row[f"{key}_mean"] = mean
            row[f"{key}_std"] = std
        summary_rows.append(row)

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "recipe_id",
            "target",
            "loss_type",
            "seeds_complete",
            "n_seeds",
            "loss_mean",
            "loss_std",
            "kl_ite_mean",
            "kl_ite_std",
            "w2_mean",
            "w2_std",
            "kde_expected_log_marginal_mean",
            "kde_expected_log_marginal_std",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    lines = ["# KDVI Debug Loss Toy Sweep", ""]
    lines.append("| Target | Loss type | Seeds | KL-ITE | W2 | KDE ELM | Train loss |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            "| {target} | `{loss_type}` | {seeds} | {kl} +/- {kl_std} | "
            "{w2} +/- {w2_std} | {elm} +/- {elm_std} | {loss} +/- {loss_std} |".format(
                target=row["target"],
                loss_type=row["loss_type"],
                seeds=row["seeds_complete"] or "none",
                kl=fmt(row["kl_ite_mean"]),
                kl_std=fmt(row["kl_ite_std"]),
                w2=fmt(row["w2_mean"]),
                w2_std=fmt(row["w2_std"]),
                elm=fmt(row["kde_expected_log_marginal_mean"]),
                elm_std=fmt(row["kde_expected_log_marginal_std"]),
                loss=fmt(row["loss_mean"]),
                loss_std=fmt(row["loss_std"]),
            )
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare KDVI debug loss types on 2D toy targets and Langevin_post."
    )
    parser.add_argument(
        "--targets",
        default=",".join(DEFAULT_TARGETS),
        help="Comma-separated targets to run.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated seeds.",
    )
    parser.add_argument("--gpu-ids", default=None, help="Comma-separated GPU IDs.")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun completed jobs.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional train.epochs override for quick smoke comparisons.",
    )
    parser.add_argument(
        "--metric-log-freq",
        type=int,
        default=None,
        help="Optional train.log.metric_log_freq override.",
    )
    parser.add_argument(
        "--loss-log-freq",
        type=int,
        default=None,
        help="Optional train.log.loss_log_freq override.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional OmegaConf dotlist overrides passed to every run.",
    )
    args = parser.parse_args()

    ensure_dirs()
    targets = parse_csv_arg(args.targets)
    seeds = parse_int_csv_arg(args.seeds)
    jobs = build_jobs(targets, seeds)
    write_manifest(jobs)

    extra_overrides = list(args.overrides)
    if args.epochs is not None:
        extra_overrides.append(f"train.epochs={args.epochs}")
    if args.metric_log_freq is not None:
        extra_overrides.append(f"train.log.metric_log_freq={args.metric_log_freq}")
    if args.loss_log_freq is not None:
        extra_overrides.append(f"train.log.loss_log_freq={args.loss_log_freq}")

    if args.dry_run:
        print(f"Manifest: {MANIFEST_PATH.relative_to(REPO_ROOT)}")
        print(f"Jobs: {len(jobs)}")
        if jobs:
            print("First command:")
            print(
                " ".join(
                    build_command(jobs[0], 0, extra_overrides, sys.executable)
                )
            )
        return 0

    if not args.summarize_only:
        gpu_ids = discover_gpus(args.gpu_ids, max(1, args.max_workers))
        gpu_queue: "queue.Queue[int]" = queue.Queue()
        for gpu_id in gpu_ids:
            gpu_queue.put(gpu_id)
        workers = min(max(1, args.max_workers), len(gpu_ids), len(jobs))
        print(f"Launching {len(jobs)} jobs with {workers} worker(s) on GPUs {gpu_ids}")

        failures: list[tuple[str, int]] = []
        print_lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    run_job,
                    job,
                    gpu_queue,
                    extra_overrides,
                    sys.executable,
                    args.force,
                )
                for job in jobs
            ]
            for future in as_completed(futures):
                job, code = future.result()
                with print_lock:
                    print(f"{job.run_id}: exit={code}")
                if code != 0:
                    failures.append((job.run_id, code))
        if failures:
            print("Failures:")
            for run_id, code in failures:
                print(f"  {run_id}: exit={code}")
            return 1

    summarize(jobs)
    print(f"Summary CSV: {SUMMARY_CSV.relative_to(REPO_ROOT)}")
    print(f"Summary MD:  {SUMMARY_MD.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
