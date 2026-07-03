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
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_SLUG = "kdvi_debug_loss_toy_sweep"
CAMPAIGN_ROOT = REPO_ROOT / "campaigns" / CAMPAIGN_SLUG
RUNTIME_ROOT = CAMPAIGN_ROOT / "runtime"
LOG_DIR = RUNTIME_ROOT / "logs"
ATTEMPT_DIR = RUNTIME_ROOT / "attempts"
DONE_DIR = RUNTIME_ROOT / "done"
FAILED_DIR = RUNTIME_ROOT / "failed"
RESULT_MAP_DIR = RUNTIME_ROOT / "result_paths"
MANIFEST_PATH = RUNTIME_ROOT / "manifest.tsv"
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
LOSS_SPECS = (
    ("mmd", "mmd"),
    ("mmd_per_dim", "mmd_per_dim"),
    ("l2", "paired_l2"),
)
LOSS_LABELS = tuple(label for label, _ in LOSS_SPECS)
DEFAULT_SEEDS = (0, 1, 7)
MAX_GPUS_DEFAULT = 10

LOSS_TAG = "train/vi_model/loss"
KL_TAG = "metric/vi_model/kl_ite"
W2_TAG = "metric/vi_model/w2"
ELM_TAG = "metric/vi_model/kde_expected_log_marginal"


@dataclass(frozen=True)
class Job:
    run_id: str
    recipe_id: str
    target: str
    loss_label: str
    loss_type: str
    seed: int
    config_path: str
    metric_family: str


def parse_csv_arg(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_int_csv_arg(value: str) -> list[int]:
    seen: set[int] = set()
    parsed: list[int] = []
    for part in parse_csv_arg(value):
        if not re.fullmatch(r"\d+", part):
            raise ValueError(f"GPU IDs must be numeric; got {part!r}")
        gpu_id = int(part)
        if gpu_id not in seen:
            parsed.append(gpu_id)
            seen.add(gpu_id)
    return parsed


def ensure_dirs() -> None:
    for path in (
        CAMPAIGN_ROOT,
        RUNTIME_ROOT,
        LOG_DIR,
        ATTEMPT_DIR,
        DONE_DIR,
        FAILED_DIR,
        RESULT_MAP_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def metric_family_for_target(target: str) -> str:
    return "elm_w2" if target == "Langevin_post" else "kl_w2"


def build_jobs(targets: list[str], seeds: list[int]) -> list[Job]:
    jobs: list[Job] = []
    for target in targets:
        config_path = f"configs/kdvi_{target}.yaml"
        if not (REPO_ROOT / config_path).is_file():
            raise FileNotFoundError(f"Missing config for target {target}: {config_path}")
        for loss_label, loss_type in LOSS_SPECS:
            recipe_id = f"KDVI-{target}-{loss_label}"
            for seed in seeds:
                run_id = f"{recipe_id}-seed{seed}"
                jobs.append(
                    Job(
                        run_id=run_id,
                        recipe_id=recipe_id,
                        target=target,
                        loss_label=loss_label,
                        loss_type=loss_type,
                        seed=seed,
                        config_path=config_path,
                        metric_family=metric_family_for_target(target),
                    )
                )
    return jobs


def write_manifest(jobs: list[Job]) -> None:
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=[
                "run_id",
                "recipe_id",
                "target",
                "loss_label",
                "loss_type",
                "seed",
                "config_path",
                "metric_family",
            ],
        )
        writer.writeheader()
        for job in jobs:
            writer.writerow(job.__dict__)


def validate_manifest(targets: list[str], seeds: list[int]) -> None:
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    groups: dict[str, list[int]] = defaultdict(list)
    run_ids = [row["run_id"] for row in rows]
    errors: list[str] = []
    for row in rows:
        groups[row["recipe_id"]].append(int(row["seed"]))
        if not (REPO_ROOT / row["config_path"]).is_file():
            errors.append(f"missing config: {row['config_path']}")
        if row["metric_family"] != metric_family_for_target(row["target"]):
            errors.append(f"{row['run_id']} has wrong metric_family")

    expected_runs = len(targets) * len(LOSS_SPECS) * len(seeds)
    expected_groups = len(targets) * len(LOSS_SPECS)
    expected_seed_list = sorted(seeds)
    if len(rows) != expected_runs:
        errors.append(f"expected {expected_runs} runs, found {len(rows)}")
    if len(set(run_ids)) != len(run_ids):
        errors.append("run IDs are not unique")
    if len(groups) != expected_groups:
        errors.append(f"expected {expected_groups} recipe groups, found {len(groups)}")
    bad_groups = {
        recipe: sorted(group_seeds)
        for recipe, group_seeds in groups.items()
        if sorted(group_seeds) != expected_seed_list
    }
    if bad_groups:
        errors.append(f"groups without exactly seeds {expected_seed_list}: {len(bad_groups)}")

    manifest_targets = {row["target"] for row in rows}
    if manifest_targets != set(targets):
        errors.append(f"unexpected targets: {sorted(manifest_targets)}")
    manifest_losses = {row["loss_label"] for row in rows}
    if manifest_losses != set(LOSS_LABELS):
        errors.append(f"unexpected loss labels: {sorted(manifest_losses)}")
    manifest_loss_types = {row["loss_type"] for row in rows}
    expected_loss_types = {loss_type for _, loss_type in LOSS_SPECS}
    if manifest_loss_types != expected_loss_types:
        errors.append(f"unexpected runner loss types: {sorted(manifest_loss_types)}")

    if errors:
        raise SystemExit("Manifest validation failed:\n- " + "\n- ".join(errors))

    print(
        "Manifest validated: "
        f"{len(rows)} runs, {len(groups)} target/loss groups, "
        f"seeds {','.join(str(seed) for seed in expected_seed_list)}."
    )


def build_command(
    job: Job,
    gpu_id: int,
    run_name: str,
    extra_overrides: list[str],
    python_executable: str,
) -> list[str]:
    cmd = [
        python_executable,
        "src.py",
        "--config",
        job.config_path,
        "use_cuda=true",
        f"cuda_visible_devices={gpu_id}",
        f"seed={job.seed}",
        f"output.results_dir=results/{CAMPAIGN_SLUG}/{run_name}",
        f"train.kdvi.loss_type={job.loss_type}",
        f"tracking.campaign={CAMPAIGN_SLUG}",
        f"tracking.group={job.recipe_id}",
        f"tracking.run_name={run_name}",
    ]
    if job.metric_family == "elm_w2":
        cmd.extend(
            [
                "metric.kl_ite.enabled=false",
                "metric.expected_log_marginal.enabled=true",
                "metric.w2.enabled=true",
            ]
        )
    cmd.extend(extra_overrides)
    return cmd


def discover_gpus(gpu_ids_arg: str | None, max_gpus: int) -> list[int]:
    if gpu_ids_arg:
        gpu_ids = parse_int_csv_arg(gpu_ids_arg)
    elif os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        gpu_ids = parse_int_csv_arg(os.environ["CUDA_VISIBLE_DEVICES"])
    else:
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
    if not gpu_ids:
        raise RuntimeError("no GPUs found; use --gpu-ids or set CUDA_VISIBLE_DEVICES")
    return gpu_ids[:max_gpus]


def result_map_path(job: Job) -> Path:
    return RESULT_MAP_DIR / f"{slug(job.run_id)}.path"


def done_path(job: Job) -> Path:
    return DONE_DIR / f"{slug(job.run_id)}.done"


def failed_path(job: Job) -> Path:
    return FAILED_DIR / f"{slug(job.run_id)}.failed"


def attempt_path(job: Job) -> Path:
    return ATTEMPT_DIR / f"{slug(job.run_id)}.txt"


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


def mapped_metrics_exist(job: Job) -> bool:
    map_path = result_map_path(job)
    if not done_path(job).is_file() or not map_path.is_file():
        return False
    mapped = Path(map_path.read_text(encoding="utf-8").strip())
    result_path = mapped if mapped.is_absolute() else REPO_ROOT / mapped
    return (result_path / "metrics.csv").is_file()


def run_job(
    job: Job,
    gpu_queue: "queue.Queue[int]",
    extra_overrides: list[str],
    python_executable: str,
    force: bool,
) -> tuple[Job, int]:
    if mapped_metrics_exist(job) and not force:
        return job, 0

    done_path(job).unlink(missing_ok=True)
    failed_path(job).unlink(missing_ok=True)
    attempt_file = attempt_path(job)
    prior_attempts = 0
    if attempt_file.is_file() and not force:
        try:
            prior_attempts = int(attempt_file.read_text(encoding="utf-8").strip())
        except ValueError:
            prior_attempts = 0

    gpu_id = gpu_queue.get()
    try:
        for attempt in range(prior_attempts + 1, 3):
            attempt_file.write_text(str(attempt), encoding="utf-8")
            run_name = slug(job.run_id if attempt == 1 else f"{job.run_id}-retry{attempt - 1}")
            log_path = LOG_DIR / f"{slug(job.run_id)}-attempt{attempt}.log"
            cmd = build_command(job, gpu_id, run_name, extra_overrides, python_executable)
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
                    result_map_path(job).write_text(rel.as_posix(), encoding="utf-8")
                except ValueError:
                    result_map_path(job).write_text(str(result_path), encoding="utf-8")
                done_path(job).touch()
                failed_path(job).unlink(missing_ok=True)
                return job, 0
            print(f"{job.run_id}: attempt {attempt} failed with exit={completed.returncode}")
        failed_path(job).touch()
        return job, 1
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


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def stdev(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(converted):
        return "-"
    return f"{converted:.6f}"


def summarize(jobs: list[Job], expected_seeds: list[int]) -> None:
    recipe_meta: dict[str, Job] = {}
    by_recipe: dict[str, list[dict[str, Any]]] = defaultdict(list)
    target_order = list(dict.fromkeys(job.target for job in jobs))

    for job in jobs:
        recipe_meta[job.recipe_id] = job
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
        required = (w2, elm) if job.metric_family == "elm_w2" else (kl, w2)
        if not all(item is not None for item in required):
            continue

        sample: dict[str, Any] = {"seed": job.seed}
        for name, point in (
            ("loss", loss),
            ("kl_ite", kl),
            ("w2", w2),
            ("kde_expected_log_marginal", elm),
        ):
            if point is not None:
                sample[name] = point[1]
                sample[f"{name}_iter"] = point[0]
        by_recipe[job.recipe_id].append(sample)

    def recipe_sort_key(recipe: str) -> tuple[int, int]:
        meta = recipe_meta[recipe]
        return (target_order.index(meta.target), LOSS_LABELS.index(meta.loss_label))

    rows: list[dict[str, Any]] = []
    for recipe in sorted(recipe_meta, key=recipe_sort_key):
        meta = recipe_meta[recipe]
        samples = sorted(by_recipe.get(recipe, []), key=lambda item: item["seed"])
        seeds = [sample["seed"] for sample in samples]
        complete = seeds == expected_seeds
        row: dict[str, Any] = {
            "recipe_id": recipe,
            "target": meta.target,
            "loss_label": meta.loss_label,
            "loss_type": meta.loss_type,
            "metric_family": meta.metric_family,
            "seeds_complete": ",".join(str(seed) for seed in seeds),
            "n_seeds": len(seeds),
            "status": "complete" if complete else "incomplete",
            "pareto": False,
        }
        for metric in ("loss", "kl_ite", "w2", "kde_expected_log_marginal"):
            values = [sample[metric] for sample in samples if metric in sample]
            iters = [sample[f"{metric}_iter"] for sample in samples if f"{metric}_iter" in sample]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = stdev(values)
            row[f"{metric}_count"] = len(values)
            row[f"{metric}_final_iter_min"] = min(iters) if iters else None
            row[f"{metric}_final_iter_max"] = max(iters) if iters else None
        rows.append(row)

    complete_rows = [row for row in rows if row["status"] == "complete"]
    for target in target_order:
        target_rows = [row for row in complete_rows if row["target"] == target]
        for candidate in target_rows:
            if candidate["metric_family"] == "elm_w2":
                candidate["pareto"] = not any(
                    other is not candidate
                    and other["kde_expected_log_marginal_mean"] >= candidate["kde_expected_log_marginal_mean"]
                    and other["w2_mean"] <= candidate["w2_mean"]
                    and (
                        other["kde_expected_log_marginal_mean"] > candidate["kde_expected_log_marginal_mean"]
                        or other["w2_mean"] < candidate["w2_mean"]
                    )
                    for other in target_rows
                )
            else:
                candidate["pareto"] = not any(
                    other is not candidate
                    and other["kl_ite_mean"] <= candidate["kl_ite_mean"]
                    and other["w2_mean"] <= candidate["w2_mean"]
                    and (
                        other["kl_ite_mean"] < candidate["kl_ite_mean"]
                        or other["w2_mean"] < candidate["w2_mean"]
                    )
                    for other in target_rows
                )

    fieldnames = [
        "recipe_id",
        "target",
        "loss_label",
        "loss_type",
        "metric_family",
        "seeds_complete",
        "n_seeds",
        "status",
        "pareto",
    ]
    for metric in ("loss", "kl_ite", "w2", "kde_expected_log_marginal"):
        fieldnames.extend(
            [
                f"{metric}_mean",
                f"{metric}_std",
                f"{metric}_count",
                f"{metric}_final_iter_min",
                f"{metric}_final_iter_max",
            ]
        )

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# KDVI Debug Loss Toy Sweep Summary",
        "",
        f"Complete target/loss groups: **{len(complete_rows)} / {len(rows)}**.",
        "Metrics are final logged values summarized as means and sample standard deviations across seeds 0, 1, and 7.",
        "",
    ]
    for target in target_order:
        target_rows = [row for row in rows if row["target"] == target]
        complete_target_rows = [row for row in target_rows if row["status"] == "complete"]
        lines.extend([f"## {target}", ""])
        if not complete_target_rows:
            lines.extend(["No complete loss groups yet.", ""])
        elif metric_family_for_target(target) == "elm_w2":
            elm_winner = max(complete_target_rows, key=lambda row: row["kde_expected_log_marginal_mean"])
            w2_winner = min(complete_target_rows, key=lambda row: row["w2_mean"])
            lines.extend(
                [
                    "### Winners",
                    "",
                    f"- **KDE ELM:** `{elm_winner['recipe_id']}` - {fmt(elm_winner['kde_expected_log_marginal_mean'])} +/- {fmt(elm_winner['kde_expected_log_marginal_std'])}",
                    f"- **W2:** `{w2_winner['recipe_id']}` - {fmt(w2_winner['w2_mean'])} +/- {fmt(w2_winner['w2_std'])}",
                    "",
                    "### ELM/W2 Pareto Front",
                    "",
                    "| Loss | KDE ELM mean +/- std | W2 mean +/- std |",
                    "|---|---:|---:|",
                ]
            )
            pareto_rows = sorted(
                (row for row in complete_target_rows if row["pareto"]),
                key=lambda row: (-row["kde_expected_log_marginal_mean"], row["w2_mean"]),
            )
            for row in pareto_rows:
                lines.append(
                    f"| `{row['loss_label']}` | {fmt(row['kde_expected_log_marginal_mean'])} +/- {fmt(row['kde_expected_log_marginal_std'])} | {fmt(row['w2_mean'])} +/- {fmt(row['w2_std'])} |"
                )
            lines.append("")
        else:
            kl_winner = min(complete_target_rows, key=lambda row: row["kl_ite_mean"])
            w2_winner = min(complete_target_rows, key=lambda row: row["w2_mean"])
            lines.extend(
                [
                    "### Winners",
                    "",
                    f"- **KL-ITE:** `{kl_winner['recipe_id']}` - {fmt(kl_winner['kl_ite_mean'])} +/- {fmt(kl_winner['kl_ite_std'])}",
                    f"- **W2:** `{w2_winner['recipe_id']}` - {fmt(w2_winner['w2_mean'])} +/- {fmt(w2_winner['w2_std'])}",
                    "",
                    "### KL/W2 Pareto Front",
                    "",
                    "| Loss | KL-ITE mean +/- std | W2 mean +/- std |",
                    "|---|---:|---:|",
                ]
            )
            pareto_rows = sorted(
                (row for row in complete_target_rows if row["pareto"]),
                key=lambda row: (row["kl_ite_mean"], row["w2_mean"]),
            )
            for row in pareto_rows:
                lines.append(
                    f"| `{row['loss_label']}` | {fmt(row['kl_ite_mean'])} +/- {fmt(row['kl_ite_std'])} | {fmt(row['w2_mean'])} +/- {fmt(row['w2_std'])} |"
                )
            lines.append("")

        lines.extend(["### All Losses", ""])
        if metric_family_for_target(target) == "elm_w2":
            lines.extend(["| Loss | Status | Seeds | KDE ELM | W2 | Train loss |", "|---|---|---|---:|---:|---:|"])
            for row in target_rows:
                lines.append(
                    f"| `{row['loss_label']}` | {row['status']} | {row['seeds_complete'] or 'none'} | {fmt(row['kde_expected_log_marginal_mean'])} +/- {fmt(row['kde_expected_log_marginal_std'])} | {fmt(row['w2_mean'])} +/- {fmt(row['w2_std'])} | {fmt(row['loss_mean'])} +/- {fmt(row['loss_std'])} |"
                )
        else:
            lines.extend(["| Loss | Status | Seeds | KL-ITE | W2 | Train loss |", "|---|---|---|---:|---:|---:|"])
            for row in target_rows:
                lines.append(
                    f"| `{row['loss_label']}` | {row['status']} | {row['seeds_complete'] or 'none'} | {fmt(row['kl_ite_mean'])} +/- {fmt(row['kl_ite_std'])} | {fmt(row['w2_mean'])} +/- {fmt(row['w2_std'])} | {fmt(row['loss_mean'])} +/- {fmt(row['loss_std'])} |"
                )
        lines.append("")

    incomplete = [row for row in rows if row["status"] != "complete"]
    lines.extend(["## Incomplete Groups", ""])
    if incomplete:
        lines.extend(["| Recipe | Complete seeds |", "|---|---|"])
        for row in incomplete:
            lines.append(f"| `{row['recipe_id']}` | {row['seeds_complete'] or 'none'} |")
    else:
        lines.append("None.")
    lines.append("")

    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare KDVI debug loss types on 2D toy targets and Langevin_post."
    )
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS), help="Comma-separated targets to run.")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS), help="Comma-separated seeds.")
    parser.add_argument("--gpu-ids", default=None, help="Comma-separated GPU IDs.")
    parser.add_argument("--max-gpus", type=int, default=MAX_GPUS_DEFAULT, help="Use at most this many visible GPUs.")
    parser.add_argument("--max-workers", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun completed jobs.")
    parser.add_argument("--epochs", type=int, default=None, help="Optional train.epochs override for quick smoke comparisons.")
    parser.add_argument("--metric-log-freq", type=int, default=None, help="Optional train.log.metric_log_freq override.")
    parser.add_argument("--loss-log-freq", type=int, default=None, help="Optional train.log.loss_log_freq override.")
    parser.add_argument("overrides", nargs="*", help="Additional OmegaConf dotlist overrides passed to every run.")
    args = parser.parse_args()

    if args.dry_run and args.summarize_only:
        raise SystemExit("--dry-run and --summarize-only are mutually exclusive")
    max_gpus = args.max_workers if args.max_workers is not None else args.max_gpus
    if max_gpus < 1:
        raise SystemExit("--max-gpus must be a positive integer")
    if max_gpus > MAX_GPUS_DEFAULT:
        print(f"Capping --max-gpus at {MAX_GPUS_DEFAULT}.")
        max_gpus = MAX_GPUS_DEFAULT

    ensure_dirs()
    targets = parse_csv_arg(args.targets)
    seeds = parse_int_csv_arg(args.seeds)
    jobs = build_jobs(targets, seeds)
    write_manifest(jobs)
    validate_manifest(targets, seeds)

    extra_overrides = list(args.overrides)
    if args.epochs is not None:
        extra_overrides.append(f"train.epochs={args.epochs}")
    if args.metric_log_freq is not None:
        extra_overrides.append(f"train.log.metric_log_freq={args.metric_log_freq}")
    if args.loss_log_freq is not None:
        extra_overrides.append(f"train.log.loss_log_freq={args.loss_log_freq}")

    if args.dry_run:
        print("Dry run complete; no experiments were launched.")
        print(f"Manifest: {MANIFEST_PATH.relative_to(REPO_ROOT)}")
        print("First five jobs:")
        with MANIFEST_PATH.open(newline="", encoding="utf-8") as handle:
            for index, row in enumerate(csv.DictReader(handle, delimiter="\t")):
                if index >= 5:
                    break
                print(row)
        if jobs:
            print("First command:")
            print(" ".join(build_command(jobs[0], 0, slug(jobs[0].run_id), extra_overrides, sys.executable)))
        return 0

    if not args.summarize_only:
        gpu_ids = discover_gpus(args.gpu_ids, max_gpus)
        gpu_queue: "queue.Queue[int]" = queue.Queue()
        for gpu_id in gpu_ids:
            gpu_queue.put(gpu_id)
        pending_jobs = [job for job in jobs if args.force or not mapped_metrics_exist(job)]
        workers = min(len(gpu_ids), len(pending_jobs))
        print(f"Using GPUs: {' '.join(str(gpu_id) for gpu_id in gpu_ids)}")
        print(f"Pending runs: {len(pending_jobs)} / {len(jobs)}")

        failures: list[tuple[str, int]] = []
        print_lock = threading.Lock()
        if pending_jobs:
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
                    for job in pending_jobs
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

    summarize(jobs, sorted(seeds))
    print(f"Summary CSV: {SUMMARY_CSV.relative_to(REPO_ROOT)}")
    print(f"Summary MD:  {SUMMARY_MD.relative_to(REPO_ROOT)}")
    if not args.summarize_only and failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
