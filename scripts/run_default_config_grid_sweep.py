from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CAMPAIGN_SLUG = "default_config_grid"
DEFAULT_METHODS = ("sivi", "uivi", "aisivi", "dsivi", "ksivi")

METRIC_MODES = {
    "metric/vi_model/elbo": "max",
    "metric/vi_model/kde_expected_log_marginal": "max",
    "metric/vi_model/kl_ite": "min",
    "metric/vi_model/w2": "min",
    "metric/vi_model/ksd": "min",
    "metric/vi_model/mmd": "min",
    "metric/vi_model/fisher_div": "min",
    "metric/vi_model/rmse": "min",
    "metric/vi_model/nll": "min",
}

LOSS_TAG = "train/vi_model/loss"
TOTAL_TRAINING_TIME_TAG = "summary/total_training_time"
AVG_EPOCH_TIME_TAG = "summary/avg_epoch_time"
ARTIFACT_MARKER = "Artifacts will be saved to: "
CONFIG_HASH_VERSION = "default-grid-effective-v1"
CONFIG_HASH_IGNORED_KEYS = {
    "config_path",
    "cuda_visible_devices",
    "device",
    "output",
}
REVERSE_RUNNERS = {"RSIVI", "AISIVI", "DSIVI"}


@dataclass
class ActiveRun:
    entry: dict[str, Any]
    gpu: int
    process: subprocess.Popen[Any]
    console_fh: Any
    console_log: Path
    started_at: str
    start_wall_time: float
    start_perf_time: float
    result_path: Path | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def relpath(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def repo_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)  # type: ignore[return-value]


def _load_omega_config(path: Path | str) -> Any:
    return OmegaConf.load(repo_path(path))


def _config_path_from(config: Any, key: str, default_path: str) -> str:
    value = config.get(key, default_path)
    return str(value)


def _resolved_effective_config(
    config_path: Path | str,
    seed: int,
    extra_overrides: list[str] | None = None,
) -> dict[str, Any]:
    config_path = repo_path(config_path)
    config = OmegaConf.load(config_path)
    dotlist = [f"seed={seed}"]
    dotlist.extend(extra_overrides or [])
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(dotlist))
    config.config_path = relpath(config_path) or str(config_path)
    config.setdefault("device", "cuda" if config.get("use_cuda", False) else "cpu")

    target_type = str(config.get("target_type"))
    target_config_path = _config_path_from(
        config,
        "target_config_path",
        f"configs/targets/{target_type}.yaml",
    )
    config.target_config_path = target_config_path
    config = OmegaConf.merge({"target": _load_omega_config(target_config_path)}, config)

    vi_model_type = str(config.get("vi_model_type"))
    vi_model_config_path = _config_path_from(
        config,
        "vi_model_config_path",
        f"configs/vi_models/{vi_model_type}.yaml",
    )
    config.vi_model_config_path = vi_model_config_path
    config = OmegaConf.merge({"vi_model": _load_omega_config(vi_model_config_path)}, config)

    runner_type = str(config.get("runner_type", ""))
    if runner_type == "UIVI":
        reverse_model_config_path = _config_path_from(
            config,
            "reverse_model_config_path",
            "configs/reverse_models/HMC.yaml",
        )
        config.reverse_model_config_path = reverse_model_config_path
        config = OmegaConf.merge({"hmc": _load_omega_config(reverse_model_config_path)}, config)
    elif runner_type in REVERSE_RUNNERS and "reverse_model_type" in config:
        reverse_model_type = str(config.get("reverse_model_type"))
        reverse_model_config_path = _config_path_from(
            config,
            "reverse_model_config_path",
            f"configs/reverse_models/{reverse_model_type}.yaml",
        )
        config.reverse_model_config_path = reverse_model_config_path
        config = OmegaConf.merge(
            {"reverse_model": _load_omega_config(reverse_model_config_path)},
            config,
        )
        if "reverse_model" in config and "vi_model" in config:
            config.reverse_model.z_dim = config.vi_model.get("z_dim", config.reverse_model.get("z_dim"))
            if "epsilon_dim" in config.reverse_model:
                config.reverse_model.epsilon_dim = config.vi_model.get(
                    "epsilon_dim",
                    config.reverse_model.get("epsilon_dim"),
                )

    return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]


def _normalize_for_config_hash(payload: Any) -> Any:
    if isinstance(payload, dict):
        normalized: dict[str, Any] = {}
        for key, value in payload.items():
            key_str = str(key)
            if key_str in CONFIG_HASH_IGNORED_KEYS or key_str.endswith("_config_path"):
                continue
            normalized[key_str] = _normalize_for_config_hash(value)
        return normalized
    if isinstance(payload, list):
        return [_normalize_for_config_hash(value) for value in payload]
    if isinstance(payload, tuple):
        return [_normalize_for_config_hash(value) for value in payload]
    return payload


def _hash_normalized_payload(payload: Any) -> str:
    encoded = json.dumps(
        {
            "config_hash_version": CONFIG_HASH_VERSION,
            "payload": _normalize_for_config_hash(payload),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def effective_config_hash(
    config_path: Path | str,
    seed: int,
    extra_overrides: list[str] | None = None,
) -> str:
    return _hash_normalized_payload(
        _resolved_effective_config(config_path, seed, extra_overrides)
    )


def artifact_config_hash(full_config_path: Path | str) -> str:
    return _hash_normalized_payload(load_config(repo_path(full_config_path)))


def classify_config_staleness(entry: dict[str, Any], previous: dict[str, Any] | None) -> str:
    if previous is None:
        return "new"
    status = previous.get("status")
    if status != "completed":
        return str(status or "unknown")
    previous_hash = previous.get("config_hash")
    if not previous_hash:
        return "unverified"
    if previous_hash == entry.get("config_hash"):
        return "fresh"
    return "stale"


def enqueue_pending_entries(
    entries: list[dict[str, Any]],
    statuses: dict[str, dict[str, Any]],
    retry_failed: bool,
    rerun_stale: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pending: list[dict[str, Any]] = []
    stale_completed: list[dict[str, Any]] = []
    unverified_completed: list[dict[str, Any]] = []
    for entry in entries:
        previous = statuses.get(entry["run_id"])
        staleness = classify_config_staleness(entry, previous)
        if previous is None:
            pending.append(entry)
        elif previous.get("status") in {"failed", "worker_error"} and retry_failed:
            pending.append(entry)
        elif staleness == "stale":
            stale_completed.append({"entry": entry, "previous": previous})
            if rerun_stale:
                pending.append(entry)
        elif staleness == "unverified":
            unverified_completed.append({"entry": entry, "previous": previous})
    return pending, stale_completed, unverified_completed


def _short_hash(value: Any) -> str:
    return str(value or "missing")[:12]


def warn_about_staleness(
    stale_completed: list[dict[str, Any]],
    unverified_completed: list[dict[str, Any]],
    rerun_stale: bool,
) -> None:
    if stale_completed:
        action = "queueing for rerun" if rerun_stale else "skipping; pass --rerun-stale to rerun"
        print(
            f"WARNING: {len(stale_completed)} completed run(s) have stale config hashes; {action}.",
            file=sys.stderr,
        )
        for item in stale_completed[:20]:
            entry = item["entry"]
            previous = item["previous"]
            print(
                "  "
                f"{entry['run_id']}: previous={_short_hash(previous.get('config_hash'))} "
                f"current={_short_hash(entry.get('config_hash'))} "
                f"config={entry['config_path']}",
                file=sys.stderr,
            )
        if len(stale_completed) > 20:
            print(f"  ... {len(stale_completed) - 20} more stale run(s)", file=sys.stderr)

    if unverified_completed:
        print(
            "WARNING: "
            f"{len(unverified_completed)} completed legacy run(s) have no config hash; "
            "skipping as unverified. Use --hash-existing-artifacts to inventory saved full_config.yaml files.",
            file=sys.stderr,
        )
        for item in unverified_completed[:20]:
            entry = item["entry"]
            print(
                f"  {entry['run_id']}: config={entry['config_path']}",
                file=sys.stderr,
            )
        if len(unverified_completed) > 20:
            print(
                f"  ... {len(unverified_completed) - 20} more unverified run(s)",
                file=sys.stderr,
            )


def write_artifact_hash_inventory(
    statuses: dict[str, dict[str, Any]],
    json_path: Path,
    csv_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_id, event in sorted(statuses.items()):
        if event.get("status") != "completed":
            continue
        result_path_value = event.get("result_path") or ""
        full_config_path = repo_path(result_path_value) / "full_config.yaml" if result_path_value else None
        row: dict[str, Any] = {
            "run_id": run_id,
            "result_path": result_path_value,
            "full_config_path": "" if full_config_path is None else str(full_config_path),
            "artifact_config_hash": "",
            "config_hash_version": CONFIG_HASH_VERSION,
            "status": "missing_result_path" if full_config_path is None else "missing_full_config",
            "error": "",
        }
        if full_config_path is not None and full_config_path.exists():
            try:
                row["artifact_config_hash"] = artifact_config_hash(full_config_path)
                row["status"] = "hashed"
            except Exception as exc:
                row["status"] = "hash_error"
                row["error"] = repr(exc)
        rows.append(row)

    write_json(json_path, rows)
    ensure_dir(csv_path.parent)
    fieldnames = [
        "run_id",
        "status",
        "artifact_config_hash",
        "config_hash_version",
        "result_path",
        "full_config_path",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run_id_for(seed: int, method: str, target: str) -> str:
    return f"seed{seed}_{method}_{target.lower()}"


def discover_default_configs(methods: tuple[str, ...] = DEFAULT_METHODS) -> list[dict[str, Any]]:
    configs_dir = REPO_ROOT / "configs"
    method_rank = {method: idx for idx, method in enumerate(methods)}
    entries: list[dict[str, Any]] = []
    for config_path in configs_dir.glob("*.yaml"):
        stem = config_path.stem
        method = None
        target = None
        for candidate in methods:
            prefix = f"{candidate}_"
            if stem.startswith(prefix):
                method = candidate
                target = stem[len(prefix):]
                break
        if method is None or target is None:
            continue

        cfg = load_config(config_path)
        entries.append(
            {
                "method_slug": method,
                "method": str(cfg.get("runner_type", method.upper())),
                "target": str(cfg.get("target_type", target)),
                "target_slug": target,
                "runner_type": str(cfg.get("runner_type", method.upper())),
                "config_path": relpath(config_path),
                "expected_epochs": cfg.get("train", {}).get("epochs", ""),
                "batch_size": cfg.get("train", {}).get("batch_size", ""),
            }
        )

    entries.sort(
        key=lambda item: (
            method_rank.get(str(item["method_slug"]), 999),
            str(item["target_slug"]).lower(),
        )
    )
    return entries


def discover_gpus() -> list[int]:
    env_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_visible:
        visible: list[int] = []
        for piece in env_visible.split(","):
            piece = piece.strip()
            if piece.isdigit():
                visible.append(int(piece))
        if visible:
            return visible

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is not None:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=index", "--format=csv,noheader"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            gpus = [
                int(line.strip())
                for line in result.stdout.splitlines()
                if line.strip().isdigit()
            ]
            if gpus:
                return gpus

    try:
        import torch

        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
    except Exception:
        pass

    return []


def build_command(
    entry: dict[str, Any],
    gpu: int,
    results_dir: str,
    tb_dir: str,
    extra_overrides: list[str],
) -> list[str]:
    cmd = [
        sys.executable,
        "src.py",
        "--config",
        entry["config_path"],
        f"seed={entry['seed']}",
        f"cuda_visible_devices={gpu}",
        f"output.results_dir={results_dir}",
        f"output.tb_dir={tb_dir}",
    ]
    cmd.extend(extra_overrides)
    return cmd


def build_manifest_entries(args: argparse.Namespace) -> list[dict[str, Any]]:
    base_entries = discover_default_configs()
    entries: list[dict[str, Any]] = []
    for seed in args.seeds:
        for base in base_entries:
            entry = dict(base)
            entry["seed"] = seed
            entry["run_id"] = run_id_for(seed, entry["method_slug"], entry["target_slug"])
            entry["campaign_slug"] = args.campaign_slug
            entry["results_dir"] = args.results_dir
            entry["tb_dir"] = args.tb_dir
            entry["status"] = "pending"
            entry["runtime_gpu"] = ""
            entry["extra_overrides"] = list(args.extra_override)
            entry["config_hash_version"] = CONFIG_HASH_VERSION
            entry["config_hash"] = effective_config_hash(
                entry["config_path"],
                seed=seed,
                extra_overrides=entry["extra_overrides"],
            )
            entry["config_hash_basis"] = (
                "resolved main config plus seed and extra overrides; "
                "target/vi/reverse config files expanded; scheduler/output/device paths ignored"
            )
            entry["command_template"] = build_command(
                entry,
                gpu=0,
                results_dir=args.results_dir,
                tb_dir=args.tb_dir,
                extra_overrides=args.extra_override,
            )
            entries.append(entry)
    if args.limit is not None:
        entries = entries[:args.limit]
    return entries


def load_events(events_path: Path) -> list[dict[str, Any]]:
    if not events_path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def latest_terminal_status(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    statuses: dict[str, dict[str, Any]] = {}
    for event in events:
        status = event.get("status")
        run_id = event.get("run_id")
        if run_id and status in {"completed", "failed", "worker_error"}:
            statuses[str(run_id)] = event
    return statuses


def append_event(events_path: Path, event: dict[str, Any]) -> None:
    ensure_dir(events_path.parent)
    with events_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_manifest(path: Path, entries: list[dict[str, Any]], statuses: dict[str, dict[str, Any]]) -> None:
    manifest: list[dict[str, Any]] = []
    for entry in entries:
        row = dict(entry)
        event = statuses.get(entry["run_id"])
        if event is not None:
            row["status"] = event.get("status", row["status"])
            row["runtime_gpu"] = event.get("gpu", row.get("runtime_gpu", ""))
            row["result_path"] = event.get("result_path", "")
            row["tb_path"] = event.get("tb_path", "")
            row["failure_reason"] = event.get("failure_reason", "")
            row["previous_config_hash"] = event.get("config_hash", "")
            row["artifact_config_hash"] = event.get("artifact_config_hash", "")
        row["config_staleness"] = classify_config_staleness(entry, event)
        manifest.append(row)
    write_json(path, manifest)


def write_manifest_csv(path: Path, entries: list[dict[str, Any]], statuses: dict[str, dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        previous = statuses.get(entry["run_id"])
        event = previous or {}
        rows.append(
            {
                "run_id": entry["run_id"],
                "status": event.get("status", entry.get("status", "pending")),
                "method": entry["method"],
                "method_slug": entry["method_slug"],
                "target": entry["target"],
                "target_slug": entry["target_slug"],
                "seed": entry["seed"],
                "config_path": entry["config_path"],
                "config_hash": entry["config_hash"],
                "config_hash_version": entry["config_hash_version"],
                "config_staleness": classify_config_staleness(entry, previous),
                "previous_config_hash": event.get("config_hash", ""),
                "artifact_config_hash": event.get("artifact_config_hash", ""),
                "expected_epochs": entry["expected_epochs"],
                "batch_size": entry["batch_size"],
                "runtime_gpu": event.get("gpu", ""),
                "result_path": event.get("result_path", ""),
                "tb_path": event.get("tb_path", ""),
                "failure_reason": event.get("failure_reason", ""),
                "command_template": " ".join(map(str, entry["command_template"])),
            }
        )

    ensure_dir(path.parent)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_current(
    path: Path,
    entries: list[dict[str, Any]],
    active: dict[int, ActiveRun],
    statuses: dict[str, dict[str, Any]],
    gpus: list[int],
) -> None:
    completed = sum(1 for entry in entries if statuses.get(entry["run_id"], {}).get("status") == "completed")
    failed = sum(1 for entry in entries if statuses.get(entry["run_id"], {}).get("status") in {"failed", "worker_error"})
    payload = {
        "status": "running" if active else "idle",
        "updated_at": utc_now(),
        "gpus": gpus,
        "total_runs": len(entries),
        "completed_runs": completed,
        "failed_runs": failed,
        "active_runs": {
            str(gpu): {
                "run_id": run.entry["run_id"],
                "config_path": run.entry["config_path"],
                "seed": run.entry["seed"],
                "method": run.entry["method"],
                "target": run.entry["target"],
                "started_at": run.started_at,
                "config_hash": run.entry.get("config_hash", ""),
                "result_path": relpath(run.result_path),
                "console_log": relpath(run.console_log),
            }
            for gpu, run in sorted(active.items())
        },
    }
    write_json(path, payload)


def tail_file(path: Path, num_lines: int = 30) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-num_lines:]


def parse_result_path_from_console_log(path: Path) -> Path | None:
    if not path.exists():
        return None
    result_path: Path | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ARTIFACT_MARKER in line:
            result_path = Path(line.split(ARTIFACT_MARKER, 1)[1].strip())
    return result_path


def infer_tb_path(result_path: Path, entry: dict[str, Any]) -> Path:
    timestamp = result_path.name
    return repo_path(entry["tb_dir"]) / entry["runner_type"] / entry["target"] / timestamp


def run_extractor(tb_path: Path) -> subprocess.CompletedProcess[str]:
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


def read_metrics_csv(path: Path) -> dict[str, list[dict[str, float]]]:
    metrics: dict[str, list[dict[str, float]]] = {}
    if not path.exists():
        return metrics
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            tag = row["tag"]
            metrics.setdefault(tag, []).append(
                {
                    "step": float(row["step"]),
                    "wall_time": float(row["wall_time"]),
                    "value": float(row["value"]),
                }
            )
    return metrics


def best_point(points: list[dict[str, float]], mode: str) -> dict[str, float] | None:
    finite_points = [point for point in points if math.isfinite(point["value"])]
    if not finite_points:
        return None
    if mode == "max":
        return max(finite_points, key=lambda point: point["value"])
    return min(finite_points, key=lambda point: point["value"])


def metric_slug(tag: str) -> str:
    return tag.replace("/", "__")


def collect_metric_summary(
    metrics: dict[str, list[dict[str, float]]],
    started_wall_time: float | None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for tag, mode in METRIC_MODES.items():
        points = metrics.get(tag, [])
        if not points:
            continue
        final = points[-1]
        best = best_point(points, mode)
        if best is None:
            continue
        slug = metric_slug(tag)
        best_elapsed = None
        if started_wall_time is not None:
            best_elapsed = best["wall_time"] - started_wall_time
        summary[f"{slug}__final_value"] = final["value"]
        summary[f"{slug}__final_iter"] = int(final["step"])
        summary[f"{slug}__best_value"] = best["value"]
        summary[f"{slug}__best_iter"] = int(best["step"])
        summary[f"{slug}__best_value_at_iter"] = f"{best['value']:.6g}@{int(best['step'])}"
        summary[f"{slug}__best_elapsed_sec"] = best_elapsed
    return summary


def collect_artifact_paths(result_path: Path | None, tb_path: Path | None, console_log: Path) -> dict[str, str]:
    artifacts = {
        "result_path": relpath(result_path) or "",
        "tb_path": relpath(tb_path) or "",
        "extracted_metrics_path": relpath(tb_path / "extracted" if tb_path is not None else None) or "",
        "console_log": relpath(console_log) or "",
        "run_log": relpath(result_path / "run.log" if result_path is not None else None) or "",
        "checkpoints_path": "",
        "samples_path": "",
        "plots_path": "",
    }
    if result_path is not None:
        for name in ("checkpoints", "samples", "plots"):
            path = result_path / name
            if path.exists():
                artifacts[f"{name}_path"] = relpath(path) or ""
    return artifacts


def summarize_completed_run(
    event: dict[str, Any],
    entry: dict[str, Any],
    console_log: Path,
    result_path: Path | None,
    tb_path: Path | None,
) -> dict[str, Any]:
    metrics_path = tb_path / "extracted" / "metrics.csv" if tb_path is not None else Path()
    metrics = read_metrics_csv(metrics_path)
    loss_points = metrics.get(LOSS_TAG, [])
    iterations = int(loss_points[-1]["step"]) if loss_points else ""
    total_training_points = metrics.get(TOTAL_TRAINING_TIME_TAG, [])
    avg_epoch_points = metrics.get(AVG_EPOCH_TIME_TAG, [])
    total_training_time = total_training_points[-1]["value"] if total_training_points else ""
    avg_iteration_time = avg_epoch_points[-1]["value"] if avg_epoch_points else ""
    if avg_iteration_time == "" and iterations:
        avg_iteration_time = float(event["duration_sec"]) / max(1, int(iterations))

    row: dict[str, Any] = {
        "run_id": entry["run_id"],
        "status": event["status"],
        "method": entry["method"],
        "method_slug": entry["method_slug"],
        "target": entry["target"],
        "target_slug": entry["target_slug"],
        "seed": entry["seed"],
        "gpu": event.get("gpu", ""),
        "config_path": entry["config_path"],
        "config_hash": entry.get("config_hash", ""),
        "config_hash_version": entry.get("config_hash_version", ""),
        "artifact_config_hash": event.get("artifact_config_hash", ""),
        "config_staleness": classify_config_staleness(entry, event),
        "wall_clock_sec": event.get("duration_sec", ""),
        "training_time_sec": total_training_time,
        "iterations": iterations,
        "avg_iteration_time_sec": avg_iteration_time,
    }
    row.update(collect_metric_summary(metrics, event.get("start_wall_time")))
    row.update(collect_artifact_paths(result_path, tb_path, console_log))
    return row


def write_summary(report_dir: Path, entries: list[dict[str, Any]], events: list[dict[str, Any]]) -> None:
    ensure_dir(report_dir)
    event_by_run: dict[str, dict[str, Any]] = {}
    entry_by_run = {entry["run_id"]: entry for entry in entries}
    for event in events:
        if event.get("status") in {"completed", "failed", "worker_error"} and event.get("run_id"):
            event_by_run[event["run_id"]] = event

    rows: list[dict[str, Any]] = []
    for run_id, event in sorted(event_by_run.items()):
        entry = entry_by_run.get(run_id)
        if entry is None:
            continue
        result_path = repo_path(event["result_path"]) if event.get("result_path") else None
        tb_path = repo_path(event["tb_path"]) if event.get("tb_path") else None
        console_log = repo_path(event["console_log"]) if event.get("console_log") else report_dir / "missing.log"
        if event.get("status") == "completed":
            rows.append(summarize_completed_run(event, entry, console_log, result_path, tb_path))
        else:
            rows.append(
                {
                    "run_id": run_id,
                    "status": event.get("status", ""),
                    "method": entry["method"],
                    "method_slug": entry["method_slug"],
                    "target": entry["target"],
                    "target_slug": entry["target_slug"],
                    "seed": entry["seed"],
                    "gpu": event.get("gpu", ""),
                    "config_path": entry["config_path"],
                    "config_hash": entry.get("config_hash", ""),
                    "config_hash_version": entry.get("config_hash_version", ""),
                    "artifact_config_hash": event.get("artifact_config_hash", ""),
                    "config_staleness": classify_config_staleness(entry, event),
                    "wall_clock_sec": event.get("duration_sec", ""),
                    "failure_reason": event.get("failure_reason", ""),
                    "console_log": event.get("console_log", ""),
                    "result_path": event.get("result_path", ""),
                    "tb_path": event.get("tb_path", ""),
                }
            )

    write_json(report_dir / "summary.json", rows)
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with (report_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    completed = sum(1 for row in rows if row.get("status") == "completed")
    failed = sum(1 for row in rows if row.get("status") in {"failed", "worker_error"})
    md_lines = [
        "# Default Config Grid Summary",
        "",
        f"- Total manifest runs: {len(entries)}",
        f"- Recorded completed runs: {completed}",
        f"- Recorded failed runs: {failed}",
        "",
        "| Run ID | Status | GPU | Wall (s) | Iterations | Result |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        wall = row.get("wall_clock_sec", "")
        if isinstance(wall, float):
            wall = f"{wall:.1f}"
        md_lines.append(
            f"| {row.get('run_id', '')} | {row.get('status', '')} | {row.get('gpu', '')} | {wall} | {row.get('iterations', '')} | {row.get('result_path', '')} |"
        )
    (report_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def poll_process_output(active_run: ActiveRun) -> None:
    if active_run.result_path is None:
        active_run.result_path = parse_result_path_from_console_log(active_run.console_log)


def drain_remaining_output(active_run: ActiveRun) -> None:
    active_run.console_fh.flush()
    if active_run.result_path is None:
        active_run.result_path = parse_result_path_from_console_log(active_run.console_log)


def launch_run(entry: dict[str, Any], gpu: int, console_root: Path) -> ActiveRun:
    console_log = console_root / f"{entry['run_id']}.log"
    console_fh = console_log.open("w", encoding="utf-8")
    cmd = build_command(
        entry,
        gpu=gpu,
        results_dir=entry["results_dir"],
        tb_dir=entry["tb_dir"],
        extra_overrides=entry.get("extra_overrides", []),
    )
    process = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=console_fh,
        stderr=subprocess.STDOUT,
    )
    return ActiveRun(
        entry=entry,
        gpu=gpu,
        process=process,
        console_fh=console_fh,
        console_log=console_log,
        started_at=utc_now(),
        start_wall_time=time.time(),
        start_perf_time=time.perf_counter(),
    )


def finalize_run(active_run: ActiveRun) -> dict[str, Any]:
    entry = active_run.entry
    active_run.process.wait()
    drain_remaining_output(active_run)
    active_run.console_fh.close()

    duration_sec = time.perf_counter() - active_run.start_perf_time
    exit_code = active_run.process.returncode
    result_path = active_run.result_path
    tb_path: Path | None = None
    run_log_path: Path | None = None
    extractor_stdout = ""
    extractor_stderr = ""
    failure_reason: str | None = None
    artifact_hash = ""

    if exit_code != 0:
        failure_reason = f"training exit code {exit_code}"
    elif result_path is None:
        failure_reason = "missing result path marker in console log"
    else:
        tb_path = infer_tb_path(result_path, entry)
        run_log_path = result_path / "run.log"
        if not result_path.exists():
            failure_reason = f"result path not found: {result_path}"
        elif not tb_path.exists():
            failure_reason = f"tb path not found: {tb_path}"
        elif not run_log_path.exists():
            failure_reason = f"run log not found: {run_log_path}"
        else:
            extractor = run_extractor(tb_path)
            extractor_stdout = extractor.stdout
            extractor_stderr = extractor.stderr
            if extractor.returncode != 0:
                failure_reason = f"extractor failed with exit code {extractor.returncode}"
        full_config_path = result_path / "full_config.yaml"
        if full_config_path.exists():
            try:
                artifact_hash = artifact_config_hash(full_config_path)
            except Exception:
                artifact_hash = ""

    status = "completed" if failure_reason is None else "failed"
    return {
        "status": status,
        "run_id": entry["run_id"],
        "seed": entry["seed"],
        "method": entry["method"],
        "method_slug": entry["method_slug"],
        "target": entry["target"],
        "target_slug": entry["target_slug"],
        "gpu": active_run.gpu,
        "config_path": entry["config_path"],
        "config_hash": entry.get("config_hash", ""),
        "config_hash_version": entry.get("config_hash_version", CONFIG_HASH_VERSION),
        "config_hash_basis": entry.get("config_hash_basis", ""),
        "artifact_config_hash": artifact_hash,
        "command": build_command(
            entry,
            gpu=active_run.gpu,
            results_dir=entry["results_dir"],
            tb_dir=entry["tb_dir"],
            extra_overrides=entry.get("extra_overrides", []),
        ),
        "started_at": active_run.started_at,
        "finished_at": utc_now(),
        "start_wall_time": active_run.start_wall_time,
        "end_wall_time": time.time(),
        "duration_sec": duration_sec,
        "exit_code": exit_code,
        "result_path": relpath(result_path),
        "tb_path": relpath(tb_path),
        "console_log": relpath(active_run.console_log),
        "run_log_tail": [] if run_log_path is None else tail_file(run_log_path),
        "console_log_tail": tail_file(active_run.console_log),
        "extractor_stdout": extractor_stdout,
        "extractor_stderr": extractor_stderr,
        "failure_reason": failure_reason,
    }


def print_dry_run(entries: list[dict[str, Any]], gpus: list[int], args: argparse.Namespace) -> None:
    print(f"campaign_slug: {args.campaign_slug}")
    print(f"discovered_gpus: {gpus if gpus else 'none'}")
    print(f"runs: {len(entries)}")
    for entry in entries:
        command = build_command(
            entry,
            gpu=gpus[0] if gpus else 0,
            results_dir=args.results_dir,
            tb_dir=args.tb_dir,
            extra_overrides=args.extra_override,
        )
        print(f"{entry['run_id']}: {' '.join(command)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the default <method>_<target> config grid with dynamic GPU scheduling."
    )
    parser.add_argument("--campaign-slug", default=DEFAULT_CAMPAIGN_SLUG)
    parser.add_argument("--results-dir", default=f"results/{DEFAULT_CAMPAIGN_SLUG}")
    parser.add_argument("--tb-dir", default=f"tb_logs/{DEFAULT_CAMPAIGN_SLUG}")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--gpus", nargs="+", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--rerun-stale",
        action="store_true",
        help="Rerun completed runs whose saved config hash differs from the current effective config hash.",
    )
    parser.add_argument(
        "--hash-existing-artifacts",
        action="store_true",
        help="Hash completed runs' result_path/full_config.yaml files and write an inventory under campaign runtime.",
    )
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--extra-override", action="append", default=[])
    return parser.parse_args()


def write_all_state(
    manifest_path: Path,
    manifest_csv_path: Path,
    current_path: Path,
    report_dir: Path,
    events_path: Path,
    entries: list[dict[str, Any]],
    active: dict[int, ActiveRun],
    gpus: list[int],
) -> dict[str, dict[str, Any]]:
    events = load_events(events_path)
    statuses = latest_terminal_status(events)
    write_manifest(manifest_path, entries, statuses)
    write_manifest_csv(manifest_csv_path, entries, statuses)
    write_current(current_path, entries, active, statuses, gpus)
    write_summary(report_dir, entries, events)
    return statuses


def main() -> None:
    args = parse_args()
    campaign_dir = REPO_ROOT / "campaigns" / args.campaign_slug
    runtime_dir = campaign_dir / "runtime"
    console_root = runtime_dir / "console_logs"
    report_dir = campaign_dir / "generated_reports"
    manifest_path = campaign_dir / "manifest.json"
    manifest_csv_path = campaign_dir / "manifest.csv"
    events_path = runtime_dir / "events.jsonl"
    current_path = runtime_dir / "current.json"

    entries = build_manifest_entries(args)

    if args.dry_run:
        gpus = args.gpus if args.gpus is not None else discover_gpus()
        print_dry_run(entries, gpus, args)
        return

    if args.hash_existing_artifacts:
        statuses = latest_terminal_status(load_events(events_path))
        rows = write_artifact_hash_inventory(
            statuses,
            runtime_dir / "artifact_config_hashes.json",
            runtime_dir / "artifact_config_hashes.csv",
        )
        hashed = sum(1 for row in rows if row.get("status") == "hashed")
        print(
            f"Wrote artifact hash inventory for {len(rows)} completed run(s); {hashed} full_config.yaml file(s) hashed."
        )
        return

    ensure_dir(console_root)
    ensure_dir(report_dir)

    if not args.resume and events_path.exists():
        events_path.unlink()

    events = load_events(events_path) if args.resume else []
    statuses = latest_terminal_status(events)

    write_manifest(manifest_path, entries, statuses)
    write_manifest_csv(manifest_csv_path, entries, statuses)

    gpus = args.gpus if args.gpus is not None else discover_gpus()
    if not gpus:
        raise RuntimeError("No GPUs discovered. Pass --gpus explicitly if discovery is unavailable.")

    pending: queue.SimpleQueue[dict[str, Any]] = queue.SimpleQueue()
    pending_entries, stale_completed, unverified_completed = enqueue_pending_entries(
        entries,
        statuses,
        retry_failed=args.retry_failed,
        rerun_stale=args.rerun_stale,
    )
    warn_about_staleness(stale_completed, unverified_completed, args.rerun_stale)
    for entry in pending_entries:
        pending.put(entry)

    active: dict[int, ActiveRun] = {}
    free_gpus = list(gpus)
    scheduler_started_at = utc_now()
    append_event(
        events_path,
        {
            "status": "scheduler_started",
            "started_at": scheduler_started_at,
            "campaign_slug": args.campaign_slug,
            "gpus": gpus,
            "seeds": args.seeds,
            "total_runs": len(entries),
            "extra_overrides": args.extra_override,
            "config_hash_version": CONFIG_HASH_VERSION,
            "rerun_stale": args.rerun_stale,
        },
    )

    try:
        while not pending.empty() or active:
            while free_gpus and not pending.empty():
                gpu = free_gpus.pop(0)
                entry = pending.get()
                started = launch_run(entry, gpu, console_root)
                active[gpu] = started
                append_event(
                    events_path,
                    {
                        "status": "started",
                        "run_id": entry["run_id"],
                        "seed": entry["seed"],
                        "method": entry["method"],
                        "method_slug": entry["method_slug"],
                        "target": entry["target"],
                        "target_slug": entry["target_slug"],
                        "gpu": gpu,
                        "config_path": entry["config_path"],
                        "config_hash": entry.get("config_hash", ""),
                        "config_hash_version": entry.get("config_hash_version", CONFIG_HASH_VERSION),
                        "config_hash_basis": entry.get("config_hash_basis", ""),
                        "started_at": started.started_at,
                        "command": build_command(entry, gpu, entry["results_dir"], entry["tb_dir"], entry.get("extra_overrides", [])),
                        "console_log": relpath(started.console_log),
                    },
                )
                write_all_state(
                    manifest_path,
                    manifest_csv_path,
                    current_path,
                    report_dir,
                    events_path,
                    entries,
                    active,
                    gpus,
                )

            finished_gpus: list[int] = []
            for gpu, active_run in list(active.items()):
                poll_process_output(active_run)
                if active_run.process.poll() is not None:
                    finished_gpus.append(gpu)

            if not finished_gpus:
                write_current(current_path, entries, active, latest_terminal_status(load_events(events_path)), gpus)
                time.sleep(args.poll_interval)
                continue

            for gpu in finished_gpus:
                active_run = active.pop(gpu)
                try:
                    event = finalize_run(active_run)
                except Exception as exc:
                    if active_run.process.poll() is None:
                        active_run.process.kill()
                    try:
                        active_run.console_fh.close()
                    except Exception:
                        pass
                    event = {
                        "status": "worker_error",
                        "run_id": active_run.entry["run_id"],
                        "gpu": gpu,
                        "config_hash": active_run.entry.get("config_hash", ""),
                        "config_hash_version": active_run.entry.get("config_hash_version", CONFIG_HASH_VERSION),
                        "config_hash_basis": active_run.entry.get("config_hash_basis", ""),
                        "started_at": active_run.started_at,
                        "finished_at": utc_now(),
                        "duration_sec": time.perf_counter() - active_run.start_perf_time,
                        "failure_reason": repr(exc),
                        "traceback": traceback.format_exc(),
                        "console_log": relpath(active_run.console_log),
                        "result_path": relpath(active_run.result_path),
                    }
                append_event(events_path, event)
                write_all_state(
                    manifest_path,
                    manifest_csv_path,
                    current_path,
                    report_dir,
                    events_path,
                    entries,
                    active,
                    gpus,
                )
                free_gpus.append(gpu)
                free_gpus.sort()

        append_event(
            events_path,
            {
                "status": "scheduler_completed",
                "started_at": scheduler_started_at,
                "finished_at": utc_now(),
                "campaign_slug": args.campaign_slug,
                "gpus": gpus,
                "config_hash_version": CONFIG_HASH_VERSION,
            },
        )
    finally:
        for active_run in active.values():
            if active_run.process.poll() is None:
                active_run.process.kill()
            try:
                active_run.console_fh.close()
            except Exception:
                pass
        write_all_state(
            manifest_path,
            manifest_csv_path,
            current_path,
            report_dir,
            events_path,
            entries,
            {},
            gpus,
        )


if __name__ == "__main__":
    main()
