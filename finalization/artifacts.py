from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import REPO_ROOT, repo_path


TARGET_ALIASES = {
    "multi_model": "multimodal",
    "multi_modal": "multimodal",
    "8_gaussian": "8_gaussians",
}


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    seed: int
    method: str
    target: str
    runner_type: str
    config_path: Path
    result_path: Path
    duration_sec: float | None
    status: str
    entry: dict[str, Any]


def normalize_target(target: str) -> str:
    return TARGET_ALIASES.get(str(target), str(target))


def load_manifest(path: Path | str) -> list[dict[str, Any]]:
    manifest_path = repo_path(path)
    if manifest_path is None or not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def resolve_repo_path(path: str | Path | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    if p.exists():
        return p
    for anchor in ("results", "tb_logs", "configs", "campaigns", "baselines"):
        if anchor in p.parts:
            idx = p.parts.index(anchor)
            return REPO_ROOT.joinpath(*p.parts[idx:])
    return repo_path(p)


def completed_runs(manifest: list[dict[str, Any]]) -> list[RunRecord]:
    records: list[RunRecord] = []
    for entry in manifest:
        status = str(entry.get("status") or entry.get("run_status") or "")
        if status not in {"completed", "process_finished"}:
            continue
        result_path = resolve_repo_path(entry.get("result_path"))
        config_path = resolve_repo_path(entry.get("config_path"))
        if result_path is None or config_path is None:
            continue
        records.append(
            RunRecord(
                run_id=str(entry["run_id"]),
                seed=int(entry["seed"]),
                method=str(entry.get("method") or entry.get("runner_type")),
                target=normalize_target(str(entry["target"])),
                runner_type=str(entry.get("runner_type") or entry.get("method")),
                config_path=config_path,
                result_path=result_path,
                duration_sec=(
                    float(entry["duration_sec"])
                    if entry.get("duration_sec") not in (None, "")
                    else None
                ),
                status=status,
                entry=entry,
            )
        )
    return records


def select_runs(
    records: list[RunRecord],
    *,
    methods: list[str],
    targets: list[str],
    seeds: list[int] | str,
) -> list[RunRecord]:
    method_set = {m.upper() for m in methods}
    target_set = {normalize_target(t) for t in targets}
    seed_set = None if seeds == "auto" else {int(s) for s in seeds}
    return [
        rec
        for rec in records
        if rec.method.upper() in method_set
        and rec.target in target_set
        and (seed_set is None or rec.seed in seed_set)
    ]


def run_index(records: list[RunRecord]) -> dict[tuple[int, str, str], RunRecord]:
    out: dict[tuple[int, str, str], RunRecord] = {}
    for rec in records:
        key = (rec.seed, rec.method.upper(), rec.target)
        out[key] = rec
    return out


def find_final_checkpoint(result_dir: Path) -> tuple[Path, int]:
    ckpt_root = result_dir / "checkpoints"
    candidates: list[tuple[int, Path]] = []
    for epoch_dir in ckpt_root.glob("epoch_*"):
        if not epoch_dir.is_dir():
            continue
        try:
            epoch = int(epoch_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if (epoch_dir / "vi_model.pt").is_file():
            candidates.append((epoch, epoch_dir))
    if not candidates:
        raise FileNotFoundError(f"No VI checkpoint under {ckpt_root}")
    epoch, ckpt_dir = max(candidates, key=lambda item: item[0])
    return ckpt_dir, epoch


def find_all_checkpoints(result_dir: Path) -> list[tuple[int, Path]]:
    """Return all checkpoint (epoch, vi_model_path) pairs, sorted by epoch."""
    ckpt_root = result_dir / "checkpoints"
    candidates: list[tuple[int, Path]] = []
    for epoch_dir in ckpt_root.glob("epoch_*"):
        if not epoch_dir.is_dir():
            continue
        try:
            epoch = int(epoch_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        model_path = epoch_dir / "vi_model.pt"
        if model_path.is_file():
            candidates.append((epoch, model_path))
    return sorted(candidates, key=lambda item: item[0])


_SAMPLE_RE = re.compile(r"samples_epoch_(?P<epoch>\d+)\.pt$")


def find_final_samples(result_dir: Path) -> tuple[Path, int]:
    sample_root = result_dir / "samples"
    candidates: list[tuple[int, Path]] = []
    for sample_path in sample_root.glob("samples_epoch_*.pt"):
        match = _SAMPLE_RE.match(sample_path.name)
        if match:
            candidates.append((int(match.group("epoch")), sample_path))
    if not candidates:
        raise FileNotFoundError(f"No sample files under {sample_root}")
    epoch, sample_path = max(candidates, key=lambda item: item[0])
    return sample_path, epoch


def load_sample_z(path: Path, *, map_location: str = "cpu") -> torch.Tensor:
    payload = torch.load(path, map_location=map_location)
    if isinstance(payload, dict):
        payload = payload.get("z", payload.get("samples"))
        if payload is None:
            raise KeyError(f"No 'z' or 'samples' tensor found in {path}")
    return torch.as_tensor(payload, dtype=torch.float32, device="cpu")


def load_baseline_samples(target: str) -> torch.Tensor:
    candidates = [
        REPO_ROOT / "baselines" / "exact" / f"{target}_exact_100k.pt",
        REPO_ROOT / "baselines" / "mcmc" / f"{target}.pt",
    ]
    for path in candidates:
        if path.exists():
            payload = torch.load(path, map_location="cpu")
            if isinstance(payload, dict):
                payload = payload["samples"]
            return torch.as_tensor(payload, dtype=torch.float32, device="cpu")
    raise FileNotFoundError(f"No baseline samples found for {target}")


# ---------------------------------------------------------------------------
# Metric series helpers (live W&B CSV first, legacy TensorBoard CSV second)
# ---------------------------------------------------------------------------

def resolve_tb_metrics_csv(rec: RunRecord) -> Path | None:
    """Resolve a live metrics.csv, falling back to a legacy extraction."""
    live_path = rec.result_path / "metrics.csv"
    if live_path.is_file():
        return live_path
    recorded_path = resolve_repo_path(rec.entry.get("metrics_path"))
    if recorded_path is not None and recorded_path.is_file():
        return recorded_path

    tb_dir = rec.entry.get("tb_dir")
    if not tb_dir:
        return None
    timestamp = Path(str(rec.result_path)).name
    relative = Path(tb_dir) / rec.runner_type / rec.target / timestamp / "extracted" / "metrics.csv"
    resolved = resolve_repo_path(relative)
    if resolved is not None and resolved.is_file():
        return resolved
    return None


def _load_wandb_series(
    rec: RunRecord, tag: str
) -> "tuple[np.ndarray, np.ndarray, np.ndarray] | None":
    """Fetch a diagnostic excluded from local CSV after an online run is synced."""
    import numpy as np

    metadata: dict[str, Any] = {}
    metadata_path = rec.result_path / "wandb_run.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
    run_path = rec.entry.get("wandb_run_path") or metadata.get("run_path")
    if not run_path:
        logging.getLogger(__name__).warning(
            "No W&B run path for %s; skipping %s", rec.run_id, tag)
        return None
    try:
        import wandb

        api_run = wandb.Api().run(str(run_path))
        rows = list(api_run.scan_history(keys=[tag, "epoch", "_timestamp"]))
    except Exception:
        logging.getLogger(__name__).warning(
            "Unable to load %s from W&B for %s", tag, rec.run_id, exc_info=True)
        return None
    valid = [row for row in rows if row.get(tag) is not None and row.get("epoch") is not None]
    if len(valid) < 2:
        return None
    steps = np.asarray([row["epoch"] for row in valid], dtype=np.float64)
    wall_times = np.asarray([row.get("_timestamp", np.nan) for row in valid], dtype=np.float64)
    values = np.asarray([row[tag] for row in valid], dtype=np.float64)
    mask = np.isfinite(steps) & np.isfinite(values)
    if mask.sum() < 2:
        return None
    order = np.argsort(steps[mask])
    return steps[mask][order], wall_times[mask][order], values[mask][order]


def load_kl_ite_series(rec: RunRecord) -> "tuple[np.ndarray, np.ndarray, np.ndarray] | None":
    """Load the KL_ITE time-series from a run's extracted metrics.csv.

    Returns:
        (steps, wall_times, values) — 1-D float64 arrays sorted by step.
        None if the file doesn't exist or contains fewer than 2 valid kl_ite rows.
    """
    import csv as _csv
    import numpy as np

    csv_path = resolve_tb_metrics_csv(rec)
    if csv_path is None:
        return None
    steps: list[int] = []
    wall_times: list[float] = []
    values: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        for row in _csv.DictReader(fh):
            if row["tag"] != "metric/vi_model/kl_ite":
                continue
            steps.append(int(row["step"]))
            wall_times.append(float(row["wall_time"]))
            values.append(float(row["value"]))
    if len(steps) < 2:
        return None
    steps_arr = np.array(steps, dtype=np.float64)
    wall_times_arr = np.array(wall_times, dtype=np.float64)
    values_arr = np.array(values, dtype=np.float64)
    # Sort by step
    order = np.argsort(steps_arr)
    steps_arr, wall_times_arr, values_arr = steps_arr[order], wall_times_arr[order], values_arr[order]
    # Drop non-finite values
    mask = np.isfinite(values_arr)
    if not mask.all():
        steps_arr, wall_times_arr, values_arr = steps_arr[mask], wall_times_arr[mask], values_arr[mask]
    if len(steps_arr) < 2:
        return None
    return steps_arr, wall_times_arr, values_arr


def load_grad_norm_series(rec: RunRecord) -> "tuple[np.ndarray, np.ndarray, np.ndarray] | None":
    """Load the gradient-norm time-series from a run's extracted metrics.csv.

    Returns:
        (steps, wall_times, values) -- 1-D float64 arrays sorted by step.
        None if the file doesn't exist or contains fewer than 2 valid rows.
    """
    import csv as _csv
    import numpy as np

    csv_path = resolve_tb_metrics_csv(rec)
    if csv_path is None:
        return _load_wandb_series(rec, "diagnostic/vi_model/grad_norm")
    steps: list[int] = []
    wall_times: list[float] = []
    values: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        for row in _csv.DictReader(fh):
            if row["tag"] != "diagnostic/vi_model/grad_norm":
                continue
            steps.append(int(row["step"]))
            wall_times.append(float(row["wall_time"]))
            values.append(float(row["value"]))
    if len(steps) < 2:
        return _load_wandb_series(rec, "diagnostic/vi_model/grad_norm")
    steps_arr = np.array(steps, dtype=np.float64)
    wall_times_arr = np.array(wall_times, dtype=np.float64)
    values_arr = np.array(values, dtype=np.float64)
    # Sort by step
    order = np.argsort(steps_arr)
    steps_arr, wall_times_arr, values_arr = steps_arr[order], wall_times_arr[order], values_arr[order]
    # Drop non-finite values
    mask = np.isfinite(values_arr)
    if not mask.all():
        steps_arr, wall_times_arr, values_arr = steps_arr[mask], wall_times_arr[mask], values_arr[mask]
    if len(steps_arr) < 2:
        return None
    return steps_arr, wall_times_arr, values_arr


def load_weight_norm_series(rec: RunRecord) -> "tuple[np.ndarray, np.ndarray, np.ndarray] | None":
    """Load the weight-norm time-series from a run's extracted metrics.csv.

    Returns:
        (steps, wall_times, values) -- 1-D float64 arrays sorted by step.
        None if the file doesn't exist or contains fewer than 2 valid rows.
    """
    import csv as _csv
    import numpy as np

    csv_path = resolve_tb_metrics_csv(rec)
    if csv_path is None:
        return _load_wandb_series(rec, "diagnostic/vi_model/weight_norm")
    steps: list[int] = []
    wall_times: list[float] = []
    values: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        for row in _csv.DictReader(fh):
            if row["tag"] != "diagnostic/vi_model/weight_norm":
                continue
            steps.append(int(row["step"]))
            wall_times.append(float(row["wall_time"]))
            values.append(float(row["value"]))
    if len(steps) < 2:
        return _load_wandb_series(rec, "diagnostic/vi_model/weight_norm")
    steps_arr = np.array(steps, dtype=np.float64)
    wall_times_arr = np.array(wall_times, dtype=np.float64)
    values_arr = np.array(values, dtype=np.float64)
    # Sort by step
    order = np.argsort(steps_arr)
    steps_arr, wall_times_arr, values_arr = steps_arr[order], wall_times_arr[order], values_arr[order]
    # Drop non-finite values
    mask = np.isfinite(values_arr)
    if not mask.all():
        steps_arr, wall_times_arr, values_arr = steps_arr[mask], wall_times_arr[mask], values_arr[mask]
    if len(steps_arr) < 2:
        return None
    return steps_arr, wall_times_arr, values_arr
