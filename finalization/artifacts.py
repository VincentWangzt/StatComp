from __future__ import annotations

import json
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
    return max(candidates, key=lambda item: item[0])


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
    return max(candidates, key=lambda item: item[0])


def load_sample_z(path: Path, *, map_location: str = "cpu") -> torch.Tensor:
    payload = torch.load(path, map_location=map_location)
    if isinstance(payload, dict):
        payload = payload["z"]
    return torch.as_tensor(payload, dtype=torch.float32, device="cpu")


def load_baseline_samples(target: str) -> torch.Tensor:
    candidates = [
        REPO_ROOT / "baselines" / "exact" / f"{target}_exact_100k_20260408.pt",
        REPO_ROOT / "baselines" / "hmc" / f"{target}.pt",
    ]
    for path in candidates:
        if path.exists():
            payload = torch.load(path, map_location="cpu")
            if isinstance(payload, dict):
                payload = payload["samples"]
            return torch.as_tensor(payload, dtype=torch.float32, device="cpu")
    raise FileNotFoundError(f"No baseline samples found for {target}")

