from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import models.vi_model as vi_models  # noqa: E402


DEFAULT_SIVI_REVERSE_CAMPAIGN = REPO_ROOT / "campaigns" / "sivi_reverse_sample_grid_20260427"
DEFAULT_TOY_CAMPAIGN = REPO_ROOT / "campaigns" / "toy_default_annealing_grid_20260427"
DEFAULT_OUT_DIR = REPO_ROOT / "campaigns" / "vi_checkpoint_variance_20260427" / "generated_reports"
TARGETS = ("banana", "multimodal", "x_shaped", "8_gaussians")
EPOCH_RE = re.compile(r"epoch_(\d+)$")


def _repo_path(path: str | Path | None) -> Path | None:
    if not path:
        return None
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _relpath(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _terminal_events(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    terminal: dict[str, dict[str, Any]] = {}
    for event in events:
        run_id = event.get("run_id")
        if not run_id:
            continue
        if event.get("status") == "completed":
            terminal[str(run_id)] = event
        elif event.get("status") == "process_finished" and event.get("run_status") == "completed":
            terminal[str(run_id)] = event
    return terminal


def _epoch_from_checkpoint(path: Path) -> int:
    match = EPOCH_RE.match(path.parent.name)
    if not match:
        raise ValueError(f"Cannot parse checkpoint epoch from {path}")
    return int(match.group(1))


def _iter_checkpoint_paths(result_path: Path) -> list[Path]:
    checkpoint_root = result_path / "checkpoints"
    if not checkpoint_root.exists():
        return []
    paths = list(checkpoint_root.glob("epoch_*/vi_model.pt"))
    return sorted(paths, key=_epoch_from_checkpoint)


def _diag_matrix(diagonal: list[float]) -> list[list[float]]:
    return [
        [value if idx == jdx else 0.0 for jdx in range(len(diagonal))]
        for idx, value in enumerate(diagonal)
    ]


def _variance_tensor(model: torch.nn.Module, epsilon: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "_variance"):
        variance = model._variance()  # type: ignore[attr-defined]
        if variance.ndim == 1:
            return variance.unsqueeze(0).expand(epsilon.shape[0], -1)
        return variance
    if hasattr(model, "_variance_from_raw"):
        output = model.net(epsilon)  # type: ignore[attr-defined]
        var_raw = output.chunk(2, dim=-1)[1]
        variance, _ = model._variance_from_raw(var_raw)  # type: ignore[attr-defined]
        return variance
    std = model.getstd(epsilon)  # type: ignore[attr-defined]
    return std.square()


def _build_model(full_config_path: Path, device: torch.device) -> torch.nn.Module:
    cfg = OmegaConf.load(full_config_path)
    vi_cfg = cfg.vi_model
    vi_cfg.device = str(device)
    model_type = str(cfg.vi_model_type)
    model_cls = getattr(vi_models, model_type)
    return model_cls(vi_cfg).to(device)


def _sample_epsilon(num_epsilon: int, epsilon_dim: int, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn((num_epsilon, epsilon_dim), generator=generator, device=device)


def _extract_one_checkpoint(
    checkpoint_path: Path,
    full_config_path: Path,
    device: torch.device,
    num_epsilon: int,
    epsilon_seed: int,
) -> dict[str, Any]:
    model = _build_model(full_config_path, device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    epsilon_dim = int(model.epsilon_dim)  # type: ignore[attr-defined]
    epsilon = _sample_epsilon(num_epsilon, epsilon_dim, device, epsilon_seed)
    with torch.no_grad():
        variance = _variance_tensor(model, epsilon).detach().float().cpu()
    mean_diag = variance.mean(dim=0)
    std_diag = variance.std(dim=0, unbiased=True) if variance.shape[0] > 1 else torch.zeros_like(mean_diag)
    min_diag = variance.min(dim=0).values
    max_diag = variance.max(dim=0).values

    diag = [float(value) for value in mean_diag.tolist()]
    std = [float(value) for value in std_diag.tolist()]
    min_values = [float(value) for value in min_diag.tolist()]
    max_values = [float(value) for value in max_diag.tolist()]
    return {
        "epoch": _epoch_from_checkpoint(checkpoint_path),
        "checkpoint_path": _relpath(checkpoint_path),
        "full_config_path": _relpath(full_config_path),
        "variance_diag_mean": diag,
        "variance_diag_std": std,
        "variance_diag_min": min_values,
        "variance_diag_max": max_values,
        "variance_matrix": _diag_matrix(diag),
        "variance_trace": float(sum(diag)),
        "variance_det": float(math.prod(diag)),
        "num_epsilon": num_epsilon,
        "epsilon_seed": epsilon_seed,
    }


def _runs_from_campaign(
    campaign_dir: Path,
    *,
    methods: set[str],
    targets: set[str],
    annealing_mode: str | None = None,
    reverse_sample_nums: set[int] | None = None,
) -> list[dict[str, Any]]:
    manifest = _load_json(campaign_dir / "manifest.json")
    events = _terminal_events(_load_events(campaign_dir / "runtime" / "events.jsonl"))
    runs: list[dict[str, Any]] = []
    for entry in manifest:
        if str(entry.get("method")) not in methods:
            continue
        if str(entry.get("target")) not in targets:
            continue
        if annealing_mode is not None and str(entry.get("annealing_mode")) != annealing_mode:
            continue
        reverse_sample_num = entry.get("reverse_sample_num")
        if reverse_sample_num not in (None, ""):
            reverse_sample_num = int(reverse_sample_num)
        if reverse_sample_nums is not None and reverse_sample_num not in reverse_sample_nums:
            continue
        event = events.get(str(entry["run_id"]))
        if event is None:
            continue
        result_path = _repo_path(event.get("result_path"))
        if result_path is None:
            continue
        full_config_path = result_path / "full_config.yaml"
        if reverse_sample_num in (None, "") and full_config_path.exists():
            full_cfg = OmegaConf.load(full_config_path)
            reverse_sample_num = full_cfg.get("train", {}).get("reverse_sample_num", "")
        runs.append(
            {
                "run_id": entry["run_id"],
                "method": entry.get("method", ""),
                "target": entry.get("target", ""),
                "seed": entry.get("seed", ""),
                "reverse_sample_num": reverse_sample_num,
                "result_path": result_path,
                "full_config_path": full_config_path,
                "campaign": campaign_dir.name,
            }
        )
    return runs


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in ("variance_diag_mean", "variance_diag_std", "variance_diag_min", "variance_diag_max", "variance_matrix"):
        out[key] = json.dumps(out[key], separators=(",", ":"))
    diag = row["variance_diag_mean"]
    for idx, value in enumerate(diag):
        out[f"variance_dim{idx}"] = value
    return out


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["method"]),
                str(row["target"]),
                str(row.get("reverse_sample_num", "")),
                int(row["epoch"]),
            )
        ].append(row)
    summaries: list[dict[str, Any]] = []
    for (method, target, reverse_sample_num, epoch), items in sorted(grouped.items()):
        dim_count = len(items[0]["variance_diag_mean"])
        out: dict[str, Any] = {
            "method": method,
            "target": target,
            "reverse_sample_num": reverse_sample_num,
            "epoch": epoch,
            "runs": len(items),
            "variance_trace_mean": sum(float(item["variance_trace"]) for item in items) / len(items),
            "variance_det_mean": sum(float(item["variance_det"]) for item in items) / len(items),
        }
        diag_mean: list[float] = []
        diag_sd: list[float] = []
        for idx in range(dim_count):
            values = [float(item["variance_diag_mean"][idx]) for item in items]
            mean = sum(values) / len(values)
            diag_mean.append(mean)
            if len(values) > 1:
                variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
                diag_sd.append(variance ** 0.5)
            else:
                diag_sd.append(0.0)
        out["variance_diag_mean_across_runs"] = json.dumps(diag_mean, separators=(",", ":"))
        out["variance_diag_sd_across_runs"] = json.dumps(diag_sd, separators=(",", ":"))
        out["variance_matrix_mean_across_runs"] = json.dumps(_diag_matrix(diag_mean), separators=(",", ":"))
        summaries.append(out)
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract mean diagonal VI covariance matrices from SIVI and DSIVI checkpoints."
    )
    parser.add_argument("--sivi-reverse-campaign", type=Path, default=DEFAULT_SIVI_REVERSE_CAMPAIGN)
    parser.add_argument("--toy-campaign", type=Path, default=DEFAULT_TOY_CAMPAIGN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--num-epsilon", type=int, default=10000)
    parser.add_argument("--epsilon-seed", type=int, default=20260427)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    targets = set(TARGETS)
    rows: list[dict[str, Any]] = []

    selected_runs = []
    selected_runs.extend(
        _runs_from_campaign(
            _repo_path(args.sivi_reverse_campaign) or args.sivi_reverse_campaign,
            methods={"SIVI"},
            targets=targets,
            annealing_mode="anneal_on",
            reverse_sample_nums={1024, 2048},
        )
    )
    selected_runs.extend(
        _runs_from_campaign(
            _repo_path(args.toy_campaign) or args.toy_campaign,
            methods={"SIVI"},
            targets=targets,
            annealing_mode="anneal_on",
            reverse_sample_nums=None,
        )
    )
    selected_runs.extend(
        _runs_from_campaign(
            _repo_path(args.toy_campaign) or args.toy_campaign,
            methods={"DSIVI"},
            targets=targets,
            annealing_mode="anneal_on",
            reverse_sample_nums=None,
        )
    )

    for run in selected_runs:
        checkpoints = _iter_checkpoint_paths(run["result_path"])
        for checkpoint_path in checkpoints:
            extracted = _extract_one_checkpoint(
                checkpoint_path,
                run["full_config_path"],
                device,
                num_epsilon=args.num_epsilon,
                epsilon_seed=args.epsilon_seed,
            )
            extracted.update(
                {
                    "campaign": run["campaign"],
                    "run_id": run["run_id"],
                    "method": run["method"],
                    "target": run["target"],
                    "seed": run["seed"],
                    "reverse_sample_num": run["reverse_sample_num"],
                    "result_path": _relpath(run["result_path"]),
                }
            )
            rows.append(extracted)

    out_dir = _repo_path(args.out_dir) or args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows.sort(
        key=lambda row: (
            str(row["method"]),
            str(row["target"]),
            str(row.get("reverse_sample_num", "")),
            int(row["seed"]),
            int(row["epoch"]),
        )
    )
    (out_dir / "vi_checkpoint_variance.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (out_dir / "vi_checkpoint_variance.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    _write_csv(out_dir / "vi_checkpoint_variance.csv", [_flatten_for_csv(row) for row in rows])
    summaries = _summary_rows(rows)
    _write_csv(out_dir / "vi_checkpoint_variance_summary.csv", summaries)

    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[f"{row['method']}|{row['target']}|{row.get('reverse_sample_num', '')}"] += 1
    metadata = {
        "num_rows": len(rows),
        "num_runs": len(selected_runs),
        "num_epsilon": args.num_epsilon,
        "epsilon_seed": args.epsilon_seed,
        "device": str(device),
        "counts": dict(sorted(counts.items())),
        "outputs": {
            "json": _relpath(out_dir / "vi_checkpoint_variance.json"),
            "jsonl": _relpath(out_dir / "vi_checkpoint_variance.jsonl"),
            "csv": _relpath(out_dir / "vi_checkpoint_variance.csv"),
            "summary_csv": _relpath(out_dir / "vi_checkpoint_variance_summary.csv"),
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
