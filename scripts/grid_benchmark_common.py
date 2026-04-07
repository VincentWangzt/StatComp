from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parent.parent
CAMPAIGN_SLUG = "grid_benchmark_20260330"
CAMPAIGN_TITLE = "Grid Benchmark 2026-03-30"
CAMPAIGN_DIR = REPO_ROOT / "campaigns" / CAMPAIGN_SLUG
GENERATED_CONFIG_DIR = REPO_ROOT / "configs" / "generated" / CAMPAIGN_SLUG
MARKDOWN_PATH = REPO_ROOT / "grid_benchmark_2026-03-30.md"

OFFICIAL_RESULTS_DIR = f"results/{CAMPAIGN_SLUG}/official"
OFFICIAL_TB_DIR = f"tb_logs/{CAMPAIGN_SLUG}/official"
SMOKE_RESULTS_DIR = f"results/{CAMPAIGN_SLUG}/smoke"
SMOKE_TB_DIR = f"tb_logs/{CAMPAIGN_SLUG}/smoke"

MANIFEST_PATH = CAMPAIGN_DIR / "manifest.json"
MANIFEST_CSV_PATH = CAMPAIGN_DIR / "manifest.csv"
SMOKE_MANIFEST_PATH = CAMPAIGN_DIR / "smoke_manifest.json"
QUEUE_GPU0_PATH = CAMPAIGN_DIR / "queue_gpu0.txt"
QUEUE_GPU1_PATH = CAMPAIGN_DIR / "queue_gpu1.txt"
README_PATH = CAMPAIGN_DIR / "README.md"
DEFAULT_QUEUE_COUNT = 2

TARGETS = [
    "banana",
    "multimodal",
    "x_shaped",
    "student_uc",
    "Langevin_post",
    "LRwaveform",
    "Bnn_boston",
    "Bnn_concrete",
    "Bnn_power",
    "Bnn_protein",
    "Bnn_winered",
    "Bnn_yacht",
]

BNN_TARGETS = {
    "Bnn_boston",
    "Bnn_concrete",
    "Bnn_power",
    "Bnn_protein",
    "Bnn_winered",
    "Bnn_yacht",
}

BASELINE_TARGETS = {
    "banana",
    "multimodal",
    "x_shaped",
    "student_uc",
    "Langevin_post",
}

METHOD_VARIANTS = [
    "sivi",
    "uivi",
    "rsivi",
    "aisivi",
    "ksivi_custom",
    "ksivi_standard_cg",
    "dsivi_default",
    "dsivi_bs4096_rbs2048",
    "dsivi_bs4096_rbs4096",
]

ANNEALING_MODES = {
    "on": True,
    "off": False,
}

VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "sivi": {
        "label": "SIVI",
        "source_method": "sivi",
        "runner_type": "SIVI",
        "cost_factor": 1.0,
    },
    "uivi": {
        "label": "UIVI",
        "source_method": "uivi",
        "runner_type": "UIVI",
        "cost_factor": 5.5,
    },
    "rsivi": {
        "label": "RSIVI",
        "source_method": "rsivi",
        "runner_type": "RSIVI",
        "cost_factor": 3.0,
    },
    "aisivi": {
        "label": "AISIVI",
        "source_method": "aisivi",
        "runner_type": "AISIVI",
        "cost_factor": 3.0,
    },
    "ksivi_custom": {
        "label": "KSIVI-custom",
        "source_method": "ksivi",
        "runner_type": "KSIVI",
        "cost_factor": 1.8,
    },
    "ksivi_standard_cg": {
        "label": "KSIVI-standard-CG",
        "source_method": "ksivi",
        "runner_type": "KSIVI",
        "cost_factor": 1.5,
    },
    "dsivi_default": {
        "label": "DSIVI-default",
        "source_method": "dsivi",
        "runner_type": "DSIVI",
        "cost_factor": 2.0,
    },
    "dsivi_bs4096_rbs2048": {
        "label": "DSIVI-bs4096-rbs2048",
        "source_method": "dsivi",
        "runner_type": "DSIVI",
        "cost_factor": 4.5,
    },
    "dsivi_bs4096_rbs4096": {
        "label": "DSIVI-bs4096-rbs4096",
        "source_method": "dsivi",
        "runner_type": "DSIVI",
        "cost_factor": 5.0,
    },
}

TARGET_COST_FACTORS = {
    "banana": 1.0,
    "multimodal": 1.0,
    "x_shaped": 1.0,
    "student_uc": 1.1,
    "Langevin_post": 2.8,
    "LRwaveform": 1.3,
    "Bnn_boston": 2.2,
    "Bnn_concrete": 1.8,
    "Bnn_power": 1.8,
    "Bnn_protein": 2.0,
    "Bnn_winered": 1.9,
    "Bnn_yacht": 1.6,
}

SMOKE_RUNS = [
    "official_on_banana_sivi",
    "official_on_bnn_yacht_uivi",
    "official_on_banana_ksivi_custom",
    "official_on_banana_ksivi_standard_cg",
    "official_on_bnn_yacht_dsivi_bs4096_rbs2048",
]

BEST_METRIC_MODES = {
    "metric/vi_model/elbo": "max",
    "metric/vi_model/kl_ite": "min",
    "metric/vi_model/w2": "min",
    "metric/vi_model/ksd": "min",
    "metric/vi_model/mmd": "min",
    "metric/vi_model/fisher_div": "min",
    "metric/vi_model/rmse": "min",
    "metric/vi_model/test_llk": "max",
    "metric/vi_model/nll": "min",
    "metric/reverse_model/kl_ite": "min",
    "metric/reverse_model/w2": "min",
    "metric/reverse_model/ksd": "min",
    "diagnostic/reverse_model/score_l2_to_target": "min",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_relpath(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def load_yaml(path: Path) -> dict[str, Any]:
    return deepcopy(OmegaConf.to_container(OmegaConf.load(path), resolve=True))


def save_yaml(data: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    OmegaConf.save(config=OmegaConf.create(data), f=str(path))


def save_json(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def queue_name_for(index: int) -> str:
    if index < 0:
        raise ValueError(f"Queue index must be non-negative, got {index}.")
    return f"gpu{index}"


def queue_names(count: int) -> list[str]:
    if count < 1:
        raise ValueError(f"Queue count must be at least 1, got {count}.")
    return [queue_name_for(idx) for idx in range(count)]


def queue_path_for(queue: str | int) -> Path:
    queue_name = queue_name_for(queue) if isinstance(queue, int) else str(queue)
    return CAMPAIGN_DIR / f"queue_{queue_name}.txt"


_QUEUE_NAME_RE = re.compile(r"^gpu(?P<index>\d+)$")


def queue_index_from_name(queue_name: str) -> int | None:
    match = _QUEUE_NAME_RE.fullmatch(queue_name)
    if match is None:
        return None
    return int(match.group("index"))


def sort_queue_names(names: list[str] | set[str]) -> list[str]:
    unique_names = {name for name in names if name}

    def _sort_key(name: str) -> tuple[int, int | str]:
        queue_index = queue_index_from_name(name)
        if queue_index is not None:
            return (0, queue_index)
        return (1, name)

    return sorted(unique_names, key=_sort_key)


def queue_names_from_manifest(manifest: list[dict[str, Any]]) -> list[str]:
    names = [str(entry["queue_name"]) for entry in manifest if entry.get("queue_name")]
    if names:
        return sort_queue_names(names)
    return queue_names(DEFAULT_QUEUE_COUNT)


def runtime_queue_names(phase: str | None = None) -> list[str]:
    rt_dir = runtime_dir()
    if not rt_dir.exists():
        return []

    names: set[str] = set()
    phase_prefix = f"{phase}_" if phase else ""
    runtime_pattern = re.compile(
        rf"^{re.escape(phase_prefix)}(?P<queue>gpu\d+)_(?:events\.jsonl|current\.json)$"
    )

    for path in rt_dir.iterdir():
        if not path.is_file():
            continue
        match = runtime_pattern.match(path.name)
        if match is not None:
            names.add(match.group("queue"))

    return sort_queue_names(names)


def discover_queue_names(
    manifest: list[dict[str, Any]] | None = None,
    phase: str | None = None,
) -> list[str]:
    names: set[str] = set()
    if manifest is not None:
        names.update(queue_names_from_manifest(manifest))
    names.update(runtime_queue_names(phase))
    if names:
        return sort_queue_names(names)
    return queue_names(DEFAULT_QUEUE_COUNT)


def target_schedule(target: str) -> tuple[float, int, float]:
    if target in BNN_TARGETS:
        return 1.0e-3, 2000, 0.7
    return 1.0e-3, 1000, 0.7


def run_id_for(target: str, variant: str, annealing_mode: str) -> str:
    target_slug = target.lower()
    return f"official_{annealing_mode}_{target_slug}_{variant}"


def display_target(target: str) -> str:
    return target


def metric_support(target: str) -> dict[str, bool]:
    has_baseline = target in BASELINE_TARGETS
    is_bnn = target in BNN_TARGETS
    kl_enabled = has_baseline and target != "Langevin_post"
    return {
        "kl": kl_enabled,
        "w2": has_baseline,
        "mmd": has_baseline,
        "ksd": True,
        "fisher": True,
        "elbo": True,
        "bnn": is_bnn,
    }


def metric_budgets(target: str) -> dict[str, int]:
    is_bnn = target in BNN_TARGETS
    return {
        "kl_num_samples": 10000,
        "w2_num_samples": 10000,
        "w2_num_projections": 1000,
        "mmd_num_samples": 1000,
        "ksd_num_samples": 1000 if is_bnn else 2000,
        "fisher_num_samples": 1000,
        "fisher_num_is_samples": 512,
        "elbo_batch_size": 256 if is_bnn else 512,
        "elbo_num_batches": 2,
        "elbo_num_z_samples": 1024,
        "bnn_num_samples": 500,
    }


def runtime_dir() -> Path:
    return CAMPAIGN_DIR / "runtime"
