"""Score comparison on a common checkpoint and a persisted HMC reference.

This module is deliberately separate from the original per-method checkpoint
analysis.  It uses one source variational checkpoint for every method-native
estimator, persists the common forward bank and posterior-HMC chain means, and
allows the reference and four method estimators to run on separate GPUs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from omegaconf import DictConfig, OmegaConf

from runner.runners import Runners

from .artifacts import (
    RunRecord,
    completed_runs,
    find_all_checkpoints,
    load_manifest,
    select_runs,
)
from .config import REPO_ROOT, repo_path
from .runner_eval import prepare_config, remove_file_handlers, set_seed
from .score_approximation import (
    _accumulator_dtype,
    assess_hmc_reference_quality,
    compute_score_metrics,
    method_native_score,
    posterior_hmc_reference_scores,
    seed_everything,
    select_progress_checkpoints,
    stable_seed,
)


DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs"
    / "finalization"
    / "score_approximation_dsivi_shared_x_shaped_grid.yaml"
)
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SharedCheckpointSpec:
    """One source checkpoint and the method configs evaluated against it."""

    source_record: RunRecord
    method_records: tuple[RunRecord, ...]
    progress: float
    epoch: int
    checkpoint_dir: Path

    @property
    def source_cell_key(self) -> str:
        method = self.source_record.method.upper()
        return (
            f"{self.source_record.run_id}|{method}|"
            f"{self.source_record.target}|{self.source_record.seed}|"
            f"{self.epoch}"
        )

    @property
    def key(self) -> str:
        return (
            f"{self.source_record.run_id}|shared|"
            f"{self.source_record.target}|{self.source_record.seed}|"
            f"{self.epoch}"
        )

    def method_record(self, method: str) -> RunRecord:
        normalized = method.upper()
        matches = [
            record
            for record in self.method_records
            if record.method.upper() == normalized
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one {normalized} method record for {self.key}, "
                f"found {len(matches)}."
            )
        return matches[0]


@dataclass(frozen=True)
class ArtifactPaths:
    input_fingerprint: str
    reference_fingerprint: str
    analysis_fingerprint: str
    forward_bank: Path
    hmc_reference: Path
    run_root: Path

    def method_score(self, spec: SharedCheckpointSpec, method: str) -> Path:
        return (
            self.run_root
            / "methods"
            / spec.source_record.target
            / f"seed_{spec.source_record.seed}"
            / f"epoch_{spec.epoch}"
            / f"{method.upper()}.pt"
        )

    def aisivi_refit_state(
        self,
        spec: SharedCheckpointSpec,
        refit_fingerprint: str | None = None,
    ) -> Path:
        directory = (
            self.run_root
            / "refits"
            / spec.source_record.target
            / f"seed_{spec.source_record.seed}"
            / f"epoch_{spec.epoch}"
            / "AISIVI"
        )
        filename = (
            "refit_state.pt"
            if refit_fingerprint is None
            else f"refit_state_{refit_fingerprint[:16]}.pt"
        )
        return directory / filename

    def aisivi_flow(self, spec: SharedCheckpointSpec) -> Path:
        return (
            self.run_root
            / "refits"
            / spec.source_record.target
            / f"seed_{spec.source_record.seed}"
            / f"epoch_{spec.epoch}"
            / "AISIVI"
            / "reverse_model.pt"
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_shared_score_config(
    path: str | Path | None,
    overrides: list[str] | None = None,
) -> DictConfig:
    config_path = DEFAULT_CONFIG if path is None else Path(path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg  # type: ignore[return-value]


def _json_fingerprint(payload: Any) -> str:
    if OmegaConf.is_config(payload):
        payload = OmegaConf.to_container(payload, resolve=True)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def analysis_fingerprint(cfg: DictConfig) -> str:
    return _json_fingerprint(cfg)


def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _one_record(
    records: Iterable[RunRecord],
    *,
    method: str,
    target: str,
    seed: int,
) -> RunRecord:
    selected = select_runs(
        records,
        methods=[method],
        targets=[target],
        seeds=[seed],
    )
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one completed {method}/{target}/seed-{seed} run, "
            f"found {len(selected)}."
        )
    return selected[0]


def build_shared_checkpoint_specs(
    cfg: DictConfig,
) -> list[SharedCheckpointSpec]:
    records = completed_runs(
        load_manifest(str(cfg.campaign.manifest_path))
    )
    source_method = str(cfg.selection.source_method).upper()
    methods = [str(value).upper() for value in cfg.selection.methods]
    targets = [str(value) for value in cfg.selection.targets]
    seeds = [int(value) for value in cfg.selection.seeds]
    progresses = [
        float(value) for value in cfg.selection.checkpoint_progress
    ]
    if len(set(methods)) != len(methods):
        raise ValueError("selection.methods contains duplicates.")
    if source_method not in methods:
        raise ValueError(
            "selection.source_method must also appear in selection.methods."
        )

    specs: list[SharedCheckpointSpec] = []
    for target in targets:
        for seed in seeds:
            source = _one_record(
                records,
                method=source_method,
                target=target,
                seed=seed,
            )
            method_records = tuple(
                _one_record(
                    records,
                    method=method,
                    target=target,
                    seed=seed,
                )
                for method in methods
            )
            source_cfg = OmegaConf.load(source.config_path)
            checkpoints = find_all_checkpoints(source.result_path)
            selected_checkpoints = select_progress_checkpoints(
                checkpoints,
                total_epochs=int(source_cfg.train.epochs),
                progresses=progresses,
            )
            for progress, epoch, checkpoint_dir in selected_checkpoints:
                vi_path = checkpoint_dir / "vi_model.pt"
                if not vi_path.is_file():
                    raise FileNotFoundError(vi_path)
                if "DSIVI" in methods:
                    reverse_path = checkpoint_dir / "reverse_model.pt"
                    if not reverse_path.is_file():
                        raise FileNotFoundError(reverse_path)
                specs.append(
                    SharedCheckpointSpec(
                        source_record=source,
                        method_records=method_records,
                        progress=progress,
                        epoch=epoch,
                        checkpoint_dir=checkpoint_dir,
                    )
                )
    return specs


def select_shared_checkpoint_specs(
    specs: Iterable[SharedCheckpointSpec],
    *,
    seeds: Iterable[int] | None = None,
    epochs: Iterable[int] | None = None,
) -> list[SharedCheckpointSpec]:
    """Select worker cells without changing the analysis configuration.

    Runtime filters deliberately operate on already-built specs.  They
    therefore leave the full OmegaConf object, and hence the artifact
    fingerprint shared by all workers, unchanged.
    """

    materialized = list(specs)
    seed_filter = (
        None if seeds is None else {int(value) for value in seeds}
    )
    epoch_filter = (
        None if epochs is None else {int(value) for value in epochs}
    )
    if seed_filter:
        available = {
            int(spec.source_record.seed) for spec in materialized
        }
        missing = seed_filter - available
        if missing:
            raise ValueError(
                "Requested worker seeds are not configured: "
                + ", ".join(str(value) for value in sorted(missing))
            )
    if epoch_filter:
        available = {int(spec.epoch) for spec in materialized}
        missing = epoch_filter - available
        if missing:
            raise ValueError(
                "Requested worker epochs are not configured: "
                + ", ".join(str(value) for value in sorted(missing))
            )
    return [
        spec
        for spec in materialized
        if (
            not seed_filter
            or int(spec.source_record.seed) in seed_filter
        )
        and (not epoch_filter or int(spec.epoch) in epoch_filter)
    ]


def _runtime_root(cfg: DictConfig) -> Path:
    root = repo_path(str(cfg.output.runtime_dir))
    if root is None:
        raise ValueError("output.runtime_dir must be configured.")
    return root


def _reference_cache_root(cfg: DictConfig) -> Path:
    configured = cfg.output.get("reference_cache_dir")
    if configured:
        root = repo_path(str(configured))
        if root is None:
            raise ValueError(
                "output.reference_cache_dir must be a valid path."
            )
        return root
    return _runtime_root(cfg) / "reference_cache"


def artifact_paths(
    cfg: DictConfig,
    spec: SharedCheckpointSpec,
) -> ArtifactPaths:
    vi_path = spec.checkpoint_dir / "vi_model.pt"
    checkpoint_hash = file_sha256(vi_path)
    forward_seed = stable_seed(spec.source_cell_key, "forward")
    input_payload = {
        "schema_version": SCHEMA_VERSION,
        "source_run_id": spec.source_record.run_id,
        "source_method": spec.source_record.method.upper(),
        "target": spec.source_record.target,
        "seed": spec.source_record.seed,
        "epoch": spec.epoch,
        "checkpoint_sha256": checkpoint_hash,
        "forward_batch_size": int(cfg.evaluation.forward_batch_size),
        "forward_seed": forward_seed,
    }
    input_fingerprint = _json_fingerprint(input_payload)
    reference_payload = {
        "schema_version": SCHEMA_VERSION,
        "estimator_version": "posterior_hmc_chain_means_v1",
        "input_fingerprint": input_fingerprint,
        "reference_seed": stable_seed(
            spec.source_cell_key,
            "reference_hmc",
        ),
        "reference": OmegaConf.to_container(
            cfg.evaluation.reference,
            resolve=True,
        ),
    }
    reference_fingerprint = _json_fingerprint(reference_payload)
    config_fingerprint = analysis_fingerprint(cfg)
    cache_dir = (
        _reference_cache_root(cfg)
        / spec.source_record.target
        / f"seed_{spec.source_record.seed}"
        / f"epoch_{spec.epoch}"
    )
    return ArtifactPaths(
        input_fingerprint=input_fingerprint,
        reference_fingerprint=reference_fingerprint,
        analysis_fingerprint=config_fingerprint,
        forward_bank=(
            cache_dir
            / f"forward_{input_fingerprint[:16]}.pt"
        ),
        hmc_reference=(
            cache_dir
            / f"hmc_{reference_fingerprint[:16]}.pt"
        ),
        run_root=(
            _runtime_root(cfg)
            / "runs"
            / config_fingerprint[:16]
        ),
    )


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.tmp.{os.getpid()}"
    )
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.tmp.{os.getpid()}"
    )
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _torch_load(
    path: Path,
    *,
    map_location: str | torch.device,
    weights_only: bool,
) -> Any:
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=weights_only,
        )
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_payload(
    path: Path,
    *,
    fingerprint_key: str,
    expected_fingerprint: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = _torch_load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a dictionary artifact in {path}.")
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise RuntimeError(f"Schema mismatch in {path}.")
    if payload.get(fingerprint_key) != expected_fingerprint:
        raise RuntimeError(f"Fingerprint mismatch in {path}.")
    return payload


def _build_runner(
    record: RunRecord,
    cfg: DictConfig,
    *,
    worker_tag: str,
) -> Any:
    runner_cfg = prepare_config(
        record,
        device=str(cfg.evaluation.device),
        scratch_results=(
            f"{cfg.output.scratch_results_dir}/{worker_tag}"
        ),
        scratch_tb=f"{cfg.output.scratch_tb_dir}/{worker_tag}",
    )
    set_seed(record.seed, runner_cfg.device == "cuda")
    runner = Runners[record.runner_type](config=runner_cfg)
    if hasattr(runner, "writer"):
        runner.writer.close()
    remove_file_handlers()
    return runner


def runner_config_fingerprint(runner: Any) -> str:
    """Fingerprint the fully merged config that controls an estimator."""
    return _json_fingerprint(
        OmegaConf.to_container(runner.config, resolve=True)
    )


def method_artifact_fingerprint(
    *,
    paths: ArtifactPaths,
    method: str,
    estimator_config_fingerprint: str,
    dsivi_reverse_sha256: str | None,
) -> str:
    return _json_fingerprint({
        "schema_version": SCHEMA_VERSION,
        "estimator_version": "shared_checkpoint_native_score_v1",
        "analysis_fingerprint": paths.analysis_fingerprint,
        "input_fingerprint": paths.input_fingerprint,
        "method": method.upper(),
        "estimator_config_fingerprint": (
            estimator_config_fingerprint
        ),
        "dsivi_reverse_sha256": dsivi_reverse_sha256,
    })


def _release_runner(runner: Any | None) -> None:
    if runner is None:
        return
    if hasattr(runner, "writer"):
        runner.writer.close()
    remove_file_handlers()
    del runner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_aligned_checkpoint(
    runner: Any,
    spec: SharedCheckpointSpec,
    *,
    load_dsivi_reverse: bool,
) -> None:
    vi_state = _torch_load(
        spec.checkpoint_dir / "vi_model.pt",
        map_location=runner.device,
        weights_only=True,
    )
    runner.vi_model.load_state_dict(vi_state)
    runner.vi_model.eval()
    for parameter in runner.vi_model.parameters():
        parameter.requires_grad_(False)

    if load_dsivi_reverse:
        if not hasattr(runner, "reverse_model"):
            raise TypeError("DSIVI evaluation requires a reverse model.")
        reverse_state = _torch_load(
            spec.checkpoint_dir / "reverse_model.pt",
            map_location=runner.device,
            weights_only=True,
        )
        runner.reverse_model.load_state_dict(reverse_state)
        runner.reverse_model.eval()
        for parameter in runner.reverse_model.parameters():
            parameter.requires_grad_(False)
    runner.curr_epoch = spec.epoch


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def prepare_forward_bank(
    cfg: DictConfig,
    spec: SharedCheckpointSpec,
    *,
    resume: bool = True,
) -> Path:
    paths = artifact_paths(cfg, spec)
    if resume and paths.forward_bank.is_file():
        _load_payload(
            paths.forward_bank,
            fingerprint_key="input_fingerprint",
            expected_fingerprint=paths.input_fingerprint,
        )
        print(f"reused_forward_bank={paths.forward_bank}", flush=True)
        return paths.forward_bank

    runner: Any | None = None
    try:
        runner = _build_runner(
            spec.source_record,
            cfg,
            worker_tag="prepare",
        )
        load_aligned_checkpoint(
            runner,
            spec,
            load_dsivi_reverse=False,
        )
        device = torch.device(runner.device)
        forward_seed = stable_seed(spec.source_cell_key, "forward")
        seed_everything(
            forward_seed,
            use_cuda=device.type == "cuda",
        )
        with torch.no_grad():
            generating_epsilon, z = runner.vi_model.sampling(
                num=int(cfg.evaluation.forward_batch_size)
            )
            target_score = runner.target_model.score(z).detach()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "input_fingerprint": paths.input_fingerprint,
            "source_cell_key": spec.source_cell_key,
            "source_run_id": spec.source_record.run_id,
            "source_method": spec.source_record.method.upper(),
            "target": spec.source_record.target,
            "seed": spec.source_record.seed,
            "epoch": spec.epoch,
            "checkpoint_dir": spec.checkpoint_dir.as_posix(),
            "checkpoint_sha256": file_sha256(
                spec.checkpoint_dir / "vi_model.pt"
            ),
            "forward_seed": forward_seed,
            "generating_epsilon": generating_epsilon.detach().cpu(),
            "z": z.detach().cpu(),
            "target_score": target_score.cpu(),
            "created_at": utc_now(),
        }
        atomic_torch_save(paths.forward_bank, payload)
    finally:
        _release_runner(runner)
    print(f"saved_forward_bank={paths.forward_bank}", flush=True)
    return paths.forward_bank


def run_hmc_reference(
    cfg: DictConfig,
    spec: SharedCheckpointSpec,
    *,
    resume: bool = True,
) -> Path:
    paths = artifact_paths(cfg, spec)
    if resume and paths.hmc_reference.is_file():
        _load_payload(
            paths.hmc_reference,
            fingerprint_key="reference_fingerprint",
            expected_fingerprint=paths.reference_fingerprint,
        )
        print(f"reused_hmc_reference={paths.hmc_reference}", flush=True)
        return paths.hmc_reference

    inputs = _load_payload(
        paths.forward_bank,
        fingerprint_key="input_fingerprint",
        expected_fingerprint=paths.input_fingerprint,
    )
    runner: Any | None = None
    try:
        runner = _build_runner(
            spec.source_record,
            cfg,
            worker_tag="reference",
        )
        load_aligned_checkpoint(
            runner,
            spec,
            load_dsivi_reverse=False,
        )
        device = torch.device(runner.device)
        generating_epsilon = inputs["generating_epsilon"].to(device)
        z = inputs["z"].to(device)
        reference_seed = stable_seed(
            spec.source_cell_key,
            "reference_hmc",
        )
        seed_everything(
            reference_seed,
            use_cuda=device.type == "cuda",
        )
        reference_cfg = cfg.evaluation.reference
        _sync(device)
        started = time.perf_counter()
        chain_score_means, diagnostics = posterior_hmc_reference_scores(
            runner.vi_model,
            z,
            generating_epsilon,
            total_samples=int(reference_cfg.total_samples),
            num_chains=int(reference_cfg.num_chains),
            burn_in_steps=int(reference_cfg.burn_in_steps),
            thinning=int(reference_cfg.thinning),
            step_size=float(reference_cfg.step_size),
            leapfrog_steps=int(reference_cfg.leapfrog_steps),
            init_jitter_scale=float(reference_cfg.init_jitter_scale),
            adapt_step_size=bool(reference_cfg.adapt_step_size),
            target_acceptance=float(reference_cfg.target_acceptance),
            adaptation_rate=float(reference_cfg.adaptation_rate),
            min_step_size=float(reference_cfg.min_step_size),
            max_step_size=float(reference_cfg.max_step_size),
            divergence_threshold=float(
                reference_cfg.divergence_threshold
            ),
            accumulator_dtype=_accumulator_dtype(
                str(reference_cfg.accumulator_dtype)
            ),
        )
        _sync(device)
        runtime = time.perf_counter() - started
        quality_status, quality_issues = assess_hmc_reference_quality(
            diagnostics,
            reference_cfg.quality,
        )
        reference_metrics = compute_score_metrics(
            None,
            chain_score_means,
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "reference_fingerprint": paths.reference_fingerprint,
            "input_fingerprint": paths.input_fingerprint,
            "source_cell_key": spec.source_cell_key,
            "source_run_id": spec.source_record.run_id,
            "target": spec.source_record.target,
            "seed": spec.source_record.seed,
            "epoch": spec.epoch,
            "reference_seed": reference_seed,
            "forward_seed": inputs["forward_seed"],
            "checkpoint_sha256": inputs["checkpoint_sha256"],
            "reference_config": OmegaConf.to_container(
                reference_cfg,
                resolve=True,
            ),
            "forward_bank_path": paths.forward_bank.as_posix(),
            "chain_score_means": chain_score_means.detach().cpu(),
            "reference_mean": (
                chain_score_means.mean(dim=0).detach().cpu()
            ),
            "diagnostics": diagnostics,
            "reference_internal_l2": reference_metrics[
                "reference_internal_l2"
            ],
            "reference_mean_mcse_l2": reference_metrics[
                "reference_mean_mcse_l2"
            ],
            "quality_status": quality_status,
            "quality_issues": quality_issues,
            "runtime_sec": runtime,
            "completed_at": utc_now(),
        }
        atomic_torch_save(paths.hmc_reference, payload)
    finally:
        _release_runner(runner)
    print(
        f"saved_hmc_reference={paths.hmc_reference} "
        f"runtime_sec={payload['runtime_sec']:.3f}",
        flush=True,
    )
    return paths.hmc_reference


def _aisivi_refit_metadata(
    runner: Any,
    cfg: DictConfig,
) -> dict[str, Any]:
    scheduler_cfg = runner.rev_train_cfg.get("scheduler")
    return {
        "steps": int(cfg.evaluation.aisivi_refit.steps),
        "batch_size": int(cfg.evaluation.aisivi_refit.get(
            "batch_size",
            runner.rev_batch_size,
        )),
        "optimizer": type(runner.training_reverse_optimizer).__name__,
        "initial_lr": float(runner.reverse_lr),
        "weight_decay": float(runner.reverse_weight_decay),
        "scheduler": (
            OmegaConf.to_container(scheduler_cfg, resolve=True)
            if OmegaConf.is_config(scheduler_cfg)
            else scheduler_cfg
        ),
        "grad_clip": (
            None
            if runner.grad_clip is None
            else float(runner.grad_clip)
        ),
    }


def _capture_rng_state(use_cuda: bool) -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch_cpu": torch.get_rng_state(),
    }
    if use_cuda:
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(
    state: dict[str, Any],
    *,
    use_cuda: bool,
) -> None:
    torch.set_rng_state(state["torch_cpu"])
    if use_cuda and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def refit_aisivi_flow(
    runner: Any,
    cfg: DictConfig,
    spec: SharedCheckpointSpec,
    paths: ArtifactPaths,
    *,
    resume: bool,
    estimator_config_fingerprint: str,
) -> dict[str, Any]:
    refit_cfg = cfg.evaluation.aisivi_refit
    steps = int(refit_cfg.steps)
    batch_size = int(
        refit_cfg.get("batch_size", runner.rev_batch_size)
    )
    checkpoint_every = int(refit_cfg.get("checkpoint_every", 500))
    log_every = int(refit_cfg.get("log_every", 250))
    if steps < 1 or batch_size < 1:
        raise ValueError("AISIVI refit steps and batch size must be positive.")
    if checkpoint_every < 1 or log_every < 1:
        raise ValueError(
            "AISIVI checkpoint_every and log_every must be positive."
        )
    optimizer = runner.training_reverse_optimizer
    scheduler = runner.training_reverse_scheduler
    if optimizer is None:
        raise RuntimeError("AISIVI refit requires an optimizer.")

    refit_fingerprint = _json_fingerprint({
        "schema_version": SCHEMA_VERSION,
        "analysis_fingerprint": paths.analysis_fingerprint,
        "input_fingerprint": paths.input_fingerprint,
        "source_cell_key": spec.source_cell_key,
        "estimator_config_fingerprint": (
            estimator_config_fingerprint
        ),
        "refit": _aisivi_refit_metadata(runner, cfg),
    })
    state_path = paths.aisivi_refit_state(
        spec,
        refit_fingerprint,
    )
    start_step = 0
    loss_sum = 0.0
    loss_count = 0
    elapsed_before = 0.0
    recent_losses: deque[float] = deque(maxlen=100)
    device = torch.device(runner.device)
    use_cuda = device.type == "cuda"

    if resume and state_path.is_file():
        state = _torch_load(
            state_path,
            map_location=device,
            weights_only=False,
        )
        if state.get("refit_fingerprint") != refit_fingerprint:
            raise RuntimeError(
                f"AISIVI refit fingerprint mismatch in {state_path}."
            )
        runner.reverse_model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        if scheduler is not None and state["scheduler_state"] is not None:
            scheduler.load_state_dict(state["scheduler_state"])
        start_step = int(state["completed_steps"])
        loss_sum = float(state["loss_sum"])
        loss_count = int(state["loss_count"])
        elapsed_before = float(state["runtime_sec"])
        recent_losses.extend(
            float(value) for value in state.get("recent_losses", [])
        )
        _restore_rng_state(state["rng_state"], use_cuda=use_cuda)
        if start_step > steps:
            raise RuntimeError(
                f"Saved AISIVI refit has {start_step} steps, requested "
                f"only {steps}."
            )
        print(
            f"resumed_aisivi_refit_step={start_step}",
            flush=True,
        )
    else:
        refit_seed = stable_seed(
            spec.source_cell_key,
            "AISIVI",
            "reverse_refit",
        )
        seed_everything(refit_seed, use_cuda=use_cuda)

    runner.reverse_model.train()
    started = time.perf_counter()

    def save_state(completed_steps: int) -> None:
        runtime = elapsed_before + (time.perf_counter() - started)
        state = {
            "schema_version": SCHEMA_VERSION,
            "refit_fingerprint": refit_fingerprint,
            "analysis_fingerprint": paths.analysis_fingerprint,
            "source_cell_key": spec.source_cell_key,
            "completed_steps": completed_steps,
            "requested_steps": steps,
            "model_state": runner.reverse_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": (
                None if scheduler is None else scheduler.state_dict()
            ),
            "loss_sum": loss_sum,
            "loss_count": loss_count,
            "recent_losses": list(recent_losses),
            "runtime_sec": runtime,
            "rng_state": _capture_rng_state(use_cuda),
            "updated_at": utc_now(),
        }
        atomic_torch_save(state_path, state)

    for step in range(start_step + 1, steps + 1):
        epsilon, z = runner.vi_model.sampling(num=batch_size)
        optimizer.zero_grad(set_to_none=True)
        loss = -runner.reverse_model.log_prob(epsilon, z).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"AISIVI refit loss became non-finite at step {step}."
            )
        loss.backward()
        if runner.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                runner.reverse_model.parameters(),
                max_norm=runner.grad_clip,
            )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        numeric_loss = float(loss.detach().item())
        loss_sum += numeric_loss
        loss_count += 1
        recent_losses.append(numeric_loss)
        if step % log_every == 0 or step == steps:
            current_lr = float(optimizer.param_groups[0]["lr"])
            print(
                f"aisivi_refit_step={step}/{steps} "
                f"loss={numeric_loss:.6f} lr={current_lr:.6g}",
                flush=True,
            )
        if step % checkpoint_every == 0 or step == steps:
            save_state(step)

    final_state = _torch_load(
        state_path,
        map_location=device,
        weights_only=False,
    )
    runner.reverse_model.load_state_dict(final_state["model_state"])
    runner.reverse_model.eval()
    atomic_torch_save(
        paths.aisivi_flow(spec),
        runner.reverse_model.state_dict(),
    )
    return {
        **_aisivi_refit_metadata(runner, cfg),
        "completed_steps": int(final_state["completed_steps"]),
        "mean_loss": (
            float(final_state["loss_sum"])
            / max(1, int(final_state["loss_count"]))
        ),
        "final_100_mean_loss": (
            sum(float(value) for value in final_state["recent_losses"])
            / max(1, len(final_state["recent_losses"]))
        ),
        "runtime_sec": float(final_state["runtime_sec"]),
        "refit_fingerprint": refit_fingerprint,
        "flow_checkpoint": paths.aisivi_flow(spec).as_posix(),
    }


def run_method_score(
    cfg: DictConfig,
    spec: SharedCheckpointSpec,
    method: str,
    *,
    resume: bool = True,
) -> Path:
    normalized = method.upper()
    configured = [
        str(value).upper() for value in cfg.selection.methods
    ]
    if normalized not in configured:
        raise ValueError(f"Method {normalized} is not configured.")
    paths = artifact_paths(cfg, spec)
    output_path = paths.method_score(spec, normalized)
    inputs = _load_payload(
        paths.forward_bank,
        fingerprint_key="input_fingerprint",
        expected_fingerprint=paths.input_fingerprint,
    )
    record = spec.method_record(normalized)
    runner: Any | None = None
    try:
        runner = _build_runner(
            record,
            cfg,
            worker_tag=f"method_{normalized.lower()}",
        )
        load_aligned_checkpoint(
            runner,
            spec,
            load_dsivi_reverse=normalized == "DSIVI",
        )
        estimator_config_hash = runner_config_fingerprint(runner)
        dsivi_reverse_hash = (
            file_sha256(spec.checkpoint_dir / "reverse_model.pt")
            if normalized == "DSIVI"
            else None
        )
        method_fingerprint = method_artifact_fingerprint(
            paths=paths,
            method=normalized,
            estimator_config_fingerprint=estimator_config_hash,
            dsivi_reverse_sha256=dsivi_reverse_hash,
        )
        if resume and output_path.is_file():
            existing = _load_payload(
                output_path,
                fingerprint_key="analysis_fingerprint",
                expected_fingerprint=paths.analysis_fingerprint,
            )
            if (
                existing.get("method_fingerprint")
                == method_fingerprint
            ):
                print(
                    f"reused_method_score={output_path}",
                    flush=True,
                )
                return output_path
            print(
                f"stale_method_score={output_path}; recomputing",
                flush=True,
            )

        refit_metadata: dict[str, Any] | None = None
        if normalized == "AISIVI":
            refit_metadata = refit_aisivi_flow(
                runner,
                cfg,
                spec,
                paths,
                resume=resume,
                estimator_config_fingerprint=estimator_config_hash,
            )
            for parameter in runner.reverse_model.parameters():
                parameter.requires_grad_(False)

        device = torch.device(runner.device)
        generating_epsilon = inputs["generating_epsilon"].to(device)
        z = inputs["z"].to(device)
        method_seed = stable_seed(
            spec.source_cell_key,
            normalized,
            "method",
        )
        seed_everything(
            method_seed,
            use_cuda=device.type == "cuda",
        )
        _sync(device)
        started = time.perf_counter()
        method_score, diagnostics = method_native_score(
            runner,
            normalized,
            z,
            generating_epsilon,
            aisivi_z_chunk_size=int(
                cfg.evaluation.get(
                    "aisivi_z_chunk_size",
                    z.shape[0],
                )
            ),
        )
        _sync(device)
        runtime = time.perf_counter() - started
        if not torch.isfinite(method_score).all():
            raise FloatingPointError(
                f"{normalized} produced a non-finite score."
            )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "analysis_fingerprint": paths.analysis_fingerprint,
            "input_fingerprint": paths.input_fingerprint,
            "method_fingerprint": method_fingerprint,
            "estimator_config_fingerprint": estimator_config_hash,
            "dsivi_reverse_sha256": dsivi_reverse_hash,
            "source_cell_key": spec.source_cell_key,
            "source_run_id": spec.source_record.run_id,
            "estimator_run_id": record.run_id,
            "estimator_config_path": record.config_path.as_posix(),
            "method": normalized,
            "target": spec.source_record.target,
            "seed": spec.source_record.seed,
            "epoch": spec.epoch,
            "method_seed": method_seed,
            "method_score": method_score.detach().cpu(),
            "diagnostics": diagnostics,
            "aisivi_refit": refit_metadata,
            "runtime_sec": runtime,
            "completed_at": utc_now(),
        }
        atomic_torch_save(output_path, payload)
    finally:
        _release_runner(runner)
    print(
        f"saved_method_score={output_path} "
        f"runtime_sec={payload['runtime_sec']:.3f}",
        flush=True,
    )
    return output_path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _mean_sample_sd(
    values: Iterable[float],
) -> tuple[float | None, float | None]:
    materialized = [float(value) for value in values]
    if not materialized:
        return None, None
    if not all(math.isfinite(value) for value in materialized):
        raise FloatingPointError(
            "Cannot summarize a metric containing non-finite values."
        )
    mean = sum(materialized) / len(materialized)
    if len(materialized) == 1:
        return mean, 0.0
    variance = sum(
        (value - mean) ** 2 for value in materialized
    ) / (len(materialized) - 1)
    return mean, math.sqrt(variance)


def summarize_shared_results(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Summarize method rows and unique HMC source cells across seeds."""

    method_groups: dict[
        tuple[str, float, int, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in rows:
        key = (
            str(row["target"]),
            float(row["progress"]),
            int(row["epoch"]),
            str(row["method"]).upper(),
        )
        method_groups[key].append(row)

    method_summary: list[dict[str, Any]] = []
    for key, items in sorted(method_groups.items()):
        target, progress, epoch, method = key
        method_mean, method_sd = _mean_sample_sd(
            float(item["method_hmc_l2"]) for item in items
        )
        relative_mean, relative_sd = _mean_sample_sd(
            float(item["method_hmc_relative_l2"]) for item in items
        )
        runtime_mean, runtime_sd = _mean_sample_sd(
            float(item["method_runtime_sec"]) for item in items
        )
        acceptance_values = [
            float(item["uivi_average_acceptance_rate"])
            for item in items
            if item.get("uivi_average_acceptance_rate") is not None
        ]
        acceptance_mean, acceptance_sd = _mean_sample_sd(
            acceptance_values
        )
        native_budgets = {
            int(item["native_auxiliary_samples"]) for item in items
        }
        if len(native_budgets) != 1:
            raise RuntimeError(
                f"Inconsistent native sample budgets for {key}: "
                f"{sorted(native_budgets)}"
            )
        method_summary.append({
            "target": target,
            "progress": progress,
            "epoch": epoch,
            "method": method,
            "n_seeds": len(items),
            "seeds": ",".join(
                str(value)
                for value in sorted(
                    int(item["seed"]) for item in items
                )
            ),
            "method_hmc_l2_mean": method_mean,
            "method_hmc_l2_sd": method_sd,
            "method_hmc_relative_l2_mean": relative_mean,
            "method_hmc_relative_l2_sd": relative_sd,
            "method_runtime_sec_mean": runtime_mean,
            "method_runtime_sec_sd": runtime_sd,
            "native_auxiliary_samples": next(iter(native_budgets)),
            "uivi_average_acceptance_rate_mean": acceptance_mean,
            "uivi_average_acceptance_rate_sd": acceptance_sd,
        })

    unique_cells: dict[
        tuple[str, int, int],
        dict[str, Any],
    ] = {}
    invariant_fields = (
        "progress",
        "hmc_internal_l2",
        "hmc_mean_mcse_l2",
        "hmc_runtime_sec",
        "hmc_quality_status",
        "hmc_quality_issues",
        "hmc_average_acceptance_rate",
        "hmc_post_burn_acceptance_rate",
        "hmc_score_rhat_p95",
        "hmc_reference_path",
    )
    for row in rows:
        cell_key = (
            str(row["target"]),
            int(row["seed"]),
            int(row["epoch"]),
        )
        existing = unique_cells.get(cell_key)
        if existing is None:
            unique_cells[cell_key] = row
            continue
        for field in invariant_fields:
            left = existing.get(field)
            right = row.get(field)
            if isinstance(left, (float, int)) and isinstance(
                right,
                (float, int),
            ):
                equal = math.isclose(
                    float(left),
                    float(right),
                    rel_tol=1.0e-12,
                    abs_tol=0.0,
                )
            else:
                equal = left == right
            if not equal:
                raise RuntimeError(
                    "Method rows disagree about shared HMC field "
                    f"{field!r} for {cell_key}."
                )

    hmc_groups: dict[
        tuple[str, float, int],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in unique_cells.values():
        key = (
            str(row["target"]),
            float(row["progress"]),
            int(row["epoch"]),
        )
        hmc_groups[key].append(row)

    hmc_summary: list[dict[str, Any]] = []
    metric_fields = (
        "hmc_internal_l2",
        "hmc_mean_mcse_l2",
        "hmc_runtime_sec",
        "hmc_average_acceptance_rate",
        "hmc_post_burn_acceptance_rate",
        "hmc_score_rhat_p95",
    )
    for key, items in sorted(hmc_groups.items()):
        target, progress, epoch = key
        summary: dict[str, Any] = {
            "target": target,
            "progress": progress,
            "epoch": epoch,
            "n_seeds": len(items),
            "seeds": ",".join(
                str(value)
                for value in sorted(
                    int(item["seed"]) for item in items
                )
            ),
            "hmc_quality_n_pass": sum(
                item["hmc_quality_status"] == "pass"
                for item in items
            ),
            "hmc_quality_n_warning": sum(
                item["hmc_quality_status"] != "pass"
                for item in items
            ),
        }
        for field in metric_fields:
            mean, sd = _mean_sample_sd(
                float(item[field]) for item in items
            )
            summary[f"{field}_mean"] = mean
            summary[f"{field}_sd"] = sd
        hmc_summary.append(summary)
    return method_summary, hmc_summary


def _metric_text(
    mean: float | None,
    sd: float | None,
    *,
    percent: bool = False,
) -> str:
    if mean is None or sd is None:
        return "—"
    scale = 100.0 if percent else 1.0
    suffix = "%" if percent else ""
    if percent:
        return f"{scale * mean:.2f} ± {scale * sd:.2f}{suffix}"
    return f"{mean:.4e} ± {sd:.4e}"


def _method_markdown_lines(
    method_summary: list[dict[str, Any]],
    methods: list[str],
) -> list[str]:
    lines = [
        "## Method–HMC L2 over training",
        "",
        "| Checkpoint | " + " | ".join(methods) + " |",
        "|---:|" + "|".join("---:" for _ in methods) + "|",
    ]
    stage_keys = sorted({
        (
            str(row["target"]),
            float(row["progress"]),
            int(row["epoch"]),
        )
        for row in method_summary
    })
    lookup = {
        (
            str(row["target"]),
            float(row["progress"]),
            int(row["epoch"]),
            str(row["method"]).upper(),
        ): row
        for row in method_summary
    }
    multiple_targets = len({key[0] for key in stage_keys}) > 1
    for target, progress, epoch in stage_keys:
        checkpoint = f"{epoch:,} ({100.0 * progress:.0f}%)"
        if multiple_targets:
            checkpoint = f"{target}: {checkpoint}"
        cells = []
        for method in methods:
            row = lookup.get((target, progress, epoch, method))
            cells.append(
                "—"
                if row is None
                else _metric_text(
                    row["method_hmc_l2_mean"],
                    row["method_hmc_l2_sd"],
                )
            )
        lines.append(f"| {checkpoint} | " + " | ".join(cells) + " |")
    return lines


def _hmc_markdown_lines(
    hmc_summary: list[dict[str, Any]],
) -> list[str]:
    lines = [
        "## HMC internal L2 over training",
        "",
        (
            "| Checkpoint | HMC internal L2 | HMC mean MCSE L2 | "
            "Post-burn acceptance | Score R-hat p95 | Quality |"
        ),
        "|---:|---:|---:|---:|---:|---:|",
    ]
    multiple_targets = (
        len({str(row["target"]) for row in hmc_summary}) > 1
    )
    for row in sorted(
        hmc_summary,
        key=lambda value: (
            str(value["target"]),
            int(value["epoch"]),
        ),
    ):
        checkpoint = (
            f"{int(row['epoch']):,} "
            f"({100.0 * float(row['progress']):.0f}%)"
        )
        if multiple_targets:
            checkpoint = f"{row['target']}: {checkpoint}"
        quality = (
            f"{int(row['hmc_quality_n_pass'])}/"
            f"{int(row['n_seeds'])} pass"
        )
        lines.append(
            f"| {checkpoint} | "
            f"{_metric_text(row['hmc_internal_l2_mean'], row['hmc_internal_l2_sd'])} | "
            f"{_metric_text(row['hmc_mean_mcse_l2_mean'], row['hmc_mean_mcse_l2_sd'])} | "
            f"{_metric_text(row['hmc_post_burn_acceptance_rate_mean'], row['hmc_post_burn_acceptance_rate_sd'], percent=True)} | "
            f"{_metric_text(row['hmc_score_rhat_p95_mean'], row['hmc_score_rhat_p95_sd'])} | "
            f"{quality} |"
        )
    return lines


def _uivi_markdown_lines(
    method_summary: list[dict[str, Any]],
) -> list[str]:
    rows = [
        row for row in method_summary if row["method"] == "UIVI"
    ]
    lines = [
        "## Native UIVI acceptance over training",
        "",
        "| Checkpoint | Average acceptance |",
        "|---:|---:|",
    ]
    for row in sorted(
        rows,
        key=lambda value: (
            str(value["target"]),
            int(value["epoch"]),
        ),
    ):
        lines.append(
            f"| {int(row['epoch']):,} "
            f"({100.0 * float(row['progress']):.0f}%) | "
            f"{_metric_text(row['uivi_average_acceptance_rate_mean'], row['uivi_average_acceptance_rate_sd'], percent=True)} |"
        )
    return lines


def _write_markdown_report(
    path: Path,
    method_summary: list[dict[str, Any]],
    hmc_summary: list[dict[str, Any]],
    *,
    methods: list[str],
    seeds: list[int],
) -> None:
    seed_text = ", ".join(str(seed) for seed in seeds)
    lines = [
        "# Shared-DSIVI score approximation",
        "",
        (
            "At each seed and training stage, all method-native estimators "
            "use that cell's x_shaped DSIVI variational checkpoint."
        ),
        (
            "The posterior-HMC reference uses 20 chains, 1,000 burn-in "
            "transitions, and 5,000 retained samples per chain. Per-chain "
            "score means are persisted at the paths in checkpoint_metrics.csv."
        ),
        "",
        (
            "Values are mean ± sample standard deviation across seeds "
            f"{seed_text}."
        ),
        "",
    ]
    lines.extend(_method_markdown_lines(method_summary, methods))
    lines.extend([""])
    lines.extend(_hmc_markdown_lines(hmc_summary))
    lines.extend([""])
    lines.extend(_uivi_markdown_lines(method_summary))
    lines.extend([
        "",
        (
            "Method–HMC L2 is the mean over the 1,024 fixed z values of "
            "the squared Euclidean difference between the method score and "
            "the mean of the 20 HMC chain-score estimates."
        ),
        (
            "HMC internal L2 is the mean squared deviation of individual "
            "HMC chain-score estimates from their 20-chain mean."
        ),
        (
            "UIVI acceptance is averaged over all native UIVI HMC "
            "transitions and z values, including its five burn-in "
            "transitions."
        ),
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _write_latex_report(
    path: Path,
    method_summary: list[dict[str, Any]],
    hmc_summary: list[dict[str, Any]],
    *,
    methods: list[str],
) -> None:
    lines = [
        r"\begin{tabular}{r" + "c" * len(methods) + "}",
        r"\toprule",
        "Checkpoint & " + " & ".join(methods) + r" \\",
        r"\midrule",
    ]
    stage_keys = sorted({
        (
            str(row["target"]),
            float(row["progress"]),
            int(row["epoch"]),
        )
        for row in method_summary
    })
    lookup = {
        (
            str(row["target"]),
            float(row["progress"]),
            int(row["epoch"]),
            str(row["method"]).upper(),
        ): row
        for row in method_summary
    }
    for target, progress, epoch in stage_keys:
        cells = []
        for method in methods:
            row = lookup[(target, progress, epoch, method)]
            cells.append(
                f"{float(row['method_hmc_l2_mean']):.4e} "
                rf"$\pm$ {float(row['method_hmc_l2_sd']):.4e}"
            )
        lines.append(
            f"{epoch} ({100.0 * progress:.0f}\\%) & "
            + " & ".join(cells)
            + r" \\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\begin{tabular}{rccc}",
        r"\toprule",
        (
            r"Checkpoint & HMC internal L2 & HMC mean MCSE L2 & "
            r"UIVI acceptance \\"
        ),
        r"\midrule",
    ])
    uivi_lookup = {
        (
            str(row["target"]),
            float(row["progress"]),
            int(row["epoch"]),
        ): row
        for row in method_summary
        if row["method"] == "UIVI"
    }
    for row in sorted(
        hmc_summary,
        key=lambda value: (
            str(value["target"]),
            int(value["epoch"]),
        ),
    ):
        key = (
            str(row["target"]),
            float(row["progress"]),
            int(row["epoch"]),
        )
        uivi = uivi_lookup[key]
        lines.append(
            f"{int(row['epoch'])} "
            f"({100.0 * float(row['progress']):.0f}\\%) & "
            f"{float(row['hmc_internal_l2_mean']):.4e} "
            rf"$\pm$ {float(row['hmc_internal_l2_sd']):.4e} & "
            f"{float(row['hmc_mean_mcse_l2_mean']):.4e} "
            rf"$\pm$ {float(row['hmc_mean_mcse_l2_sd']):.4e} & "
            f"{100.0 * float(uivi['uivi_average_acceptance_rate_mean']):.2f}"
            rf"\% $\pm$ "
            f"{100.0 * float(uivi['uivi_average_acceptance_rate_sd']):.2f}"
            r"\% \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def aggregate_shared_results(
    cfg: DictConfig,
    specs: list[SharedCheckpointSpec],
) -> tuple[list[dict[str, Any]], Path]:
    methods = [str(value).upper() for value in cfg.selection.methods]
    rows: list[dict[str, Any]] = []
    hmc_paths: list[Path] = []
    for spec in specs:
        paths = artifact_paths(cfg, spec)
        reference = _load_payload(
            paths.hmc_reference,
            fingerprint_key="reference_fingerprint",
            expected_fingerprint=paths.reference_fingerprint,
        )
        _load_payload(
            paths.forward_bank,
            fingerprint_key="input_fingerprint",
            expected_fingerprint=paths.input_fingerprint,
        )
        hmc_scores = reference["chain_score_means"]
        if not torch.isfinite(hmc_scores).all():
            raise FloatingPointError(
                f"Non-finite HMC score in {paths.hmc_reference}."
            )
        hmc_paths.append(paths.hmc_reference)
        for method in methods:
            method_artifact = _load_payload(
                paths.method_score(spec, method),
                fingerprint_key="analysis_fingerprint",
                expected_fingerprint=paths.analysis_fingerprint,
            )
            if (
                method_artifact.get("input_fingerprint")
                != paths.input_fingerprint
            ):
                raise RuntimeError(
                    f"Input mismatch for {method} at {spec.key}."
                )
            method_score = method_artifact["method_score"]
            metrics = compute_score_metrics(
                method_score,
                hmc_scores,
            )
            diagnostics = method_artifact["diagnostics"]
            refit = method_artifact.get("aisivi_refit") or {}
            rows.append({
                "analysis_fingerprint": paths.analysis_fingerprint,
                "source_run_id": spec.source_record.run_id,
                "source_method": spec.source_record.method.upper(),
                "source_checkpoint_dir": (
                    spec.checkpoint_dir.as_posix()
                ),
                "estimator_run_id": method_artifact[
                    "estimator_run_id"
                ],
                "estimator_config_path": method_artifact[
                    "estimator_config_path"
                ],
                "estimator_config_fingerprint": method_artifact.get(
                    "estimator_config_fingerprint"
                ),
                "method_fingerprint": method_artifact.get(
                    "method_fingerprint"
                ),
                "dsivi_reverse_sha256": method_artifact.get(
                    "dsivi_reverse_sha256"
                ),
                "target": spec.source_record.target,
                "seed": spec.source_record.seed,
                "progress": spec.progress,
                "epoch": spec.epoch,
                "method": method,
                "method_hmc_l2": metrics["method_l2"],
                "method_hmc_l2_z_sd": metrics["method_l2_z_sd"],
                "method_hmc_relative_l2": metrics[
                    "method_relative_l2"
                ],
                "hmc_internal_l2": metrics[
                    "reference_internal_l2"
                ],
                "hmc_mean_mcse_l2": metrics[
                    "reference_mean_mcse_l2"
                ],
                "native_auxiliary_samples": int(
                    diagnostics["native_auxiliary_samples"]
                ),
                "uivi_average_acceptance_rate": diagnostics.get(
                    "uivi_hmc_acceptance_rate"
                ),
                "method_runtime_sec": method_artifact["runtime_sec"],
                "aisivi_refit_steps": refit.get("completed_steps"),
                "aisivi_refit_runtime_sec": refit.get("runtime_sec"),
                "aisivi_refit_final_100_mean_loss": refit.get(
                    "final_100_mean_loss"
                ),
                "hmc_runtime_sec": reference["runtime_sec"],
                "hmc_quality_status": reference["quality_status"],
                "hmc_quality_issues": json.dumps(
                    reference["quality_issues"],
                    separators=(",", ":"),
                ),
                "hmc_average_acceptance_rate": reference[
                    "diagnostics"
                ]["hmc_acceptance_rate"],
                "hmc_post_burn_acceptance_rate": reference[
                    "diagnostics"
                ]["hmc_post_burn_acceptance_rate"],
                "hmc_score_rhat_p95": reference[
                    "diagnostics"
                ]["hmc_score_rhat_p95"],
                "hmc_reference_path": paths.hmc_reference.as_posix(),
                "forward_bank_path": paths.forward_bank.as_posix(),
                "completed_at": method_artifact["completed_at"],
            })

    report_dir = repo_path(str(cfg.output.report_dir))
    if report_dir is None:
        raise ValueError("output.report_dir must be configured.")
    report_dir.mkdir(parents=True, exist_ok=True)
    method_summary, hmc_summary = summarize_shared_results(rows)
    methods = [
        str(value).upper() for value in cfg.selection.methods
    ]
    seeds = [int(value) for value in cfg.selection.seeds]
    _write_csv(report_dir / "checkpoint_metrics.csv", rows)
    _write_csv(
        report_dir / "checkpoint_summary.csv",
        method_summary,
    )
    _write_csv(
        report_dir / "hmc_checkpoint_summary.csv",
        hmc_summary,
    )
    _write_markdown_report(
        report_dir / "score_approximation_table.md",
        method_summary,
        hmc_summary,
        methods=methods,
        seeds=seeds,
    )
    (report_dir / "method_hmc_l2_table.md").write_text(
        "\n".join(_method_markdown_lines(method_summary, methods))
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (report_dir / "hmc_internal_l2_table.md").write_text(
        "\n".join(_hmc_markdown_lines(hmc_summary)) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_latex_report(
        report_dir / "score_approximation_table.tex",
        method_summary,
        hmc_summary,
        methods=methods,
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "analysis_fingerprint": analysis_fingerprint(cfg),
        "generated_at": utc_now(),
        "git_commit": _git_commit(),
        "rows": len(rows),
        "source_cells": len(specs),
        "methods": methods,
        "method_summary_rows": len(method_summary),
        "hmc_summary_rows": len(hmc_summary),
        "hmc_reference_paths": [
            path.as_posix() for path in hmc_paths
        ],
        "hmc_scores_persisted": True,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    atomic_json_save(report_dir / "run_metadata.json", metadata)
    return rows, report_dir


def validate_production_budget(cfg: DictConfig) -> None:
    reference = cfg.evaluation.reference
    num_chains = int(reference.num_chains)
    total_samples = int(reference.total_samples)
    if num_chains != 20:
        raise ValueError("Production reference requires exactly 20 chains.")
    if int(reference.burn_in_steps) != 1000:
        raise ValueError(
            "Production reference requires 1,000 burn-in steps."
        )
    if total_samples // num_chains != 5000:
        raise ValueError(
            "Production reference requires 5,000 retained samples "
            "per chain."
        )
    if total_samples % num_chains:
        raise ValueError(
            "reference.total_samples must be divisible by num_chains."
        )
