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
import os
import subprocess
import time
from collections import deque
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
    / "score_approximation_dsivi_shared_x_shaped_10k.yaml"
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


def _runtime_root(cfg: DictConfig) -> Path:
    root = repo_path(str(cfg.output.runtime_dir))
    if root is None:
        raise ValueError("output.runtime_dir must be configured.")
    return root


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
        _runtime_root(cfg)
        / "reference_cache"
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


def _write_markdown_report(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    hmc_path: Path,
    source_checkpoint: Path,
) -> None:
    lines = [
        "# Shared-DSIVI score approximation",
        "",
        (
            "All method-native estimators use the same x_shaped DSIVI "
            f"variational checkpoint: `{source_checkpoint.as_posix()}`."
        ),
        (
            "The posterior-HMC reference uses 20 chains, 1,000 burn-in "
            "transitions, and 5,000 retained samples per chain. Its saved "
            f"per-chain score means are in `{hmc_path.as_posix()}`."
        ),
        "",
        (
            "| Method | Method–HMC L2 | HMC internal L2 | "
            "Native auxiliaries | UIVI acceptance |"
        ),
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        acceptance = row.get("uivi_average_acceptance_rate")
        acceptance_text = (
            "—"
            if acceptance is None
            else f"{100.0 * float(acceptance):.2f}%"
        )
        lines.append(
            f"| {row['method']} | "
            f"{float(row['method_hmc_l2']):.6e} | "
            f"{float(row['hmc_internal_l2']):.6e} | "
            f"{int(row['native_auxiliary_samples'])} | "
            f"{acceptance_text} |"
        )
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
    rows: list[dict[str, Any]],
) -> None:
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        (
            r"Method & Method--HMC L2 & HMC internal L2 & "
            r"Auxiliaries & UIVI acceptance \\"
        ),
        r"\midrule",
    ]
    for row in rows:
        acceptance = row.get("uivi_average_acceptance_rate")
        acceptance_text = (
            "--"
            if acceptance is None
            else f"{100.0 * float(acceptance):.2f}\\%"
        )
        lines.append(
            f"{row['method']} & "
            f"{float(row['method_hmc_l2']):.6e} & "
            f"{float(row['hmc_internal_l2']):.6e} & "
            f"{int(row['native_auxiliary_samples'])} & "
            f"{acceptance_text} \\\\"
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
    _write_csv(report_dir / "checkpoint_metrics.csv", rows)
    _write_markdown_report(
        report_dir / "score_approximation_table.md",
        rows,
        hmc_path=hmc_paths[0],
        source_checkpoint=specs[0].checkpoint_dir,
    )
    _write_latex_report(
        report_dir / "score_approximation_table.tex",
        rows,
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "analysis_fingerprint": analysis_fingerprint(cfg),
        "generated_at": utc_now(),
        "git_commit": _git_commit(),
        "rows": len(rows),
        "source_cells": len(specs),
        "methods": methods,
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
