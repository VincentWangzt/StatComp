"""DSIVI score analysis with a terminal-particle posterior SGLD reference.

For each fixed forward sample ``z_i``, the sampler targets

    q_phi(epsilon | z_i) propto q_0(epsilon) q_phi(z_i | epsilon)

and applies Fisher's identity by averaging the conditional Gaussian score
over terminal epsilon particles.  Groups of independently evolved particles
are retained as the replication unit for the within-SGLD L2 calculation.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from .config import REPO_ROOT, repo_path
from .score_approximation import (
    CellSpec,
    _accumulator_dtype,
    _build_runner,
    _finite_tensor_summary,
    _flatten_cell,
    _load_checkpoint,
    _release_runner,
    _summary_rows,
    _sync,
    _write_csv,
    atomic_write_json,
    build_cell_specs,
    cell_record_path,
    compute_score_metrics,
    config_fingerprint,
    native_dsivi_score,
    pending_cell_specs,
    posterior_log_prob_and_grad,
    seed_everything,
    select_cell_specs,
    shard_cell_specs,
    stable_seed,
    utc_now,
)


DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs"
    / "finalization"
    / "score_approximation_sgld_20x10k_20k.yaml"
)
SGLD_IMPLEMENTATION_VERSION = "posterior-sgld-terminal-v1"


def load_sgld_score_config(
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


def _device_rng_state(device: torch.device) -> torch.Tensor:
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device)
    return torch.get_rng_state()


def _set_device_rng_state(device: torch.device, state: torch.Tensor) -> None:
    if device.type == "cuda":
        torch.cuda.set_rng_state(state.cpu(), device)
    else:
        torch.set_rng_state(state.cpu())


def _group_score(
    vi_model: torch.nn.Module,
    z_particles: torch.Tensor,
    epsilon: torch.Tensor,
    *,
    accumulator_dtype: torch.dtype,
) -> torch.Tensor:
    """Return group means with shape ``[G,N,Dz]``."""
    with torch.no_grad():
        score = vi_model.score(z_particles, epsilon).detach()
        score = score.to(dtype=accumulator_dtype)
        return score.mean(dim=2).permute(1, 0, 2).contiguous()


def _validate_sgld_inputs(
    z: torch.Tensor,
    generating_epsilon: torch.Tensor,
    *,
    num_groups: int,
    chains_per_group: int,
    num_steps: int,
    step_size: float,
    init_jitter_scale: float,
    finite_check_interval: int,
) -> None:
    if z.ndim != 2 or generating_epsilon.ndim != 2:
        raise ValueError("z and generating_epsilon must both be rank two.")
    if z.shape[0] != generating_epsilon.shape[0]:
        raise ValueError("z and generating_epsilon batch sizes must match.")
    if num_groups < 2:
        raise ValueError("SGLD internal L2 requires at least two groups.")
    if chains_per_group < 1 or num_steps < 1:
        raise ValueError("SGLD chains and steps must be positive.")
    if step_size <= 0:
        raise ValueError("SGLD step_size must be positive.")
    if init_jitter_scale < 0:
        raise ValueError("SGLD init_jitter_scale must be non-negative.")
    if finite_check_interval < 1:
        raise ValueError("finite_check_interval must be positive.")


def posterior_sgld_group_scores(
    vi_model: torch.nn.Module,
    z: torch.Tensor,
    generating_epsilon: torch.Tensor,
    *,
    num_groups: int,
    chains_per_group: int,
    num_steps: int,
    step_size: float,
    init_jitter_scale: float,
    diagnostic_steps: Iterable[int] = (),
    finite_check_interval: int = 1000,
    accumulator_dtype: torch.dtype = torch.float64,
    resume_state: dict[str, Any] | None = None,
    checkpoint_interval: int = 0,
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Evolve posterior particles and return terminal group score means.

    The state layout is ``[N,G,C,D_epsilon]``.  The update convention matches
    ``utils.mcmc.SGLDSampler``:

    ``epsilon += 0.5 * h * grad_log_posterior + sqrt(h) * Normal(0, I)``.

    ``resume_state`` and ``checkpoint_callback`` make a long z tile resumable
    without changing its random-number stream.
    """
    _validate_sgld_inputs(
        z,
        generating_epsilon,
        num_groups=num_groups,
        chains_per_group=chains_per_group,
        num_steps=num_steps,
        step_size=step_size,
        init_jitter_scale=init_jitter_scale,
        finite_check_interval=finite_check_interval,
    )
    if checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must be non-negative.")

    requested_snapshots = {
        int(step)
        for step in diagnostic_steps
        if 0 < int(step) <= num_steps
    }
    requested_snapshots.add(num_steps)

    n_z, z_dim = z.shape
    epsilon_dim = generating_epsilon.shape[-1]
    device = z.device
    dtype = generating_epsilon.dtype
    particle_shape = (
        n_z,
        num_groups,
        chains_per_group,
        epsilon_dim,
    )
    z_particles = z.detach()[:, None, None, :].expand(
        n_z,
        num_groups,
        chains_per_group,
        z_dim,
    )

    snapshots: dict[int, torch.Tensor]
    initial_diagnostics: dict[str, Any]
    if resume_state is None:
        center = generating_epsilon.detach()[:, None, None, :]
        center = center.expand(particle_shape)
        jitter = torch.randn(
            particle_shape,
            dtype=dtype,
            device=device,
        )
        epsilon_current = (
            center + init_jitter_scale * jitter
        ).detach()
        initial_log_prob, initial_gradient = posterior_log_prob_and_grad(
            vi_model,
            epsilon_current,
            z_particles,
        )
        initial_gradient_norm = initial_gradient.norm(dim=-1)
        initial_center_distance_sq = (
            epsilon_current - center
        ).square().sum(dim=-1)
        initial_diagnostics = {
            "sgld_initial_log_prob_mean": float(
                initial_log_prob.to(torch.float64).mean().item()
            ),
            "sgld_initial_gradient_norm_mean": float(
                initial_gradient_norm.to(torch.float64).mean().item()
            ),
            "sgld_initial_center_distance_sq_mean": float(
                initial_center_distance_sq.to(torch.float64).mean().item()
            ),
            **_finite_tensor_summary(
                initial_gradient_norm,
                prefix="sgld_initial_gradient_norm",
            ),
        }
        current_step = 0
        cached_gradient: torch.Tensor | None = initial_gradient
        snapshots = {}
    else:
        expected = tuple(int(value) for value in particle_shape)
        saved_epsilon = resume_state["epsilon_current"]
        if tuple(saved_epsilon.shape) != expected:
            raise ValueError(
                "SGLD resume state shape does not match the requested tile."
            )
        current_step = int(resume_state["step"])
        if not 0 <= current_step < num_steps:
            raise ValueError("SGLD resume step is outside the run horizon.")
        epsilon_current = saved_epsilon.to(
            device=device,
            dtype=dtype,
        )
        snapshots = {
            int(step): value.to(
                device=device,
                dtype=accumulator_dtype,
            )
            for step, value in resume_state.get("snapshots", {}).items()
        }
        initial_diagnostics = dict(
            resume_state.get("initial_diagnostics", {})
        )
        _set_device_rng_state(device, resume_state["rng_state"])
        cached_gradient = None

    noise_scale = math.sqrt(step_size)
    for step in range(current_step + 1, num_steps + 1):
        if cached_gradient is None:
            _, gradient = posterior_log_prob_and_grad(
                vi_model,
                epsilon_current,
                z_particles,
            )
        else:
            gradient = cached_gradient
            cached_gradient = None
        noise = torch.randn_like(epsilon_current)
        epsilon_current = (
            epsilon_current
            + 0.5 * step_size * gradient
            + noise_scale * noise
        ).detach()

        if (
            step % finite_check_interval == 0
            or step == num_steps
        ):
            if not torch.isfinite(epsilon_current).all():
                raise FloatingPointError(
                    f"SGLD produced non-finite particles at step {step}."
                )

        if step in requested_snapshots:
            snapshots[step] = _group_score(
                vi_model,
                z_particles,
                epsilon_current,
                accumulator_dtype=accumulator_dtype,
            )

        if (
            checkpoint_callback is not None
            and checkpoint_interval > 0
            and step < num_steps
            and step % checkpoint_interval == 0
        ):
            checkpoint_callback({
                "step": step,
                "epsilon_current": epsilon_current.detach().cpu(),
                "rng_state": _device_rng_state(device).cpu(),
                "snapshots": {
                    key: value.detach().cpu()
                    for key, value in snapshots.items()
                },
                "initial_diagnostics": initial_diagnostics,
            })

    final_log_prob, final_gradient = posterior_log_prob_and_grad(
        vi_model,
        epsilon_current,
        z_particles,
    )
    final_gradient_norm = final_gradient.norm(dim=-1)
    center = generating_epsilon.detach()[:, None, None, :].expand(
        particle_shape
    )
    final_center_distance_sq = (
        epsilon_current - center
    ).square().sum(dim=-1)
    final_group_scores = snapshots[num_steps]
    final_mean = final_group_scores.mean(dim=0)

    diagnostics: dict[str, Any] = {
        "sgld_num_groups": num_groups,
        "sgld_chains_per_group": chains_per_group,
        "sgld_total_terminal_particles_per_z": (
            num_groups * chains_per_group
        ),
        "sgld_num_steps": num_steps,
        "sgld_step_size": step_size,
        "sgld_langevin_time": num_steps * step_size,
        "sgld_init_jitter_scale": init_jitter_scale,
        "sgld_terminal_nonfinite_fraction": float(
            1.0
            - torch.isfinite(epsilon_current).sum().item()
            / epsilon_current.numel()
        ),
        "sgld_final_log_prob_mean": float(
            final_log_prob.to(torch.float64).mean().item()
        ),
        "sgld_final_gradient_norm_mean": float(
            final_gradient_norm.to(torch.float64).mean().item()
        ),
        "sgld_final_center_distance_sq_mean": float(
            final_center_distance_sq.to(torch.float64).mean().item()
        ),
        **initial_diagnostics,
        **_finite_tensor_summary(
            final_gradient_norm,
            prefix="sgld_final_gradient_norm",
        ),
    }
    for step, group_scores in sorted(snapshots.items()):
        if step == num_steps:
            continue
        earlier_mean = group_scores.mean(dim=0)
        diagnostics[
            f"sgld_score_drift_step_{step}_to_{num_steps}_l2"
        ] = float(
            (
                (earlier_mean - final_mean)
                .square()
                .sum(dim=-1)
                .mean()
                .item()
            )
        )
    return final_group_scores, diagnostics


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    os.replace(temp_path, path)


def _load_torch_payload(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def sgld_chunk_dir(run_root: Path, spec: CellSpec) -> Path:
    return (
        run_root
        / "sgld_chunks"
        / spec.record.target
        / spec.record.method.upper()
        / f"seed_{spec.record.seed}"
        / f"epoch_{spec.epoch}"
    )


def sgld_chunk_path(
    run_root: Path,
    spec: CellSpec,
    start: int,
    stop: int,
) -> Path:
    return sgld_chunk_dir(run_root, spec) / f"z_{start:04d}_{stop:04d}.pt"


def sgld_active_state_path(
    run_root: Path,
    spec: CellSpec,
    start: int,
    stop: int,
) -> Path:
    return (
        sgld_chunk_dir(run_root, spec)
        / f"z_{start:04d}_{stop:04d}.active.pt"
    )


def _validate_chunk_payload(
    payload: dict[str, Any],
    *,
    fingerprint: str,
    spec: CellSpec,
    start: int,
    stop: int,
    implementation_version: str,
) -> None:
    expected = {
        "analysis_fingerprint": fingerprint,
        "cell_key": spec.key,
        "z_start": start,
        "z_stop": stop,
        "implementation_version": implementation_version,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(
                f"SGLD chunk metadata mismatch for {key}: "
                f"{payload.get(key)!r} != {value!r}"
            )


def _validate_completed_chunk(
    payload: dict[str, Any],
    *,
    num_groups: int,
    z_count: int,
    z_dim: int,
    chains_per_group: int,
    num_steps: int,
    step_size: float,
    init_jitter_scale: float,
    accumulator_dtype: torch.dtype,
) -> None:
    group_scores = payload.get("group_scores")
    if not isinstance(group_scores, torch.Tensor):
        raise RuntimeError("Completed SGLD chunk has no score tensor.")
    expected_shape = (num_groups, z_count, z_dim)
    if tuple(group_scores.shape) != expected_shape:
        raise RuntimeError(
            "Completed SGLD chunk score shape mismatch: "
            f"{tuple(group_scores.shape)} != {expected_shape}."
        )
    if group_scores.dtype != accumulator_dtype:
        raise RuntimeError(
            "Completed SGLD chunk accumulator dtype mismatch: "
            f"{group_scores.dtype} != {accumulator_dtype}."
        )
    if not torch.isfinite(group_scores).all():
        raise RuntimeError("Completed SGLD chunk contains non-finite scores.")

    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise RuntimeError("Completed SGLD chunk has no diagnostics.")
    expected_integer = {
        "sgld_num_groups": num_groups,
        "sgld_chains_per_group": chains_per_group,
        "sgld_num_steps": num_steps,
    }
    for key, expected in expected_integer.items():
        if int(diagnostics.get(key, -1)) != expected:
            raise RuntimeError(
                f"Completed SGLD chunk diagnostic {key} is incompatible."
            )
    expected_float = {
        "sgld_step_size": step_size,
        "sgld_init_jitter_scale": init_jitter_scale,
    }
    for key, expected in expected_float.items():
        actual = float(diagnostics.get(key, math.nan))
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-15):
            raise RuntimeError(
                f"Completed SGLD chunk diagnostic {key} is incompatible."
            )


def _merge_chunk_diagnostics(
    values: list[tuple[int, dict[str, Any]]],
) -> dict[str, Any]:
    if not values:
        return {}
    total_weight = sum(weight for weight, _ in values)
    keys = sorted({key for _, item in values for key in item})
    result: dict[str, Any] = {}
    constants = {
        "sgld_num_groups",
        "sgld_chains_per_group",
        "sgld_total_terminal_particles_per_z",
        "sgld_num_steps",
        "sgld_step_size",
        "sgld_langevin_time",
        "sgld_init_jitter_scale",
    }
    for key in keys:
        present = [
            (weight, item[key])
            for weight, item in values
            if key in item
        ]
        if key in constants:
            unique = {value for _, value in present}
            if len(unique) != 1:
                raise RuntimeError(
                    f"Inconsistent SGLD chunk diagnostic {key}."
                )
            result[key] = present[0][1]
            continue
        numeric = [
            (weight, float(value))
            for weight, value in present
            if isinstance(value, (int, float))
        ]
        if not numeric:
            continue
        if key.endswith("_max"):
            result[key] = max(value for _, value in numeric)
        elif key.endswith("_p95"):
            result[f"{key}_max_across_z_chunks"] = max(
                value for _, value in numeric
            )
        elif key.endswith("_median"):
            result[f"{key}_mean_across_z_chunks"] = sum(
                weight * value for weight, value in numeric
            ) / sum(weight for weight, _ in numeric)
        else:
            result[key] = sum(
                weight * value for weight, value in numeric
            ) / sum(weight for weight, _ in numeric)
    result["sgld_z_chunks"] = len(values)
    result["sgld_completed_z"] = total_weight
    return result


def streamed_posterior_sgld_reference_scores(
    vi_model: torch.nn.Module,
    z: torch.Tensor,
    generating_epsilon: torch.Tensor,
    *,
    spec: CellSpec,
    run_root: Path,
    fingerprint: str,
    reference_seed: int,
    num_groups: int,
    chains_per_group: int,
    num_steps: int,
    step_size: float,
    init_jitter_scale: float,
    z_chunk_size: int,
    diagnostic_steps: Iterable[int],
    finite_check_interval: int,
    checkpoint_interval: int,
    accumulator_dtype: torch.dtype,
    implementation_version: str,
    resume: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run or resume independently seeded z tiles."""
    if z_chunk_size < 1:
        raise ValueError("z_chunk_size must be positive.")
    if implementation_version != SGLD_IMPLEMENTATION_VERSION:
        raise ValueError(
            "Unsupported SGLD implementation_version: "
            f"{implementation_version!r}."
        )
    group_chunks: list[torch.Tensor] = []
    chunk_diagnostics: list[tuple[int, dict[str, Any]]] = []
    chunk_runtimes: list[float] = []
    total_chunks = math.ceil(z.shape[0] / z_chunk_size)

    for chunk_index, start in enumerate(
        range(0, z.shape[0], z_chunk_size),
        start=1,
    ):
        stop = min(z.shape[0], start + z_chunk_size)
        completed_path = sgld_chunk_path(
            run_root,
            spec,
            start,
            stop,
        )
        active_path = sgld_active_state_path(
            run_root,
            spec,
            start,
            stop,
        )
        if not resume:
            if completed_path.is_file():
                completed_path.unlink()
            if active_path.is_file():
                active_path.unlink()
        if resume and completed_path.is_file():
            payload = _load_torch_payload(completed_path)
            _validate_chunk_payload(
                payload,
                fingerprint=fingerprint,
                spec=spec,
                start=start,
                stop=stop,
                implementation_version=implementation_version,
            )
            _validate_completed_chunk(
                payload,
                num_groups=num_groups,
                z_count=stop - start,
                z_dim=z.shape[-1],
                chains_per_group=chains_per_group,
                num_steps=num_steps,
                step_size=step_size,
                init_jitter_scale=init_jitter_scale,
                accumulator_dtype=accumulator_dtype,
            )
            group_scores = payload["group_scores"].to(
                device=z.device,
                dtype=accumulator_dtype,
            )
            diagnostics = dict(payload["diagnostics"])
            chunk_runtime = float(payload.get("runtime_sec", 0.0))
            status = "resumed-complete"
        else:
            chunk_seed = stable_seed(
                reference_seed,
                "sgld_z_chunk",
                start,
                stop,
            )
            seed_everything(
                chunk_seed,
                use_cuda=z.device.type == "cuda",
            )
            resume_state: dict[str, Any] | None = None
            previous_runtime = 0.0
            if resume and active_path.is_file():
                active_payload = _load_torch_payload(active_path)
                _validate_chunk_payload(
                    active_payload,
                    fingerprint=fingerprint,
                    spec=spec,
                    start=start,
                    stop=stop,
                    implementation_version=implementation_version,
                )
                resume_state = dict(active_payload["state"])
                previous_runtime = float(
                    active_payload.get("elapsed_sec", 0.0)
                )
            tile_started = time.perf_counter()

            def save_active(state: dict[str, Any]) -> None:
                _atomic_torch_save(
                    active_path,
                    {
                        "analysis_fingerprint": fingerprint,
                        "cell_key": spec.key,
                        "z_start": start,
                        "z_stop": stop,
                        "implementation_version": implementation_version,
                        "elapsed_sec": (
                            previous_runtime
                            + time.perf_counter()
                            - tile_started
                        ),
                        "state": state,
                    },
                )

            group_scores, diagnostics = posterior_sgld_group_scores(
                vi_model,
                z[start:stop],
                generating_epsilon[start:stop],
                num_groups=num_groups,
                chains_per_group=chains_per_group,
                num_steps=num_steps,
                step_size=step_size,
                init_jitter_scale=init_jitter_scale,
                diagnostic_steps=diagnostic_steps,
                finite_check_interval=finite_check_interval,
                accumulator_dtype=accumulator_dtype,
                resume_state=resume_state,
                checkpoint_interval=checkpoint_interval,
                checkpoint_callback=save_active,
            )
            chunk_runtime = (
                previous_runtime + time.perf_counter() - tile_started
            )
            completed_payload = {
                "analysis_fingerprint": fingerprint,
                "cell_key": spec.key,
                "z_start": start,
                "z_stop": stop,
                "implementation_version": implementation_version,
                "runtime_sec": chunk_runtime,
                "group_scores": group_scores.detach().cpu(),
                "diagnostics": diagnostics,
            }
            _validate_chunk_payload(
                completed_payload,
                fingerprint=fingerprint,
                spec=spec,
                start=start,
                stop=stop,
                implementation_version=implementation_version,
            )
            _validate_completed_chunk(
                completed_payload,
                num_groups=num_groups,
                z_count=stop - start,
                z_dim=z.shape[-1],
                chains_per_group=chains_per_group,
                num_steps=num_steps,
                step_size=step_size,
                init_jitter_scale=init_jitter_scale,
                accumulator_dtype=accumulator_dtype,
            )
            _atomic_torch_save(
                completed_path,
                completed_payload,
            )
            if active_path.is_file():
                active_path.unlink()
            status = "computed"

        group_chunks.append(group_scores)
        chunk_diagnostics.append((stop - start, diagnostics))
        chunk_runtimes.append(chunk_runtime)
        print(
            f"  SGLD z tile {chunk_index}/{total_chunks} "
            f"[{start}:{stop}] {status}",
            flush=True,
        )

    merged_diagnostics = _merge_chunk_diagnostics(chunk_diagnostics)
    merged_diagnostics["sgld_compute_runtime_sec_total"] = sum(
        chunk_runtimes
    )
    return torch.cat(group_chunks, dim=1), merged_diagnostics


def _assess_sgld_quality(
    diagnostics: dict[str, Any],
    metrics: dict[str, Any],
    quality_cfg: DictConfig,
) -> tuple[str, list[str]]:
    issues: list[str] = []
    nonfinite = float(
        diagnostics.get("sgld_terminal_nonfinite_fraction", math.inf)
    )
    maximum_nonfinite = float(
        quality_cfg.get("max_nonfinite_fraction", 0.0)
    )
    if not math.isfinite(nonfinite) or nonfinite > maximum_nonfinite:
        issues.append(
            "sgld_terminal_nonfinite_fraction="
            f"{nonfinite:.6g} > {maximum_nonfinite:.6g}"
        )

    drift_keys = [
        key
        for key in diagnostics
        if key.startswith("sgld_score_drift_step_")
        and key.endswith("_l2")
    ]
    if drift_keys:
        latest_key = max(
            drift_keys,
            key=lambda value: int(
                value.split("sgld_score_drift_step_", 1)[1].split(
                    "_to_",
                    1,
                )[0]
            ),
        )
        drift = float(diagnostics[latest_key])
        mcse = float(metrics["reference_mean_mcse_l2"])
        ratio = drift / max(mcse, np.finfo(np.float64).eps)
        diagnostics["sgld_latest_horizon_drift_l2"] = drift
        diagnostics["sgld_latest_horizon_drift_to_mcse"] = ratio
        maximum_ratio = float(
            quality_cfg.get("max_horizon_drift_to_mcse", math.inf)
        )
        if not math.isfinite(ratio) or ratio > maximum_ratio:
            issues.append(
                f"{latest_key}/gold_mcse={ratio:.6g} > "
                f"{maximum_ratio:.6g}"
            )
    return ("pass" if not issues else "warning"), issues


def evaluate_sgld_cell(
    runner: Any,
    spec: CellSpec,
    cfg: DictConfig,
    *,
    fingerprint: str,
    run_root: Path,
    resume_chunks: bool,
) -> dict[str, Any]:
    if spec.record.method.upper() != "DSIVI":
        raise ValueError("The SGLD analysis currently supports DSIVI only.")
    _load_checkpoint(runner, spec)
    device = torch.device(runner.device)
    use_cuda = device.type == "cuda"
    forward_count = int(cfg.evaluation.forward_batch_size)
    reference_cfg = cfg.evaluation.reference
    estimator = str(reference_cfg.estimator).lower()
    if estimator != "posterior_sgld_terminal":
        raise ValueError(
            "Expected reference.estimator=posterior_sgld_terminal."
        )
    implementation_version = str(reference_cfg.implementation_version)
    if implementation_version != SGLD_IMPLEMENTATION_VERSION:
        raise ValueError(
            "Configured SGLD implementation version does not match code: "
            f"{implementation_version!r} != "
            f"{SGLD_IMPLEMENTATION_VERSION!r}."
        )
    num_groups = int(reference_cfg.num_groups)
    chains_per_group = int(reference_cfg.chains_per_group)
    accumulator_dtype = _accumulator_dtype(
        str(reference_cfg.accumulator_dtype)
    )

    forward_seed = stable_seed(spec.key, "forward")
    seed_everything(forward_seed, use_cuda=use_cuda)
    generating_epsilon, z = runner.vi_model.sampling(num=forward_count)

    method_seed = stable_seed(spec.key, "method")
    seed_everything(method_seed, use_cuda=use_cuda)
    _sync(device)
    method_started = time.perf_counter()
    method_score, method_diagnostics = native_dsivi_score(runner, z)
    _sync(device)
    method_runtime = time.perf_counter() - method_started
    with torch.no_grad():
        target_score = runner.target_model.score(z).detach()
    if method_score.shape != z.shape:
        raise RuntimeError("DSIVI score shape does not match sampled z.")
    if target_score.shape != z.shape:
        raise RuntimeError("Target score shape does not match sampled z.")
    if not torch.isfinite(method_score).all():
        raise FloatingPointError("DSIVI score contains non-finite values.")
    if not torch.isfinite(target_score).all():
        raise FloatingPointError("Target score contains non-finite values.")

    reference_seed = stable_seed(spec.key, "reference_sgld")
    _sync(device)
    reference_started = time.perf_counter()
    reference_scores, reference_diagnostics = (
        streamed_posterior_sgld_reference_scores(
            runner.vi_model,
            z,
            generating_epsilon,
            spec=spec,
            run_root=run_root,
            fingerprint=fingerprint,
            reference_seed=reference_seed,
            num_groups=num_groups,
            chains_per_group=chains_per_group,
            num_steps=int(reference_cfg.num_steps),
            step_size=float(reference_cfg.step_size),
            init_jitter_scale=float(reference_cfg.init_jitter_scale),
            z_chunk_size=int(reference_cfg.z_chunk_size),
            diagnostic_steps=[
                int(value)
                for value in reference_cfg.get("diagnostic_steps", [])
            ],
            finite_check_interval=int(
                reference_cfg.finite_check_interval
            ),
            checkpoint_interval=int(
                reference_cfg.checkpoint_interval
            ),
            accumulator_dtype=accumulator_dtype,
            implementation_version=implementation_version,
            resume=resume_chunks,
        )
    )
    _sync(device)
    current_invocation_reference_runtime = (
        time.perf_counter() - reference_started
    )
    reference_runtime = float(
        reference_diagnostics["sgld_compute_runtime_sec_total"]
    )

    metrics = compute_score_metrics(
        method_score,
        reference_scores,
        target_score,
    )
    z_dim = int(z.shape[-1])
    metrics.update({
        "method_mse_per_coordinate": (
            float(metrics["method_l2"]) / z_dim
        ),
        "reference_internal_mse_per_coordinate": (
            float(metrics["reference_internal_l2"]) / z_dim
        ),
        "reference_mean_mcse_mse_per_coordinate": (
            float(metrics["reference_mean_mcse_l2"]) / z_dim
        ),
    })
    quality_status, quality_issues = _assess_sgld_quality(
        reference_diagnostics,
        metrics,
        reference_cfg.quality,
    )

    gpu_name = (
        torch.cuda.get_device_name(device) if use_cuda else ""
    )
    return {
        "analysis_fingerprint": fingerprint,
        "cell_key": spec.key,
        "run_id": spec.record.run_id,
        "method": spec.record.method.upper(),
        "target": spec.record.target,
        "seed": spec.record.seed,
        "progress": spec.progress,
        "epoch": spec.epoch,
        "checkpoint_dir": spec.checkpoint_dir.as_posix(),
        "forward_batch_size": forward_count,
        "reference_estimator": estimator,
        "reference_implementation_version": implementation_version,
        "reference_total_samples": num_groups * chains_per_group,
        "reference_num_groups": num_groups,
        "reference_group_size": chains_per_group,
        "reference_repeats": num_groups,
        "reference_replication_unit": "sgld_group_mean",
        "reference_quality_status": quality_status,
        "reference_quality_issues": quality_issues,
        "accumulator_dtype": str(accumulator_dtype),
        "forward_seed": forward_seed,
        "method_seed": method_seed,
        "reference_seed": reference_seed,
        "method_runtime_sec": method_runtime,
        "method_status": "ok",
        "method_error": "",
        "reference_runtime_sec": reference_runtime,
        "current_invocation_reference_runtime_sec": (
            current_invocation_reference_runtime
        ),
        "reference_groups_evolved_jointly": True,
        "total_runtime_sec": method_runtime + reference_runtime,
        "device": str(device),
        "gpu_name": gpu_name,
        "diagnostics": {
            **method_diagnostics,
            **reference_diagnostics,
        },
        **metrics,
        "completed_at": utc_now(),
    }


def _fixed(value: float | None, digits: int = 8) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def _fixed_mean_sd(row: dict[str, Any], name: str) -> str:
    mean = row.get(f"{name}_mean")
    sd = row.get(f"{name}_sd")
    if mean is None or sd is None:
        return "NA"
    return f"{_fixed(float(mean))} ± {_fixed(float(sd))}"


def _write_markdown_report(
    path: Path,
    summary_rows: list[dict[str, Any]],
    *,
    seeds: Iterable[int],
    records: Iterable[dict[str, Any]],
) -> None:
    materialized_records = list(records)
    first_record = materialized_records[0] if materialized_records else {}
    num_groups = int(first_record.get("reference_num_groups", 0))
    group_size = int(first_record.get("reference_group_size", 0))
    seed_text = ", ".join(str(int(seed)) for seed in seeds)
    lines = [
        "# DIVI (DSIVI checkpoint)–SGLD Score-Approximation Analysis",
        "",
        "All values are mean ± sample standard deviation across seeds "
        f"{seed_text}. Values use fixed-point notation.",
        "",
        f"The reference is the average of {num_groups:,} independent "
        "SGLD-group score means; each group averages "
        f"{group_size:,} terminal epsilon particles. "
        "Within-SGLD L2 is calculated across those group means.",
        "",
        "> A small within-SGLD L2 measures Monte Carlo agreement between "
        "groups; it does not by itself establish mixing or remove common "
        "finite-horizon and fixed-step bias.",
    ]
    warnings = [
        record
        for record in materialized_records
        if record.get("reference_quality_status") != "pass"
    ]
    if warnings:
        lines.extend(["", "## Diagnostic warnings", ""])
        for record in warnings:
            issues = "; ".join(record["reference_quality_issues"])
            lines.append(
                f"- {record['target']} / seed {record['seed']}: {issues}"
            )
    lines.extend([
        "",
        "| Target | Epoch | DIVI (DSIVI)–SGLD L2 | Within-SGLD L2 | "
        "Golden-score MCSE L2 | DIVI–SGLD per-coordinate MSE |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in summary_rows:
        lines.append(
            "| {target} | {epoch} | {method} | {internal} | {mcse} | "
            "{per_coordinate} |".format(
                target=row["target"],
                epoch=int(row["epoch"]),
                method=_fixed_mean_sd(row, "method_l2"),
                internal=_fixed_mean_sd(row, "reference_internal_l2"),
                mcse=_fixed_mean_sd(row, "reference_mean_mcse_l2"),
                per_coordinate=_fixed_mean_sd(
                    row,
                    "method_mse_per_coordinate",
                ),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex_report(
    path: Path,
    summary_rows: list[dict[str, Any]],
) -> None:
    lines = [
        r"\begin{tabular}{lrcccc}",
        r"\toprule",
        r"Target & Epoch & DIVI (DSIVI)--SGLD L2 & Within-SGLD L2 & "
        r"Golden MCSE L2 & Per-coordinate MSE \\",
        r"\midrule",
    ]
    for row in summary_rows:
        values = [
            str(row["target"]).replace("_", r"\_"),
            str(int(row["epoch"])),
            _fixed_mean_sd(row, "method_l2").replace("±", r"$\pm$"),
            _fixed_mean_sd(
                row,
                "reference_internal_l2",
            ).replace("±", r"$\pm$"),
            _fixed_mean_sd(
                row,
                "reference_mean_mcse_l2",
            ).replace("±", r"$\pm$"),
            _fixed_mean_sd(
                row,
                "method_mse_per_coordinate",
            ).replace("±", r"$\pm$"),
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def aggregate_sgld_results(
    cfg: DictConfig,
    specs: list[CellSpec],
    *,
    fingerprint: str,
    require_complete: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runtime_root = repo_path(str(cfg.output.runtime_dir))
    report_dir = repo_path(str(cfg.output.report_dir))
    assert runtime_root is not None and report_dir is not None
    run_root = runtime_root / fingerprint[:16]

    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for spec in specs:
        path = cell_record_path(run_root, spec)
        if not path.is_file():
            missing.append(spec.key)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("analysis_fingerprint") != fingerprint:
            raise RuntimeError(f"Cell fingerprint mismatch in {path}.")
        records.append(payload)
    if require_complete and missing:
        raise RuntimeError(
            f"Cannot aggregate incomplete SGLD analysis: "
            f"{len(missing)} cells are missing."
        )

    checkpoint_rows = [_flatten_cell(record) for record in records]
    repeat_rows: list[dict[str, Any]] = []
    for record in records:
        for index, value in enumerate(
            record["reference_repeat_internal_l2"]
        ):
            repeat_rows.append({
                "run_id": record["run_id"],
                "target": record["target"],
                "method": record["method"],
                "seed": record["seed"],
                "epoch": record["epoch"],
                "group": index,
                "replication_unit": "sgld_group_mean",
                "reference_internal_l2": value,
            })
    summary_rows = _summary_rows(records)
    for row in summary_rows:
        matching = [
            record
            for record in records
            if record["target"] == row["target"]
            and record["method"] == row["method"]
            and int(record["epoch"]) == int(row["epoch"])
        ]
        for metric in (
            "method_mse_per_coordinate",
            "reference_internal_mse_per_coordinate",
            "reference_mean_mcse_mse_per_coordinate",
        ):
            values = np.asarray(
                [float(record[metric]) for record in matching],
                dtype=np.float64,
            )
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )

    if require_complete:
        expected_rows = (
            len(cfg.selection.targets)
            * len(cfg.selection.methods)
            * len(cfg.selection.checkpoint_progress)
        )
        if len(records) != len(specs):
            raise RuntimeError("Unexpected SGLD cell count.")
        if len(summary_rows) != expected_rows:
            raise RuntimeError("Unexpected SGLD summary-row count.")
        expected_seeds = len(cfg.selection.seeds)
        if any(
            int(row["n_seeds"]) != expected_seeds
            for row in summary_rows
        ):
            raise RuntimeError("An SGLD summary row is missing seeds.")

    report_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(report_dir / "checkpoint_metrics.csv", checkpoint_rows)
    _write_csv(report_dir / "sgld_group_metrics.csv", repeat_rows)
    _write_csv(report_dir / "seed_summary.csv", summary_rows)
    _write_markdown_report(
        report_dir / "score_sgld_table.md",
        summary_rows,
        seeds=cfg.selection.seeds,
        records=records,
    )
    _write_latex_report(
        report_dir / "score_sgld_table.tex",
        summary_rows,
    )
    atomic_write_json(
        report_dir / "run_metadata.json",
        {
            "analysis_fingerprint": fingerprint,
            "git_commit": _git_commit(),
            "generated_at": utc_now(),
            "expected_cells": len(specs),
            "completed_cells": len(records),
            "summary_rows": len(summary_rows),
            "reference_quality_warnings": sum(
                record.get("reference_quality_status") != "pass"
                for record in records
            ),
            "estimator_note": (
                "Fixed-step terminal-particle ULA/SGLD reference; "
                "between-group agreement of group means does not certify "
                "convergence."
            ),
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
    )
    return records, summary_rows


def run_sgld_analysis(
    cfg: DictConfig,
    *,
    limit: int | None = None,
    resume: bool = True,
    aggregate_only: bool = False,
    shard_count: int = 1,
    shard_index: int = 0,
    aggregate_after_run: bool = True,
    cell_keys: set[str] | None = None,
) -> tuple[int, int]:
    fingerprint = config_fingerprint(cfg)
    all_specs = build_cell_specs(cfg)
    runtime_root = repo_path(str(cfg.output.runtime_dir))
    assert runtime_root is not None
    run_root = runtime_root / fingerprint[:16]

    if aggregate_only:
        records, summary = aggregate_sgld_results(
            cfg,
            all_specs,
            fingerprint=fingerprint,
            require_complete=True,
        )
        return len(records), len(summary)

    specs = shard_cell_specs(
        all_specs,
        shard_count=shard_count,
        shard_index=shard_index,
    )
    specs = select_cell_specs(specs, cell_keys)
    if str(cfg.evaluation.device) == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the production configuration.")

    pending = pending_cell_specs(
        specs,
        run_root=run_root,
        fingerprint=fingerprint,
        resume=resume,
    )
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive.")
        pending = pending[:limit]

    runner: Any | None = None
    active_run_id: str | None = None
    completed_now = 0
    try:
        for spec in pending:
            if spec.record.run_id != active_run_id:
                _release_runner(runner)
                runner = _build_runner(spec.record, cfg)
                active_run_id = spec.record.run_id
            assert runner is not None
            started = time.perf_counter()
            record = evaluate_sgld_cell(
                runner,
                spec,
                cfg,
                fingerprint=fingerprint,
                run_root=run_root,
                resume_chunks=resume,
            )
            atomic_write_json(
                cell_record_path(run_root, spec),
                record,
            )
            completed_now += 1
            print(
                f"[{completed_now}/{len(pending)}] DSIVI "
                f"{spec.record.target} seed={spec.record.seed} "
                f"epoch={spec.epoch}: "
                f"method_l2={record['method_l2']:.6f}, "
                f"within_sgld_l2="
                f"{record['reference_internal_l2']:.6f}, "
                f"runtime={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    finally:
        _release_runner(runner)

    if limit is None and aggregate_after_run:
        records, summary = aggregate_sgld_results(
            cfg,
            all_specs,
            fingerprint=fingerprint,
            require_complete=True,
        )
        return len(records), len(summary)
    return completed_now, 0
