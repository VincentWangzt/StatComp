#!/usr/bin/env python
"""
KDVI 8-Gaussian kernel-fix sweep driver (sequential, GPU-friendly).

Tests three additive changes on top of the previous sweep's winning recipe:
  - new ``GaussianKernelMMD`` kernel (textbook MMD median heuristic)
  - new ``LaplaceL2Kernel`` (Laplace-on-L2 distance, heavier tails)
  - wider epsilon dim (8, 16, 32)

Anchor recipe (from ``campaigns/kdvi_8gauss_adaptive_sweep/summary.md``)::

    train.kdvi.mcmc_type=sgld
    train.kdvi.mcmc_steps=10
    train.kdvi.mcmc_step_size=0.02
    train.kdvi.step_size_schedule.type=coupled

Phases:

    Phase A — Kernel + heuristic isolation (6 single-seed runs)
        A0: gaussian + fb=x       (reproduce sweep baseline)
        A1: gaussian_mmd + fb=x   (heuristic fix in isolation)
        A2: gaussian_mmd + fb=xy  (joint-set bandwidth fit)
        A3: laplace_l2 + fb=x     (heavier-tail kernel)
        A4: laplace_l2 + fb=xy    (heavier tails + joint fit)
        A5: gaussian + h=0.5      (sanity vs prior winner)

    Phase B — eps-dim widening on Phase-A winner (4 single-seed runs)
        B0: ConditionalGaussian (default, eps=z_dim=2)
        B1: ConditionalGaussian-Eps8     (eps=8, base arch)
        B2: ConditionalGaussian-Eps16    (eps=16, base arch)
        B3: ConditionalGaussian-Notebook (eps=32, deeper+wider net)

    Phase C — Finalists × 3 seeds (6 runs)
        C-top1: A-winner + B-winner-eps × {seed=42, 0, 1}
        C-top2: A-winner + default eps × {seed=42, 0, 1}

For each config, the driver:
  1. Subprocess-launches ``python src.py --config configs/kdvi_8_gaussians.yaml <overrides>``
  2. Polls the TB events for ``metric/vi_model/kl_ite`` every poll_interval_s.
  3. Applies an early-stop rule (no improvement in best KL_ITE over the last
     early_stop_window epochs).
  4. Records final/best metrics + wallclock to ``log.jsonl``.
  5. Generates a ``summary.md`` at the end.

Outputs land at ``campaigns/kdvi_8gauss_kernel_fix_sweep/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEP_ROOT = Path(__file__).resolve().parent / "kdvi_8gauss_kernel_fix_sweep"
RUNS_ROOT = SWEEP_ROOT / "runs"
LOG_PATH = SWEEP_ROOT / "log.jsonl"
SUMMARY_PATH = SWEEP_ROOT / "summary.md"

# On the local box use .venv; on remote, system python (or conda env python)
# is on $PATH so we just use whatever invokes us.
PYTHON = sys.executable
MAIN_CONFIG = "configs/kdvi_8_gaussians.yaml"

# Anchor recipe from prior sweep's winning config.
ANCHOR_OVERRIDES = [
    "train.kdvi.mcmc_type=sgld",
    "train.kdvi.mcmc_steps=10",
    "train.kdvi.mcmc_step_size=0.02",
    "train.kdvi.step_size_schedule.type=coupled",
]

# Common overrides for every run.
BASE_OVERRIDES = [
    "use_cuda=true",
    "tracking.campaign=kdvi_8gauss_kernel_fix_sweep",
    "metric.elbo.num_batches=1",
    "metric.elbo.num_z_samples=500",
    "train.epochs=100000",
    "train.log.metric_log_freq=500",
    # Plot/checkpoint less often to save disk and time.
    "train.checkpoint.freq=20000",
    "train.plot.freq=20000",
    "train.sample.freq=20000",
]

KL_ITE_TAG = "metric/vi_model/kl_ite"
W2_TAG = "metric/vi_model/w2"
MMD_TAG = "metric/vi_model/mmd"
ELBO_TAG = "metric/vi_model/elbo"


@dataclass
class RunResult:
    run_id: str
    phase: str
    overrides: list[str]
    config_dict: dict[str, Any]
    final_kl_ite: float | None
    best_kl_ite: float | None
    final_w2: float | None
    final_mmd: float | None
    final_elbo: float | None
    epochs_completed: int
    wallclock_s: float
    status: str  # 'completed' | 'early_stopped' | 'failed' | 'skipped'
    seed: int
    notes: str = ""
    kl_ite_curve: list[tuple[int, float]] = field(default_factory=list)


# ----------------------------------------------------------------------
# TB scraping
# ----------------------------------------------------------------------

def _find_tb_event_file(timestamp_dir: Path) -> Path | None:
    if not timestamp_dir.exists():
        return None
    candidates = sorted(timestamp_dir.glob("events.out.tfevents.*"))
    return candidates[-1] if candidates else None


def _read_scalar_curve(event_file: Path, tag: str) -> list[tuple[int, float]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return []


def _read_metrics_curve(metrics_file: Path, tag: str) -> list[tuple[int, float]]:
    if not metrics_file.is_file():
        return []
    try:
        with metrics_file.open("r", encoding="utf-8", newline="") as fh:
            return [
                (int(row["step"]), float(row["value"]))
                for row in csv.DictReader(fh)
                if row.get("tag") == tag
            ]
    except (OSError, KeyError, TypeError, ValueError):
        return []
    try:
        ea = EventAccumulator(str(event_file.parent),
                              size_guidance={"scalars": 100000})
        ea.Reload()
        if tag not in ea.Tags()["scalars"]:
            return []
        return [(ev.step, float(ev.value)) for ev in ea.Scalars(tag)]
    except Exception:
        return []


def _find_results_timestamp(stdout_text: str) -> str | None:
    marker = "Artifacts will be saved to: results/KDVI/8_gaussians/"
    for line in stdout_text.splitlines():
        if marker in line:
            return line.split(marker)[1].strip()
    return None


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def _make_run_id(phase: str, idx: int, label: str) -> str:
    safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in label)
    return f"{phase}_{idx:02d}_{safe}"[:80]


def _build_cmd(overrides: list[str]) -> list[str]:
    return [PYTHON, "src.py", "--config", MAIN_CONFIG] + BASE_OVERRIDES + list(overrides)


def _parse_overrides_to_dict(overrides: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ov in overrides:
        if "=" not in ov:
            continue
        k, v = ov.split("=", 1)
        try:
            if "." in v or "e" in v.lower():
                out[k] = float(v)
            else:
                out[k] = int(v)
        except ValueError:
            out[k] = v
    return out


def run_once(
    *,
    phase: str,
    idx: int,
    label: str,
    overrides: list[str],
    seed: int,
    poll_interval_s: float = 30.0,
    early_stop_window: int = 20000,
) -> RunResult:
    run_id = _make_run_id(phase, idx, label)
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.log"

    full_overrides = list(overrides) + [f"seed={seed}"]
    cmd = _build_cmd(full_overrides)

    print(f"\n{'=' * 70}")
    print(f"[{phase}-{idx:02d} {label}]")
    print(f"  overrides: {' '.join(overrides)} seed={seed}")
    print(f"  run_dir:   {run_dir}")
    print(f"{'=' * 70}", flush=True)

    t0 = time.perf_counter()
    with stdout_path.open("w") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=f,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

    timestamp: str | None = None
    tb_event_file: Path | None = None
    metrics_file: Path | None = None
    for _ in range(60):
        if proc.poll() is not None:
            break
        try:
            txt = stdout_path.read_text()
        except FileNotFoundError:
            txt = ""
        timestamp = _find_results_timestamp(txt)
        if timestamp:
            metrics_file = (REPO_ROOT / "results" / "KDVI" /
                            "8_gaussians" / timestamp / "metrics.csv")
            if metrics_file.exists():
                break
            tb_dir = REPO_ROOT / "tb_logs" / "KDVI" / "8_gaussians" / timestamp
            tb_event_file = _find_tb_event_file(tb_dir)
            if tb_event_file is not None:
                break
        time.sleep(2)

    kl_curve: list[tuple[int, float]] = []
    last_known_step = 0
    best_kl = float("inf")
    best_kl_step = 0
    status = "completed"

    while proc.poll() is None:
        time.sleep(poll_interval_s)
        if metrics_file is not None and metrics_file.exists():
            kl_curve = _read_metrics_curve(metrics_file, KL_ITE_TAG)
        elif tb_event_file is not None and tb_event_file.exists():
            kl_curve = _read_scalar_curve(tb_event_file, KL_ITE_TAG)
        else:
            continue
        if not kl_curve:
            continue
        last_known_step = kl_curve[-1][0]
        cur_min = min(v for _, v in kl_curve)
        if cur_min < best_kl:
            best_kl = cur_min
            best_kl_step = next(s for s, v in kl_curve if v == cur_min)
        elapsed = time.perf_counter() - t0
        print(
            f"  [poll t={elapsed:6.0f}s step={last_known_step:6d}] "
            f"kl_ite cur={kl_curve[-1][1]:.4f} best={best_kl:.4f} "
            f"@step={best_kl_step}",
            flush=True,
        )
        if (
            best_kl < float("inf")
            and last_known_step - best_kl_step >= early_stop_window
            and last_known_step >= early_stop_window
        ):
            print(
                f"  ↳ early-stop: no KL_ITE improvement in last "
                f"{early_stop_window} epochs",
                flush=True,
            )
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            status = "early_stopped"
            break

    proc.wait()
    wallclock = time.perf_counter() - t0

    final_kl: float | None = None
    final_w2: float | None = None
    final_mmd: float | None = None
    final_elbo: float | None = None
    epochs_done = last_known_step

    if tb_event_file is not None and tb_event_file.exists():
        kl_curve = _read_scalar_curve(tb_event_file, KL_ITE_TAG)
        if kl_curve:
            epochs_done = kl_curve[-1][0]
            final_kl = kl_curve[-1][1]
            best_kl = min(v for _, v in kl_curve)
        w2_curve = _read_scalar_curve(tb_event_file, W2_TAG)
        if w2_curve:
            final_w2 = w2_curve[-1][1]
        mmd_curve = _read_scalar_curve(tb_event_file, MMD_TAG)
        if mmd_curve:
            final_mmd = mmd_curve[-1][1]
        elbo_curve = _read_scalar_curve(tb_event_file, ELBO_TAG)
        if elbo_curve:
            final_elbo = elbo_curve[-1][1]

    if proc.returncode not in (0, -signal.SIGTERM, signal.SIGTERM, None) \
            and status == "completed":
        status = "failed"

    result = RunResult(
        run_id=run_id,
        phase=phase,
        overrides=list(overrides),
        config_dict=_parse_overrides_to_dict(overrides),
        final_kl_ite=final_kl,
        best_kl_ite=(best_kl if best_kl < float("inf") else None),
        final_w2=final_w2,
        final_mmd=final_mmd,
        final_elbo=final_elbo,
        epochs_completed=epochs_done,
        wallclock_s=wallclock,
        status=status,
        seed=seed,
        kl_ite_curve=kl_curve[-200:] if kl_curve else [],
    )

    print(
        f"  ↳ done: status={status}  best_kl_ite={result.best_kl_ite}  "
        f"final_w2={result.final_w2}  wallclock={wallclock:.0f}s",
        flush=True,
    )

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(asdict(result)) + "\n")

    return result


def _load_log() -> list[RunResult]:
    if not LOG_PATH.exists():
        return []
    out: list[RunResult] = []
    for line in LOG_PATH.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out.append(RunResult(**d))
    return out


def best_run(results: list[RunResult]) -> RunResult | None:
    valid = [r for r in results if r.best_kl_ite is not None]
    return min(valid, key=lambda r: r.best_kl_ite) if valid else None


# ----------------------------------------------------------------------
# Phase configs
# ----------------------------------------------------------------------

def phase_a_configs() -> list[tuple[str, list[str]]]:
    """Phase A — Kernel + heuristic isolation, on the anchor recipe."""
    out: list[tuple[str, list[str]]] = []

    # A0: existing gaussian + default fb=x  -- reproduces sweep baseline
    out.append((
        "gaussian_fb_x",
        ANCHOR_OVERRIDES + [
            "train.kdvi.kernel=gaussian",
            "train.kdvi.fit_bandwidth_on=x",
        ],
    ))
    # A1: NEW gaussian_mmd + fb=x
    out.append((
        "gaussian_mmd_fb_x",
        ANCHOR_OVERRIDES + [
            "train.kdvi.kernel=gaussian_mmd",
            "train.kdvi.fit_bandwidth_on=x",
        ],
    ))
    # A2: NEW gaussian_mmd + fb=xy (matches notebook's joint-set median)
    out.append((
        "gaussian_mmd_fb_xy",
        ANCHOR_OVERRIDES + [
            "train.kdvi.kernel=gaussian_mmd",
            "train.kdvi.fit_bandwidth_on=xy",
        ],
    ))
    # A3: NEW laplace_l2 + fb=x
    out.append((
        "laplace_l2_fb_x",
        ANCHOR_OVERRIDES + [
            "train.kdvi.kernel=laplace_l2",
            "train.kdvi.fit_bandwidth_on=x",
        ],
    ))
    # A4: NEW laplace_l2 + fb=xy
    out.append((
        "laplace_l2_fb_xy",
        ANCHOR_OVERRIDES + [
            "train.kdvi.kernel=laplace_l2",
            "train.kdvi.fit_bandwidth_on=xy",
        ],
    ))
    # A5: existing gaussian + fixed h=0.5 -- sanity check vs prior winner
    out.append((
        "gaussian_hfix_0p5",
        ANCHOR_OVERRIDES + [
            "train.kdvi.kernel=gaussian",
            "train.kdvi.kernel_bandwidth=0.5",
        ],
    ))
    return out


def phase_b_configs(a_winner_overrides: list[str]
                    ) -> list[tuple[str, list[str]]]:
    """Phase B — eps-dim sweep on top of the Phase-A winner."""
    return [
        ("eps_default",
         a_winner_overrides),
        ("eps8_base_arch",
         a_winner_overrides + [
             "vi_model_config_path=configs/vi_models/ConditionalGaussian-Eps8.yaml",
         ]),
        ("eps16_base_arch",
         a_winner_overrides + [
             "vi_model_config_path=configs/vi_models/ConditionalGaussian-Eps16.yaml",
         ]),
        ("eps32_notebook_arch",
         a_winner_overrides + [
             "vi_model_config_path=configs/vi_models/ConditionalGaussian-Notebook.yaml",
         ]),
    ]


def phase_c_configs(
    a_winner_overrides: list[str],
    b_winner_overrides: list[str],
) -> list[tuple[str, list[str], int]]:
    """Phase C — finalists × 3 seeds.

    Two recipes get re-run with seeds {42, 0, 1}:
      C-top1: full A-winner + B-winner combination
      C-top2: A-winner + default eps (control: how much does eps add?)
    """
    out: list[tuple[str, list[str], int]] = []
    for s in (42, 0, 1):
        out.append((f"top1_seed{s}", b_winner_overrides, s))
    for s in (42, 0, 1):
        out.append((f"top2_seed{s}", a_winner_overrides, s))
    return out


# ----------------------------------------------------------------------
# Driver loop
# ----------------------------------------------------------------------

def confirm(prompt: str, *, auto: bool) -> bool:
    if auto:
        return True
    sys.stdout.write(prompt)
    sys.stdout.flush()
    line = sys.stdin.readline().strip().lower()
    if line in ("", "y", "yes"):
        return True
    if line in ("q", "quit", "exit"):
        sys.exit(0)
    return False


def write_summary(results: list[RunResult]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# KDVI 8-Gaussian Kernel-Fix Sweep — Summary\n")

    # Phase C: group by overrides
    c_runs = [r for r in results if r.phase == "C"
              and r.best_kl_ite is not None]
    if c_runs:
        groups: dict[str, list[RunResult]] = {}
        for r in c_runs:
            key = " ".join(sorted(r.overrides))
            groups.setdefault(key, []).append(r)

        def _mean(vals: list[float]) -> float:
            return sum(vals) / len(vals) if vals else float("nan")

        def _std(vals: list[float]) -> float:
            if len(vals) < 2:
                return 0.0
            m = _mean(vals)
            return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5

        rows = sorted(
            groups.items(),
            key=lambda kv: _mean(
                [r.best_kl_ite for r in kv[1] if r.best_kl_ite is not None]),
        )

        lines.append("## Phase C — finalists × seeds\n")
        lines.append("| rank | overrides | seeds | KL_ITE mean±std | "
                     "W2 mean | MMD mean | wall mean (s) |")
        lines.append("|---|---|---|---|---|---|---|")
        for i, (key, rs) in enumerate(rows[:5], 1):
            kls = [r.best_kl_ite for r in rs if r.best_kl_ite is not None]
            w2s = [r.final_w2 for r in rs if r.final_w2 is not None]
            mmds = [r.final_mmd for r in rs if r.final_mmd is not None]
            walls = [r.wallclock_s for r in rs]
            lines.append(
                f"| {i} | `{key}` | {len(rs)} | "
                f"{_mean(kls):.4f} ± {_std(kls):.4f} | "
                f"{_mean(w2s):.4f} | {_mean(mmds):.4f} | "
                f"{_mean(walls):.0f} |"
            )

    valid = [r for r in results if r.best_kl_ite is not None]
    valid.sort(key=lambda r: r.best_kl_ite or float("inf"))
    lines.append("\n## All runs — top 20 by best KL_ITE\n")
    lines.append("| rank | run_id | phase | KL_ITE | W2 | MMD | ELBO | "
                 "epochs | wall(s) | overrides |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(valid[:20], 1):
        lines.append(
            f"| {i} | `{r.run_id}` | {r.phase} | "
            f"{r.best_kl_ite:.4f} | "
            f"{(r.final_w2 if r.final_w2 is not None else float('nan')):.4f} | "
            f"{(r.final_mmd if r.final_mmd is not None else float('nan')):.4f} | "
            f"{(r.final_elbo if r.final_elbo is not None else float('nan')):.4f} | "
            f"{r.epochs_completed} | {r.wallclock_s:.0f} | "
            f"`{' '.join(r.overrides) if r.overrides else '(baseline)'}` |"
        )

    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    print(f"\nSummary written to {SUMMARY_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true",
                        help="Run all phases without confirmation prompts.")
    parser.add_argument("--phase", choices=("A", "B", "C", "all"),
                        default="all",
                        help="Only run a specific phase.")
    parser.add_argument("--early-stop-window", type=int, default=20000,
                        help="Epochs of no improvement before kill.")
    parser.add_argument("--poll-interval-s", type=float, default=30.0,
                        help="Seconds between TB polls.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override train.epochs for all runs (debug).")
    args = parser.parse_args()

    if args.epochs is not None:
        # Replace the train.epochs override in BASE_OVERRIDES.
        for i, ov in enumerate(BASE_OVERRIDES):
            if ov.startswith("train.epochs="):
                BASE_OVERRIDES[i] = f"train.epochs={args.epochs}"
                break

    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = _load_log()
    print(f"Loaded {len(results)} prior runs from {LOG_PATH}")

    def go(phase: str, idx: int, label: str, overrides: list[str],
           seed: int = 42) -> RunResult:
        prompt = (f"\nProceed with {phase}-{idx:02d} {label}? "
                  f"[Enter=yes, s=skip, q=quit] ")
        if not confirm(prompt, auto=args.auto):
            print("  -> skipped")
            return RunResult(
                run_id=_make_run_id(phase, idx, label),
                phase=phase, overrides=overrides,
                config_dict=_parse_overrides_to_dict(overrides),
                final_kl_ite=None, best_kl_ite=None,
                final_w2=None, final_mmd=None, final_elbo=None,
                epochs_completed=0, wallclock_s=0.0, status="skipped",
                seed=seed,
            )
        return run_once(
            phase=phase, idx=idx, label=label,
            overrides=overrides, seed=seed,
            poll_interval_s=args.poll_interval_s,
            early_stop_window=args.early_stop_window,
        )

    a_winner: RunResult | None = None
    b_winner: RunResult | None = None

    if args.phase in ("A", "all"):
        for idx, (label, ov) in enumerate(phase_a_configs()):
            r = go("A", idx, label, ov)
            results.append(r)

    a_runs = [r for r in results if r.phase == "A" and r.best_kl_ite is not None]
    a_winner = best_run(a_runs)

    if args.phase in ("B", "all"):
        if a_winner is None:
            print("Phase B needs a Phase-A winner; skipping.")
        else:
            print(f"\n** Phase A winner: {a_winner.run_id}  "
                  f"best_kl={a_winner.best_kl_ite:.4f}")
            for idx, (label, ov) in enumerate(
                    phase_b_configs(a_winner.overrides)):
                r = go("B", idx, label, ov)
                results.append(r)

    b_runs = [r for r in results if r.phase == "B" and r.best_kl_ite is not None]
    b_winner = best_run(b_runs)

    if args.phase in ("C", "all"):
        if a_winner is None or b_winner is None:
            print("Phase C needs A and B winners; skipping.")
        else:
            print(f"\n** Phase B winner: {b_winner.run_id}  "
                  f"best_kl={b_winner.best_kl_ite:.4f}")
            for idx, (label, ov, seed) in enumerate(
                    phase_c_configs(a_winner.overrides, b_winner.overrides)):
                r = go("C", idx, label, ov, seed=seed)
                results.append(r)

    write_summary(results)


if __name__ == "__main__":
    main()
