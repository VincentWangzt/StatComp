#!/usr/bin/env python
"""
Adaptive KDVI 8-Gaussian sweep driver (CPU-sequential).

Runs the four phases described in
``/root/.claude-internal/plans/well-the-thing-is-encapsulated-river.md``:

    Phase A: anchor + K + mcmc_step_size + mcmc_type
    Phase B: bandwidth + VI structure
    Phase C: step-size / beta schedules
    Phase D: top-3 candidates × 3 seeds

For each config, this driver:
    1. Subprocess-launches `python src.py --config <main> <overrides...>`.
    2. While the run is in flight, polls the live metrics CSV (falling back to
       legacy TensorBoard events) every
       ``poll_interval_s`` seconds for ``metric/vi_model/kl_ite``.
    3. Applies an early-stop rule: if KL_ITE has not improved its running
       min by more than ``early_stop_rel_tol`` over the last
       ``early_stop_window`` iterations, the process is killed.
    4. After completion (or kill), records the final-iteration metrics
       (KL_ITE, W2, MMD, ELBO) plus wallclock to ``log.jsonl``.
    5. Picks the next config via ``next_phase_config(...)`` and either
       prompts the user (default) or proceeds (``--auto``).

The sweep always uses these CPU-friendly base overrides on top of the
main config ``configs/kdvi_8_gaussians.yaml``::

    use_cuda=false
    metric.elbo.num_batches=1
    metric.elbo.num_z_samples=500
    train.epochs=100000
    train.log.metric_log_freq=500

Outputs land at::

    campaigns/kdvi_8gauss_adaptive_sweep/
    ├── log.jsonl                 ← one JSON row per completed run
    ├── runs/<run_id>/stdout.log  ← captured subprocess stdout
    └── summary.md                ← top-3 multi-seed summary (Phase D)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEP_ROOT = Path(__file__).resolve().parent / "kdvi_8gauss_adaptive_sweep"
RUNS_ROOT = SWEEP_ROOT / "runs"
LOG_PATH = SWEEP_ROOT / "log.jsonl"
SUMMARY_PATH = SWEEP_ROOT / "summary.md"

PYTHON = str(REPO_ROOT / ".venv" / "bin" / "python")
MAIN_CONFIG = "configs/kdvi_8_gaussians.yaml"

BASE_OVERRIDES = [
    "use_cuda=false",
    "tracking.campaign=kdvi_8gauss_adaptive_sweep",
    "metric.elbo.num_batches=1",
    "metric.elbo.num_z_samples=500",
    "train.epochs=100000",
    "train.log.metric_log_freq=500",
    # Plot/checkpoint less often to save disk and CPU time
    "train.checkpoint.freq=20000",
    "train.plot.freq=20000",
    "train.sample.freq=20000",
]

# ---- Metric tags as written by base_runner.py ----
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
# Live metric polling with legacy TensorBoard fallback
# ----------------------------------------------------------------------

def _find_tb_event_file(timestamp_dir: Path) -> Path | None:
    """Locate the (single) TB event file for a run timestamp directory."""
    if not timestamp_dir.exists():
        return None
    candidates = sorted(timestamp_dir.glob("events.out.tfevents.*"))
    return candidates[-1] if candidates else None


def _read_scalar_curve(event_file: Path, tag: str) -> list[tuple[int, float]]:
    """Read all (step, value) entries for a given tag from a TB event file."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return []


def _read_metrics_curve(metrics_file: Path, tag: str) -> list[tuple[int, float]]:
    """Read a live W&B-era metrics.csv, tolerating a partially written tail."""
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
        # Try to coerce to numeric
        try:
            if "." in v or "e" in v.lower():
                out[k] = float(v)
            else:
                out[k] = int(v)
        except ValueError:
            out[k] = v
    return out


def _find_results_timestamp(stdout_text: str) -> str | None:
    """Pull the auto-generated timestamp dir from the subprocess stdout."""
    marker = "Artifacts will be saved to: results/KDVI/8_gaussians/"
    for line in stdout_text.splitlines():
        if marker in line:
            return line.split(marker)[1].strip()
    return None


def run_once(
    *,
    phase: str,
    idx: int,
    label: str,
    overrides: list[str],
    seed: int,
    poll_interval_s: float = 30.0,
    early_stop_window: int = 20000,
    early_stop_rel_tol: float = 0.05,
    max_epochs: int = 100000,
) -> RunResult:
    """Launch one run; poll its TB log; early-stop if stalled."""
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
            preexec_fn=os.setsid,  # so we can kill the process group
        )

    # Wait for the run to start writing to results/, find timestamp.
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
        # Early-stop rule: best kl hasn't improved in the last
        # `early_stop_window` epochs and we are at least that far in.
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

    # Final scrape
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


def best_in_phase(results: list[RunResult], phase: str) -> RunResult | None:
    return best_run([r for r in results if r.phase == phase])


# ----------------------------------------------------------------------
# Phase configs
# ----------------------------------------------------------------------

def _ovs(**kwargs: Any) -> list[str]:
    return [f"{k.replace('__', '.')}={v}" for k, v in kwargs.items()]


def phase_a_configs() -> list[tuple[str, list[str]]]:
    """Phase A: anchor + K + mcmc_step_size + mcmc_type axes."""
    return [
        ("baseline", []),
        ("K1",  _ovs(train__kdvi__mcmc_steps=1)),
        ("K10", _ovs(train__kdvi__mcmc_steps=10)),
        ("K20", _ovs(train__kdvi__mcmc_steps=20)),
    ]


def phase_a2_configs(best_K: int) -> list[tuple[str, list[str]]]:
    """Phase A2: vary mcmc_step_size on top of best K from A."""
    return [
        (f"step0p01_K{best_K}",
         _ovs(train__kdvi__mcmc_steps=best_K,
              train__kdvi__mcmc_step_size=0.01)),
        (f"step0p02_K{best_K}",
         _ovs(train__kdvi__mcmc_steps=best_K,
              train__kdvi__mcmc_step_size=0.02)),
        (f"step0p1_K{best_K}",
         _ovs(train__kdvi__mcmc_steps=best_K,
              train__kdvi__mcmc_step_size=0.1)),
    ]


def phase_a3_configs(best_K: int, best_step: float) -> list[tuple[str, list[str]]]:
    """Phase A3: vary mcmc_type on top of (best K, best step_size)."""
    return [
        (f"mala_K{best_K}_s{best_step}",
         _ovs(train__kdvi__mcmc_steps=best_K,
              train__kdvi__mcmc_step_size=best_step,
              train__kdvi__mcmc_type="mala")),
        (f"hmc_K{best_K}_s{best_step}",
         _ovs(train__kdvi__mcmc_steps=best_K,
              train__kdvi__mcmc_step_size=best_step,
              train__kdvi__mcmc_type="hmc")),
    ]


def phase_b_configs(best_overrides: list[str]) -> list[tuple[str, list[str]]]:
    """Phase B: bandwidth & VI structure."""
    out: list[tuple[str, list[str]]] = []
    for fb in ("y", "xy"):
        out.append((f"fb_{fb}",
                    best_overrides + _ovs(train__kdvi__fit_bandwidth_on=fb)))
    for h in (0.5, 1.0):
        label = f"hfix_{str(h).replace('.', 'p')}"
        out.append((label,
                    best_overrides + _ovs(train__kdvi__kernel_bandwidth=h)))
    out.append((
        "vi_wide",
        best_overrides + [
            "vi_model_config_path=configs/vi_models/ConditionalGaussian-Wide.yaml",
        ],
    ))
    out.append((
        "vi_notebook",
        best_overrides + [
            "vi_model_config_path=configs/vi_models/ConditionalGaussian-Notebook.yaml",
        ],
    ))
    return out


def phase_c_configs(best_overrides: list[str], best_step: float
                    ) -> list[tuple[str, list[str]]]:
    """Phase C: schedules."""
    return [
        ("cosine_step",
         best_overrides + _ovs(
             train__kdvi__step_size_schedule__type="cosine",
             train__kdvi__step_size_schedule__start=max(best_step * 2, 0.1),
             train__kdvi__step_size_schedule__end=max(best_step / 4, 0.005),
             train__kdvi__step_size_schedule__steps=50000,
         )),
        ("coupled_step",
         best_overrides + _ovs(
             train__kdvi__step_size_schedule__type="coupled",
         )),
        ("anneal_50k",
         best_overrides + _ovs(train__annealing__steps=50000)),
    ]


def phase_d_configs(top3: list[RunResult]) -> list[tuple[str, list[str], int]]:
    """Phase D: top-3 candidates × seeds {42, 0, 1}."""
    out: list[tuple[str, list[str], int]] = []
    for r in top3:
        for s in (42, 0, 1):
            label = f"{r.phase}_seed{s}"
            out.append((label, r.overrides, s))
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
    """Group by overrides+phase, compute mean/std, sort by mean kl_ite."""
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# KDVI 8-Gaussian Adaptive Sweep — Summary\n")

    # Phase D: group by overrides
    d_runs = [r for r in results if r.phase.startswith("D")
              and r.best_kl_ite is not None]
    if d_runs:
        groups: dict[str, list[RunResult]] = {}
        for r in d_runs:
            key = " ".join(sorted(r.overrides))
            groups.setdefault(key, []).append(r)

        rows: list[tuple[str, list[RunResult]]] = []
        for key, rs in groups.items():
            rows.append((key, rs))

        def _mean(vals: list[float]) -> float:
            return sum(vals) / len(vals)

        def _std(vals: list[float]) -> float:
            if len(vals) < 2:
                return 0.0
            m = _mean(vals)
            return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5

        rows.sort(key=lambda kv: _mean(
            [r.best_kl_ite for r in kv[1] if r.best_kl_ite is not None]))

        lines.append("## Phase D — top candidates × seeds\n")
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
                f"{(_mean(w2s) if w2s else float('nan')):.4f} | "
                f"{(_mean(mmds) if mmds else float('nan')):.4f} | "
                f"{_mean(walls):.0f} |"
            )

    # All phases: top-10 by best KL_ITE
    valid = [r for r in results if r.best_kl_ite is not None]
    valid.sort(key=lambda r: r.best_kl_ite or float("inf"))
    lines.append("\n## All runs — top 15 by best KL_ITE\n")
    lines.append("| rank | run_id | phase | KL_ITE | W2 | MMD | ELBO | "
                 "epochs | wall(s) | overrides |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(valid[:15], 1):
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


def step_size_of(r: RunResult) -> float:
    return float(r.config_dict.get("train.kdvi.mcmc_step_size", 0.05))


def K_of(r: RunResult) -> int:
    return int(r.config_dict.get("train.kdvi.mcmc_steps", 5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true",
                        help="Run all phases without confirmation prompts.")
    parser.add_argument("--phase", choices=("A", "B", "C", "D", "all"),
                        default="all",
                        help="Only run a specific phase.")
    parser.add_argument("--early-stop-window", type=int, default=20000,
                        help="Epochs of no improvement before kill.")
    parser.add_argument("--poll-interval-s", type=float, default=30.0,
                        help="Seconds between TB polls.")
    args = parser.parse_args()

    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = _load_log()
    print(f"Loaded {len(results)} prior runs from {LOG_PATH}")

    def go(phase: str, idx: int, label: str, overrides: list[str],
           seed: int = 42) -> RunResult:
        prompt = f"\nProceed with {phase}-{idx:02d} {label}? [Enter=yes, s=skip, q=quit] "
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

    if args.phase in ("A", "all"):
        # Phase A: anchor + K
        for idx, (label, ov) in enumerate(phase_a_configs()):
            r = go(f"A", idx, label, ov)
            results.append(r)

        a_runs = [r for r in results if r.phase == "A"]
        winner_K = best_run(a_runs)
        if winner_K is None:
            print("Phase A produced no valid runs; aborting.")
            return
        best_K = K_of(winner_K)
        print(f"\n** Phase A K-winner: K={best_K} (best_kl={winner_K.best_kl_ite:.4f})")

        # Phase A2: step_size on top of K winner
        for idx, (label, ov) in enumerate(phase_a2_configs(best_K)):
            r = go("A", 10 + idx, label, ov)
            results.append(r)

        # Pick winner across A & A2
        a_so_far = [r for r in results if r.phase == "A"]
        winner_step = best_run(a_so_far)
        if winner_step is None:
            print("Phase A failed to produce a step-size winner; aborting.")
            return
        best_step = step_size_of(winner_step)
        best_K_after = K_of(winner_step)
        print(
            f"\n** Phase A step-winner: K={best_K_after} step={best_step} "
            f"(best_kl={winner_step.best_kl_ite:.4f})"
        )

        # Phase A3: mcmc_type on top of (best K, best step)
        for idx, (label, ov) in enumerate(
                phase_a3_configs(best_K_after, best_step)):
            r = go("A", 20 + idx, label, ov)
            results.append(r)

    if args.phase in ("B", "all"):
        a_runs = [r for r in results if r.phase == "A"]
        winner = best_run(a_runs) if a_runs else None
        if winner is None:
            print("Phase B needs a Phase-A winner; skipping.")
        else:
            print(f"\n** Building Phase B on top of A winner: {winner.run_id}")
            for idx, (label, ov) in enumerate(phase_b_configs(winner.overrides)):
                r = go("B", idx, label, ov)
                results.append(r)

    if args.phase in ("C", "all"):
        ab_runs = [r for r in results if r.phase in ("A", "B")]
        winner = best_run(ab_runs) if ab_runs else None
        if winner is None:
            print("Phase C needs an A/B winner; skipping.")
        else:
            best_step = step_size_of(winner)
            print(f"\n** Building Phase C on top of A/B winner: {winner.run_id}")
            for idx, (label, ov) in enumerate(
                    phase_c_configs(winner.overrides, best_step)):
                r = go("C", idx, label, ov)
                results.append(r)

    if args.phase in ("D", "all"):
        abc_runs = [r for r in results
                    if r.phase in ("A", "B", "C") and r.best_kl_ite is not None]
        # Dedupe by overrides set
        uniq: dict[str, RunResult] = {}
        for r in abc_runs:
            key = " ".join(sorted(r.overrides))
            if key not in uniq or (uniq[key].best_kl_ite or 1e9) > (r.best_kl_ite or 1e9):
                uniq[key] = r
        sorted_uniq = sorted(uniq.values(),
                             key=lambda r: r.best_kl_ite or 1e9)
        top3 = sorted_uniq[:3]
        if not top3:
            print("Phase D needs A/B/C winners; skipping.")
        else:
            print(f"\n** Phase D top-3 candidates:")
            for r in top3:
                print(f"     {r.run_id}  best_kl={r.best_kl_ite:.4f}  "
                      f"overrides={' '.join(r.overrides) or '(baseline)'}")
            for idx, (label, ov, seed) in enumerate(phase_d_configs(top3)):
                r = go("D", idx, label, ov, seed=seed)
                results.append(r)

    write_summary(results)


if __name__ == "__main__":
    main()
