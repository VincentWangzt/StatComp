# IVI ↔ KDVI Parity Validation Progress

> **Archived historical note (2026-06-23):** The lockstep harness and the
> production-only parity helpers documented below were intentionally removed
> after validation. This file is retained as an investigation record; its
> commands and symbol names are no longer part of the supported KDVI interface.

Goal: make IVI (`IVI-via-mcmc-distillation/run_ivi.py::ImVIDrift`) and KDVI
(`runner/kdvi.py::KDVIRunner`) produce **identical** training trajectories
under the same seed, first on `8_gaussians_small`, then `8_gaussians`.

Method: an in-process lockstep harness
(`scripts/compare_ivi_kdvi_lockstep.py`) force-syncs initial parameters and
diffs every per-step phase (sampled `z`, MCMC-refined `z'`, bandwidth `h`,
loss, per-parameter gradients, post-step params), locating the **first** point
of divergence. Then locate → fix → re-verify, progressively.

---

## Equivalence facts established by code reading (pre-run analysis)

These were derived analytically from the source and will be confirmed
numerically by the harness:

- **VI net architecture** is structurally identical:
  `[Linear(2,256),ELU,Linear(256,256),ELU,Linear(256,256),ELU,Linear(256,4)]`.
  IVI `Transform.model.{0,2,4,6}` ↔ KDVI `ConditionalGaussian.net.{0,2,4,6}`.
- **Reparameterization** (`logstd`, `log_std_min=-3`): `std=exp(log_std)`,
  `z=mu+std*u` — identical in both.
- **MALA parameterization**: KDVI `step_size = 2 * IVI drift_stepsz`
  (`0.02 ↔ 0.01`); KDVI `coupled` step `tau/beta` ↔ IVI `stepsz/anneal_coef`;
  forward/backward proposal densities and the `log(rand) < ratio` accept test
  are algebraically equal when `beta == anneal_coef`.
- **Annealing**: KDVI `annealing(scheme='ivi', steps=50000)` ==
  IVI `anneal_coef = min(1, 0.1 + i/50000)`.
- **Optimizer/scheduler**: Adam(lr=1e-3, betas=(0.9,0.999)); StepLR
  `base*gamma^((e-1)//5000)` matches IVI manual `lr*=0.75` every 5000 iters.
- **Within-step RNG draw order/shape** matches: `randn[128,2]`, `randn[128,2]`,
  `randn[128,2]`, `rand[128]`.
- **Target** is the same project class `EightGaussiansSmall` for both (IVI wraps
  it via `_ProjectGMMAdapter`).

## Suspected real discrepancies (to confirm/fix progressively)

1. **Thread nondeterminism** — IVI sets `torch.set_num_threads(1)`; KDVI/`src.py`
   do not. (Harness sets it.)
2. **MMD distance mismatch** — IVI uses `torch.cdist(p=2)` for both bandwidth
   and kernel (diagonal exactly 0); `utils/kernels.py::LaplaceL2Kernel.pair_eval`
   uses matmul-expansion `sqrt(.+1e-12)` (diagonal = 1e-6). Different rounding,
   enters the gradient every step.
3. **Pre-training / construction RNG phase** — `src.py` omits `np.random.seed`
   and seeds before runner construction; IVI seeds after reference gen and draws
   a pre-training `sample(2000)`. (Harness neutralizes via per-step reseed +
   force-sync.)
4. **Mid-training eval RNG** — differing sample counts/cadence between the two
   eval paths.

---

## Progress log

### [setup] Lockstep harness scaffold — DONE
- Created `scripts/compare_ivi_kdvi_lockstep.py` (CPU, single-thread,
  force-syncs IVI→KDVI init params, two RNG regimes: `reseed` / `global`).
- Confirmed: initial parameters **byte-identical** after force-sync
  (`max-abs param diff = 0.000e+00`).

### [fix 1] `z` forward divergence — std computation — DONE
- **Symptom**: step-1 `d_z = 4.77e-07`; `d_grad = 1.89e-04` despite identical loss.
- **Root cause**: KDVI `logstd` path computed `std = sqrt(exp(2*log_std))`
  (`_variance_from_raw` → `sqrt(var)`), whereas IVI computes `std = exp(log_std)`.
  Algebraically equal but different float32 value AND different autograd graph.
- **Fix**: `models/vi_model.py::ConditionalGaussian` — added `_std_from_raw`
  which, for `logstd`, returns `exp(clamp(log_std))` directly. `reparameterize`
  and `getstd` now use it. Other parameterizations unchanged.
- **Result**: step-1 `d_z = 0.000e+00`.

### [fix 2] MMD distance computation — cdist parity — DONE
- **Symptom**: with `d_z=0` and `d_loss=0`, step-1 `d_grad = 1.2e-04`,
  `d_param = 2e-03`. The large param jump is **Adam sign-flip amplification**
  (step-1 update ≈ `lr·sign(grad)`, so any near-zero grad that flips sign moves
  the param by `2·lr = 2e-3`). Requires bit-identical gradients.
- **Root cause**: IVI's `maximum_mean_discrepancy` uses `torch.cdist(...,p=2)`
  for both bandwidth and kernel; `utils/kernels.py::LaplaceL2Kernel.pair_eval`
  used a manual `sqrt(x_norm+y_norm-2xyᵀ + 1e-12)` (different float value and
  different backward; diagonal = 1e-6 vs exact 0).
- **Fix**: `LaplaceL2Kernel.fit_h` and `.pair_eval` now use
  `torch.cdist(...,p=2)` (default compute_mode), matching IVI. Removed the
  1e-12 diagonal epsilon (cdist gives a well-defined zero subgradient).
  Only affects the `laplace_l2` kernel (KDVI-only; `grad_all` unused).
- **Result**: step-1 `d_grad = 1.16e-09`, `d_param = 1.11e-06`. The MMD distance
  was the dominant gradient-discrepancy source.

### [fix 3] MALA proposal — IVI-exact arithmetic — DONE
- **Symptom**: step-1 `d_z_refined = 1.19e-07`, slowly cascading
  (`d_param ~5e-6` after 5 steps) via the chaotic optimizer.
- **Root cause**: float-ordering in the Langevin proposal/density. IVI applies
  a single folded scalar `(stepsz*anneal)*score` and uses the drawn noise tensor
  directly in the forward density; the generic `mala_transition` applies
  `0.5*step_size*(beta*score)` (two tensor mults, different scalar) and
  recomputes the forward density from differences, with `sqrt(step_size)` vs
  IVI `sqrt(2*stepsz)`.
- **Fix**: added `utils/mcmc_kernels.py::mala_transition_ivi` — a line-by-line
  batched replica of `ImVIDrift.mala`. The KDVI `mala` parity path calls it with
  RAW target score/logp (annealing applied internally via `anneal_coef=beta`)
  and `stepsz = current_step_size/2` (so noise scale `sqrt(2*stepsz) ==
  sqrt(current_step_size)` and drift `stepsz*anneal == drift_stepsz`).
- **Result**: step-1 `d_z_refined = 0.000e+00`. New first divergence: `d_grad`.

### [fix 4] MMD kernel expression + gradient accumulation order — DONE
- **Symptom (after fix 3)**: with `d_z=0`, `d_z_refined=0`, `d_loss=0`, step-1
  `d_grad = 2.33e-10` (float32), cascading to `d_param ~1.7e-6`.
- **Isolation** (`scripts/_tmp_isolate_grad.py`, temp, deleted): backprop both
  MMD forms into a SHARED `z` leaf (model graph factored out). Residual
  `d_grad = 5.8e-11` in float32 but `1.08e-19` in float64 → confirms a pure
  float32 reduction-ordering effect in the MMD backward, not a structural bug.
- **Root cause**: gradient **accumulation order**. IVI's
  `maximum_mean_discrepancy` creates the CROSS term `cdist(samples, y)` FIRST,
  then the `xx` term `cdist(y, y)`. KDVI's `mmd_ivi_drift` created `K_xx` first,
  then `K_yx`. Autograd accumulates the two contributions to `x.grad` in
  node-creation order; float32 addition is non-associative → ULP-level diff.
- **Fix**: (a) `utils/mmd.py::mmd_ivi_drift` now creates `K_yx` (cross) BEFORE
  `K_xx`, matching IVI's statement order. (b) `LaplaceL2Kernel.pair_eval` uses
  the textually identical `(-d / (2 * h)).exp()` form.
- **Result**: shared-`z` `d_grad = 0.000e+00` in BOTH float32 and float64.

### [MILESTONE] Bit-identical lockstep on 8_gaussians_small (reseed mode) — DONE
- With fixes 1–4, the harness in `--mode reseed` (identical RNG stream fed to
  both per step) reports **`d_anneal = d_z = d_z_refined = d_loss = d_grad =
  d_param = 0.000e+00` for all 30 steps**. IVI and KDVI are now bit-identical
  in algorithm + autograd graph + float32 reduction order.
- `--mode global` still diverges immediately, but that is **by harness design**:
  it runs `ivi_step` then `kdvi_step` sequentially sharing ONE global RNG
  stream, so the two never observe the same draws. The per-step RNG draw
  order/shape itself is proven identical by the `reseed` result. End-to-end
  RNG-stream alignment (construction-time offset + eval/plot draws) is the next
  milestone for full standalone-run identity.

---

## Production wiring status
- [x] `models/vi_model.py` — `_std_from_raw` (fix 1). Affects `logstd` path only.
- [x] `utils/kernels.py` — `LaplaceL2Kernel` cdist + exp form (fix 2, 4b).
- [x] `utils/mmd.py` — `mmd_ivi_drift` K_yx-before-K_xx order (fix 4a).
- [x] `utils/mcmc_kernels.py` — added `mala_transition_ivi` (fix 3).
- [x] `runner/kdvi.py` — `mala` path now uses `mala_transition_ivi`.

### [fix 5] Standalone-run RNG-stream alignment — DONE
- **Goal**: make the *actual CLI programs* (`src.py` KDVI vs `run_ivi.py` IVI)
  byte-identical, not just the per-step algorithm. Two obstacles:
  (a) construction / pre-training contour-plot draws offset the training RNG;
  (b) mid-training eval/plot/sample draws perturb it at differing cadences.
- **Verified first**: natural initialization is **byte-identical** without any
  force-sync — both build the same 4-`Linear` net as the FIRST torch-RNG
  consumer after seeding (`--no-sync` harness run: init diff = 0).
- **Fix** (opt-in, guarded; default off → zero impact on other runs):
  - `runner/base_runner.py`: `train.parity_rng_isolation` flag. When set,
    `torch.manual_seed(seed)` right before the training loop, and snapshot/
    restore the RNG around the eval/sample/plot block so diagnostics are
    RNG-neutral.
  - `IVI-via-mcmc-distillation/run_ivi.py`: `--rng-isolation` flag doing the
    same (re-seed before loop; snapshot/restore around KSD-print + eval_callback).
  - `configs/kdvi_8_gaussians_small.yaml`: `train.parity_rng_isolation: true`.
- **Result**:
  - Harness `--mode standalone --no-sync` (each impl its OWN continuous RNG
    stream from the same init): **0 divergence across 2000 steps** (seeds 7, 42).
  - Real `learn()` methods end-to-end (KDVI `runner.learn()` + IVI
    `model.learn(rng_isolation=True)`), eval disabled: **final max-abs param
    diff = 0.000e+00 after 300 training steps** — byte-identical trained models.

### [MILESTONE] Full algorithm + standalone-run parity on 8_gaussians_small — DONE
- IVI and KDVI are now byte-identical: per-step (reseed), per-stream
  (standalone), natural init, and end-to-end via the real training loops.

### [CONFIRM] Both targets, both RNG modes, long horizon — DONE
- `8_gaussians_small`: standalone `--no-sync` **0 divergence / 10000 steps**
  (covers multiple StepLR decays); reseed/standalone 0 diff at 30/50/200/2000.
- `8_gaussians` (large; `configs/kdvi_8_gaussians.yaml`, IVI drift_stepsz=0.25,
  KDVI mcmc_step_size=0.50): standalone `--no-sync` **0 divergence / 200 steps**,
  reseed **0 divergence / 100 steps** (seed 13). Natural init byte-identical.
- `configs/kdvi_8_gaussians.yaml` also gets `train.parity_rng_isolation: true`.

## Final residual analysis
- The ONLY non-zero residual encountered after fixes 1–4 was a ULP-level
  (~5.8e-11 float32; ~1e-19 float64) gradient-accumulation effect, which fix 4
  eliminated by matching node-creation order. With all fixes, every compared
  quantity (z, z', h, loss, grad, params) is exactly `0.0` in float32 across
  thousands of steps on both targets. No irreducible residual remains for these
  toy targets at batch=128.

## How to reproduce parity in standalone CLI runs
- KDVI: `src.py --config configs/kdvi_8_gaussians_small.yaml`
  (config already sets `train.parity_rng_isolation: true`).
- IVI:  `IVI-via-mcmc-distillation/run_ivi.py --target 8_gaussians_small
  --drift-stepsz 0.01 --rng-isolation` (same seed).
- Both build the same net first after seeding (natural init identical), reset
  the training RNG before the loop, and keep eval/plot/sample RNG-neutral, so
  the trained models are byte-identical.

## Temporary debug outputs to remove before finishing (CHECKLIST)
- [x] `scripts/compare_ivi_kdvi_lockstep.py` — KEPT as a permanent regression
      harness (clean, documented, no stray debug; opt-in CLI tool). Re-run any
      time to assert IVI<->KDVI byte-identity.
- [x] `scripts/_tmp_isolate_grad.py` — deleted after isolating fix 4.
- [x] `scripts/_tmp_e2e_compare.py` — deleted after confirming fix 5.
- [x] No `print`/`logger.debug` left inside production modules (verified by
      grep: no DEBUG/_tmp_/FIXME markers in the edited files).
- [x] No leftover `results/ivi_kdvi_lockstep`, `tb_logs/ivi_kdvi_lockstep`, or
      `results/_tmp_e2e` artifact directories.
