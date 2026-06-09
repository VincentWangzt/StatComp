# KDVI 8-Gaussian Adaptive Sweep — Summary

## TL;DR

**Best recipe (3-seed mean):** `mcmc_steps=10, mcmc_step_size=0.02, mcmc_type=sgld, kernel_bandwidth=0.5 (fixed), step_size_schedule.type=coupled` → **KL_ITE = 0.509 ± 0.064** (seeds 42/0/1).

**Improvement over default KDVI on this target:** baseline KL_ITE ≈ 0.78 → 0.51, a **35% reduction**, in 27 runs and ~10.8 h total CPU time.

## Findings (ranked by effect size)

1. **Fixed kernel bandwidth dominates the median heuristic.** Switching from
   median-of-batch (current default `fit_bandwidth_on=x`) to a fixed `h=0.5`
   (matching the σ of the target's component Gaussians) dropped KL_ITE from
   0.599 → 0.536 — the single largest single-knob win. h=1.0 was nearly as
   good; both fit-on-y and fit-on-xy hurt slightly. The median heuristic
   shrinks during training as q_φ collapses, choking the mode-coverage
   gradient signal.

2. **Coupled step-size annealing (notebook style) helps.** Adding
   `step_size_schedule.type=coupled` (i.e. `step / β(t)`) on top of
   fixed-bandwidth dropped KL_ITE another 0.04 (0.536 → 0.497 single-seed,
   0.509 across 3 seeds). The cosine variant was slightly worse on average
   but had higher variance (σ=0.21).

3. **K=10 is the right MCMC chain length.** K=5 (default) → 0.78,
   K=10 → 0.61, K=20 → 0.75, K=1 (notebook style) → 0.65. K=10 is the
   sweet spot — enough drift to escape modes, not so many no-ops that
   later steps just add noise.

4. **MCMC step size 0.02 > 0.05 (default) > 0.01 ≈ 0.1.** On the FFJORD
   r=4/σ=0.5 target, the default 0.05 turned out too aggressive once
   K=10 was applied; 0.02 gave a finer-grained drift.

5. **VI capacity helps modestly when ε-dim grows with it.** The
   notebook-style net (ε=32, hidden=256, 3 layers, 141k params) gave
   KL_ITE = 0.567 — better than default 128/2 but worse than the same
   network with fixed bandwidth (0.536). A wider network without ε-dim
   widening (ConditionalGaussian-Wide.yaml) was *worse* (0.754).

6. **Slower β annealing hurts.** `train.annealing.steps: 50000` (over
   100K epochs) dropped to 0.71 — significantly worse than the default
   25K-step β ramp.

7. **MALA & HMC underperform SGLD with K=10.** With K=10/step=0.02,
   SGLD → 0.599, MALA → 0.670, HMC → 0.640. The M-H rejections in
   MALA/HMC slow effective mixing per gradient step, more than the bias
   correction is worth at this batch + K. (HMC accept rate held ≥ 0.5
   throughout — not a tuning issue.)

## Caveats

- **W2 increased with the winning recipes.** Top-3 by KL_ITE all show W2
  ≥ 1.0 (vs ~0.34 for the Phase-A K=10 baseline). KL_ITE rewards mode
  coverage; W2 penalizes inflated spread. The fixed-bandwidth h=0.5 (a
  *large* bandwidth on the σ=0.5 target) appears to push q_φ toward an
  over-spread distribution that covers all 8 modes well in density but
  isn't tightly concentrated on each.
- **The h=0.5 + coupled_step recipe variance is non-trivial across
  seeds** (σ=0.064 on KL_ITE, plus one cosine_step seed=0 run that
  catastrophically diverged to 0.880 / W2=2.93). Suggests this recipe
  needs either a longer run, a learning-rate warm-up, or seed-averaged
  evaluation.
- **B_02_hfix_0p5 didn't early-stop** — it improved monotonically across
  the full 100K epochs. With more iterations, the no-schedule recipe
  could potentially close the gap with `coupled_step` (which plateaus
  earlier).

## Recommended default for KDVI on 8-Gaussian (FFJORD r=4)

```yaml
train:
  kdvi:
    mcmc_type: sgld
    mcmc_steps: 10
    mcmc_step_size: 0.02
    kernel: gaussian
    kernel_bandwidth: 0.5         # fixed; forces fit_bandwidth_on=none
    step_size_schedule:
      type: coupled               # notebook-style: step / beta(t)
```

(All other knobs unchanged from `configs/kdvi_8_gaussians.yaml`.)

## Phase D — top candidates × seeds

| rank | overrides | seeds | KL_ITE mean±std | W2 mean | MMD mean | wall mean (s) |
|---|---|---|---|---|---|---|
| 1 | `train.kdvi.kernel_bandwidth=0.5 train.kdvi.mcmc_step_size=0.02 train.kdvi.mcmc_steps=10 train.kdvi.step_size_schedule.type=coupled` | 3 | 0.5088 ± 0.0638 | 1.1574 | 0.0568 | 1157 |
| 2 | `train.kdvi.kernel_bandwidth=0.5 train.kdvi.mcmc_step_size=0.02 train.kdvi.mcmc_steps=10` | 3 | 0.5719 ± 0.0331 | 1.0292 | 0.0541 | 2125 |
| 3 | `train.kdvi.kernel_bandwidth=0.5 train.kdvi.mcmc_step_size=0.02 train.kdvi.mcmc_steps=10 train.kdvi.step_size_schedule.end=0.005 train.kdvi.step_size_schedule.start=0.1 train.kdvi.step_size_schedule.steps=50000 train.kdvi.step_size_schedule.type=cosine` | 3 | 0.6342 ± 0.2132 | 1.6741 | 0.1172 | 1182 |

## All runs — top 15 by best KL_ITE

| rank | run_id | phase | KL_ITE | W2 | MMD | ELBO | epochs | wall(s) | overrides |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `D_02_C_seed1` | D | 0.4519 | 1.3717 | 0.0515 | -0.8546 | 43000 | 1161 | `train.kdvi.mcmc_steps=10 train.kdvi.mcmc_step_size=0.02 train.kdvi.kernel_bandwidth=0.5 train.kdvi.step_size_schedule.type=coupled` |
| 2 | `C_01_coupled_step` | C | 0.4969 | 0.9757 | 0.0353 | -0.6397 | 44500 | 1170 | `train.kdvi.mcmc_steps=10 train.kdvi.mcmc_step_size=0.02 train.kdvi.kernel_bandwidth=0.5 train.kdvi.step_size_schedule.type=coupled` |
| 3 | `D_00_C_seed42` | D | 0.4969 | 1.0211 | 0.0441 | -0.5899 | 43000 | 1258 | `train.kdvi.mcmc_steps=10 train.kdvi.mcmc_step_size=0.02 train.kdvi.kernel_bandwidth=0.5 train.kdvi.step_size_schedule.type=coupled` |
| 4 | `D_05_C_seed1` | D | 0.5013 | 1.0008 | 0.0345 | -0.5660 | 43500 | 1235 | `train.kdvi.mcmc_steps=10 train.kdvi.mcmc_step_size=0.02 train.kdvi.kernel_bandwidth=0.5 train.kdvi.step_size_schedule.type=cosine train.kdvi.step_size_schedule.start=0.1 train.kdvi.step_size_schedule.end=0.005 train.kdvi.step_size_schedule.steps=50000` |
| 5 | `C_00_cosine_step` | C | 0.5212 | 1.0572 | 0.0495 | -0.6579 | 40500 | 1054 | `train.kdvi.mcmc_steps=10 train.kdvi.mcmc_step_size=0.02 train.kdvi.kernel_bandwidth=0.5 train.kdvi.step_size_schedule.type=cosine train.kdvi.step_size_schedule.start=0.1 train.kdvi.step_size_schedule.end=0.005 train.kdvi.step_size_schedule.steps=50000` |
| 6 | `D_03_C_seed42` | D | 0.5212 | 1.0958 | 0.0565 | -0.7450 | 43000 | 1162 | `train.kdvi.mcmc_steps=10 train.kdvi.mcmc_step_size=0.02 train.kdvi.kernel_bandwidth=0.5 train.kdvi.step_size_schedule.type=cosine train.kdvi.step_size_schedule.start=0.1 train.kdvi.step_size_schedule.end=0.005 train.kdvi.step_size_schedule.steps=50000` |
| 7 | `B_02_hfix_0p5` | B | 0.5357 | 0.8894 | 0.0347 | -0.5796 | 100000 | 2889 | `train.kdvi.mcmc_steps=10 train.kdvi.mcmc_step_size=0.02 train.kdvi.kernel_bandwidth=0.5` |
| 8 | `D_06_B_seed42` | D | 0.5357 | 0.8894 | 0.0374 | -0.5796 | 100000 | 2663 | `train.kdvi.mcmc_steps=10 train.kdvi.mcmc_step_size=0.02 train.kdvi.kernel_bandwidth=0.5` |
| 9 | `B_03_hfix_1p0` | B | 0.5385 | 0.4688 | 0.0124 | -0.6194 | 64000 | 1750 | `train.kdvi.mcmc_steps=10 train.kdvi.mcmc_step_size=0.02 train.kdvi.kernel_bandwidth=1.0` |
| 10 | `B_05_vi_notebook` | B | 0.5673 | 0.5347 | 0.0081 | -3.4652 | 60000 | 2069 | `train.kdvi.mcmc_steps=10 train.kdvi.mcmc_step_size=0.02 vi_model_config_path=configs/vi_models/ConditionalGaussian-Notebook.yaml` |
| 11 | `D_01_C_seed0` | D | 0.5778 | 1.0795 | 0.0749 | -0.6937 | 38000 | 1053 | `train.kdvi.mcmc_steps=10 train.kdvi.mcmc_step_size=0.02 train.kdvi.kernel_bandwidth=0.5 train.kdvi.step_size_schedule.type=coupled` |
| 12 | `D_08_B_seed1` | D | 0.5794 | 1.1047 | 0.0568 | -0.6588 | 100000 | 2655 | `train.kdvi.mcmc_steps=10 train.kdvi.mcmc_step_size=0.02 train.kdvi.kernel_bandwidth=0.5` |
| 13 | `A_11_step0p02_K10` | A | 0.5994 | 0.4523 | 0.0104 | -0.6425 | 36500 | 1031 | `train.kdvi.mcmc_steps=10 train.kdvi.mcmc_step_size=0.02` |
| 14 | `D_07_B_seed0` | D | 0.6006 | 1.0936 | 0.0680 | -0.7618 | 40000 | 1058 | `train.kdvi.mcmc_steps=10 train.kdvi.mcmc_step_size=0.02 train.kdvi.kernel_bandwidth=0.5` |
| 15 | `A_02_K10` | A | 0.6117 | 0.3447 | 0.0048 | -0.8577 | 42500 | 1158 | `train.kdvi.mcmc_steps=10` |
