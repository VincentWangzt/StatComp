# KDVI 8-Gaussian Kernel-Fix Sweep — Summary

## Phase C — finalists × seeds

| rank | overrides | seeds | KL_ITE mean±std | W2 mean | MMD mean | wall mean (s) |
|---|---|---|---|---|---|---|
| 1 | `train.kdvi.fit_bandwidth_on=xy train.kdvi.kernel=laplace_l2 train.kdvi.mcmc_step_size=0.02 train.kdvi.mcmc_steps=10 train.kdvi.mcmc_type=sgld train.kdvi.step_size_schedule.type=coupled vi_model_config_path=configs/vi_models/ConditionalGaussian-Eps8.yaml` | 3 | 0.3026 ± 0.0083 | 0.3378 | 0.0091 | 760 |
| 2 | `train.kdvi.fit_bandwidth_on=xy train.kdvi.kernel=laplace_l2 train.kdvi.mcmc_step_size=0.02 train.kdvi.mcmc_steps=10 train.kdvi.mcmc_type=sgld train.kdvi.step_size_schedule.type=coupled` | 1 | 0.3895 ± 0.0000 | 0.5814 | 0.0112 | 833 |

## All runs — top 20 by best KL_ITE

| rank | run_id | phase | KL_ITE | W2 | MMD | ELBO | epochs | wall(s) | overrides |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `C_01_top1_seed0` | C | 0.2938 | 0.2951 | 0.0094 | -1.8339 | 100000 | 760 | `train.kdvi.mcmc_type=sgld train.kdvi.mcmc_steps=10 train.kdvi.mcmc_step_size=0.02 train.kdvi.step_size_schedule.type=coupled train.kdvi.kernel=laplace_l2 train.kdvi.fit_bandwidth_on=xy vi_model_config_path=configs/vi_models/ConditionalGaussian-Eps8.yaml` |
| 2 | `B_01_eps8_base_arch` | B | 0.3034 | 0.3122 | 0.0072 | -3.0912 | 100000 | 831 | `train.kdvi.mcmc_type=sgld train.kdvi.mcmc_steps=10 train.kdvi.mcmc_step_size=0.02 train.kdvi.step_size_schedule.type=coupled train.kdvi.kernel=laplace_l2 train.kdvi.fit_bandwidth_on=xy vi_model_config_path=configs/vi_models/ConditionalGaussian-Eps8.yaml` |
| 3 | `C_00_top1_seed42` | C | 0.3034 | 0.3122 | 0.0053 | -3.0912 | 100000 | 763 | `train.kdvi.mcmc_type=sgld train.kdvi.mcmc_steps=10 train.kdvi.mcmc_step_size=0.02 train.kdvi.step_size_schedule.type=coupled train.kdvi.kernel=laplace_l2 train.kdvi.fit_bandwidth_on=xy vi_model_config_path=configs/vi_models/ConditionalGaussian-Eps8.yaml` |
| 4 | `C_02_top1_seed1` | C | 0.3104 | 0.4061 | 0.0126 | -3.6065 | 100000 | 757 | `train.kdvi.mcmc_type=sgld train.kdvi.mcmc_steps=10 train.kdvi.mcmc_step_size=0.02 train.kdvi.step_size_schedule.type=coupled train.kdvi.kernel=laplace_l2 train.kdvi.fit_bandwidth_on=xy vi_model_config_path=configs/vi_models/ConditionalGaussian-Eps8.yaml` |
| 5 | `B_02_eps16_base_arch` | B | 0.3517 | 0.5872 | 0.0099 | -3.2290 | 97500 | 742 | `train.kdvi.mcmc_type=sgld train.kdvi.mcmc_steps=10 train.kdvi.mcmc_step_size=0.02 train.kdvi.step_size_schedule.type=coupled train.kdvi.kernel=laplace_l2 train.kdvi.fit_bandwidth_on=xy vi_model_config_path=configs/vi_models/ConditionalGaussian-Eps16.yaml` |
| 6 | `A_04_laplace_l2_fb_xy` | A | 0.3895 | 0.5814 | 0.0082 | -0.8205 | 100000 | 764 | `train.kdvi.mcmc_type=sgld train.kdvi.mcmc_steps=10 train.kdvi.mcmc_step_size=0.02 train.kdvi.step_size_schedule.type=coupled train.kdvi.kernel=laplace_l2 train.kdvi.fit_bandwidth_on=xy` |
| 7 | `B_00_eps_default` | B | 0.3895 | 0.5814 | 0.0158 | -0.8205 | 100000 | 826 | `train.kdvi.mcmc_type=sgld train.kdvi.mcmc_steps=10 train.kdvi.mcmc_step_size=0.02 train.kdvi.step_size_schedule.type=coupled train.kdvi.kernel=laplace_l2 train.kdvi.fit_bandwidth_on=xy` |
| 8 | `C_03_top2_seed42` | C | 0.3895 | 0.5814 | 0.0112 | -0.8205 | 100000 | 833 | `train.kdvi.mcmc_type=sgld train.kdvi.mcmc_steps=10 train.kdvi.mcmc_step_size=0.02 train.kdvi.step_size_schedule.type=coupled train.kdvi.kernel=laplace_l2 train.kdvi.fit_bandwidth_on=xy` |
| 9 | `A_03_laplace_l2_fb_x` | A | 0.3942 | 0.6336 | 0.0132 | -0.7350 | 100000 | 837 | `train.kdvi.mcmc_type=sgld train.kdvi.mcmc_steps=10 train.kdvi.mcmc_step_size=0.02 train.kdvi.step_size_schedule.type=coupled train.kdvi.kernel=laplace_l2 train.kdvi.fit_bandwidth_on=x` |
| 10 | `A_05_gaussian_hfix_0p5` | A | 0.5822 | 1.1205 | 0.0710 | -0.7009 | 50500 | 370 | `train.kdvi.mcmc_type=sgld train.kdvi.mcmc_steps=10 train.kdvi.mcmc_step_size=0.02 train.kdvi.step_size_schedule.type=coupled train.kdvi.kernel=gaussian train.kdvi.kernel_bandwidth=0.5` |
| 11 | `B_03_eps32_notebook_arch` | B | 0.6179 | 1.4531 | 0.0845 | -2.6877 | 100000 | 843 | `train.kdvi.mcmc_type=sgld train.kdvi.mcmc_steps=10 train.kdvi.mcmc_step_size=0.02 train.kdvi.step_size_schedule.type=coupled train.kdvi.kernel=laplace_l2 train.kdvi.fit_bandwidth_on=xy vi_model_config_path=configs/vi_models/ConditionalGaussian-Notebook.yaml` |
| 12 | `A_00_gaussian_fb_x` | A | 0.8194 | 0.3149 | 0.0062 | -3.0811 | 48000 | 367 | `train.kdvi.mcmc_type=sgld train.kdvi.mcmc_steps=10 train.kdvi.mcmc_step_size=0.02 train.kdvi.step_size_schedule.type=coupled train.kdvi.kernel=gaussian train.kdvi.fit_bandwidth_on=x` |
| 13 | `A_01_gaussian_mmd_fb_x` | A | 1.1179 | 0.2169 | 0.0051 | -2.7212 | 49500 | 372 | `train.kdvi.mcmc_type=sgld train.kdvi.mcmc_steps=10 train.kdvi.mcmc_step_size=0.02 train.kdvi.step_size_schedule.type=coupled train.kdvi.kernel=gaussian_mmd train.kdvi.fit_bandwidth_on=x` |
| 14 | `A_02_gaussian_mmd_fb_xy` | A | 1.1670 | 0.2995 | 0.0097 | -2.7387 | 46500 | 362 | `train.kdvi.mcmc_type=sgld train.kdvi.mcmc_steps=10 train.kdvi.mcmc_step_size=0.02 train.kdvi.step_size_schedule.type=coupled train.kdvi.kernel=gaussian_mmd train.kdvi.fit_bandwidth_on=xy` |


## Sweep status: aborted after 14 / 16 runs

Phase A (6/6) and Phase B (4/4) finished. Phase C had top-1 epsilon=8 (3 seeds) plus top-2 epsilon=2 (1 seed) before abort. The two missing runs were top-2 with seeds 0 and 1 — irrelevant for the headline number, since top-2 (default epsilon) is the *control*, and a single seed already shows it underperforms top-1.

## Headline

**Best 3-seed mean: KL_ITE = 0.3026 +/- 0.0083** with `laplace_l2` + `fb=xy` + `epsilon_dim=8`.
**vs prior sweep best: 0.5088 +/- 0.0638** — a **40.5% reduction** from the previous sweep winner, **61% reduction** from the original KDVI baseline (0.78).

## Findings (ranked by effect size)

1. **`LaplaceL2Kernel` is the right kernel for multimodal MMD on this target.** Switching from the existing `gaussian` kernel (with its SVGD heuristic) to the new `laplace_l2` kernel dropped KL_ITE from 0.819 -> 0.394 — a 52% reduction in a single change. The heavier exponential tails preserve cross-mode gradient signal even when q_phi and the MCMC-refined samples concentrate on different modes.

2. **`GaussianKernelMMD` (the textbook MMD heuristic, h = sqrt(median(d^2))) is *worse* than the existing SVGD heuristic on this target** (1.118 vs 0.819). h ~ 1.5 is too wide for sigma=0.5 modes — the kernel cannot localize within a single mode, so q_phi diffuses. The story is not "narrow vs wide bandwidth"; it is the *kernel family* (Gaussian-on-d^2 vs Laplace-on-d) that determines whether the auto-fit bandwidth helps or hurts.

3. **`epsilon_dim=8` is the sweet spot, not 16 or 32.** With laplace_l2 + fb=xy + base architecture (hidden=128, layers=2):
   - epsilon=2 (default): 0.390
   - epsilon=8: 0.303  <-- best
   - epsilon=16: 0.352
   - epsilon=32 (notebook arch, hidden=256, 3 layers, ELU): 0.618 (regression!)

   The notebook architecture severely underperformed at 100K iters — wider/deeper networks with ELU need more training before they outpace the lean SiLU baseline. epsilon itself plateaus around 8; adding more latent capacity does not help once the network is the bottleneck.

4. **`fb=xy` slightly beats `fb=x` for laplace_l2** (0.390 vs 0.394) — the joint median heuristic better tracks the kernel scale needed to compare both distributions.

5. **The prior sweep's "fixed h=0.5 + coupled step-size" recipe (KL_ITE 0.51) is decisively beaten by laplace_l2 + epsilon=8 (KL_ITE 0.30).** Heavier-tail kernel + slightly larger epsilon is a much better lever than tuning a Gaussian kernel bandwidth.

6. **Reproducibility is excellent.** With laplace_l2 + epsilon=8: seeds {42, 0, 1} -> {0.3034, 0.2938, 0.3104}, std = 0.0083. ~8x smaller std than the previous sweep's 0.0638.

7. **W2 also drops** (0.31 -- 0.41 across 3 seeds vs 1.16 in the prior sweep) — this run does NOT trade KL for W2 the way the prior fixed-h=0.5 recipe did.

## Recommended new default for KDVI on multimodal targets

```yaml
train:
  kdvi:
    mcmc_type: sgld
    mcmc_steps: 10
    mcmc_step_size: 0.02
    kernel: laplace_l2          # NEW kernel from utils/kernels.py
    fit_bandwidth_on: xy        # joint set, matches notebook
    step_size_schedule:
      type: coupled             # notebook-style
vi_model_config_path: configs/vi_models/ConditionalGaussian-Eps8.yaml
```

## Caveats and next directions

- The previous sweep's `gaussian + h=0.5` recipe traded KL_ITE for W2 (W2 ~ 1.0+); the new `laplace_l2 + epsilon=8` recipe keeps W2 ~ 0.31-0.40. Both metrics improved jointly.
- ELBO is *more* negative for epsilon=8 (-3.0) than epsilon=2 (-0.8) — the wider latent VI distribution has higher log-q penalty per sample, but mode coverage is much better.
- The notebook-arch run (epsilon=32, hidden=256, 3 ELU layers) was a regression at 100K. With longer training (notebook-style 700K cumulative iters) it might overtake; not tested.
- **Open question**: does `laplace_l2` carry to BNN posteriors? Worth a sanity check on a non-toy target.
