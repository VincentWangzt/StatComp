# Experiment Results — Comprehensive Multi-Method Benchmark

**Date**: 2026-03-25
**Metrics**: ELBO↑, KL↓, W2↓ (primary for targets with baselines); ELBO↑, KSD↓ (for data-dependent); BNN RMSE↓, NLL↓ (for Bnn_boston)

---

## Banana (z_dim=2)

| Method | Anneal | ELBO | KL↓ | W2↓ | KSD | MMD | EpTime | Epochs |
|--------|--------|------|-----|-----|-----|-----|--------|--------|
| SIVI | on | -0.55 | 0.548 | 1.292 | 0.197 | 0.065 | 0.017s | 20K |
| KSIVI | off | ★div | 10.49 | 5434 | ★div | 0.135 | 0.007s | 50K |
| UIVI | on | 0.002 | 0.029 | 0.536 | 0.005 | 0.002 | 0.091s | 10K |
| RSIVI | on | -414 | 7.92 | 9.99 | 1692 | 0.145 | 0.048s | 10K |
| AISIVI | on | **1.56** | **0.009** | **0.244** | -0.007 | **0.002** | 0.031s | 10K |
| DSIVI | on | 2.96 | 0.012 | 0.326 | 0.003 | 0.002 | **0.011s** | 10K |
| DSIVI | off | 0.54 | 0.014 | 0.283 | 0.007 | **0.001** | **0.010s** | 10K |

## Multimodal (z_dim=2)

| Method | Anneal | ELBO | KL↓ | W2↓ | KSD | MMD | EpTime | Epochs |
|--------|--------|------|-----|-----|-----|-----|--------|--------|
| SIVI | on | -0.21 | 0.234 | 0.330 | 0.049 | 0.025 | 0.017s | 20K |
| KSIVI | on | ★div | 5.24 | 1042 | 5.39 | 0.527 | 0.007s | 50K |
| UIVI | on | -0.03 | 0.052 | 0.161 | 0.014 | 0.007 | 0.094s | 10K |
| RSIVI | on | -0.25 | 0.272 | 0.514 | 0.069 | 0.029 | 0.052s | 10K |
| AISIVI | on | **0.03** | **-0.002** | **0.030** | **-0.001** | **0.002** | 0.033s | 10K |
| DSIVI | on | 0.01 | -0.002 | 0.049 | 0.001 | 0.001 | **0.010s** | 10K |
| DSIVI | off | 0.22 | 0.032 | 0.068 | 0.001 | 0.002 | **0.010s** | 10K |

## X-Shaped (z_dim=2)

| Method | Anneal | ELBO | KL↓ | W2↓ | KSD | MMD | EpTime | Epochs |
|--------|--------|------|-----|-----|-----|-----|--------|--------|
| SIVI | on | 0.005 | 0.020 | 0.047 | -0.001 | 0.002 | 0.017s | 20K |
| KSIVI | off | ★div | 12.15 | 18.9M | ★div | 0.076 | 0.007s | 50K |
| UIVI | on | -0.09 | 0.091 | 0.278 | 0.006 | 0.007 | 0.099s | 10K |
| RSIVI | — | ★crash | — | — | — | — | — | — |
| AISIVI | — | ★crash | — | — | — | — | — | — |
| DSIVI | on | 0.05 | **0.013** | **0.037** | -0.003 | **0.002** | **0.011s** | 10K |
| DSIVI | off | **0.61** | **0.012** | 0.051 | -0.0002 | **0.002** | **0.011s** | 10K |

## Student-UC (z_dim=2)

| Method | Anneal | ELBO | KL↓ | W2↓ | KSD | MMD | EpTime | Epochs |
|--------|--------|------|-----|-----|-----|-----|--------|--------|
| SIVI | on | -2.70 | **0.016** | **0.051** | 0.028 | **0.002** | 0.016s | 20K |
| KSIVI | on | -2.52 | 0.094 | 0.138 | 0.075 | 0.003 | 0.005s | 50K |
| UIVI | on | -2.73 | 0.040 | 0.236 | 0.087 | 0.003 | 0.094s | 10K |
| RSIVI | — | ★crash | — | — | — | — | — | — |
| AISIVI | on | -2.71 | 0.025 | 0.049 | 0.056 | 0.003 | 0.030s | 10K |
| DSIVI | off | -2.47 | 0.009* | 0.132* | 0.021 | 0.001 | **0.009s** | 10K |

*DSIVI student_uc: 10K with noanneal only. Missing anneal variant.

## Langevin_post (z_dim=100)

| Method | Anneal | ELBO | KL↓ | W2↓ | KSD | MMD | EpTime | Epochs |
|--------|--------|------|-----|-----|-----|-----|--------|--------|
| SIVI | on | -176 | 17.5 | 0.035 | 0.503 | 0.023 | 0.047s | 20K |
| KSIVI | off | ★div | 318 | 75.3M | ★div | 0.004 | 0.011s | 100K |
| UIVI | on | **-77.4** | **5.43** | **0.007** | **0.014** | **0.002** | 0.090s | 10K |
| RSIVI | on | -180 | 22.5 | 0.033 | 0.707 | 0.021 | 0.615s | 10K |
| AISIVI | on | 63.3 | 8.95 | 0.074 | 1.011 | 0.004 | 0.620s | 10K |
| DSIVI | on | -75.2 | 7.00 | 0.008 | 0.094 | 0.002 | 0.021s | 10K |
| DSIVI | off | 44.4 | 7.37 | 0.010 | 0.097 | 0.002 | **0.022s** | 10K |

## LRwaveform (z_dim=22, no KL/W2 baseline)

| Method | Anneal | ELBO↑ | KSD↓ | Fisher | EpTime | Epochs |
|--------|--------|-------|------|--------|--------|--------|
| SIVI | on | -33.5 | 1.78 | 5922 | 0.022s | 20K |
| KSIVI | — | ★crash (config) | — | — | — | — |
| UIVI | on | **-24.2** | **0.031** | 95852 | 0.097s | 10K |
| RSIVI | — | ★crash (RealNVP) | — | — | — | — |
| AISIVI | — | ★crash (RealNVP) | — | — | — | — |
| DSIVI | off | -56.4 | 134 | 75933 | **0.011s** | 2K |

## Bnn_boston (z_dim=751, no KL/W2 baseline)

| Method | Anneal | ELBO | KSD | RMSE↓ | NLL↓ | EpTime | Epochs |
|--------|--------|------|-----|-------|------|--------|--------|
| SIVI | on | -1201 | -0.37 | 5.63 | 3.41 | 0.032s | 20K |
| KSIVI | — | ★crash (config) | — | — | — | — | — |
| UIVI | on | **-915** | 21.6 | 5.26 | 3.43 | 0.115s | 10K |
| RSIVI | — | not run | — | — | — | — | — |
| AISIVI | — | not run | — | — | — | — | — |
| DSIVI | on | -4432 | 116 | 3.60 | **2.69** | 0.115s | 10K |
| DSIVI | off | -4331 | 101 | **3.53** | **2.68** | 0.124s | 10K |

---

## Summary Rankings

### By KL↓ (toy 2D + Langevin)

| Target | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| Banana | AISIVI (0.009) | DSIVI-on (0.012) | DSIVI-off (0.014) |
| Multimodal | AISIVI/DSIVI-on (~0) | DSIVI-off (0.032) | UIVI (0.052) |
| X-Shaped | DSIVI-off (0.012) | DSIVI-on (0.013) | SIVI (0.020) |
| Student-UC | DSIVI-off (0.009) | SIVI (0.016) | AISIVI (0.025) |
| Langevin | UIVI (5.43) | DSIVI-on (7.00) | DSIVI-off (7.37) |

### By ELBO↑ (data-dependent)

| Target | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| LRwaveform | UIVI (-24.2) | SIVI (-33.5) | DSIVI (-56.4) |
| Bnn_boston | UIVI (-915) | SIVI (-1201) | DSIVI-off (-4331) |

### By BNN prediction (Bnn_boston)

| Target | 1st RMSE | 1st NLL |
|--------|----------|---------|
| Bnn_boston | DSIVI-off (**3.53**) | DSIVI-off (**2.68**) |

---

## Key Findings

1. **DSIVI is consistently top-2 on toy 2D** targets (by KL), and top-2 on Langevin_post
2. **AISIVI is strongest on banana/multimodal** but crashes on x_shaped/student_uc/LRwaveform (RealNVP instability)
3. **KSIVI diverges numerically** on 5/7 targets — only student_uc works; needs investigation
4. **RSIVI RealNVP crashes** on 4/7 targets — unreliable
5. **UIVI is the most robust baseline** — works on all targets, best on Langevin/LRwaveform ELBO
6. **DSIVI epoch time is 5-10x faster** than UIVI on toy 2D (0.01s vs 0.09s)
7. **DSIVI dominates on BNN prediction** — RMSE 3.53 vs UIVI 5.26 (33% better), NLL 2.68 vs 3.43 (22% better)
8. **DSIVI anneal vs off**: Similar on most targets; annealing gives marginal KL improvement on banana/multimodal

## Crashes/Failures Summary

| Method | Targets that crash | Root cause |
|--------|-------------------|------------|
| KSIVI | banana, multimodal, x_shaped, Langevin_post, LRwaveform, Bnn_boston | Numerical divergence (KSD loss) / config issues |
| RSIVI | banana (collapse), x_shaped, student_uc, LRwaveform | RealNVP sampling failure |
| AISIVI | x_shaped, LRwaveform | RealNVP sampling failure |
| SIVI | Bnn_boston (OOM with default reverse_sample_num=4096) | Fixed with reduced to 512 |
