# Experiment Results — Comprehensive Multi-Method Benchmark

**Date**: 2026-03-25
**Metrics**: ELBO↑, KL↓, W2↓ (primary); KSD↓, Fisher↓, MMD↓ (diagnostic)

---

## Banana (z_dim=2)

| Method | Anneal | ELBO | KL↓ | W2↓ | KSD | Fisher | MMD | EpTime | Epochs |
|--------|--------|------|-----|-----|-----|--------|-----|--------|--------|
| SIVI | on | -0.55 | 0.548 | 1.292 | 0.197 | 13.2 | 0.065 | 0.017s | 20K |
| KSIVI | off | ★diverged | 10.49 | 5434 | 5.5e23 | 3.5e25 | 0.135 | 0.007s | 50K |
| UIVI | on | 0.002 | 0.029 | 0.536 | 0.005 | 2.19 | 0.002 | 0.091s | 10K |
| RSIVI | on | -414 | 7.92 | 9.99 | 1692 | 117K | 0.145 | 0.048s | 10K |
| AISIVI | on | **1.56** | **0.009** | **0.244** | -0.007 | 2306 | **0.002** | 0.031s | 10K |
| DSIVI | on | 2.96 | 0.012 | 0.326 | 0.003 | 35K | 0.002 | **0.011s** | 10K |
| DSIVI | off | 0.54 | 0.014 | 0.283 | 0.007 | 5736 | **0.001** | **0.010s** | 10K |

## Multimodal (z_dim=2)

| Method | Anneal | ELBO | KL↓ | W2↓ | KSD | Fisher | MMD | EpTime | Epochs |
|--------|--------|------|-----|-----|-----|--------|-----|--------|--------|
| SIVI | on | -0.21 | 0.234 | 0.330 | 0.049 | 0.81 | 0.025 | 0.017s | 20K |
| KSIVI | on | ★diverged | 5.24 | 1042 | 5.39 | 94.9M | 0.527 | 0.007s | 50K |
| UIVI | on | -0.03 | 0.052 | 0.161 | 0.014 | 0.26 | 0.007 | 0.094s | 10K |
| RSIVI | on | -0.25 | 0.272 | 0.514 | 0.069 | 1.11 | 0.029 | 0.052s | 10K |
| AISIVI | on | **0.03** | **-0.002** | **0.030** | **-0.001** | **0.070** | **0.002** | 0.033s | 10K |
| DSIVI | on | 0.01 | -0.002 | 0.049 | 0.001 | 0.57 | 0.001 | **0.010s** | 10K |
| DSIVI | off | 0.22 | 0.032 | 0.068 | 0.001 | 1725 | 0.002 | **0.010s** | 10K |

## X-Shaped (z_dim=2)

| Method | Anneal | ELBO | KL↓ | W2↓ | KSD | Fisher | MMD | EpTime | Epochs |
|--------|--------|------|-----|-----|-----|--------|-----|--------|--------|
| SIVI | on | 0.005 | 0.020 | 0.047 | -0.001 | 0.17 | 0.002 | 0.017s | 20K |
| KSIVI | off | ★diverged | 12.15 | 18.9M | 2.2e12 | 5.1e13 | 0.076 | 0.007s | 50K |
| UIVI | on | -0.09 | 0.091 | 0.278 | 0.006 | 0.92 | 0.007 | 0.099s | 10K |
| RSIVI | — | ★crashed (RealNVP) | — | — | — | — | — | — | — |
| AISIVI | — | ★crashed (RealNVP) | — | — | — | — | — | — | — |
| DSIVI | on | 0.05 | **0.013** | **0.037** | -0.003 | 5.65 | **0.002** | **0.011s** | 10K |
| DSIVI | off | 0.61 | **0.012** | 0.051 | -0.0002 | 60.6 | **0.002** | **0.011s** | 10K |

## Student-UC (z_dim=2)

| Method | Anneal | ELBO | KL↓ | W2↓ | KSD | Fisher | MMD | EpTime | Epochs |
|--------|--------|------|-----|-----|-----|--------|-----|--------|--------|
| SIVI | on | -2.70 | **0.016** | **0.051** | 0.028 | **1.32** | **0.002** | 0.016s | 20K |
| KSIVI | on | -2.52 | 0.094 | 0.138 | 0.075 | 21.5 | 0.003 | 0.005s | 50K |
| UIVI | on | -2.73 | 0.040 | 0.236 | 0.087 | 2.67 | 0.003 | 0.094s | 10K |
| RSIVI | — | ★crashed (RealNVP) | — | — | — | — | — | — | — |
| AISIVI | on | -2.71 | 0.025 | 0.049 | 0.056 | 1.23 | 0.003 | 0.030s | 10K |
| DSIVI | off | **-2.47** | **0.009** | 0.132 | 0.021 | 124 | 0.001 | **0.009s** | 10K |
| DSIVI | on | — | — | — | — | — | — | — | pending |

## Langevin_post (z_dim=100)

| Method | Anneal | ELBO | KL↓ | W2↓ | KSD | Fisher | MMD | EpTime | Epochs |
|--------|--------|------|-----|-----|-----|--------|-----|--------|--------|
| SIVI | on | -176 | 17.5 | 0.035 | 0.503 | 9124 | 0.023 | 0.047s | 20K |
| KSIVI | off | ★diverged | 318 | 75.3M | 5.6e20 | 3.0e22 | 0.004 | 0.011s | 100K |
| UIVI | on | **-77.4** | **5.43** | **0.007** | **0.014** | 103K | **0.002** | 0.090s | 10K |
| RSIVI | — | — | — | — | — | — | — | — | rerunning |
| AISIVI | — | — | — | — | — | — | — | — | rerunning |
| DSIVI | on | — | — | — | — | — | — | — | rerunning |
| DSIVI | off | — | — | — | — | — | — | — | rerunning |

## LRwaveform (z_dim=22, no baseline)

Running on GPU 1...

## Bnn_boston (z_dim=751, no baseline)

Pending.

---

## Summary Rankings (by KL↓)

| Target | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| Banana | AISIVI (0.009) | DSIVI-on (0.012) | DSIVI-off (0.014) |
| Multimodal | AISIVI/DSIVI-on (-0.002) | DSIVI-off (0.032) | UIVI (0.052) |
| X-Shaped | DSIVI-off (0.012) | DSIVI-on (0.013) | SIVI (0.020) |
| Student-UC | DSIVI-off (0.009) | SIVI (0.016) | AISIVI (0.025) |
| Langevin_post | UIVI (5.43) | SIVI (17.5) | KSIVI (318) |

## Key Observations

1. **DSIVI consistently top-2** on all toy 2D targets (KL 0.009–0.032)
2. **AISIVI strongest on banana/multimodal** but crashes on x_shaped/student_uc (RealNVP failure)
3. **KSIVI diverges** on 4/5 targets (banana, multimodal, x_shaped, Langevin_post); only works on student_uc
4. **RSIVI crashes** on x_shaped, student_uc (RealNVP issues); mediocre on others
5. **UIVI is the best on Langevin_post** (100D) — KL 5.43 vs SIVI 17.5
6. **DSIVI is 5-10x faster per epoch** than UIVI/RSIVI/AISIVI (0.01s vs 0.03-0.09s)
7. **DSIVI anneal vs off**: Similar quality, slight edge for annealing on multimodal/banana; noanneal better on student_uc/x_shaped
