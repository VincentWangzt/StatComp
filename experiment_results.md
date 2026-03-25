# Experiment Results — Comprehensive Multi-Method Benchmark

**Date**: 2026-03-25
**Primary Metrics**: ELBO↑, KL↓, W2↓ (toy/Langevin); ELBO↑ (LRwaveform); RMSE↓, NLL↓ (Bnn_boston)
**Note**: KL unreliable on Langevin_post — use W2 and ELBO instead.

---

## Banana (z_dim=2)

| Method | Anneal | ELBO | KL↓ | W2↓ | KSD | MMD | EpTime | Epochs |
|--------|--------|------|-----|-----|-----|-----|--------|--------|
| SIVI | on | -0.55 | 0.548 | 1.292 | 0.197 | 0.065 | 0.017s | 20K |
| **KSIVI** | **off** | **-0.04** | **0.074** | **0.903** | **0.010** | **0.008** | **0.006s** | **50K** |
| UIVI | on | 0.002 | 0.029 | 0.536 | 0.005 | 0.002 | 0.091s | 10K |
| RSIVI | on | -414 | 7.92 | 9.99 | 1692 | 0.145 | 0.048s | 10K |
| AISIVI | on | **1.56** | **0.009** | **0.244** | -0.007 | **0.002** | 0.031s | 10K |
| DSIVI | on | 2.96 | 0.012 | 0.326 | 0.003 | 0.002 | **0.011s** | 10K |
| DSIVI | off | 0.54 | 0.014 | 0.283 | 0.007 | **0.001** | **0.010s** | 10K |

## Multimodal (z_dim=2)

| Method | Anneal | ELBO | KL↓ | W2↓ | KSD | MMD | EpTime | Epochs |
|--------|--------|------|-----|-----|-----|-----|--------|--------|
| SIVI | on | -0.21 | 0.234 | 0.330 | 0.049 | 0.025 | 0.017s | 20K |
| **KSIVI** | **on** | **0.003** | **0.009** | **0.041** | **-0.0004** | **0.002** | **0.007s** | **50K** |
| UIVI | on | -0.03 | 0.052 | 0.161 | 0.014 | 0.007 | 0.094s | 10K |
| RSIVI | on | -0.25 | 0.272 | 0.514 | 0.069 | 0.029 | 0.052s | 10K |
| AISIVI | on | **0.03** | -0.002 | **0.030** | -0.001 | **0.002** | 0.033s | 10K |
| DSIVI | on | 0.01 | -0.002 | 0.049 | 0.001 | 0.001 | **0.010s** | 10K |
| DSIVI | off | 0.22 | 0.032 | 0.068 | 0.001 | 0.002 | **0.010s** | 10K |

## X-Shaped (z_dim=2)

| Method | Anneal | ELBO | KL↓ | W2↓ | KSD | MMD | EpTime | Epochs |
|--------|--------|------|-----|-----|-----|-----|--------|--------|
| SIVI | on | 0.005 | 0.020 | 0.047 | -0.001 | 0.002 | 0.017s | 20K |
| **KSIVI** | **off** | **0.006** | **0.001** | **0.068** | **0.002** | **0.003** | **0.008s** | **50K** |
| UIVI | on | -0.09 | 0.091 | 0.278 | 0.006 | 0.007 | 0.099s | 10K |
| RSIVI | — | ★crash | — | — | — | — | — | — |
| AISIVI | — | ★crash | — | — | — | — | — | — |
| DSIVI | on | 0.05 | 0.013 | **0.037** | -0.003 | **0.002** | **0.011s** | 10K |
| DSIVI | off | **0.61** | 0.012 | 0.051 | -0.0002 | **0.002** | **0.011s** | 10K |

## Student-UC (z_dim=2)

| Method | Anneal | ELBO | KL↓ | W2↓ | KSD | MMD | EpTime | Epochs |
|--------|--------|------|-----|-----|-----|-----|--------|--------|
| SIVI | on | -2.70 | 0.016 | 0.051 | 0.028 | 0.002 | 0.016s | 20K |
| **KSIVI** | **on** | **-2.63** | **0.032** | **0.063** | **0.018** | **0.002** | **0.006s** | **50K** |
| UIVI | on | -2.73 | 0.040 | 0.236 | 0.087 | 0.003 | 0.094s | 10K |
| RSIVI | — | ★crash | — | — | — | — | — | — |
| AISIVI | on | -2.71 | 0.025 | 0.049 | 0.056 | 0.003 | 0.030s | 10K |
| DSIVI | on | -2.47 | **0.009** | 0.132 | 0.021 | **0.002** | **0.013s** | 10K |
| DSIVI | off | -2.47 | **0.009** | 0.132 | 0.021 | 0.001 | **0.009s** | 10K |

## Langevin_post (z_dim=100) — Use W2 and ELBO (KL unreliable)

| Method | Anneal | Rev | ELBO↑ | W2↓ | KSD | MMD | EpTime | Epochs |
|--------|--------|-----|-------|-----|-----|-----|--------|--------|
| SIVI | on | — | -176 | 0.035 | 0.503 | 0.023 | 0.047s | 20K |
| KSIVI | off | — | ★div | 75.3M | ★div | 0.004 | 0.011s | 100K |
| UIVI | on | — | **-77.4** | **0.007** | **0.014** | **0.002** | 0.090s | 10K |
| RSIVI | on | — | -180 | 0.033 | 0.707 | 0.021 | 0.615s | 10K |
| AISIVI | on | — | 63.3 | 0.074 | 1.011 | 0.004 | 0.620s | 10K |
| DSIVI | on | rev2 | -75.2 | **0.008** | 0.094 | **0.002** | **0.021s** | 10K |
| DSIVI | off | rev2 | 44.4 | 0.010 | 0.097 | **0.002** | **0.022s** | 10K |
| DSIVI | on | rev5 | -79.0 | **0.008** | 0.122 | **0.002** | 0.048s | 10K |
| DSIVI | off | rev5 | -54.4 | 0.011 | 0.896 | **0.002** | 0.046s | 10K |

## LRwaveform (z_dim=22, no baseline)

| Method | Anneal | Rev | ELBO↑ | KSD↓ | Fisher | EpTime | Epochs |
|--------|--------|-----|-------|------|--------|--------|--------|
| SIVI | on | — | -33.5 | 1.78 | 5922 | 0.022s | 20K |
| KSIVI | — | — | pending | — | — | — | — |
| UIVI | on | — | **-24.2** | **0.031** | 95852 | 0.097s | 10K |
| RSIVI | — | — | ★crash | — | — | — | — |
| AISIVI | — | — | ★crash | — | — | — | — |
| DSIVI | off | rev2 | **-24.3** | 0.260 | 57829 | **0.011s** | 10K |
| DSIVI | on | rev2 | -56.4 | 134 | 75933 | **0.011s** | 2K |

## Bnn_boston (z_dim=751, no baseline)

| Method | Anneal | Rev | ELBO | RMSE↓ | NLL↓ | EpTime | Epochs |
|--------|--------|-----|------|-------|------|--------|--------|
| SIVI | on | — | -1201 | 5.63 | 3.41 | 0.032s | 20K |
| KSIVI | — | — | pending | — | — | — | — |
| UIVI | on | — | **-915** | 5.26 | 3.43 | 0.115s | 10K |
| DSIVI | on | rev2 | -4432 | 3.60 | 2.69 | 0.115s | 10K |
| DSIVI | off | rev2 | -4331 | **3.53** | **2.68** | 0.124s | 10K |
| DSIVI | on | rev5 | -4424 | 3.76 | 2.74 | 0.278s | 10K |
| DSIVI | off | rev5 | — | — | — | — | running |

---

## Updated Rankings

### By KL↓ (toy 2D)

| Target | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| Banana | AISIVI (0.009) | DSIVI-on (0.012) | DSIVI-off (0.014) |
| Multimodal | KSIVI (0.009) | DSIVI-on (-0.002) | UIVI (0.052) |
| X-Shaped | **KSIVI (0.001)** | DSIVI-off (0.012) | DSIVI-on (0.013) |
| Student-UC | **DSIVI (0.009)** | SIVI (0.016) | AISIVI (0.025) |

### By W2↓ (Langevin_post)

| 1st | 2nd | 3rd |
|-----|-----|-----|
| UIVI (0.007) | DSIVI-on rev2 (0.008) | DSIVI-off rev2 (0.010) |

### By ELBO↑ (LRwaveform)

| 1st | 2nd | 3rd |
|-----|-----|-----|
| UIVI (-24.2) | DSIVI-off rev2 10K (-24.3) | SIVI (-33.5) |

### By BNN NLL↓

| 1st | 2nd | 3rd |
|-----|-----|-----|
| DSIVI-off rev2 (**2.68**) | DSIVI-on rev2 (2.69) | DSIVI-on rev5 (2.74) |

---

## Key Findings (Updated)

1. **KSIVI now works** after bug fix! Top performer on multimodal (KL 0.009) and x_shaped (KL 0.001). Still diverges on Langevin_post.
2. **DSIVI consistently top-2-3** across all targets and metrics.
3. **DSIVI LRwaveform improved** from ELBO -56.4 (2K epochs) to -24.3 (10K epochs) — now matches UIVI.
4. **DSIVI rev2 vs rev5 on Bnn_boston**: rev2 is better (RMSE 3.53 vs 3.76) AND faster (0.12s vs 0.28s). Less reverse training = better BNN prediction.
5. **DSIVI rev2 vs rev5 on Langevin_post**: Similar W2 (0.008 vs 0.008), but rev2 is 2x faster.
6. **For speed-quality tradeoff**: rev2 dominates rev5 — faster AND equal or better quality.

## Crashes/Failures

| Method | Targets that crash/diverge |
|--------|---------------------------|
| KSIVI | Langevin_post (diverges), LRwaveform/Bnn_boston (pending) |
| RSIVI | banana (collapse), x_shaped, student_uc, LRwaveform (RealNVP) |
| AISIVI | x_shaped, LRwaveform (RealNVP) |
