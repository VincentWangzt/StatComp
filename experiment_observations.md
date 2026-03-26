# Experiment Observations — Comprehensive Benchmark

---

## Phase 3: Multi-Method Benchmark (2026-03-25)

### What was run
- 6 methods × 7 targets = 42 intended runs
- 34 completed successfully, 8 crashed (KSIVI divergence, RealNVP failures, OOM, config)

### Method Reliability

| Method | Success rate | Failure modes |
|--------|------------|---------------|
| SIVI | 6/7 (86%) | OOM on Bnn_boston with default config; fixed with smaller reverse_sample_num |
| KSIVI | 1/7 (14%) | Numerical divergence on most targets; config issues on data-dependent |
| UIVI | 7/7 (100%) | No failures |
| RSIVI | 3/7 (43%) | RealNVP crashes on x_shaped, student_uc, LRwaveform |
| AISIVI | 4/7 (57%) | RealNVP crashes on x_shaped, LRwaveform |
| DSIVI | 7/7 (100%) | No failures (with rev2 config) |

**UIVI and DSIVI are the only methods that work on all 7 targets.**

### DSIVI Performance Summary

**Toy 2D (by KL):** Top-1 or top-2 on all 4 targets. Competitive with AISIVI where AISIVI doesn't crash.

**Langevin_post (100D):** KL 7.0 (anneal on), close to UIVI (5.4). Both DSIVI variants beat SIVI (17.5) and RSIVI (22.5).

**LRwaveform (22D):** ELBO -56.4, worse than UIVI (-24.2). The rev2 configuration with only 2K epochs may be undertrained. Needs more epochs.

**Bnn_boston (751D):** Dominates on BNN prediction (RMSE 3.53 vs UIVI 5.26). ELBO is unreliable (broken as seen in Phase 1).

### DSIVI Speed Advantage

| Target dim | DSIVI ep time | UIVI ep time | Speedup |
|-----------|---------------|--------------|---------|
| 2D | 0.010s | 0.092s | **9.2x** |
| 100D | 0.021s | 0.090s | **4.3x** |
| 22D | 0.011s | 0.097s | **8.8x** |
| 751D | 0.115s | 0.115s | 1.0x |

DSIVI with rev2 is 4-9x faster than UIVI on low-to-mid dimensional targets. On Bnn_boston (751D), the speed is similar because the VI model forward pass dominates.

### KSIVI Analysis

KSIVI diverges on banana, multimodal, x_shaped, Langevin_post. The ELBO/KSD/Fisher values become astronomical (10^15 to 10^25). Only student_uc converges (KL 0.094).

**Hypothesis:** The Gaussian kernel KSD estimator has variance issues when the target score has large norms. The student_uc target has bounded scores, which is why it works. The other targets may have score magnitudes that destabilize the kernel computation. The `detach_kernel=false` setting may also contribute by allowing gradient through the kernel bandwidth, leading to degenerate bandwidth selection.

### RSIVI/AISIVI RealNVP Issues

The ConditionalRealNVP reverse model produces NaN/Inf samples on x_shaped, student_uc, and LRwaveform. This happens both with default and larger batch sizes. The RealNVP architecture may need better conditioning or regularization for these target geometries.

### Missing/TODO
- [x] DSIVI LRwaveform needs more epochs → done (10K: ELBO -24.3)
- [x] KSIVI Bnn_boston/LRwaveform config fix → done (missing metric sub-keys fixed)
- [ ] KSIVI Langevin_post 50K, LRwaveform 10K, Bnn_boston 10K reruns (crashed initially, re-running now)
- [ ] RSIVI/AISIVI Bnn_boston (not attempted — RealNVP likely to crash)

---

## Phase 4: KSIVI Half-Epoch Reruns + New BNN Targets (2026-03-26)

### What was run
- KSIVI half-epoch reruns: 7 original targets with halved epochs (25K toy/LR/BNN, 50K Langevin)
- New BNN targets (concrete, power, protein, winered, yacht): SIVI/UIVI/DSIVI × anneal on/off × 10K/20K
- KSIVI on new BNN targets: 10K/20K/30K (pending re-run after config fix)

### Infrastructure changes
- Added 5 new UCI BNN regression datasets to pipeline
- Fixed KSIVI configs for high-dim targets (missing `num_samples`/`num_z_samples` metric sub-keys caused silent crashes)

### KSIVI 25K vs 50K: half-epoch is NOT sufficient for toy 2D

| Target | KL (50K) | KL (25K) | Δ |
|--------|----------|----------|---|
| banana | **0.061** | 0.148 | 2.4× worse |
| multimodal | **~0** | 0.015 | acceptable |
| x_shaped | **0.001** | 0.031 | 31× worse |
| student_uc | 0.032 | **0.015** | better at 25K |

25K is clearly insufficient for banana and x_shaped. Recommendation: keep 50K for toy 2D.

### New BNN Targets: DSIVI annealing pattern

A clear pattern emerges across the 5 new targets:

**Annealing ON wins on**: yacht (N=308), power (N=9568), concrete (N=1030) — smaller/medium datasets.
**Annealing OFF wins on**: protein (N=45730) — large dataset, more stable gradient signal.
**UIVI wins on**: winered (N=1599) — all methods within 5%, DSIVI doesn't dominate.

**Hypothesis**: On small datasets, the posterior is well-concentrated and annealing helps DSIVI learn the reverse model before the target becomes too sharp. On large datasets (protein), stochastic gradients provide sufficient noise that annealing is not needed.

### DSIVI speed on new BNN targets

| Target (z_dim) | DSIVI ep | UIVI ep | Speedup |
|----------------|----------|---------|---------|
| yacht (401) | 0.119s | 0.095s | 0.8× (UIVI faster!) |
| concrete (501) | 0.140s | 0.094s | 0.7× |
| power (301) | 0.094s | 0.095s | ~1× |
| protein (551) | 0.163s | 0.093s | 0.6× |
| winered (651) | 0.175s | 0.091s | 0.5× |

**Important observation**: Unlike Bnn_boston where DSIVI matched UIVI speed, on these new targets DSIVI is actually *slower* per epoch. The new BNN targets have fewer training samples (protein aside), so the VI forward pass is cheaper but DSIVI still pays the reverse model cost. The 9× speedup observed on toy 2D does not carry over to BNN targets beyond boston.

### Missing/TODO — ALL COMPLETE ✅
- [x] KSIVI on new BNN targets 10K/20K/30K — diverges on all 5 (KSD explodes to ±millions/billions)
- [x] KSIVI Langevin_post 50K — diverges (KSD=3531), halving epochs makes no difference
- [x] KSIVI LRwaveform 10K — broken (KSD=1283), worse than 20K
- [x] KSIVI Bnn_boston 10K — broken ELBO, RMSE=12.16 (better than 20K's 142.4 but still far behind DSIVI)
- [ ] DSIVI on new BNN targets with 20K anneal-on for winered/power (optional, low priority)

---
