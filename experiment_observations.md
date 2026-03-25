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
- [ ] DSIVI student_uc with annealing on (only noanneal was run with 10K)
- [ ] DSIVI LRwaveform needs more epochs (only 2K due to config default)
- [ ] KSIVI Bnn_boston/LRwaveform config fix
- [ ] RSIVI/AISIVI Bnn_boston (not attempted — RealNVP likely to crash)

---
