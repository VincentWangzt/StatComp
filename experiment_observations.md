# Experiment Observations — DSIVI on BNN (Bnn_boston)

Running log of experiments, observations, and hypotheses.

---

## 2026-03-24: Experiment Round

### Context
- Reworked metric system (Fisher divergence, BNN RMSE/NLL, restructured TensorBoard)
- All results below are from this round only

### Step 0: UIVI Baselines — COMPLETED ✅

| Run | Annealing | Epochs | Best ELBO | BNN RMSE | BNN NLL |
|-----|-----------|--------|-----------|----------|---------|
| UIVI-A | 5000 | 10K | -909.81 | 4.86 | 3.42 |
| UIVI-B | 1000 | 5K | -1072.66 | 4.85 | 3.39 |

### Step 1: Annealing Study — COMPLETED ✅

**Key finding**: Shorter annealing (500 steps) is optimal for DSIVI. Longer annealing causes ELBO estimation breakdown.

| Anneal | ELBO Stable? | nan onset | Best ELBO |
|--------|-------------|-----------|-----------|
| 500 | ✅ | never | -1095 (10K) |
| 1000 | ❌ | epoch 590 | -1383 |
| 5000 | ❌ | epoch 800 | -1367 |

**Root cause**: During annealing, the VI distribution shifts. With longer annealing, the shift is more gradual but the reverse model must track it for longer. The VI's variance collapses in some dimensions, causing numerical issues in the ELBO estimator (log q(z) underestimates). With anneal=500, the transition completes quickly and the VI stabilizes.

### Step 2: Reverse Model Training — COMPLETED ✅

**Key finding**: 10 reverse inner epochs per main step is necessary for stability. Fewer (2) causes ELBO breakdown.

| Rev Epochs | ELBO Stable? | Best ELBO | Best BNN |
|-----------|-------------|-----------|----------|
| 10 (anneal=500) | ✅ | -1095 | 4.46 |
| 2 (anneal=500) | ❌ nan@600 | -1308 | **3.25** |

### Step 3: LR Schedule & Long Training — IN PROGRESS

**Key finding**: LR decay is the primary bottleneck beyond 10K epochs. Standard StepLR(2000, 0.7) reduces LR too fast.

| LR Step | Epochs | Best ELBO | Peak Epoch | Gap to UIVI-A |
|---------|--------|-----------|------------|---------------|
| 2000 | 20K | -999 | 14950 | 89 |
| 4000 | 20K | **-940** | 19440 | **30** |
| 5000 | 30K | -1059 | 11520 | — (running) |
| 6000 | 30K | — | — | — (running) |

**LR comparison at key epochs**:

| Epoch | step=2000 LR | step=4000 LR | step=5000 LR |
|-------|-------------|-------------|-------------|
| 10K | 0.000168 | 0.000490 | 0.000700 |
| 15K | 0.000058 | 0.000240 | 0.000490 |
| 20K | 0.000020 | 0.000118 | 0.000240 |
| 25K | — | — | 0.000118 |
| 30K | — | — | 0.000058 |

The step=5000 run maintains 4x higher LR at epoch 15K and 12x higher at epoch 20K compared to step=2000.

### Step 4: Next Directions

**If slowlr-30K or veryslow-30K reach below -940:**
- Try even longer training (40K+ epochs) with step=8000
- Try cosine annealing instead of StepLR
- Try higher base LR (0.002) with appropriate warmup

**If neither improves beyond -940:**
- The gap to UIVI-A (-910) may be inherent to DSIVI vs UIVI
- Try different reverse model architecture (wider/deeper)
- Try different VI model architecture

**Interesting open questions:**
1. Why does DSIVI find better BNN modes but worse ELBO in some configurations?
2. Can we combine the best of both — DSIVI's BNN quality with better ELBO?
3. Is the -910 UIVI-A ELBO a ceiling or does UIVI also improve with more epochs?

---
