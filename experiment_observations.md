# Experiment Observations — DSIVI on BNN (Bnn_boston)

Running log of experiments, observations, and hypotheses.

---

## 2026-03-24: Experiment Round Start

### Context
- Reworked metric system (Fisher divergence, BNN RMSE/NLL, restructured TensorBoard)
- Previous best DSIVI ELBO: -1210 at epoch 6040 (not directly comparable — old metrics)
- Previous UIVI baseline: -1165 at 5K epochs (not directly comparable — old metrics)
- All results below are from this round only

### Step 0: UIVI Baselines (re-establishing with new metrics)

| Run | GPU | Config | Annealing | Epochs | Status |
|-----|-----|--------|-----------|--------|--------|
| UIVI-A | cuda:0 | uivi_Bnn_boston.yaml | linear, 5000 steps | 10,000 | Launched |
| UIVI-B | cuda:1 | uivi_Bnn_boston.yaml | linear, 1000 steps | 5,000 | Launched |

**UIVI-A**: Standard long annealing baseline — matches the default uivi_Bnn_boston config but with KSD and Fisher enabled.

**UIVI-B**: Short annealing baseline — same epoch count as previous DSIVI best, tests whether shorter annealing helps UIVI too.

---
