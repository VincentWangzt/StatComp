# DSIVI Quick Reference

## What is DSIVI?
- **Denoising Semi-Implicit Variational Inference**
- Uses a **learned reverse denoising model** to estimate the score function
- Optimizes the ELBO: `loss = -E[log p(z) - log q_phi(z)]`

## Key Files
- `runner/dsivi.py` - Main DSIVI runner implementation
- `runner/base_reverse_runner.py` - Base class for reverse model training
- `runner/base_runner.py` - Core SIVI infrastructure

## Loss Computation
1. Sample `epsilon ~ p(epsilon)` and `z = g_phi(epsilon)`
2. Compute `log p(z)` using target model score
3. **Estimate `log q_phi(z)` using reverse denoising model** ← DSIVI difference
4. Loss = `-mean(log p(z) - log q_phi(z))`

## Reverse Model Loss
- MSE between predicted and true scores:
  ```
  score_pred = reverse_model.score(z)
  score_true = vi_model.score(z, epsilon)
  loss_reverse = mean((score_pred - score_true)^2)
  ```

## Checkpoint Format
```
checkpoints/epoch_{N}/
├── vi_model.pt           ✓ Always saved
├── vi_optim.pt           ✓ Always saved
├── vi_sched.pt           ✓ Always saved
├── reverse_model.pt      ✓ Always saved
├── reverse_optim.pt      ✓ If use_optimizer=True
└── reverse_sched.pt      ✓ If scheduler configured
```

## Checkpoint Timing
- **Periodic only**: Saved every `ckpt_freq` epochs (default: 1000)
- **No best checkpoint**: Only periodic saves
- **Resumable**: Contains optimizer & scheduler state

## TensorBoard Metrics

### Training Loss (every `loss_log_freq` epochs)
```
train/vi_model/loss              ← Main ELBO loss
train/reverse_model/loss         ← Reverse model MSE loss
```

### Evaluation Metrics (every `metric_log_freq` epochs)
```
metric/vi_model/{kl_ite, w2, elbo, fisher_div, ksd, mmd}
metric/reverse_model/{ksd, kl_ite, w2}     ← DSIVI-specific
```

### Diagnostics (every epoch)
```
diagnostic/vi_model/{grad_norm, z_norm_avg, z_norm_std, ...}
diagnostic/reverse_model/{avg_score_norm, score_l2_to_target, ...}
```

### Timing (every epoch)
```
time/{vi_sample, neg_score, backward, reverse_train, kl_estimation, ...}
time_avg/{key}  ← Moving average over time_avg_window
```

## Common Targets
- **Toy**: `banana`, `8_gaussians`, `multimodal`, `x_shaped`, `student_uc`
- **Real Data**: `Bnn_boston`, `Bnn_concrete`, `Bnn_power`, `Bnn_protein`, `Bnn_winered`, `Bnn_yacht`
- **Posterior**: `Langevin_post`, `LRwaveform`

## Important Config Parameters
```yaml
train:
  epochs: 2000                    # Training epochs
  batch_size: 1024                # VI batch size
  reverse:
    lr: 1.0e-3
    batch_size: 2048              # Reverse model batch size
    epochs: 10                     # Inner epochs per update
    update_freq: 1                 # Update every N outer epochs
  
  log:
    metric_log_freq: 10            # Expensive metrics every N epochs
    loss_log_freq: 100             # Log loss every N epochs
    reverse_log_freq: 500          # Log reverse loss every N epochs
  
  checkpoint:
    enabled: true
    freq: 1000                     # Checkpoint every 1000 epochs
  
  sample:
    freq: 500                      # Save samples every 500 epochs
    num: 10000                     # Number of samples to save
```

## Evaluation
- **During training**: Metrics logged periodically to TensorBoard
- **Post-hoc**: `finalization/runner_eval.py` loads final checkpoint and re-evaluates
  - Loads only VI model (not reverse model)
  - Computes ELBO, W2, KDE ELM, BNN metrics with higher precision
  - Reports standard errors across seeds

## What's NOT Saved
❌ Reverse model is not used during evaluation (only VI model)
❌ Training samples (can be reconstructed from checkpoint)
❌ Best checkpoint tracking

## Key Metrics
| Metric | What | When |
|--------|------|------|
| ELBO | Evidence Lower Bound | Training + Evaluation |
| W2 | Sliced Wasserstein-2 | Training + Evaluation |
| KL | KL divergence to baseline | Training (if baseline exists) |
| KSD | Kernelized Stein Discrepancy | Training (if enabled) |
| Fisher | Fisher divergence | Training (if enabled) |
| BNN RMSE | Test error (BNN targets) | Training + Evaluation |
| NLL | Negative test log-likelihood (BNN) | Training + Evaluation |

## Typical Training Timeline
1. **VI pretraining** (BNN targets only): 100 steps on dev split
2. **Reverse model warmup** (optional): Few epochs
3. **Main training loop**: For each epoch:
   - VI model update (1 step)
   - Reverse model training (every `update_freq` epochs)
   - Evaluate metrics (every `metric_log_freq` epochs)
   - Save samples (every `sample_freq` epochs)
   - Save checkpoint (every `ckpt_freq` epochs)

---

For detailed analysis, see `DSIVI_ANALYSIS.md`
