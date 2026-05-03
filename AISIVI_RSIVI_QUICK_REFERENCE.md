# AISIVI & RSIVI Quick Reference

## What are AISIVI & RSIVI?
- **Learned reverse model variants** (unlike SIVI, KSIVI, UIVI)
- Use neural network to learn score function
- Train reverse model via score matching

## AISIVI: Importance-Weighted
```python
# Score estimation via importance sampling
score_pred = reverse_model.score(z)

# Reverse model loss
loss_reverse = mean((score_pred - score_true)^2)

# Training loss uses importance-weighted density:
log_q = logsumexp(log_q_conditional + log_weights)
log_q_z = log_q - log(M)
```

## RSIVI: Direct Averaging
```python
# Same reverse model training
score_pred = reverse_model.score(z)
loss_reverse = mean((score_pred - score_true)^2)

# Training loss uses direct score averaging:
score_avg = mean(score_samples)  # Simple average
log_q_z = sum(score_avg * z)
```

## AISIVI vs RSIVI

| Feature | AISIVI | RSIVI |
|---------|--------|-------|
| Score Est | Importance-weighted | Direct average |
| Complexity | Higher | Lower |
| Robustness | Better | Needs good reverse |
| Speed | Slower | Faster |
| Use Case | Production | Research |

## Configuration
```yaml
train:
  epochs: 2000
  batch_size: 1024
  loss_log_freq: 100
  metric_log_freq: 500
  
  reverse:
    type: 'mlp'           # 'mlp' or 'gaussian'
    lr: 1.0e-3
    train_steps: 10       # Inner training steps
    sample_num: 10        # M: auxiliary samples

reverse_model:
  hidden_dim: 128
  num_layers: 3
  activation: 'silu'
```

## TensorBoard Metrics
```
train/vi_model/loss
train/reverse_model/loss         ← MSE of score matching

metric/reverse_model/{
  ksd,                          ← Reverse model quality
  kl_ite,
  w2
}

diagnostic/reverse_model/{
  avg_score_norm,
  avg_epsilon_distance,
  score_l2_to_target           ← Alignment with target
}
```

## Checkpoint Format
```
checkpoints/epoch_N/
  ├── vi_model.pt
  ├── vi_optim.pt
  ├── vi_sched.pt
  ├── reverse_model.pt         ✓ Always saved
  ├── reverse_optim.pt         ✓ If use_optimizer=True
  └── reverse_sched.pt         ✓ If scheduler configured
```

**Key: Periodic only (no best model tracking)**

## When to Use

**AISIVI**:
✓ Production code  
✓ Robustness to reverse model errors  
✓ High-dimensional problems  

**RSIVI**:
✓ Research/comparison  
✓ Speed critical  
✓ Well-trained reverse model

## Training Dynamics
```
1. VI model trained on ELBO
2. Reverse model trained on score matching (MSE)
3. Inner loop: vi_step + reverse_step
4. Diagnostics: Check score_l2_to_target
```

## Tuning
- **Increase `reverse:train_steps`** if reverse not converging
- **Increase `reverse:lr`** if reverse training slow
- **Increase `reverse:sample_num`** if gradients noisy
- Monitor `diagnostic/reverse_model/score_l2_to_target` → should decrease

## Safety Features
**AISIVI**:
- Importance weight clipping (max=10.0)
- NaN checks on weights
- Extensive error handling

**RSIVI**:
- Finite score checking
- Zero-count detection
- Simpler error handling
