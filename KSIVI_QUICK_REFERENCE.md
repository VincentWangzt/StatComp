# KSIVI Quick Reference

## What is KSIVI?
- **Kernel Semi-Implicit Variational Inference**
- Minimizes KSD²: `KSD²(q_phi || p) = E[k(x,x') * <f,f>]`
- Uses adaptive kernel bandwidth
- No need to estimate log q_phi(z)

## Key Loss Computation
```python
# 1. Sample z1, z2 from VI model (independent)
z1, s1 = vi_model(eps1)
z2, s2 = vi_model(eps2)

# 2. Compute combined scores
f1 = anneal * score_target(z1) + score_implicit(z1)
f2 = anneal * score_target(z2) + score_implicit(z2)

# 3. Kernel matrix (adaptive bandwidth)
K = kernel.pair_eval(z1, z2, fit_h=True)

# 4. KSD² loss
loss = mean(K * (f1 @ f2.T))
```

## Configuration
```yaml
ksivi:
  statistic: 'v'              # 'v' (V-stat) or 'u' (U-stat)
  kernel: 'gaussian'          # 'gaussian', 'imq', 'laplace', 'riesz'
  detach_kernel: True         # Don't backprop through kernel
  log_p_reg: 0.0              # Optional log p(z) regularization
  log_p_reg_mode: 'warmup_only'

pretrain:
  enabled: False
  steps: 1000
  lr: 1.0e-3
```

## Statistic Choices
- **V-statistic**: Two independent batches (unbiased, default)
- **U-statistic**: Single batch with diagonal zeroed (biased but lower variance)

## TensorBoard Metrics
```
train/vi_model/loss
metric/vi_model/{kl_ite, w2, elbo, ksd, mmd, fisher_div}
ksivi/kernel_bandwidth          ← Adaptive kernel parameter
ksivi/score_product_mean        ← Kernel effectiveness
pretrain/vi_model/loss          ← If pretraining enabled
```

## Checkpoint Format
```
checkpoints/best/
  ├── vi_model.pt
  ├── vi_optim.pt
  └── vi_sched.pt
```

## When to Use KSIVI
✓ Avoid marginal density estimation  
✓ Kernel-based objective  
✓ BNN targets (with pretraining)  
✗ When ELBO interpretation needed

## Pretraining (for BNNs)
```yaml
ksivi:
  pretrain:
    enabled: true
    steps: 5000
    lr: 1.0e-3
    batch_size: 256
```
Minimizes MSE on dev set before main training.

## Tuning
- **Kernel choice**: gaussian → imq (long-range) → laplace (robust)
- **log_p_reg**: Small value (0.01-0.1) biases toward high-density
- **detach_kernel**: Usually True (more stable)
