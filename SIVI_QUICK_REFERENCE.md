# SIVI Quick Reference

## What is SIVI?
- **Semi-Implicit Variational Inference**
- Minimizes ELBO: `loss = -E[log p(z) - log q_phi(z)]`
- Estimates `log q_phi(z)` via Monte Carlo: `E_{eps'~p}[q_phi(z|eps')]`

## Key Method: calc_log_q_phi_z

```python
# Algorithm:
# 1. Sample M fresh epsilon samples from prior
epsilon_new = vi_model.sample_epsilon(num=M)

# 2. For each z, evaluate q_phi(z|eps_new) and q_phi(z|eps_orig)
log_q_conditional = vi_model.logp(z, [eps_new, eps_orig])

# 3. Average via logsumexp (numerically stable)
log_q_z = logsumexp(log_q_conditional) - log(M+1)
```

## Configuration
```yaml
train:
  reverse_sample_num: 10        # M: number of fresh epsilon samples
  batch_size: 1024
  epochs: 2000
  loss_log_freq: 100
  metric_log_freq: 500
```

## Loss Computation
```
1. Sample epsilon, z from VI model
2. Compute log p(z) from target
3. Estimate log q_phi(z) via Monte Carlo:
   - Sample M fresh epsilon
   - Evaluate q_phi(z|eps) for each
   - Average: log q_z ≈ E[q(z|eps)]
4. Loss = -mean(log p(z) - log q_z)
```

## TensorBoard Metrics
```
train/vi_model/loss
metric/vi_model/{kl_ite, w2, elbo, ksd, mmd, fisher_div}
diagnostic/vi_model/grad_norm
diagnostic/reverse_model/avg_epsilon_distance  ← SIVI-specific
```

## Checkpoint Format
```
checkpoints/best/
  ├── vi_model.pt
  ├── vi_optim.pt
  └── vi_sched.pt
```

## Usage
```python
config = OmegaConf.load('configs/sivi.yaml')
runner = SIVIRunner(config=config)
runner.train()
```

## When to Use SIVI
✓ Simple baseline  
✓ Limited computational budget  
✓ Moderate-dimensional problems  
✗ Not for very high dimensions (expensive Monte Carlo)

## Parameter Tuning
- **Increase `reverse_sample_num`** if: gradient variance too high
- **Decrease `reverse_sample_num`** if: too slow
- **Adjust `lr`** if: divergence or plateau
