# UIVI Quick Reference

## What is UIVI?
- **Unbiased Semi-Implicit Variational Inference**
- Uses HMC to sample epsilon conditional on z
- No learned reverse model (stateless)
- Minimizes ELBO with HMC-sampled auxiliary variables

## Key Algorithm: HMC Sampling
```python
# For each z:
# 1. Initialize epsilon from training data
# 2. Burn-in steps of HMC (discard samples)
# 3. Collect M HMC samples
# 4. Use for score estimation

z_aux, eps_aux, acc_rate = sample_epsilon_hmc(
    z, eps_init,
    num_samples=M,
    burn_in_steps=5,
    step_size=0.1,
    leapfrog_steps=10
)
```

## Configuration
```yaml
hmc:
  step_size: 0.1
  leapfrog_steps: 10
  burn_in_steps: 5

train:
  reverse_sample_num: 10      # M: HMC samples to collect
  batch_size: 1024
```

## HMC Parameter Tuning
```
Acceptance rate targets: 0.6 - 0.8

If acc_rate < 0.6:
  - Decrease step_size (too large steps)
  - Or increase leapfrog_steps (better proposals)

If acc_rate > 0.8:
  - Increase step_size (too small steps)
  - Or decrease leapfrog_steps
```

## TensorBoard Metrics
```
train/vi_model/loss
train/reverse_model/hmc_accept_rate  ← HMC-specific!
metric/vi_model/{kl_ite, w2, elbo, ksd, mmd}
metric/reverse_model/{ksd, kl_ite, w2}
diagnostic/reverse_model/avg_epsilon_distance
```

## Checkpoint Format
```
checkpoints/best/
  ├── vi_model.pt
  ├── vi_optim.pt
  └── vi_sched.pt
(No HMC state - stateless MCMC)
```

## When to Use UIVI
✓ Adaptive epsilon sampling  
✓ No learned model to train  
✓ Stateless MCMC advantages  
✗ High computational cost (HMC expensive)  
✗ Requires tuning HMC parameters

## Key Hyperparameters
- **step_size**: Control proposal step (adjust for 0.65 acceptance)
- **leapfrog_steps**: More → better mixing but slower
- **burn_in_steps**: Typically 5-10, larger if poor mixing
- **reverse_sample_num**: More samples → better estimate but slower

## Usage
```python
runner = UIVIRunner(config=config)
runner.train()  # HMC happens internally
```
Check `train/reverse_model/hmc_accept_rate` to monitor sampling quality.
