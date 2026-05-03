# SIVI Runner Comprehensive Analysis

## 1. SIVI Training Overview

### What is SIVI?
**Semi-Implicit Variational Inference (SIVI)** is the baseline variational inference method in this codebase that:
- Uses a VI model `q_phi(z|epsilon)` to encode noise `epsilon` into latent samples `z`
- Estimates the marginal density `log q_phi(z)` via Monte Carlo averaging
- Uses implicit score functions from the reparameterization trick

### Key Inheritance Chain
```
BaseSIVIRunner
    ↓
SIVIRunner (runner/sivi.py)
```

---

## 2. SIVI Loss Computation

### Training Loss
The SIVI training optimizes the **ELBO-based objective**:

```python
loss = -E_q[log p(z) - log q_phi(z)]
```

**Detailed Steps (lines 1211-1249 in base_runner.py):**

1. **Sample epsilon** from VI model: `epsilon ~ p(epsilon)`
2. **Forward through VI model** to get samples: `z = g_phi(epsilon)`
3. **Compute target log-density** using score-gradient trick:
   ```python
   log_prob_target = target_model.score(z) * z  # [Batch]
   log_prob_target = log_prob_target.sum(dim=-1)
   ```
4. **Apply optional annealing** for warm-up phase
5. **Estimate marginal log q_phi(z)** via `calc_log_q_phi_z()` ← **SIVI-specific**
6. **Compute ELBO loss**:
   ```python
   loss = -mean(log_prob_target - log_q_phi_z)
   ```

### SIVI-Specific: Marginal Density Estimation

**Location:** `sivi.py` lines 20-81 (`calc_log_q_phi_z`)

The key to SIVI is how it estimates the marginal density `log q_phi(z)`:

```python
# Step 1: Sample fresh epsilon samples (not the training epsilon)
epsilon_new = vi_model.sample_epsilon(num=training_reverse_sample_num)
# Shape: [num_samples, epsilon_dim]

# Step 2: Replicate z to create auxiliary samples
# Shape: [batch_size, num_samples, z_dim]
z_aux = z.unsqueeze(1).repeat(1, num_samples, 1)

# Step 3: Include original epsilon in auxiliary set (plus fresh samples)
# Shape: [batch_size, num_samples + 1, z_dim]
epsilon_aux = torch.cat([epsilon_new.repeat(batch_size, 1, 1), 
                          epsilon.unsqueeze(1)], dim=1)

# Step 4: Evaluate conditional densities
# Shape: [batch_size, num_samples + 1]
log_q_conditional = vi_model.logp(z_aux, epsilon_aux)

# Step 5: Average via logsumexp (numerically stable)
log_q_phi_z = torch.logsumexp(log_q_conditional, dim=1) \
              - torch.log(torch.tensor(num_samples + 1))
```

**Mathematical Interpretation:**
```
log q_phi(z) ≈ log E_{epsilon' ~ p(epsilon)}[q_phi(z|epsilon')]
            = log (1/(M+1)) * sum_{i=1}^{M+1} q_phi(z|epsilon_i)
            = log sum_i - log(M+1)  [computed via logsumexp]
```

### Why Include Original Epsilon?
The algorithm includes the **original training epsilon** plus `M` fresh samples (lines 47, 50-54):
- Gives importance to the actual noise that generated this z
- More stable when `M` is small
- Total: `M + 1` samples for averaging

### Diagnostics Logged
```python
# Average distance between auxiliary epsilons and original epsilon
avg_eps_distance = torch.mean(torch.norm(epsilon_aux - epsilon.unsqueeze(1), dim=-1))
```
Logged to TensorBoard: `diagnostic/reverse_model/avg_epsilon_distance`

---

## 3. SIVI Checkpoint System

### Checkpoint Directory Structure
```
results/SIVI/{target_type}/{timestamp}/
├── checkpoints/
│   ├── best/
│   │   ├── vi_model.pt           ✓ Best model (lowest KL)
│   │   ├── vi_optim.pt           ✓ Optimizer state
│   │   └── vi_sched.pt           ✓ Scheduler state
│   ├── latest/
│   │   ├── vi_model.pt           ✓ Most recent
│   │   ├── vi_optim.pt
│   │   └── vi_sched.pt
│   └── epoch_{N}/
│       └── [same structure]
├── samples/
│   ├── samples_epoch_500.pt
│   ├── samples_epoch_1000.pt
│   └── ...
├── plots/
│   ├── contour_epoch_500.png
│   ├── contour_epoch_1000.png
│   └── ...
├── tensorboard/
│   └── events.out.tfevents...
└── config.yaml
```

### Checkpoint Timing
- **Best model**: Saved when KL divergence improves (best-based tracking)
- **Latest model**: Saved after every epoch
- **Periodic checkpoints**: Saved every `ckpt_freq` epochs (default: 1000)
- **Resumable**: Contains optimizer & scheduler state

### Loading Checkpoint
```python
checkpoint = torch.load(checkpoint_path)
vi_model.load_state_dict(checkpoint['vi_model_state_dict'])
optimizer.load_state_dict(checkpoint['vi_optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['vi_scheduler_state_dict'])
```

---

## 4. SIVI TensorBoard Metrics

### Training Loss (every `loss_log_freq` epochs)
```
train/vi_model/loss              ← Main ELBO loss
```

### Evaluation Metrics (every `metric_log_freq` epochs)
```
metric/vi_model/{
    kl_ite,                      ← KL divergence (ITE baseline)
    w2,                          ← Wasserstein-2 distance
    elbo,                        ← ELBO value
    elbo_std_total,              ← ELBO uncertainty
    elbo_std_q,                  ← ELBO q-estimation uncertainty
    elbo_ci_half,                ← ELBO 95% CI half-width
    ksd,                         ← Kernel Stein Discrepancy
    fisher_div,                  ← Fisher divergence
    mmd,                         ← Maximum Mean Discrepancy
    rmse,                        ← BNN RMSE (if applicable)
    test_llk,                    ← Test log-likelihood (if applicable)
    nll                          ← Negative log-likelihood (if applicable)
}
```

### Diagnostics (every epoch)
```
diagnostic/vi_model/{
    grad_norm,                   ← Gradient norm (stability indicator)
    z_norm_avg,                  ← Average norm of sampled z
    z_norm_std,                  ← Std dev of z norms
    epsilon_norm_avg,            ← Average norm of epsilon
    epsilon_norm_std,            ← Std dev of epsilon norms
    marginal_conditional_score_l2_gap,
    kde_expected_log_marginal_std,
    kde_expected_log_marginal_clamped_dims
}
diagnostic/reverse_model/avg_epsilon_distance  ← SIVI-specific
```

### Timing (every epoch)
```
time/{vi_sample, neg_score, backward, ...}
time_avg/{key}  ← Moving average over time_avg_window
```

---

## 5. SIVI Configuration Parameters

### Required Parameters
```yaml
train:
  epochs: 2000                    # Total training epochs
  batch_size: 1024                # VI training batch size
  
  loss_log_freq: 100             # Log loss every N epochs
  metric_log_freq: 500           # Evaluate metrics every N epochs
  ckpt_freq: 1000                # Save checkpoint every N epochs
  
  reverse_sample_num: 10         # M: number of fresh epsilon samples
  
  optimizer:
    lr: 1.0e-3                   # Learning rate
    betas: [0.9, 0.999]          # Adam betas
    weight_decay: 0.0
  
  scheduler:
    type: 'cosine'               # LR scheduler type
    total_steps: ${train.epochs}
```

### Optional Parameters
```yaml
train:
  grad_clip: 1.0                 # Gradient clipping (None = disabled)
  warmup_epochs: 100             # Warm-up phase
  annealing:
    enabled: true
    scheme: 'linear'             # 'linear' or 'sqrt'
    steps: 200
```

### VI Model Parameters
```yaml
vi_model:
  epsilon_dim: 3                 # Dimension of epsilon
  z_dim: ${target.z_dim}         # Dimension of z
  hidden_dim: 10                 # Hidden layer size
  num_layers: 2                  # Number of hidden layers
  activation: 'relu'             # 'relu' or 'silu'
  variance_parameterization: 'logvar'  # 'logvar' or 'softplus_var'
```

### Logging Parameters
```yaml
logging:
  tensorboard_dir: './runs'
  log_freq: 1
  save_samples: true
```

---

## 6. SIVI Computational Flow

### Forward Pass
```
epsilon ~ p(epsilon)
    ↓
[VI Model: MLP]
    ↓
z = mu(epsilon) + sigma(epsilon) * u
    ↓
score = u / sigma  (implicit score)
```

### Loss Computation
```
Sample epsilon_train
    ↓
Forward: z = g_phi(epsilon_train)
    ↓
Evaluate log p(z) using target
    ↓
For each z, sample M fresh epsilon_new
    ↓
Compute log q(z|epsilon_new) for each new epsilon
    ↓
Average via logsumexp: log q_phi(z) ≈ E[q(z|eps)]
    ↓
ELBO = log p(z) - log q_phi(z)
    ↓
Loss = -ELBO
```

---

## 7. SIVI Code Implementation

### Class Definition
```python
class SIVIRunner(BaseSIVIRunner):

    def __init__(
        self,
        config: DictConfig,
        name: str = "SIVI",
    ):
        super().__init__(config=config, name=name)
        self.reverse_model_type: str = 'prior q(epsilon)'
        self.training_reverse_sample_num = self.training_cfg.reverse_sample_num
```

**Key Attributes:**
- `reverse_model_type`: String identifier ('prior q(epsilon)')
- `training_reverse_sample_num`: Number M of fresh epsilon samples

### Main Method: calc_log_q_phi_z

**Location:** `runner/sivi.py`, lines 20-81

```python
def calc_log_q_phi_z(
    self,
    z: torch.Tensor,        # [batch_size, z_dim]
    epsilon: torch.Tensor,  # [batch_size, epsilon_dim]
) -> torch.Tensor:          # [batch_size]
    
    # Sample fresh epsilon samples (Monte Carlo)
    epsilon_new = self.vi_model.sample_epsilon(
        num=self.training_reverse_sample_num)  # [M, epsilon_dim]

    # Tile z to match fresh samples
    epsilon_aux = epsilon_new.repeat(z.shape[0], 1, 1)  # [B, M, epsilon_dim]

    # Concatenate with original epsilon
    epsilon_aux = torch.cat(
        [epsilon_aux, epsilon.unsqueeze(1)], dim=1)  # [B, M+1, epsilon_dim]

    # Tile z for all auxiliary epsilon
    z_aux = z.unsqueeze(1).repeat(
        1, self.training_reverse_sample_num + 1, 1)  # [B, M+1, z_dim]

    # Evaluate conditional log-densities
    log_q_phi_z_given_epsilon = self.vi_model.logp(z_aux, epsilon_aux)  # [B, M+1]

    # Average via logsumexp (numerically stable)
    log_q_phi_z = torch.logsumexp(
        log_q_phi_z_given_epsilon, dim=1
    ) - torch.log(
        torch.tensor(
            self.training_reverse_sample_num + 1,
            device=z.device,
        ))

    # Log diagnostic: average epsilon distance
    avg_eps_distance = torch.mean(
        torch.norm(
            epsilon_aux - epsilon.unsqueeze(1),
            dim=-1,
        )).item()
    self.writer.add_scalar(
        "diagnostic/reverse_model/avg_epsilon_distance",
        avg_eps_distance,
        self.curr_epoch,
    )

    return log_q_phi_z  # [batch_size]
```

---

## 8. Comparison with Other Runners

| Aspect | SIVI | KSIVI | UIVI | DSIVI/AISIVI/RSIVI |
|--------|------|-------|------|-------------------|
| Loss Type | ELBO | KSD² | ELBO | ELBO |
| Marginal Density | Monte Carlo | N/A | HMC | Learned model |
| Reverse Model | None | None | HMC | Neural network |
| Checkpoint Frequency | Best + Latest | Best + Latest | Best + Latest | Periodic only |
| Computational Cost | Low-Medium | Medium | Medium-High | High |
| Gradient Stability | Good | Good | Good | Can be unstable |

---

## 9. Practical Usage Guide

### Basic Initialization
```python
from omegaconf import OmegaConf
from runner.sivi import SIVIRunner

config = OmegaConf.load('configs/sivi.yaml')
runner = SIVIRunner(config=config)
```

### Training Loop
```python
runner.pretrain_vi()              # Optional pretraining
runner.train()                    # Main training
```

### Custom Training
```python
for epoch in range(num_epochs):
    # Forward pass
    epsilon = runner.vi_model.sample_epsilon(num=batch_size)
    z, neg_score = runner.vi_model.forward(epsilon)
    
    # Compute loss
    loss_dict = runner._compute_loss_and_step(epoch)
    
    # Evaluate metrics
    if epoch % metric_log_freq == 0:
        runner.evaluate_metrics(epoch)
```

### Loading and Resuming
```python
checkpoint = torch.load('checkpoints/best/vi_model.pt')
runner.vi_model.load_state_dict(checkpoint['model_state_dict'])
runner.optimizer_vi.load_state_dict(checkpoint['optimizer_state_dict'])
runner.train()  # Resume from checkpoint
```

---

## 10. Key Formulas

### Marginal Density Estimation
```
log q_phi(z) = log E_{epsilon' ~ p(epsilon)}[q_phi(z|epsilon')]

Estimated as:
log_q_hat = logsumexp([log q(z|eps_1), ..., log q(z|eps_M), log q(z|eps_orig)])
          - log(M + 1)
```

### ELBO Loss
```
ELBO = E[log p(z) - log q_phi(z)]
Loss = -ELBO = -E[log p(z) - log q_phi(z)]
```

### Gradient Flow
```
z = mu(eps) + sigma(eps) * u
score_implicit = u / sigma = -∇_z log q(z|eps)
```

---

## 11. Debugging & Troubleshooting

### Common Issues

**Issue: NaN gradients**
- Solution: Check `grad_norm` in TensorBoard
- Likely cause: unstable log q_phi estimation
- Fix: Increase `reverse_sample_num`

**Issue: Slow convergence**
- Solution: Check learning rate and annealing schedule
- Likely cause: Suboptimal epsilon dimension
- Fix: Adjust `reverse_sample_num` or learning rate

**Issue: Mode collapse**
- Solution: Monitor `z_norm_avg` and `z_norm_std`
- Likely cause: Insufficient exploration
- Fix: Increase variance in VI model

---

## Summary

SIVI is the baseline SIVI variant that:
1. Uses Monte Carlo averaging for marginal density estimation
2. Requires sampling `M+1` auxiliary epsilon samples per training step
3. Provides a simple, interpretable baseline
4. Has moderate computational cost
5. Works well for moderate-dimensional problems
