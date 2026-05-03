# AISIVI & RSIVI Runner Comprehensive Analysis

## 1. Overview

### AISIVI: Adaptive Importance-Weighted SIVI
Uses a learned reverse model with **importance-weighted** score estimation:

```
log q_phi(z) = log E_{eps' ~ q_psi(eps|z)} [
    q_phi(z|eps') * q(eps') / q_psi(eps'|z)
]
```

### RSIVI: Reverse Score SIVI
Uses a learned reverse model with **direct score averaging** (simpler):

```
∇_z log q_phi(z) ≈ E_{eps' ~ q_psi(eps|z)} [∇_z log q_phi(z|eps')]
```

### Key Inheritance Chain
```
BaseSIVIRunner
    ↓
BaseReverseConditionalRunner
    ├─ AISIVIRunner (runner/aisivi.py)
    └─ RSIVIRunner (runner/rsivi.py)
```

---

## 2. AISIVI Loss Computation

### Main Difference from SIVI
**SIVI**: Uses samples from prior `p(epsilon)`

**AISIVI**: Uses samples from learned reverse model `q_psi(epsilon | z)` with importance weighting

### Algorithm

**Location:** `aisivi.py` lines 77-205 (`calc_log_q_phi_z`)

```python
# Step 1: Sample from reverse model (conditional on z)
# Shape: [B, M, Dz], [B, M, De], [B, M]
z_aux, epsilon_aux, log_q_psi_epsilon_given_z = reverse_model.sample(
    z, num_samples=M)

# Step 2: Compute importance weights
# log w_i = log q(eps_i) - log q_psi(eps_i | z)
log_q_epsilon = vi_model.log_q_epsilon(epsilon_aux)  # [B, M]
log_w = log_q_epsilon - log_q_psi_epsilon_given_z    # [B, M]
w = exp(log_w)  # Importance weights

# Step 3: Clip importance weights (stability)
log_w = log_w.clamp(max=10.0)

# Step 4: Compute weighted conditional densities
# log q_phi(z|eps_i) * w_i
log_q_phi_z_aux = vi_model.logp(z_aux, epsilon_aux) + log_w  # [B, M]

# Step 5: Logsumexp (numerically stable averaging)
log_q_phi_z = torch.logsumexp(log_q_phi_z_aux, dim=1) - log(M)

# Step 6: Backprop through z to get score
score = ∇_z log_q_phi_z

# Step 7: Approximate log q_phi(z) ≈ score · z
log_q_phi_z_approx = torch.sum(score * z, dim=-1)
```

### Importance Weight Clipping
```python
importance_sampling_weights = log_w.clamp(max=10.0)
```
Prevents gradients from exploding if reverse model is poorly calibrated.

### Error Handling
Multiple levels of safety checks (lines 95-132):
1. Try-except around reverse model sampling
2. Check for NaN in log_q_epsilon
3. Check for NaN in log_q_psi_epsilon_given_z
4. Check for non-finite importance weights
5. Check for fully invalid samples

---

## 3. RSIVI Loss Computation

### Main Difference from AISIVI
**AISIVI**: Importance-weighted averaging (more robust but complex)

**RSIVI**: Direct score averaging (simpler, more efficient)

### Algorithm

**Location:** `rsivi.py` lines 88-160 (`calc_log_q_phi_z`)

```python
# Step 1: Sample from reverse model
z_aux, epsilon_aux, _ = reverse_model.sample(z, num_samples=M)

# Step 2: Compute scores directly (no importance weights!)
# ∇_z log q_phi(z|eps_i)
score_samples = vi_model.score(z_aux, epsilon_aux)  # [B, M, Dz]

# Step 3: Filter finite samples
finite_mask = torch.isfinite(score_samples).all(dim=-1)  # [B, M]
valid_counts = finite_mask.sum(dim=1)  # [B]

# Step 4: Safe averaging (zero out invalid entries)
safe_score_samples = torch.where(
    finite_mask.unsqueeze(-1),
    score_samples,
    zeros_like(score_samples)
)
score = safe_score_samples.sum(dim=1) / valid_counts  # [B, Dz]

# Step 5: Approximate log q_phi(z) ≈ score · z
log_q_phi_z = torch.sum(score * z, dim=-1)
```

### Key Differences
1. **No importance weighting** (simpler gradients)
2. **No clipping needed** (working with scores, not log-weights)
3. **Faster computation** (fewer operations)
4. **Potentially less stable** if reverse model far from truth

---

## 4. Reverse Model Training

### Shared Base: BaseReverseConditionalRunner

**Location:** `runner/base_reverse_runner.py`

```python
class BaseReverseConditionalRunner(BaseSIVIRunner):
    def train_reverse_model(self, epoch_outer: int):
        """Train reverse model via score matching"""
        for inner_step in range(self.train_reverse_steps):
            # Sample z from VI model
            _, z_samples = vi_model.sampling(num=batch_size)
            
            # True score: ∇_z log q_phi(z|eps)
            score_true = vi_model.score(z_samples, epsilon_samples)
            
            # Predicted score: reverse model prediction
            score_pred = reverse_model.score(z_samples)
            
            # MSE loss
            loss = torch.mean((score_pred - score_true)**2)
            
            # Gradient update
            loss.backward()
            reverse_optimizer.step()
```

### Reverse Model Types
1. **MLPDenoiseModel**: Neural network-based (learned parameters)
2. **GaussianReverse**: Closed-form Gaussian solution (fitted)

---

## 5. AISIVI vs RSIVI Comparison

| Aspect | AISIVI | RSIVI |
|--------|--------|-------|
| Score Estimation | Importance-weighted | Direct averaging |
| Formula | log q using logsumexp | Direct score mean |
| Complexity | Higher | Lower |
| Gradient Flow | Through weights & scores | Through scores only |
| Robustness | Better if reverse model poor | Requires good reverse |
| Computational Cost | Higher (extra clamp/checks) | Lower |
| Error Handling | Extensive | Basic (finite checks) |
| Use Case | Production | Research/comparison |

---

## 6. AISIVI Reverse KSD

**Location:** `aisivi.py` lines 20-69 (`calculate_rev_KSD`)

```python
def calculate_rev_KSD(self):
    # Sample z from VI model
    _, z_samples = vi_model.sampling(num=n_ksd)
    
    # Sample epsilon from reverse model
    z_aux, epsilon_aux, log_q_psi_epsilon_given_z = (
        reverse_model.sample(z_samples, num_samples=M)
    )
    
    # Importance weights
    log_w = vi_model.log_q_epsilon(epsilon_aux) - log_q_psi_epsilon_given_z
    log_w = log_w.clamp(max=10.0)
    
    # Weighted log density
    log_q = vi_model.logp(z_aux, epsilon_aux) + log_w
    
    # Marginal estimate
    log_q_z = torch.logsumexp(log_q, dim=1) - log(M)
    
    # Gradient for score
    score = torch.autograd.grad(log_q_z.sum(), z_aux)[0]
    score = score.sum(dim=1)
    
    # KSD
    return compute_ksd(z_samples, score)
```

---

## 7. RSIVI Reverse KSD

**Location:** `rsivi.py` lines 49-80 (`calculate_rev_KSD`)

```python
def calculate_rev_KSD(self):
    # Sample z
    _, z_samples = vi_model.sampling(num=n_ksd)
    
    # Sample epsilon from reverse model
    z_aux, epsilon_aux, _ = reverse_model.sample(
        z_samples, num_samples=M
    )
    
    # Compute scores directly
    score = _estimate_reverse_score(z_aux, epsilon_aux, "evaluation")
    
    # KSD
    return compute_ksd(z_samples, score)
```

---

## 8. Configuration Parameters

### Shared Parameters
```yaml
train:
  epochs: 2000
  batch_size: 1024
  loss_log_freq: 100
  metric_log_freq: 500
  
  reverse:
    type: 'mlp'           # 'mlp' or 'gaussian'
    lr: 1.0e-3
    batch_size: 1024
    train_steps: 10       # Inner steps per epoch
    sample_num: 10        # M: auxiliary samples
```

### Reverse Model Types
```yaml
# MLPDenoiseModel
reverse_model:
  type: 'mlp'
  hidden_dim: 128
  num_layers: 3
  activation: 'silu'

# GaussianReverse
reverse_model:
  type: 'gaussian'
  # No hyperparameters (closed-form fit)
```

---

## 9. Checkpoint System

### Files Saved
```
checkpoints/epoch_N/
├── vi_model.pt
├── vi_optim.pt
├── vi_sched.pt
├── reverse_model.pt           ✓ Always
├── reverse_optim.pt           ✓ If use_optimizer=True
└── reverse_sched.pt           ✓ If scheduler configured
```

### Checkpoint Timing
- **Periodic only**: Every `ckpt_freq` epochs (default: 1000)
- **No best model tracking** (only periodic saves)
- **Resumable**: Full state for both VI and reverse models

---

## 10. Diagnostics & Metrics

### AISIVI/RSIVI-Specific Metrics
```
metric/reverse_model/{
    ksd,              ← Reverse model KSD
    kl_ite,
    w2
}

diagnostic/reverse_model/{
    avg_epsilon_distance,    ← How far samples are from original
    avg_score_norm,          ← Score magnitude
    norm_of_avg_score,       ← Magnitude of mean score
    score_l2_to_target       ← Alignment with target score
}

train/reverse_model/{
    loss,             ← MSE between predicted and true scores
    steps
}
```

---

## 11. Computational Comparison

### Tokens per Training Step

**SIVI**:
- VI forward: 1 sample z
- Density est: M+1 forward evals
- Total density evals: O(M)

**AISIVI**:
- VI forward: 1 sample z
- Reverse sample: M epsilon samples
- Importance weight: M evals
- Density evals: M (same as SIVI but with weights)
- Plus: Reverse model training M samples

**RSIVI**:
- VI forward: 1 sample z
- Reverse sample: M epsilon samples
- Score eval: M evals (no importance weights)
- Plus: Reverse model training M samples

---

## 12. Practical Recommendations

### Use AISIVI if:
- Reverse model may be poorly trained
- Robustness to reverse model errors important
- Extra computation acceptable

### Use RSIVI if:
- Reverse model is well-trained
- Want maximum speed
- Simplicity preferred

### For BNNs:
- Both work well
- AISIVI more stable during early training
- RSIVI better for final quality

---

## 13. Error Handling

### AISIVI Safety Checks
```python
# 1. Reverse sampling failure
try:
    z_aux, eps_aux, log_q_psi = reverse_model.sample(z, M)
except RuntimeError:
    return NaN  # Skip update

# 2. Check for NaN in intermediate computations
if torch.isnan(log_q_epsilon).any():
    logger.debug("NaN in log_q_epsilon")

# 3. Check importance weights
if torch.isnan(importance_sampling_weights).any():
    return NaN  # Skip

# 4. Check logsumexp input
if not torch.isfinite(log_q_phi_z_aux).all():
    return NaN  # Skip
```

### RSIVI Safety Checks
```python
# 1. Finite score check
finite_mask = torch.isfinite(score_samples).all(dim=-1)

# 2. Zero valid counts check
if (valid_counts == 0).any():
    logger.warning("Samples with no finite scores")
    return NaN
```

---

## Summary

**AISIVI & RSIVI** are learned reverse model SIVI variants:

**AISIVI** (Adaptive Importance-weighted):
- More robust through importance weighting
- Higher complexity
- Better for challenging targets

**RSIVI** (Reverse Score):
- Simpler and faster
- Direct score averaging
- Best for well-trained reverse models

Both require learning a reverse model, which adds complexity but improves convergence.
