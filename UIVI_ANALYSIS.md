# UIVI Runner Comprehensive Analysis

## 1. UIVI Training Overview

### What is UIVI?
**Unbiased Semi-Implicit Variational Inference (UIVI)** uses Hamiltonian Monte Carlo (HMC) to sample epsilon conditional on z:

- **VI Model**: Maps epsilon to z
- **HMC Sampler**: Samples epsilon ~ q_phi(epsilon | z) conditionally
- **Score Estimation**: Uses VI model score directly (u/σ)

### Key Features
- Uses HMC instead of learned reverse model
- Adaptive epsilon sampling based on current z
- Automatic acceptance rate monitoring
- No learned parameters for reverse model

### Key Inheritance Chain
```
BaseSIVIRunner
    ↓
UIVIRunner (runner/uivi.py)
```

---

## 2. UIVI HMC Sampling

### Core HMC Implementation

**Location:** Lines 83-161 (`sample_epsilon_hmc`)

```python
def sample_epsilon_hmc(
    self,
    z: torch.Tensor,              # [B, Dz]
    eps_init: torch.Tensor,       # [B, De]
    num_samples: int,             # S
    burn_in_steps: int,
    step_size: float,
    leapfrog_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Returns:
        z_aux: [B, S, Dz]
        eps_aux: [B, S, De]
        acc_rate: float (0-1)
    """
```

### HMC Algorithm

Per z, run sequential HMC targeting q_phi(z|epsilon) * q(epsilon):

```python
for step in range(burn_in_steps + num_samples):
    # 1. Resample momentum
    p0 = torch.randn(B, De)
    
    # 2. Current energy
    logp0 = log q_phi(eps|z) = log q(eps) + log q_phi(z|eps)
    K0 = 0.5 * ||p0||²
    
    # 3. Leapfrog integrator
    p = p0 + 0.5 * step_size * ∇ log q_phi(eps|z)
    for _ in range(leapfrog_steps):
        eps_prop = eps_prop + step_size * p
        grad = ∇ log q_phi(eps_prop|z)
        p = p + step_size * grad
    p = p + 0.5 * step_size * grad
    
    # 4. MH acceptance
    logp_prop = log q_phi(eps_prop|z)
    K_prop = 0.5 * ||p||²
    
    dH = (K_prop - logp_prop) - (K0 - logp0)
    accept_prob = exp(-dH)
    u ~ U[0,1]
    
    if u < accept_prob:
        eps_current = eps_prop
    
    # 5. After burn-in, collect sample
    if step >= burn_in_steps:
        collect eps_current
```

### Conditional Log-Probability

**Location:** Lines 44-59 (`_log_q_phi_eps_given_z`)

```python
def _log_q_phi_eps_given_z(
    self,
    epsilon: torch.Tensor,  # [B, De]
    z: torch.Tensor,        # [B, Dz]
) -> torch.Tensor:          # [B]
    return (
        vi_model.log_q_epsilon(epsilon) +    # log q(eps)
        vi_model.logp(z, epsilon)            # log q(z|eps)
    )
```

### Gradient Computation

**Location:** Lines 61-81 (`_grad_log_q_phi`)

```python
def _grad_log_q_phi(
    self,
    epsilon: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    epsilon = epsilon.clone().detach().requires_grad_(True)
    logp = _log_q_phi_eps_given_z(epsilon, z)
    logp_sum = logp.sum()
    grad = torch.autograd.grad(
        logp_sum,
        epsilon,
        retain_graph=False,
        create_graph=False
    )[0]
    return grad.detach()
```

---

## 3. UIVI Loss Computation

### Training Loss

**Location:** Inherited from base_runner.py, but uses HMC for epsilon sampling

```python
# Step 1: Sample initial epsilon
epsilon_init ~ p(epsilon)

# Step 2: Forward through VI model
z, neg_score = vi_model(epsilon_init)

# Step 3: Run HMC conditional on z
z_aux, epsilon_aux, acc_rate = sample_epsilon_hmc(
    z, epsilon_init,
    num_samples=M,
    burn_in_steps=5,
    step_size=0.1,
    leapfrog_steps=10
)  # [B, M, Dz], [B, M, De]

# Step 4: Compute VI scores
score = vi_model.score(z_aux, epsilon_aux)  # [B, M, Dz]
score = score.mean(dim=1)                    # [B, Dz]

# Step 5: ELBO loss
log_q_phi_z = sum(score * z)  # [B]
loss = -mean(log_p(z) - log_q_phi_z)
```

---

## 4. UIVI Configuration

### HMC Parameters
```yaml
hmc:
  step_size: 0.1              # HMC step size
  leapfrog_steps: 10          # Leapfrog steps per HMC transition
  burn_in_steps: 5            # Burn-in transitions before sampling

train:
  reverse_sample_num: 10      # M: HMC samples to collect
```

### Configuration Path
**Location:** Lines 22-37 (initialization)

```python
if 'reverse_model_config_path' not in config:
    default_path = 'configs/reverse_models/HMC.yaml'
    config.reverse_model_config_path = default_path

_reverse_model_config = {
    'hmc': OmegaConf.load(config.reverse_model_config_path)
}
config = OmegaConf.merge(_reverse_model_config, config)
```

---

## 5. UIVI Diagnostics

### HMC Acceptance Rate
```
train/reverse_model/hmc_accept_rate

Target: 0.6 - 0.8
If < 0.6: increase step_size or increase leapfrog_steps
If > 0.8: decrease step_size or increase leapfrog_steps
```

### Standard Diagnostics
```
diagnostic/reverse_model/avg_epsilon_distance
diagnostic/reverse_model/avg_score_norm
diagnostic/reverse_model/norm_of_avg_score
metric/reverse_model/ksd
```

---

## 6. UIVI Checkpoint System

### Files Saved
```
checkpoints/
├── best/
│   └── vi_model.pt, vi_optim.pt, vi_sched.pt
├── latest/
│   └── vi_model.pt, vi_optim.pt, vi_sched.pt
└── epoch_{N}/
    └── [same]
```

Note: No HMC state saved (stateless MCMC)

---

## 7. UIVI Computational Flow

### Per-Sample HMC Update
```
For each z_i:
    Initialize eps_i from training
    ↓
    For burn_in_steps:
        Run 1 HMC transition
        (leapfrog + MH acceptance)
    ↓
    For num_samples:
        Run 1 HMC transition + collect eps
    ↓
    Compute average score from M samples
    ↓
    Use for loss computation
```

---

## 8. UIVI Code Implementation

### Key Methods

**sample_epsilon_hmc** (lines 83-161):
Main HMC sampling routine - see section 2.

**calculate_rev_KSD** (lines 163-195):
```python
def calculate_rev_KSD(self):
    z_samples, _ = vi_model.sampling(num=n_ksd_samples)
    
    # HMC sample epsilon conditional on z
    z_aux, epsilon_aux, _ = sample_epsilon_hmc(
        z_samples, epsilon_init,
        num_samples=M, burn_in_steps=5,
        step_size=step_size, leapfrog_steps=leapfrog_steps
    )
    
    # VI model score
    score = vi_model.score(z_aux, epsilon_aux).mean(dim=1)
    
    # KSD between score and target
    return compute_ksd(z_samples, score)
```

**calc_log_q_phi_z** (lines 203-255):
```python
def calc_log_q_phi_z(self, z, epsilon):
    # HMC sample epsilon conditional on z
    z_aux, epsilon_aux, acc_rate = sample_epsilon_hmc(
        z, epsilon,
        num_samples=M, burn_in_steps=burn_in,
        step_size=hmc_step_size,
        leapfrog_steps=hmc_leapfrog_steps
    )
    
    # Log acceptance rate
    self.writer.add_scalar(
        "train/reverse_model/hmc_accept_rate",
        acc_rate, epoch
    )
    
    # VI score
    score = vi_model.score(z_aux, epsilon_aux).mean(dim=1)
    
    # Log q_phi(z) ≈ score · z
    return torch.sum(score * z, dim=-1), score
```

---

## 9. UIVI vs Other Runners

| Aspect | UIVI | SIVI | KSIVI | DSIVI |
|--------|------|------|-------|-------|
| Epsilon Sampling | HMC | Prior | Prior | Prior |
| Reverse Model | Stateless | None | None | Learned NN |
| Computational Cost | High | Low-Med | Low-Med | High |
| Hyperparameters | HMC params | None | Kernel | NN params |
| Acceptance Rate | Monitored | N/A | N/A | N/A |

---

## 10. UIVI Tuning Guide

### Step Size Too Large
- **Symptom**: Acceptance rate < 0.6
- **Fix**: Decrease `hmc_step_size` (e.g., 0.1 → 0.05)

### Step Size Too Small
- **Symptom**: Acceptance rate > 0.8
- **Fix**: Increase `hmc_step_size` (e.g., 0.1 → 0.2)

### Poor Mixing
- **Symptom**: Slow epsilon_distance change
- **Fix**: Increase `leapfrog_steps` (e.g., 10 → 20)

### Memory Issues
- **Symptom**: Out of memory during sampling
- **Fix**: Reduce `reverse_sample_num` or `burn_in_steps`

---

## 11. Example HMC Config

```yaml
hmc:
  step_size: 0.1           # Adjust for 0.65 acceptance
  leapfrog_steps: 10       # More steps → better mixing
  burn_in_steps: 5         # Discard first 5 samples

train:
  reverse_sample_num: 10   # Collect 10 samples per z
  batch_size: 256
```

---

## Summary

UIVI is an HMC-based SIVI variant that:
1. Uses Hamiltonian Monte Carlo for conditional epsilon sampling
2. Requires no learned reverse model
3. Monitors HMC acceptance rate
4. Has higher computational cost than SIVI
5. Provides more sophisticated conditional sampling
