# KSIVI Runner Comprehensive Analysis

## 1. KSIVI Training Overview

### What is KSIVI?
**Kernel Semi-Implicit Variational Inference (KSIVI)** minimizes the squared Kernel Stein Discrepancy between the variational approximation and the target distribution:

```
KSD²(q_phi || p) = E_{q(x,z), q(x',z')} [ k(x,x') * <f(x), f(x')> ]

where f(x) = ∇ log p(x) + ∇ log q(x|z) is the combined score
```

### Key Features
- **No marginal density estimation** needed (avoids intractable log q(z))
- **Kernel-based objective** (not ELBO-based)
- **Implicit score functions** from reparameterization
- **Kernel choices**: Gaussian, IMQ, Laplace, Riesz
- **Optional pretraining** for BNN targets
- **Optional log p(z) regularization** during warm-up

### Key Inheritance Chain
```
BaseSIVIRunner
    ↓
KSIVIRunner (runner/ksivi.py)
```

---

## 2. KSIVI Loss Computation

### KSD² Formulation

**Location:** `ksivi.py` lines 149-296 (`_compute_loss_and_step`)

The loss is based on comparing combined scores:

```python
# Combined scores (line 199-200)
f1 = anneal_factor * target_score1 + neg_score_implicit1
f2 = anneal_factor * target_score2 + neg_score_implicit2

# Kernel matrix (line 207-220)
K = kernel.pair_eval(z1, z2, fit_h=True, detach_h=True)

# Score product matrix (line 230)
score_product = f1 @ f2.T  # [N, N]

# KSD² loss (line 236)
loss = (score_product * K).mean()
```

### V-Statistic vs U-Statistic

**V-Statistic** (statistic='v', recommended):
- Uses two **independent** batches z1 and z2
- Unbiased estimator
- Line 162-167: Samples two separate epsilon batches

```python
eps1 = vi_model.sample_epsilon(num=batch_size)
eps2 = vi_model.sample_epsilon(num=batch_size)
z1, neg_score1 = vi_model.forward(eps1)
z2, neg_score2 = vi_model.forward(eps2)
```

**U-Statistic** (statistic='u'):
- Uses same batch z1 == z2 with diagonal zeroed
- Biased but lower variance
- Line 169-173: Reuses first batch

```python
z2 = z1
neg_score2 = neg_score1
eps2 = eps1
score_product = score_product.fill_diagonal_(0)
```

### Optional Log p(z) Regularization

**Location:** Lines 238-251

During warm-up (when anneal_factor < 1.0):

```python
apply_log_p_reg = (
    log_p_reg > 0 and 
    (log_p_reg_mode == 'always' or anneal_factor < 1.0)
)

if apply_log_p_reg:
    log_p = target_model.logp(z1)
    reg_scale = anneal_factor if log_p_reg_mode == 'warmup_only' else 1.0
    loss = loss - log_p_reg * log_p.mean() * reg_scale
```

Biases the solution toward high density regions.

### Kernel Bandwidth Fitting

**Location:** Lines 207-220

```python
K = kernel.pair_eval(
    z1.detach(),
    z2.detach(),
    fit_h=True,      # Compute bandwidth h
    detach_h=True,   # Don't backprop through h
)
```

If `detach_kernel=False`: allows gradient flow through kernel matrix.

---

## 3. KSIVI Pretraining

### Purpose
For BNN targets: pretrain VI model on development set to improve initial samples.

**Location:** Lines 104-147 (`pretrain_vi`)

### Pretraining Objective
```python
# Minimize MSE between predicted mean and target values
loss = ((pred_y.mean(0) - y_dev)**2).mean()
```

### Configuration
```yaml
ksivi:
  pretrain:
    enabled: True/False
    steps: 1000
    lr: 1.0e-3
    batch_size: 256
```

### Pretraining Flow
```
1. Load development set from target_model.dev_data
2. For N steps:
   - Sample epsilon
   - Forward through VI model: z = g_phi(epsilon)
   - Predict y = target.predict_y(z, X_dev, ...)
   - Compute MSE: loss = mean((y_pred - y_dev)^2)
   - Gradient update
3. Log pretrain/vi_model/loss every 1/10 of steps
```

---

## 4. KSIVI Checkpoint System

### Checkpoint Management
Same as SIVI (parent class):
```
checkpoints/
├── best/
│   ├── vi_model.pt
│   ├── vi_optim.pt
│   └── vi_sched.pt
├── latest/
│   ├── vi_model.pt
│   ├── vi_optim.pt
│   └── vi_sched.pt
└── epoch_{N}/
    └── [same]
```

### Key Difference
- **No reverse model** (KSD doesn't need one)
- Only VI model saved

---

## 5. KSIVI TensorBoard Metrics

### Training Loss
```
train/vi_model/loss              ← KSD² objective
```

### Evaluation Metrics
```
metric/vi_model/{kl_ite, w2, elbo, ksd, mmd, fisher_div, ...}
```

### KSIVI-Specific Diagnostics
```
ksivi/kernel_bandwidth           ← Kernel bandwidth h (adaptive)
ksivi/score_product_mean         ← Mean of K ⊙ f1f2.T
```

### General Diagnostics
```
diagnostic/vi_model/{grad_norm, z_norm_avg, z_norm_std, ...}
pretrain/vi_model/loss          ← Pretraining phase loss
```

---

## 6. KSIVI Configuration Parameters

### Core KSIVI Parameters
```yaml
ksivi:
  statistic: 'v'                # 'v' (V-statistic) or 'u' (U-statistic)
  kernel: 'gaussian'            # 'gaussian', 'imq', 'laplace', 'riesz'
  detach_kernel: True           # Detach kernel from computation graph
  log_p_reg: 0.0                # Coefficient for log p(z) regularization
  log_p_reg_mode: 'warmup_only' # 'warmup_only' or 'always'
  affine_invariant: False       # Use affine-invariant scaling
```

### Pretraining Parameters
```yaml
ksivi:
  pretrain:
    enabled: False
    steps: 1000
    lr: 1.0e-3
    batch_size: 256
```

### Kernel Types
- **'gaussian'**: RBF kernel (default, most common)
- **'imq'**: Inverse multiquadratic (long-range)
- **'laplace'**: L1 distance kernel
- **'riesz'**: Riesz kernel

---

## 7. KSIVI Computational Flow

### Loss Computation Pipeline
```
Sample z1 from VI model
Sample z2 from VI model (independent if V-stat)
    ↓
Compute target scores: ∇ log p(z1), ∇ log p(z2)
    ↓
Compute VI scores: u/σ (implicit from reparameterization)
    ↓
Combine: f1 = ∇ log p + u/σ
         f2 = ∇ log p + u/σ
    ↓
Compute kernel: K = kernel(z1, z2)  [adaptive bandwidth h]
    ↓
Score products: f1 @ f2.T
    ↓
KSD²: mean(K ⊙ score_products)
    ↓
Backprop through VI model parameters
```

---

## 8. KSIVI Code Implementation

### Class Definition
```python
class KSIVIRunner(BaseSIVIRunner):

    def __init__(self, config: DictConfig, name: str = "KSIVI"):
        super().__init__(config=config, name=name)
        
        # Parse KSIVI config
        ksivi_cfg = training_cfg.get('ksivi', {})
        self.statistic_type = ksivi_cfg.get('statistic', 'v')
        self.kernel = Kernels[ksivi_cfg.get('kernel', 'gaussian')]()
        self.detach_kernel = ksivi_cfg.get('detach_kernel', True)
        self.log_p_reg = ksivi_cfg.get('log_p_reg', 0.0)
        
        # No reverse model for KSIVI
        self.reverse_train = False
```

### Key Methods

**calc_log_q_phi_z** (lines 89-98):
```python
def calc_log_q_phi_z(self, z, epsilon):
    """Not used by KSIVI. Raises if called."""
    raise NotImplementedError(
        "KSIVI does not estimate log q(z). "
        "This method should not be called in KSIVI."
    )
```

**train_reverse_model** (lines 100-102):
```python
def train_reverse_model(self, epoch_outer: int):
    """No-op: KSIVI has no reverse model."""
    pass
```

**_compute_loss_and_step** (lines 149-296):
Main loss computation - see section 2 above.

---

## 9. KSIVI vs Other Runners

| Aspect | KSIVI | SIVI | UIVI |
|--------|-------|------|------|
| Loss Type | KSD² | ELBO | ELBO |
| Needs log q(z) | No | Yes | Yes |
| Kernel-based | Yes | No | No |
| Pretraining | Optional | No | No |
| Checkpoint saves | VI only | VI only | VI only |
| Bandwidth adaptation | Automatic | N/A | N/A |

---

## 10. KSIVI Configuration Examples

### Toy Problem (Banana)
```yaml
ksivi:
  statistic: 'v'
  kernel: 'gaussian'
  detach_kernel: true
  log_p_reg: 0.0
  affine_invariant: false
```

### BNN with Pretraining
```yaml
ksivi:
  statistic: 'v'
  kernel: 'gaussian'
  detach_kernel: true
  log_p_reg: 0.1        # Bias toward high-density regions
  log_p_reg_mode: 'warmup_only'
  affine_invariant: false
  pretrain:
    enabled: true
    steps: 5000
    lr: 1.0e-3
    batch_size: 256
```

### Affine-Invariant Version
```yaml
ksivi:
  statistic: 'v'
  kernel: 'gaussian'
  affine_invariant: true  # Scale by covariance matrix
```

---

## 11. Troubleshooting

### NaN in KSD Loss
- **Cause**: Kernel bandwidth adaptation failing
- **Solution**: Manually set kernel bandwidth or use different kernel

### Divergence
- **Cause**: Score products too large
- **Solution**: Reduce learning rate or enable log_p_reg

### Slow Convergence
- **Cause**: Kernel bandwidth not adaptive enough
- **Solution**: Enable pretraining for BNN targets

---

## Summary

KSIVI is a kernel-based SIVI variant that:
1. Minimizes kernel Stein discrepancy instead of ELBO
2. Avoids intractable marginal density estimation
3. Includes optional log p(z) regularization
4. Supports adaptive kernel bandwidth
5. Includes optional pretraining for BNN tasks
6. Works well for both toy and real data problems
