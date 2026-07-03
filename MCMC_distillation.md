# KDVI: Kernel Distillation Variational Inference

> **Status**: Implemented design
> **Method name**: Kernel Distillation Variational Inference (KDVI)
> **Last updated**: 2026-06-23

---

## 1. Core Idea

Given any implicit variational model $q_\phi$ that can produce samples, we train it by **distilling MCMC transition kernels**. Specifically:

1. Draw samples $\{z_i\}_{i=1}^N \sim q_\phi$ from the current variational distribution.
2. Apply $K$ steps of an MCMC transition kernel (targeting the true posterior $p$) to each $z_i$, producing improved samples $\{z_i'\}_{i=1}^N$.
3. Minimize the **Maximum Mean Discrepancy (MMD)** between the original samples and the MCMC-improved samples:

$$
\mathcal{L}(\phi) = \widehat{\text{MMD}}^2\bigl(\{z_i\},\; \{\text{sg}(z_i')\}\bigr)
$$

where $\text{sg}(\cdot)$ denotes stop-gradient (MCMC targets are treated as fixed).

**Intuition**: The MCMC kernel moves samples toward higher-probability regions of $p$. By training $q_\phi$ to match these improved samples, we iteratively "distill" the MCMC dynamics into the variational model. At convergence, $q_\phi = p$ implies the MCMC kernel is a fixed point (samples don't move), so $\text{MMD} = 0$.

---

## 2. Algorithm

### 2.1 Training Loop (one iteration)

```
Input: VI model q_phi, target log-density log p(z), MCMC kernel T, steps K, batch size N,
       annealing factor beta(t)

1. Sample epsilon ~ p(epsilon)                    [N noise vectors]
2. z = f_phi(epsilon)                             [N variational samples, differentiable]
3. z' = T^K(z; beta * log p)                     [apply K MCMC steps on annealed target, NO gradient]
4. Compute MMD^2({z}, {z'}) with RBF kernel
5. Backprop through MMD^2 w.r.t. phi only (z' is detached)
6. Update phi via optimizer step
```

### 2.2 Pseudocode

```python
for epoch in range(num_epochs):
    # Step 0: Compute annealing factor
    beta = anneal_schedule(epoch)              # beta in [beta_min, 1.0]
    annealed_logp = lambda z: beta * target.logp(z)
    
    # Step 1-2: Sample from variational model
    epsilon = vi_model.sample_epsilon(N)
    z, neg_score = vi_model(epsilon)         # z is differentiable w.r.t. phi
    
    # Step 3: MCMC improvement (no gradient)
    with torch.no_grad():
        z_improved = mcmc_transition(z.detach(), annealed_logp, K_steps)
    
    # Step 4-5: MMD loss
    loss = mmd_squared(z, z_improved.detach(), kernel, bandwidth)
    
    # Step 6: Optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 2.3 MMD^2 Estimator

Using the biased V-statistic:

$$
\widehat{\text{MMD}}^2_V = \frac{1}{N^2}\sum_{i,j} k(z_i, z_j) - \frac{2}{N^2}\sum_{i,j} k(z_i, z_j') + \frac{1}{N^2}\sum_{i,j} k(z_i', z_j')
$$

Or the unbiased U-statistic (zero diagonal):

$$
\widehat{\text{MMD}}^2_U = \frac{1}{N(N-1)}\sum_{i \neq j} \bigl[k(z_i,z_j) - 2k(z_i,z_j') + k(z_i',z_j')\bigr]
$$

**Note**: The third term $k(z_i', z_j')$ does not depend on $\phi$ (since targets are detached). Its gradient is zero, but it contributes to the loss magnitude for logging/convergence monitoring.

---

## 3. MCMC Transition Kernels

### 3.1 Supported Kernels

The framework is kernel-agnostic. Two initial implementations:

#### Hamiltonian Monte Carlo (HMC)

One transition step:
1. Sample momentum $p \sim \mathcal{N}(0, M)$
2. Leapfrog integration for $L$ steps with step size $\varepsilon$
3. Metropolis accept/reject (optional)

Parameters: `step_size`, `num_leapfrog_steps`, `mass_matrix` (diagonal)

#### Metropolis-Adjusted Langevin Algorithm (MALA)

One transition step:
1. Propose: $z^* = z + \frac{\tau}{2}\nabla\log p(z) + \sqrt{\tau}\,\xi$, where $\xi \sim \mathcal{N}(0,I)$
2. Accept/reject via MH ratio (optional — unadjusted variant = ULA)

Parameters: `step_size` ($\tau$)

### 3.2 Accept/Reject Options

| Mode | Description | Trade-off |
|------|-------------|-----------|
| **Unadjusted** (default) | Always accept proposal | Biased for finite step sizes, but provides consistent gradient signal |
| **Adjusted** | Standard MH correction | Unbiased transitions, but rejected proposals → $z' = z$ → zero MMD gradient for that sample |

Default: **Unadjusted** for training stability. The bias vanishes as $q_\phi \to p$.

### 3.3 Number of Steps ($K$)

- **Fixed $K$**: Simple baseline. Typical values: $K \in \{1, 3, 5, 10\}$.
- **Scheduled**: Increase $K$ over training.
  - Motivation: Early in training, $q_\phi$ is far from $p$, so even one MCMC step provides a useful gradient direction. As $q_\phi$ improves, more steps give stronger signal.
  - Example schedule: $K(t) = K_{\min} + \lfloor (K_{\max} - K_{\min}) \cdot t / T \rfloor$

### 3.4 Target Annealing

Instead of running MCMC against the full target $p$ from the start, we can anneal the target distribution used by the MCMC kernel:

$$
\log p_\beta(z) = \beta(t) \cdot \log p(z)
$$

where $\beta(t)$ increases from $\beta_{\min}$ to $1$ over training.

**Motivation**: Early in training, $q_\phi$ is far from $p$. Running MCMC against the full (potentially sharply peaked or multimodal) target may produce samples that are too far from the current $q_\phi$, leading to uninformative MMD gradients (all kernel values $\approx 0$). Annealing the target "flattens" it initially, keeping MCMC-improved samples within a useful radius of the variational samples.

**Schedules**:

| Schedule | Formula | Behavior |
|----------|---------|----------|
| **Linear** | $\beta(t) = \beta_{\min} + (1 - \beta_{\min}) \cdot t/T_{\text{anneal}}$ | Simplest; uniform ramp |
| **Offset linear** | $\beta(t) = \min(1, 0.1 + t/T_{\text{anneal}})$ | Starts at 0.1 and reaches 1 at $0.9T_{\text{anneal}}$; config value `offset_linear` |
| **Cosine** | $\beta(t) = 1 - (1 - \beta_{\min}) \cdot \frac{1}{2}(1 + \cos(\pi t / T_{\text{anneal}}))$ | Slow start/end, fast middle |
| **Sigmoid** | $\beta(t) = \beta_{\min} + (1-\beta_{\min}) \cdot \sigma(a(t/T_{\text{anneal}} - 0.5))$ | Smooth S-curve, tunable steepness $a$ |

**Interaction with step scheduling**: Target annealing and step scheduling ($K$) serve complementary roles. Annealing controls *where* the MCMC targets lie (closer to $q_\phi$ vs. closer to $p$), while step scheduling controls *how far* the MCMC chain moves. They can be composed: early training uses low $\beta$ + low $K$, late training uses $\beta=1$ + high $K$.

**Note**: When $\beta < 1$, the MCMC kernel targets a tempered distribution $p_\beta \neq p$, so the fixed-point argument (Section 6.1) only holds once $\beta = 1$. The annealing phase is purely a training heuristic for stability.

---

## 4. MMD Kernel Choice

### 4.1 RBF (Gaussian) Kernel — Default

$$
k(x, y) = \exp\!\Bigl(-\frac{\|x - y\|^2}{2h^2}\Bigr)
$$

### 4.2 Bandwidth Selection

| Method | Config | Notes |
|--------|--------|-------|
| **Variational adaptive** (default) | `fit_bandwidth_on: x` | Refit each step from detached $q_\phi$ samples. |
| **Joint adaptive** | `fit_bandwidth_on: xy` | Refit each step from the pooled detached variational and MCMC-refined samples. |
| **Fixed** | `kernel_bandwidth: <positive float>` | Pins the bandwidth and takes precedence over adaptive fitting. |

Bandwidth is **detached** (stop-gradient) by default to avoid degenerate solutions where the kernel collapses.
The former `y`, `ivi`, and `none` fitting values are not supported.

### 4.3 Other Kernels (from KSIVI infrastructure)

- **IMQ**: $(1 + \|x-y\|^2/h)^{-1/2}$ — heavier tails, may be better for multimodal targets
- **Laplace**: $\exp(-\|x-y\|_1/h)$ — L1-based, potentially more robust in high dimensions
- All available via `utils/kernels.py` with the same `pair_eval` interface.

---

## 5. Gradient Analysis

### 5.1 Gradient of the Loss

Since $z' = \text{sg}(z')$, only the first two terms of MMD^2 contribute gradients:

$$
\nabla_\phi \mathcal{L} = \nabla_\phi \Bigl[\frac{1}{N^2}\sum_{i,j} k(z_i, z_j) - \frac{2}{N^2}\sum_{i,j} k(z_i, z_j')\Bigr]
$$

The gradient flows through $z_i = f_\phi(\epsilon_i)$ via the reparameterization trick.

**Intuitive interpretation**:
- The first term (repulsive) pushes variational samples apart.
- The second term (attractive) pulls variational samples toward the MCMC targets.

### 5.2 Variance Analysis: Paired vs. Unpaired

#### Paired Estimator (Default)

Draw one batch $\{z_i\}_{i=1}^N$ from $q_\phi$. The MCMC targets $\{z_i'\}$ are derived from the **same** $z_i$. The MMD cross-term is:

$$
C_{\text{paired}} = -\frac{2}{N^2}\sum_{i,j} k(z_i, z_j')
$$

Since $z_i'$ is correlated with $z_i$ (it started from $z_i$), the cross-term has **lower variance** due to positive covariance between $k(z_i, z_i')$ across iterations. The diagonal terms $k(z_i, z_i')$ are especially informative (large when the MCMC step was small, i.e., the sample was already good).

**Variance bound** (informal): For $K=1$ MCMC step with step size $\tau \to 0$, the displacement $z_i' - z_i = O(\tau)$, so $k(z_i, z_i') \approx 1 - O(\tau^2/h^2)$. The gradient signal is $O(\tau/h^2)$ per sample, with variance $O(\tau^2/h^4 \cdot N^{-1})$.

#### Unpaired Estimator

Draw **two independent** batches from $q_\phi$: $\{z_i\}$ for the MMD left side, $\{\tilde{z}_j\}$ as MCMC seeds (producing $\{\tilde{z}_j'\}$ as targets). The cross-term is:

$$
C_{\text{unpaired}} = -\frac{2}{NM}\sum_{i,j} k(z_i, \tilde{z}_j')
$$

This is a standard two-sample MMD with independent draws. **Higher variance** than paired because $z_i$ and $\tilde{z}_j'$ are independent (no correlation to exploit).

#### Summary

| Estimator | Variance | Bias | Compute |
|-----------|----------|------|---------|
| Paired (default) | Lower (correlated samples) | Unbiased (for U-stat) | $N$ MCMC chains |
| Unpaired | Higher (independent samples) | Unbiased (for U-stat) | $M$ MCMC chains + $N$ fresh samples |

**Recommendation**: Use paired by default. The variance reduction is free (same compute budget). Consider unpaired only if there's concern about the estimator being too local (e.g., when $K$ is very small and all $z_i' \approx z_i$).

---

## 6. Theoretical Motivation

### 6.1 Fixed-Point Argument

If $q_\phi = p$, then:
- Samples $z_i \sim p$ are already distributed according to the target.
- MCMC transitions preserve $p$ (by detailed balance or invariance): $z_i' \sim p$.
- Therefore $\text{MMD}^2(q_\phi, T^K q_\phi) = 0$.

This exact invariance argument applies to corrected kernels such as MALA and
HMC. Finite-step SGLD is an intentionally biased training transition, so its
fixed-point interpretation is approximate.

Conversely, if $q_\phi \neq p$ and the MCMC kernel is ergodic, then $T^K q_\phi$ is closer to $p$ (in some sense) than $q_\phi$, so minimizing MMD drives $q_\phi$ toward $p$.

### 6.2 Relationship to Other Methods

| Method | How it uses MCMC | How it trains q_phi |
|--------|-----------------|-------------------|
| **UIVI** | HMC targeting $q_\phi(\epsilon|z) \cdot q(\epsilon)$ in noise space | ELBO with reverse model |
| **KSIVI** | None | Kernel Stein Discrepancy directly |
| **KDVI (ours)** | MCMC targeting $p$ in sample space | MMD between q_phi samples and MCMC-improved samples |

Key distinction from KSIVI: KSIVI uses the **score** of $p$ directly in a Stein discrepancy. KDVI uses MCMC **transitions** (which also require the score) but measures closeness via MMD, which doesn't require score access at gradient time — the score is only needed inside the MCMC step (which is detached).

Key distinction from UIVI: UIVI runs MCMC in the **noise space** $\epsilon$ to approximate the reverse conditional, then uses ELBO. KDVI runs MCMC in the **sample space** $z$ and uses MMD. No reverse model is needed.

### 6.3 Convergence Properties (Informal)

Under mild conditions:
1. The MCMC kernel $T$ contracts toward $p$ (geometric ergodicity).
2. MMD with characteristic kernel metrizes convergence in distribution.
3. If $q_\phi$ is sufficiently expressive, the minimizer of $\text{MMD}^2(q_\phi, T^K q_\phi)$ is $q_\phi = p$.

**Caveat**: Finite-capacity $q_\phi$ may not achieve zero MMD. The quality of approximation depends on the expressiveness of the implicit model.

---

## 7. Implementation Plan

### 7.1 Runner Architecture

```
KDVI Runner (standalone, like KSIVI)
├── Inherits: BaseSIVIRunner (from runner/base_runner.py)
├── No reverse model
├── Components:
│   ├── VI Model (models/vi_model.py)          — generates samples
│   ├── MCMC Kernel (new: utils/mcmc_kernels.py) — transition operator
│   ├── MMD Kernel (utils/kernels.py)           — computes MMD^2
│   └── Target (models/target_models.py)        — provides log p / score
└── Loss: MMD^2(vi_samples, mcmc_improved_samples)
```

### 7.2 Main Implementation Files

| File | Description |
|------|-------------|
| `runner/kdvi.py` | KDVI runner and production configuration parsing |
| `utils/mcmc_kernels.py` | Batched SGLD, HMC, and analytic-score MALA transitions |
| `utils/mmd.py` | Differentiable biased V-statistic MMD² objective |
| `configs/kdvi_*.yaml` | Separate convenience configuration for each target |

### 7.3 Config Structure

```yaml
runner_type: KDVI
target_type: banana
vi_model_type: ConditionalGaussian

train:
  epochs: 50000
  batch_size: 128
  vi:
    lr: 1.0e-3
    scheduler:
      type: StepLR
      step_size: 2000
      gamma: 0.9

  annealing:
    enabled: true
    scheme: offset_linear       # linear | sigmoid | offset_linear
    steps: 25000
  
  kdvi:
    mcmc_type: sgld             # sgld | hmc | mala
    mcmc_steps: 5
    mcmc_step_size: 0.05
    hmc_leapfrog_steps: 10
    kernel: gaussian
    fit_bandwidth_on: x         # x | xy; default x
    kernel_bandwidth: null      # positive float selects fixed bandwidth
    
    # Step schedule (optional)
    mcmc_steps_schedule:
      enabled: false
      min_steps: 1
      max_steps: 10
      warmup_epochs: 10000      # linearly increase K over this many epochs
    
    step_size_schedule:
      type: none                # none | cosine | coupled
```

---

## 8. Potential Extensions (Future Work)

1. **Hybrid loss**: Combine MMD with ELBO or KSD for faster early convergence.
2. **Adaptive MCMC step sizes**: Tune step size to target a specific acceptance rate (when using MH correction).
3. **Multi-scale MMD**: Sum of MMD at multiple bandwidths for robustness.
4. **Amortized MCMC**: Learn a parameterized MCMC kernel (e.g., L2HMC-style) jointly with $q_\phi$.
5. **Importance weighting**: Weight MMD samples by importance ratios $p(z)/q_\phi(z)$ for asymptotic efficiency.
6. **Stein + MMD hybrid**: Use KSD on the improved samples for a tighter bound.

---

## 9. Expected Advantages and Limitations

### Advantages

- **No reverse model needed**: Simpler architecture (like KSIVI), fewer hyperparameters than RSIVI/AISIVI/DSIVI.
- **No score at gradient time**: Unlike KSIVI, the target score is only needed inside the (detached) MCMC step. This makes gradient computation cheaper and avoids Hessian terms.
- **Flexible MCMC backbone**: Can plug in any transition kernel. Better kernels → better targets → faster convergence.
- **Natural curriculum**: MCMC steps provide a difficulty-adaptive training signal. Early in training, even one step helps. Later, more steps maintain signal.

### Limitations

- **Computational cost**: Each training iteration requires $K$ MCMC steps per sample (each involving a score evaluation). Cost is $O(N \cdot K)$ score evaluations per iteration.
- **MCMC step size sensitivity**: Poor step sizes lead to either too-small moves (weak signal) or too-large moves with high rejection (wasted compute).
- **Mode collapse risk**: If $q_\phi$ concentrates on one mode, MCMC steps from that mode may not reach other modes within $K$ steps. Mitigation: use multimodal initialization or large $K$.
- **High dimensions**: MCMC transitions become less efficient in high dimensions (step size must shrink). May need HMC with many leapfrog steps.

---

## 10. Experimental Plan

### Phase 1: Toy 2D targets
- Targets: `banana`, `multimodal`, `x_shaped`, `8_gaussians`
- Baselines: SIVI, KSIVI, RSIVI
- Metrics: KL divergence (via ITE), Wasserstein-2, visual comparison
- Ablations: MALA vs HMC, K=1 vs K=5 vs K=10, paired vs unpaired

### Phase 2: High-dimensional
- Target: `Langevin_post` (dim=100)
- Focus on MCMC kernel choice and step size tuning

### Phase 3: Data-dependent
- Targets: BNN and LR regression targets
- Metric: ELBO (no ground truth available)
- Compare convergence speed against KSIVI and RSIVI

---

## Appendix A: Notation Reference

| Symbol | Meaning |
|--------|---------|
| $q_\phi$ | Implicit variational distribution (parameterized by $\phi$) |
| $p$ | Target distribution (unnormalized density accessible via `logp`) |
| $T$ | MCMC transition kernel targeting $p$ |
| $T^K$ | $K$-fold composition of $T$ |
| $z_i$ | Sample from $q_\phi$ |
| $z_i'$ | MCMC-improved sample: $z_i' = T^K(z_i)$ |
| $k(\cdot,\cdot)$ | MMD kernel (RBF by default) |
| $h$ | Kernel bandwidth |
| $\text{sg}(\cdot)$ | Stop-gradient operator |
| $\beta(t)$ | Target annealing factor at epoch $t$; $\beta \in [\beta_{\min}, 1]$ |
| $p_\beta$ | Annealed target: $\log p_\beta(z) = \beta \log p(z)$ |
| $\tau$ / $\varepsilon$ | MCMC step size |
| $K$ | Number of MCMC transition steps |
| $N$ | Batch size |

---

## Implementation Status

> **Last updated**: 2026-06-23

### V1 — Implemented and Validated

| Component | File | Status |
|-----------|------|--------|
| SGLD transition kernel (batched, K-step) | `utils/mcmc_kernels.py` | Done |
| HMC transition kernel (batched, leapfrog + M-H) | `utils/mcmc_kernels.py` | Done |
| MALA transition kernel (Langevin + M-H) | `utils/mcmc_kernels.py` | Done |
| K-step scheduling (linear ramp) | `utils/annealing.py` | Done |
| Differentiable MMD² (V-statistic, paired) | `utils/mmd.py` | Done |
| KDVI Runner (inherits BaseSIVIRunner) | `runner/kdvi.py` | Done |
| Banana experiment config | `configs/kdvi_banana.yaml` | Done |
| Runner registry update | `runner/runners.py` | Done |

**V1 Scope:**
- MCMC kernels: SGLD (no accept/reject) + HMC (with M-H correction) + MALA (Langevin + M-H)
- K-step scheduling: linear ramp from K_min to K_max over warmup_epochs
- MMD loss: biased V-statistic, paired mode
- Target annealing: linear, sigmoid, or offset-linear schedule
- Bandwidth fitting: adaptive on `x`/`xy`, or fixed via `kernel_bandwidth`
- Full integration with base runner: metrics (KL, W2, ELBO), TensorBoard logging, contour plots, checkpointing, EMA

### V2 — To Be Implemented

| Feature | Priority | Notes |
|---------|----------|-------|
| U-statistic estimator | Medium | Diagonal exclusion, unbiased |
| Unpaired mode | Low | Two independent batches for MMD |
| Cosine / sigmoid annealing | Low | Add to `utils/annealing.py` |
| Multi-scale MMD | Medium | Sum over multiple bandwidths |
| Adaptive MCMC step size | Medium | Dual-averaging targeting ~0.65 acceptance |
| Data-dependent targets | High | Use `score_on_batch` for BNN/LR targets |
| Additional target configs | Medium | multimodal, x_shaped, 8_gaussians, Langevin_post |
| Full comparison experiments | High | KDVI vs KSIVI vs SIVI on all targets |
