# DSIVI Runner Comprehensive Analysis

## 1. DSIVI Training Overview

### What is DSIVI?
**Denoising Semi-Implicit Variational Inference (DSIVI)** is a variant of SIVI that uses:
- **VI Model (`q_phi(z|epsilon)`)**: Encodes noise `epsilon` into latent samples `z`
- **Reverse Denoising Model (`reverse_model`)**: Learns to estimate the score function `∇_z log q_phi(z)` using a denoising network

### Key Inheritance Chain
```
BaseSIVIRunner
    ↓
BaseReverseConditionalRunner
    ↓
DSIVIRunner
```

---

## 2. DSIVI Loss Computation

### Training Loss
The DSIVI training optimizes the **ELBO-based objective**:

```python
# From _compute_loss_and_step() in base_runner.py
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
5. **Estimate marginal log q_phi(z)** via `calc_log_q_phi_z()`:
   - This is where DSIVI **differs from standard SIVI**
   - Standard SIVI: uses implicit score estimation
   - DSIVI: uses learned reverse denoising model to estimate score

6. **Compute ELBO loss**:
   ```python
   loss = -mean(log_prob_target - log_q_phi_z)
   ```

### DSIVI-Specific: Reverse Model Loss
**Location:** `dsivi.py` lines 82-143 (`_train_reverse_model`)

The reverse denoising model is trained to **predict the score function** of the VI model:

```python
# Generate samples from VI model
epsilon_samples, z_samples = vi_model.sampling(num=batch_size)

# Compute true score using VI model
score_true = vi_model.score(z_samples, epsilon_samples)  # ∇_z log q_phi(z|eps)

# Reverse model tries to predict this
score_pred = reverse_model.score(z_samples)

# MSE loss for score matching
loss_reverse = mean((score_pred - score_true)**2)
```

**Key Properties:**
- The reverse model is trained to match **marginal scores** (gradient w.r.t. z only)
- Uses **Mean Squared Error (MSE)** loss (line 121: `loss = torch.mean((score_pred - score)**2)`)
- Can be either:
  - **Neural network-based** (MLPDenoiseModel): optimized with Adam
  - **Statistic-based** (GaussianReverse): fitted using closed-form solution

---

## 3. Checkpoint System

### Checkpoint Directory Structure
```
results/DSIVI/{target_type}/{timestamp}/
├── checkpoints/
│   ├── epoch_1000/
│   │   ├── vi_model.pt           # VI model state dict
│   │   ├── vi_optim.pt           # VI optimizer state dict
│   │   ├── vi_sched.pt           # VI scheduler state dict
│   │   ├── reverse_model.pt      # Reverse denoising model state dict
│   │   ├── reverse_optim.pt      # Reverse optimizer state dict (if applicable)
│   │   └── reverse_sched.pt      # Reverse scheduler state dict (if applicable)
│   ├── epoch_2000/
│   │   └── [same structure]
│   └── ...
├── samples/
│   ├── samples_epoch_500.pt
│   ├── samples_epoch_1000.pt
│   └── ...
├── plots/
│   ├── contour_epoch_500.png
│   ├── contour_epoch_1000.png
│   └── ...
└── full_config.yaml
```

### What's Saved in Each Checkpoint

**From `save_checkpoint()` in base_reverse_runner.py (lines 455-488):**

1. **VI Model Checkpoint** (`vi_model.pt`):
   - Model state dict: `vi_model.state_dict()`
   
2. **VI Optimizer Checkpoint** (`vi_optim.pt`):
   - Optimizer state dict for gradient-based optimization
   
3. **VI Scheduler Checkpoint** (`vi_sched.pt`):
   - Learning rate scheduler state dict
   
4. **Reverse Model Checkpoint** (`reverse_model.pt`):
   - Reverse denoising model state dict
   
5. **Reverse Optimizer Checkpoint** (`reverse_optim.pt`):
   - Only saved if `use_optimizer=True` for reverse model
   - Not saved for statistic-based reverse models
   
6. **Reverse Scheduler Checkpoint** (`reverse_sched.pt`):
   - Only saved if scheduler is configured for reverse optimizer

### Checkpoint Configuration
**From config files (e.g., `dsivi.yaml` lines 38-40):**
```yaml
checkpoint:
  enabled: true
  freq: 1000  # Save every 1000 epochs
```

### Checkpoint Frequency
- **Periodic checkpoints**: Saved every `ckpt_freq` epochs (typically 1000)
- **No final checkpoint**: Only periodic checkpoints are saved
- **No best checkpoint**: No tracking of best metrics - just periodic saves

---

## 4. TensorBoard Metrics Logging

### TensorBoard Directory
```
tb_logs/DSIVI/{target_type}/{timestamp}/
├── events.out.tfevents.*
└── [TensorBoard event files]
```

### Metrics Logged During Training

#### Training Loss (logged every `loss_log_freq` epochs):
```
train/vi_model/loss                      # Main ELBO loss
train/reverse_model/loss                 # Reverse model MSE loss (lines 186)
train/reverse_model/steps                # Reverse model training steps
```

#### Diagnostics (every epoch):
```
diagnostic/vi_model/grad_norm             # Gradient norm of VI model
diagnostic/vi_model/z_norm_avg            # Average L2 norm of z samples
diagnostic/vi_model/z_norm_std            # Std of z norms
diagnostic/vi_model/epsilon_norm_avg      # Average L2 norm of epsilon
diagnostic/vi_model/epsilon_norm_std      # Std of epsilon norms
diagnostic/vi_model/marginal_conditional_score_l2_gap  # Score gap for reverse models
diagnostic/reverse_model/avg_score_norm   # Avg L2 norm of predicted score
diagnostic/reverse_model/norm_of_avg_score # L2 norm of mean score
diagnostic/reverse_model/score_l2_to_target # L2 gap between reverse and target score
```

#### Evaluation Metrics (logged every `metric_log_freq` epochs):
**From base_runner.py and dsivi.py:**

```
metric/vi_model/kl_ite                    # KL divergence to baseline (ITE package)
metric/vi_model/w2                        # Sliced Wasserstein-2 distance
metric/vi_model/elbo                      # ELBO estimate
metric/vi_model/elbo_std_total            # ELBO std
metric/vi_model/elbo_std_q                # ELBO std from q estimation
metric/vi_model/elbo_ci_half              # 95% CI half-width
metric/vi_model/expected_log_marginal     # KDE-based expected log marginal
metric/vi_model/kde_expected_log_marginal # Same as above (lines 751-760)
metric/vi_model/fisher_div                # Fisher divergence
metric/vi_model/ksd                       # Kernelized Stein Discrepancy (VI model)
metric/vi_model/mmd                       # Maximum Mean Discrepancy
metric/vi_model/rmse                      # BNN test RMSE (for BNN targets)
metric/vi_model/test_llk                  # BNN test log-likelihood
metric/vi_model/nll                       # BNN negative log-likelihood
```

**Reverse Model Metrics (DSIVI-specific, from dsivi.py lines 76-80):**
```
metric/reverse_model/ksd                  # KSD of reverse model
metric/reverse_model/ksd_h                # KSD kernel bandwidth
metric/reverse_model/kl_ite               # KL of reverse joint distribution
metric/reverse_model/w2                   # W2 of reverse joint distribution
```

**Warmup Metrics (lines 153-160 in dsivi.py):**
```
warmup/reverse_model_loss                 # Reverse model loss during warmup
warmup/rev_model_ksd                      # KSD of reverse model
warmup/fisher_div                         # Fisher divergence during warmup
```

#### Diagnostic Metrics:
```
diagnostic/vi_model/kde_expected_log_marginal_std              # Std across reference samples
diagnostic/vi_model/kde_expected_log_marginal_clamped_dims     # Num dims with clamped bandwidth
```

#### Timing Metrics (every epoch, lines 1543-1554):
```
time/vi_sample              # Time for sampling and forward pass
time/neg_score              # Time for log q estimation
time/backward               # Time for backward + optimizer step
time/reverse_train          # Time for reverse model training
time/sampling               # Time for saving samples
time/checkpoint             # Time for saving checkpoints
time/kl_estimation          # Time for KL metric
time/w2_estimation          # Time for W2 metric
time/elbo_estimation        # Time for ELBO metric
time/expected_log_marginal_estimation    # Time for ELM metric
time/mmd_estimation         # Time for MMD metric
time/ksd_estimation         # Time for KSD metric
time/bnn_estimation         # Time for BNN metrics
time/fisher_estimation      # Time for Fisher metric
time/metric_eval_tot        # Total time for all metrics
time/plot                   # Time for plotting
time/epoch                  # Total epoch time

time_avg/{key}              # Moving average (over time_avg_window)
```

#### Summary Metrics (at end of training):
```
summary/total_training_time    # Total wall-clock time
summary/avg_epoch_time         # Average per-epoch time
summary/warmup_time            # Warmup wall-clock time
summary/warmup_avg_epoch_time  # Average warmup epoch time
```

#### Configuration Logging:
```
config/full_config            # Full YAML configuration (as text)
```

### Pretrain Metrics (for BNN targets):
```
pretrain/vi_model/loss        # VI pretraining loss
```

---

## 5. DSIVI Config Examples

### Config 1: Toy Problem (`dsivi.yaml`)
**Target:** Banana (2D bimodal distribution)
```yaml
runner_type: DSIVI
target_type: banana
vi_model_type: ConditionalGaussian
reverse_model_type: MLPDenoiseModel

train:
  epochs: 2000
  batch_size: 1024
  reverse:
    lr: 1.0e-3
    batch_size: 2048
    epochs: 10
    update_freq: 1

metric:
  kl_ite:
    enabled: true
    num_samples: 10000
  w2:
    enabled: true
    num_samples: 10000
    num_projections: 1000
  elbo:
    enabled: true
    batch_size: 512
    num_batches: 10
    num_z_samples: 5000
  fisher:
    enabled: true
    num_samples: 1000
    num_is_samples: 512
```

### Config 2: BNN Target (`dsivi_Bnn_boston.yaml`)
**Target:** Boston housing with BNN posterior
```yaml
runner_type: DSIVI
target_type: Bnn_boston
vi_model_type: ConditionalGaussian
reverse_model_type: MLPDenoiseModel

train:
  epochs: 10000
  batch_size: 128
  grad_clip: 1.0
  pretrain:
    enabled: true
    steps: 100
    lr: 1.0e-2
  ema:
    enabled: true
    beta: 0.999
  reverse:
    lr: 1.0e-3
    batch_size: 2048
    epochs: 2
    update_freq: 1

metric:
  kl_ite:
    enabled: false
  bnn:
    enabled: true
    num_samples: 500
```

**Common DSIVI Targets:**
- Toy problems: `banana`, `8_gaussians`, `multimodal`, `x_shaped`, `student_uc`
- Real data: `Bnn_boston`, `Bnn_concrete`, `Bnn_power`, `Bnn_protein`, `Bnn_winered`, `Bnn_yacht`
- Posterior inference: `Langevin_post`, `LRwaveform`

---

## 6. Evaluation and Checkpoint Loading

### Loading Checkpoints (from `finalization/runner_eval.py`)

**Function: `build_runner()` lines 161-171:**
```python
def build_runner(rec: RunRecord, cfg: Any):
    # Find the final (highest epoch) checkpoint
    ckpt_dir, epoch = find_final_checkpoint(rec.result_path)
    
    # Initialize runner
    runner = Runners[rec.runner_type](config=cfg)
    
    # Load VI model state
    state = torch.load(ckpt_dir / "vi_model.pt", map_location=runner.device)
    runner.vi_model.load_state_dict(state)
    runner.vi_model.eval()
    
    return runner, ckpt_dir, epoch
```

**Note:** Reverse model is NOT loaded during evaluation - only the VI model is used.

### Post-hoc Evaluation Metrics
**From `finalization/runner_eval.py` lines 312-417:**

The finalization pipeline computes metrics **after training** by loading checkpoints:

1. **ELBO** (`evaluate_elbo()`):
   - 5000 z samples with 10 batches of 512 auxiliary epsilon per z
   
2. **W2 Distance** (`evaluate_w2_budgeted()`):
   - 10000 VI samples vs 10000 baseline samples
   - 5000 random projections
   
3. **Truncated W2** (`constrained_w2()`):
   - W2 computed only on samples within specified box width
   - Used for bounded targets like `student_uc`
   
4. **KDE Expected Log Marginal** (`evaluate_expected_log_marginal()`):
   - For Langevin_post: 1D KDE on reference baseline samples
   
5. **BNN Metrics**:
   - **RMSE**: Mean prediction error of ensemble mean
   - **NLL**: Negative log-likelihood (Test log-likelihood)

### Metrics Not Logged During Training But Computed Post-hoc
- The finalization pipeline **re-evaluates** metrics with **higher precision** than during training
- Uses configurable sample sizes (typically larger than during training)
- Provides standard error estimates across seeds

---

## 7. Key Implementation Details

### Sample Save Format (`save_samples()` lines 968-996)
```python
sample_dict = {
    'z': z_sample,              # Latent samples [num_samples, z_dim]
    'epsilon': epsilon_sample,  # Noise samples [num_samples, epsilon_dim]
    'epoch': epoch,             # Training epoch
    'time': elapsed_time,       # Wall-clock time since training start
    'exp_name': self.name,      # Runner name (e.g., "DSIVI")
    'target_type': self.target_type,
    'vi_model_type': self.vi_model_type,
}
torch.save(sample_dict, f"samples_epoch_{epoch}.pt")
```

### Reverse Model Training Loop
**From `train_reverse_model()` in base_reverse_runner.py lines 569-590:**
- Called every `rev_update_freq` epochs (typically every epoch)
- Trains for `rev_epochs` inner epochs (typically 2-10)
- Uses batch size of `rev_batch_size` (typically 2048)
- Generates new VI samples for each training round

### Warmup Phase
**From `warmup()` in base_reverse_runner.py lines 403-453:**
- Optional pre-training phase for reverse model
- Configured via `reverse_model.warmup` config
- Logs warmup metrics to TensorBoard with "warmup/" prefix
- Reports total warmup time and per-epoch time

### Resume Training
**From `load_checkpoints()` in base_reverse_runner.py lines 490-567:**
- Loads VI model, optimizer, scheduler from checkpoint
- Loads reverse model, optimizer, scheduler
- Optionally recovers epoch number to resume from correct epoch

---

## 8. Summary: What's Tracked

### During Training:
✅ Training loss (VI + reverse model)
✅ Diagnostics (gradient norms, sample norms)
✅ Evaluation metrics (KL, W2, ELBO, KSD, Fisher, BNN metrics)
✅ Timing breakdown
✅ Reverse model-specific metrics (KSD, KL, W2 of reverse joint)
✅ Warmup metrics

### In Checkpoints:
✅ VI model parameters
✅ VI optimizer state (for resuming)
✅ VI scheduler state (for resuming)
✅ Reverse model parameters
✅ Reverse optimizer state (if applicable)
✅ Reverse scheduler state (if applicable)

### Post-hoc Evaluation:
✅ Same metrics re-computed with higher precision
✅ Standard error estimates
✅ Aggregated statistics across seeds

---

## 9. Configuration Details

### Metric Enable/Disable
Each metric can be individually enabled/disabled in config:
```yaml
metric:
  kl_ite:
    enabled: true/false
    num_samples: 10000
  w2:
    enabled: true/false
    num_samples: 10000
    num_projections: 1000
  elbo:
    enabled: true/false
  fisher:
    enabled: true/false
  ksd:
    enabled: true/false
  mmd:
    enabled: true/false
  bnn:
    enabled: true/false  # Auto-enabled for BNN targets
  expected_log_marginal:
    enabled: true/false
```

### Logging Frequencies
```yaml
log:
  metric_log_freq: 10      # Evaluate expensive metrics every N epochs
  loss_log_freq: 100       # Log training loss every N epochs
  reverse_log_freq: 500    # Log reverse model loss every N epochs
  time_avg_window: 500     # Moving average window for timing
```

### Sampling During Training
```yaml
sample:
  freq: 500                # Save samples every N epochs
  num: 10000               # Number of samples to save per checkpoint
plot:
  freq: 500                # Generate plots every N epochs
  num: 10000               # Number of samples for plotting
```

