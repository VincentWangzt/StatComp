# Conditional variance learning-rate investigation

This campaign returns to StatComp's width-128 SiLU, softplus-variance model.
Every primary run uses 50,000 optimizer updates, two independent batches of 128,
Adam (0.9, 0.999), mean LR 0.001, StepLR 0.9 per 1,000 updates, and target-score
annealing 0.1 to 1 over 25,000 updates. The effective mixing-noise dimension is
2, matching production target binding. Variance starts at softplus(1), unless
the explicitly named initialization ablation changes it. Diagnostics preserve
training RNG. All run specifications, source hashes, optimizer checkpoints and
metrics stay under `/root/ruivi/results/ksivi_variance_lr_20261011`.

## Atomic hypotheses and experimental design

1. A smaller variance-head LR improves the shared model: sweep variance/mean
   LR ratios 0, .001, .003, .01, .03, .1, .3, 1, 3, seeds 42--44.
2. Sharing limits the effect of a head-only LR: repeat exactly that sweep with
   a separate, initially copied width-128 two-layer variance backbone. All its
   parameters receive the variance LR. Include a zero-LR freeze control.
3. Capacity or gradient routing matters: test separate variance backbones with
   width 32, one hidden layer, no hidden layer, and a shared stop-gradient
   variant, each at ratios .01, .1, 1. Change one architectural factor at a time.
4. A slower head may merely underfit: compare 0.2 versus 1.3133 initialization,
   including a frozen 0.2 variance, on identical paired seeds.
5. The annealing/decay schedule can interact with head speed: disable annealing
   alone for matched selected configurations and controls.
6. Improvement can depend on estimator/floor: apply Stein loss or a 0.2 floor
   to the selected shared LR, retaining otherwise identical settings.
7. More steps alone may provide little optimization budget after StepLR:
   fork exact 50K checkpoints to 200K, either retain decay or restart at 1e-4
   and hold it constant. Compare all methods at 50K, 100K, 200K; no run exceeds
   200K. The restart is labeled separately from additional update count.

Discovery uses seeds 42--44. Select the positive LR giving the smallest mean
50K sliced-Wasserstein error separately for shared, separate, and alternative
backbones. Commit that selection before five held-out seeds 45--49. Controls
include the unchanged legacy conditional model, equal-LR shared and separate
models, global variance, shared Stein loss, and the 0.2-floor model. The Stein
treatment also detaches the fitted bandwidth, matching the earlier investigation;
fixed-bandwidth probes separate estimator changes from that policy.

`ConditionalGaussianHeads` preserves the combined final Linear arithmetic for
the shared model but has separate parameter blocks for Adam. It exposes real
optimizer parameter groups; multiplying gradients is not a substitute for an
Adam learning-rate change. The separate model's mean initialization matches
the shared model, and every initial variance is constant and matched. Shared
backbone gradients still combine both paths; stop-gradient routing does not
freeze its features against changes driven by the mean.

## Commands

Run locally with `.\.venv\Scripts\python.exe`; on the server use the ruivi
Conda interpreter from the checked-out worktree. Commit and push code locally,
then fetch/pull remotely before running. Long campaigns run inside tmux.

```bash
python campaigns/ksivi_variance_lr_20261011/experiment.py validate
python campaigns/ksivi_variance_lr_20261011/experiment.py benchmark
python campaigns/ksivi_variance_lr_20261011/experiment.py campaign --phase sweep --jobs 8
python campaigns/ksivi_variance_lr_20261011/experiment.py campaign --phase architecture --jobs 8
python campaigns/ksivi_variance_lr_20261011/experiment.py select
python campaigns/ksivi_variance_lr_20261011/experiment.py campaign --phase confirmation --jobs 8
python campaigns/ksivi_variance_lr_20261011/experiment.py campaign --phase ablation --jobs 8
python campaigns/ksivi_variance_lr_20261011/experiment.py campaign --phase extension --jobs 8
```

The hypothesis list describes the planned comparisons. Results and causal
conclusions are added only after the corresponding experiments and audits.
