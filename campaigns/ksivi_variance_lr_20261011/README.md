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
   Also test linear heads at .001 and .003: an exact finite-budget Adam
   coordinate bound predicts a reachable-function obstruction at these rates.
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

## Completed evidence

The final campaign contains **190 primary 50K runs** and **36 continuations to
200K**, for 14.9 million optimizer updates counting each shared parent once.
All 226 trajectories finished without a nonfinite-loss failure. Each saved
endpoint was independently evaluated; the audit regenerated 262 sample arrays
and checked their metrics and loss gradients. The unchanged legacy model and
the split-head equal-LR implementation have bitwise identical parameters at
every checked checkpoint, for three paired seeds.

Discovery selects shared ratio **0.01**, separate 128x2 ratio **0.03**, and an
independent linear variance map at ratio **1**. Selection was recorded in commit
`808b97f` before held-out seeds 45--49 were launched. Mean held-out 50K SW2:

| Configuration | Mean error |
|---|---:|
| Shared, equal LR | 0.870 |
| Shared, variance-head LR 0.00001 | 0.129 |
| Separate 128x2, variance LR 0.00003 | 0.240 |
| Independent linear variance, equal LR | 0.129 |
| Global variance | 0.159 |
| Shared, Stein loss | 0.262 |
| Shared, variance floor 0.2 | 0.397 |
| Separate 128x2, equal LR | 0.589 |

The slow shared head improves all five paired equal-LR runs. It beats global
variance on three of five seeds; these data do **not** establish superiority
over global variance. The linear variance map learns nearly global variance:
its mean conditional coefficient of variation is about 3.3% on held-out seeds.
Making an independent variance branch arbitrarily slow fails: linear ratios
0.001--0.1 under-adapt, and the exact finite-budget Adam reachability theorem
rules out an exact target fit at the smallest rates in the stated budgets.

At 200K, continued StepLR adds almost no improvement. A constant 0.0001 mean-LR
restart gives mean errors about 0.109 (selected shared), 0.109 (selected
separate), 0.103 (global), and 0.225 (Stein) on the three discovery seeds.
Equal-LR shared remains much worse, about 1.461. This comparison changes both
the number of steps and the cumulative LR; it is labeled as a separate policy.

The report derives the exact shared-head functional drift identity, a local
block-SGD descent inequality, a finite-budget Adam bound, a frozen-scale
characteristic-function obstruction, and Gaussian-kernel gradient-noise
asymptotics with an unchanged marginal. The last result proves that original
gradient variance can grow as `1 / (N^2 v^2)` in a specified direction, while
the Stein counterpart has variance `O(v)`. These are scoped results, not a
universal estimator ordering or an Adam convergence guarantee. The X-shaped
target itself has an unbounded log-density Hessian; the original paper's
global smoothness assumptions do not apply mechanically to this target.

All 14 atomic training ablations, per-seed tables, independent marginal metrics,
gradient-path/freeze probes, exact formulas and proofs, limitations, and figures
are in [the report](report/report.pdf). Machine-readable records are under
`report/`; raw checkpoints, source manifests, RNG states and samples remain at
the server path above. `floor_fraction` in the original training-diagnostic
record always denotes the fraction at the reference `1e-4` threshold, including
in the explicitly larger-floor runs; use variance quantiles to inspect those.

## Enable the production model

`ConditionalGaussianHeads` is an opt-in model; existing canonical configs keep
their original behavior. For the selected shared setting, choose the model
type and set actual Adam parameter-group LRs:

```text
vi_model_type=ConditionalGaussianHeads
train.vi.lr=0.001 train.vi.var_lr=0.00001
vi_model.variance_backbone=shared
```

For the separate network use `vi_model.variance_backbone=separate` and
`train.vi.var_lr=0.00003`. For the linear variance branch also use
`vi_model.variance_num_layers=0` and `train.vi.var_lr=0.001`. Keep the mean
architecture, annealing, transform and schedule matched when interpreting
these recommendations. The legacy `ConditionalGaussian` does not expose
the new separate parameter groups.

## Evaluate and rebuild

After all training phases, evaluate each phase (parallel processes must write
different output JSON files), merge them with `analyze.py merge --inputs ...`,
then export and audit. The default evaluation command covers all phases:

```bash
python campaigns/ksivi_variance_lr_20261011/analyze.py evaluate --probes
python campaigns/ksivi_variance_lr_20261011/analyze.py export --output campaigns/ksivi_variance_lr_20261011/report
python campaigns/ksivi_variance_lr_20261011/audit.py
python campaigns/ksivi_variance_lr_20261011/theory_checks.py
python campaigns/ksivi_variance_lr_20261011/build_report.py --compile latexmk
```

The report builder requires all 226 runs, 262 independent evaluations, and a
passed audit unless `--draft` is explicitly requested. For the remote cached
compiler use `--compile tectonic --compiler` with the absolute compiler path.
`typesetting.json` records the compiler version and source/PDF hashes. Generate
final figures and report artifacts on the server from pushed code, commit and
push them there, then pull them locally for PDF rendering and visual QA.
