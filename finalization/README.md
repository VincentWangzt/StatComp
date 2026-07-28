# Finalization Module

Run the default one-pass workflow on the remote repo after syncing code:

```bash
python scripts/run_finalization.py
```

Full finalization reevaluation should be run in the remote GPU environment
(`~/ruivi/`, conda env `ruivi`) after syncing the branch with git. The local
Windows workspace usually does not contain the full `results/default_config_grid`
checkpoint tree required by `--only evaluate --set evaluation.overwrite=true`;
use local runs only for unit tests or lightweight table/plot wiring checks.

Final figures and tables should also be generated on the remote host from
committed code. Commit and push those generated artifacts on the remote branch,
then pull them into the local checkout through git.

Useful overrides:

```bash
python scripts/run_finalization.py --only scatter_grid
python scripts/run_finalization.py --only scatter_hist_grid
python scripts/run_finalization.py --only evaluate --set evaluation.overwrite=true
python scripts/run_finalization.py --set selection.seeds=[42] --set evaluation.device=cpu
```

Default outputs are written under:

```text
campaigns/default_config_grid/generated_reports/finalization/
```

## Score-approximation study

The checkpoint score study has a separate resumable entrypoint:

```bash
python scripts/run_score_approximation.py --dry-run
python -u scripts/run_score_approximation.py
```

Its defaults are defined in
`configs/finalization/score_approximation.yaml`. The reference sampler uses
posterior HMC for `q_phi(epsilon | z)`, initialized from each fixed sample's
generating epsilon. Each of ten chains retains 1,000 samples after 500 warm-up
transitions. The ten chain means form the reference replicates, and the reports
include the training-style method-to-target comparison, log-scale figures, and
acceptance, step-size, divergence, and R-hat diagnostics. Cells that miss the
configured sampler-quality thresholds remain resumable records but are marked
with a reference-quality warning.
Runtime cell metrics remain under
`results/default_config_grid/score_approximation/`; aggregate CSV, Markdown,
LaTeX, and metadata files are generated under
`campaigns/default_config_grid/generated_reports/finalization/score_approximation/`.

### Terminal-particle SGLD score reference

The DSIVI-only SGLD analysis uses the epoch-10,000 checkpoints for
`x_shaped` and `8_gaussians`:

```bash
python scripts/run_score_sgld_approximation.py --dry-run
python -u scripts/run_score_sgld_approximation.py
```

Its production configuration is
`configs/finalization/score_approximation_sgld_10x1k_5k.yaml`. For each fixed
z, it evolves 10 groups of 1,000 posterior-epsilon particles for 5,000
fixed-size Langevin steps at step size 0.0001 and retains only the terminal
state. Each group mean is one reference replicate. Completed z tiles and the
active tile state are fingerprinted and resumable. The 1,000-, 2,500-, and
5,000-step score snapshots diagnose finite-horizon drift.

The matched small-initialization-noise run uses
`configs/finalization/score_approximation_sgld_10x1k_5k_jitter_0p1.yaml`.
It changes only the initialization jitter from unit standard-normal noise to
0.1 times standard-normal noise and writes to a separate runtime and report
namespace. Forward samples, initial standard-normal draws, and Langevin noise
use the same deterministic seeds as the unit-jitter run.

This sampler uses exact posterior gradients, so it is technically fixed-step
unadjusted Langevin (ULA), despite the conventional SGLD label. A small
within-SGLD L2 measures agreement among group means but does not certify
mixing or rule out shared finite-horizon and discretization bias.

### HMC initialization-jitter ablation

The focused jitter ablation uses DSIVI on `8_gaussians` at epoch 10,000 and
seeds 42--44:

```bash
python scripts/run_score_jitter_ablation.py --dry-run
python -u scripts/run_score_jitter_ablation.py
```

It evaluates jitter scales 0, 1e-4, 1e-3, and 1e-2 with the production HMC
budget. Forward samples and all HMC random draws are common across scales, so
the resulting differences isolate chain initialization. In addition to the
method-to-HMC and internal L2 metrics, it reports pairwise L2 distances between
the four HMC reference means. Runtime records are resumable by seed under
`results/default_config_grid/score_jitter_ablation/`; aggregate tables and
figures are written to the matching finalization report directory.
