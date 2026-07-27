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
