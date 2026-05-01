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
