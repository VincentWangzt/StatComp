# Scripts

This directory keeps runnable maintenance and experiment entrypoints. Historical
one-off campaign scripts and generated-grid compatibility utilities have been
removed; use git history if an old workflow is needed for provenance.

Use the project virtual environment when running scripts locally:

```powershell
.\.venv\Scripts\python.exe scripts\<script>.py
```

Remote experiment runs should follow `AGENTS.md`: make code/config changes
locally, test when feasible, commit, push, pull on the remote host, then run
under the remote environment.

Final report artifacts follow the same git-backed remote workflow. Generate
figures and tables on the remote host from committed code, commit those generated
artifacts on the remote branch, push, then pull the artifact commit locally.
Avoid using direct copy, tar extraction, or `scp` as the primary path for final
generated images and tables.

## Current Workflow

The only local campaign directory kept in this checkout is:

```text
campaigns/default_config_grid/
```

Use these scripts for the current workflow:

| Script | Purpose | Main arguments |
| --- | --- | --- |
| `run_default_config_grid_sweep.py` | Dynamic GPU scheduler for the default `<method>_<target>` grid. | `--campaign-slug`, `--results-dir`, `--tb-dir`, `--seeds`, `--methods`, `--exclude-methods`, `--targets`, `--gpus`, `--limit`, `--dry-run`, `--resume/--no-resume`, `--retry-failed`, `--rerun-stale`, `--hash-existing-artifacts`, `--poll-interval`, `--extra-override`, finalization knobs |
| `run_finalization.py` | Runs final evaluation, figures, and tables for the default campaign. | `--config`, `--set`, `--only` |
| `fetch_grid_benchmark_artifacts.py` | Fetches compact runtime metadata for inspection only; not the primary workflow for final figures/tables. | `--host`, `--port`, `--remote-repo`, `--campaign-slug`, `--remote-artifact-root` |

## Examples

Preview the current default grid without launching jobs:

```
python scripts/run_default_config_grid_sweep.py \
  --dry-run
```

Run the main default-config sweep with five seeds while excluding RSIVI. This
uses all discovered GPUs unless `--gpus` is provided, writes run artifacts under
`results/default_config_grid/` and TensorBoard logs under
`tb_logs/default_config_grid/`, and resumes already-completed fresh runs by
default:

```
python scripts/run_default_config_grid_sweep.py \
  --seeds 42 43 44 45 46 \
  --exclude-methods rsivi \
  --finalize-mode async \
  --finalize-workers 1
```

If configs changed and completed artifacts need to be checked against the new
effective configs, generate the hash inventory first, then rerun stale entries.
The inventory command only writes hash files under campaign runtime and exits:

```
python scripts/run_default_config_grid_sweep.py \
  --seeds 42 43 44 45 46 \
  --exclude-methods rsivi \
  --hash-existing-artifacts

python scripts/run_default_config_grid_sweep.py \
  --seeds 42 43 44 45 46 \
  --exclude-methods rsivi \
  --rerun-stale
```

Run the full default finalization pass after the campaign manifest has completed.
The default finalization config already selects `[SIVI, UIVI, AISIVI, DSIVI,
KSIVI]`, matching the RSIVI-excluded sweep:

```
python scripts/run_finalization.py
```

Run only evaluation, overwriting existing reevaluation outputs:

```
python scripts/run_finalization.py \
  --only evaluate \
  --set evaluation.overwrite=true
```

Regenerate only the tables and figure grids from existing reevaluation outputs:

```
python scripts/run_finalization.py \
  --only scatter_grid \
  --only scatter_hist_grid \
  --only toy_tables \
  --only toy_method_grid \
  --only langevin_table \
  --only student_edge_table \
  --only langevin_trace_grid \
  --only bnn_table
```

## Utility Scripts

| Script | Purpose | Main arguments |
| --- | --- | --- |
| `config_review_server.py` | Launches the config reviewer web tool. Currently known broken until migrated off removed legacy grid modules; see `tools/config_reviewer/README.md`. | `--port`, `--host` |
| `run_sgld_baseline.py` | Generates saved target samples with SGLD. | `--target`, `--num-samples`, `--burn-in`, `--step-size`, `--thinning`, `--num-chains`, `--seed`, `--device`, `--max-grad-norm`, `--output-dir`, `--overwrite`, `--plot` |
| `grid_finalization.py` | Shared event/config-hash/finalization helpers used by dynamic sweeps. | library module |

The standalone ELM evaluator scripts were removed. The adopted ELM metric is
the coordinate-wise KDE estimator implemented in `utils/elm/`.
