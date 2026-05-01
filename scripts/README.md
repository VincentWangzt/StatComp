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
| `fetch_grid_benchmark_artifacts.py` | Fetches compact runtime/result artifacts from the configured remote server. | `--host`, `--port`, `--remote-repo`, `--campaign-slug`, `--remote-artifact-root` |

## Utility Scripts

| Script | Purpose | Main arguments |
| --- | --- | --- |
| `config_review_server.py` | Launches the config reviewer web tool. Currently known broken until migrated off removed legacy grid modules; see `tools/config_reviewer/README.md`. | `--port`, `--host` |
| `run_sgld_baseline.py` | Generates saved target samples with SGLD. | `--target`, `--num-samples`, `--burn-in`, `--step-size`, `--thinning`, `--num-chains`, `--seed`, `--device`, `--max-grad-norm`, `--output-dir`, `--overwrite`, `--plot` |
| `grid_finalization.py` | Shared event/config-hash/finalization helpers used by dynamic sweeps. | library module |

The standalone ELM evaluator scripts were removed. The adopted ELM metric is
the coordinate-wise KDE estimator implemented in `utils/elm/`.
