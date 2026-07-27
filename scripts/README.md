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
| `run_default_config_grid_sweep.py` | Dynamic GPU scheduler for the default `<method>_<target>` grid and independent variants based on those configs. | `--campaign-slug`, `--results-dir`, `--tb-dir`, `--seeds`, `--methods`, `--exclude-methods`, `--targets`, `--variant`, `--gpus`, `--limit`, `--dry-run`, `--resume/--no-resume`, `--retry-failed`, `--rerun-stale`, `--hash-existing-artifacts`, `--poll-interval`, `--extra-override`, finalization knobs |
| `run_finalization.py` | Runs final evaluation, figures, and tables for the default campaign. | `--config`, `--set`, `--only` |
| `run_score_approximation.py` | Compares native checkpoint score approximations with a multi-chain posterior-HMC marginal-score reference. | `--config`, `--set`, `--dry-run`, `--limit`, `--resume/--no-resume`, `--aggregate-only` |
| `run_score_jitter_ablation.py` | Measures DSIVI/8-Gaussians posterior-HMC score sensitivity to chain-initialization jitter with common random numbers. | `--config`, `--set`, `--dry-run`, `--limit`, `--resume/--no-resume`, `--aggregate-only` |
| `fetch_grid_benchmark_artifacts.py` | Fetches compact runtime metadata for inspection only; not the primary workflow for final figures/tables. | `--host`, `--port`, `--remote-repo`, `--campaign-slug`, `--remote-artifact-root` |

## Examples

Preview the current default grid without launching jobs:

```
python scripts/run_default_config_grid_sweep.py \
  --dry-run
```

Define independent variants with repeated `--variant NAME [KEY=VALUE ...]`
arguments. Every variant starts from the same selected default config, so the
scheduler does not form a Cartesian product between variants:

```
python scripts/run_default_config_grid_sweep.py \
  --methods dsivi \
  --targets banana \
  --variant baseline \
  --variant reverse_steps_1 train.reverse.epochs=1 \
  --variant reverse_batch_512 train.reverse.batch_size=512 \
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

Validate and run the score-approximation study. The production configuration
selects SIVI/UIVI/AISIVI/DSIVI, the `x_shaped` and `8_gaussians` targets,
seeds 42--46, and five checkpoint stages. For every fixed `z`, the reference
targets `q_phi(epsilon | z)` with ten HMC chains and retains 1,000 posterior
samples per chain after 500 warm-up transitions. Chain-level score means
provide the ten internal-L2 replicates. The report places the training-style
method-to-target L2 next to the method-to-HMC-reference L2 and renders
sampler-diagnostic figures. Quality-threshold misses are retained and flagged
rather than silently discarded:

```
python scripts/run_score_approximation.py --dry-run
python -u scripts/run_score_approximation.py
```

Cell records are resumable under
`results/default_config_grid/score_approximation/`. Final CSV, Markdown, and
LaTeX reports are written under the default finalization report directory.

Run the three-seed DSIVI/8-Gaussians initialization-jitter ablation with:

```
python scripts/run_score_jitter_ablation.py --dry-run
python -u scripts/run_score_jitter_ablation.py
```

The four jitter settings reuse the same fixed `z` samples, momenta, and
accept/reject uniforms. Its report includes pairwise HMC-reference L2 distances
to distinguish a changed reference mean from chain-level internal dispersion.

## Utility Scripts

| Script | Purpose | Main arguments |
| --- | --- | --- |
| `config_review_server.py` | Launches the config reviewer web tool. Currently known broken until migrated off removed legacy grid modules; see `tools/config_reviewer/README.md`. | `--port`, `--host` |
| `run_sgld_baseline.py` | Generates saved target samples with SGLD. | `--target`, `--num-samples`, `--burn-in`, `--step-size`, `--thinning`, `--num-chains`, `--seed`, `--device`, `--max-grad-norm`, `--output-dir`, `--overwrite`, `--plot` |
| `run_when_gpu_free.py` | Queues a command behind a blocker process/campaign and sustained GPU-idle telemetry. | `--gpu`, `--wait-pid`, `--wait-manifest`, `--poll-seconds`, `--idle-seconds`, `--max-utilization`, `--max-used-memory-mib`, `--log-file` |
| `grid_finalization.py` | Shared event/config-hash/finalization helpers used by dynamic sweeps. | library module |

The standalone ELM evaluator scripts were removed. The adopted ELM metric is
the coordinate-wise KDE estimator implemented in `utils/elm/`.
