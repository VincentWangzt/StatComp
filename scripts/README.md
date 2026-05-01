# Scripts

This directory keeps runnable maintenance and experiment entrypoints. Historical
campaign one-offs live under `scripts/archive/` so the top-level script list
stays focused on workflows that are still expected to be used.

Use the project virtual environment when running scripts locally:

```powershell
.\.venv\Scripts\python.exe scripts\<script>.py
```

Remote experiment runs should still follow the repository workflow in
`AGENTS.md`: change code/config locally, test when feasible, commit, push, pull
on the remote host, then run under the remote environment.

## Active Scripts

| Script | Purpose | Main arguments |
| --- | --- | --- |
| `config_review_server.py` | Launches the config reviewer web tool from `tools/config_reviewer`. | none |
| `run_default_config_grid_sweep.py` | Current dynamic GPU scheduler for the default `<method>_<target>` grid. | `--campaign-slug`, `--results-dir`, `--tb-dir`, `--seeds`, `--methods`, `--exclude-methods`, `--targets`, `--gpus`, `--limit`, `--dry-run`, `--resume/--no-resume`, `--retry-failed`, `--rerun-stale`, `--hash-existing-artifacts`, `--poll-interval`, `--extra-override`, finalization knobs |
| `run_finalization.py` | Runs final evaluation, figures, and tables for a completed campaign. | `--config`, `--set`, `--only` |
| `run_sgld_baseline.py` | Generates saved target samples with SGLD. | `--target`, `--num-samples`, `--burn-in`, `--step-size`, `--thinning`, `--num-chains`, `--seed`, `--device`, `--max-grad-norm`, `--output-dir`, `--overwrite`, `--plot` |

## Legacy Reusable Grid Tools

These are retained because older generated campaigns still use their manifest
and runtime format. Prefer `run_default_config_grid_sweep.py` for new sweeps.

| Script | Purpose | Main arguments |
| --- | --- | --- |
| `fetch_grid_benchmark_artifacts.py` | Fetches compact artifacts from the configured remote server. | `--host`, `--port`, `--remote-repo`, `--campaign-slug`, `--remote-artifact-root` |
| `run_grid_queue.py` | Runs one queue from a generated campaign manifest. | `--phase`, `--queue`, `--gpu`, `--manifest`, `--campaign-dir`, `--limit`, `--continue-past-failed` |
| `show_grid_status.py` | Displays queue progress for generated campaign manifests. | `--phase`, `--manifest`, `--campaign-dir` |
| `summarize_grid_benchmark.py` | Summarizes completed runs from generated campaign manifests. | `--phase`, `--manifest`, `--campaign-dir` |
| `grid_benchmark_common.py` | Shared constants and helpers for the legacy generated-grid scripts. | library module |
| `grid_finalization.py` | Shared event/config-hash/finalization helpers used by dynamic sweeps. | library module |

## Archived Scripts

`scripts/archive/campaigns/` contains one-off campaign generators, launchers,
and report renderers for completed experiment campaigns. These files are useful
as provenance, but they should not be the starting point for new experiment
work.

`scripts/archive/ksivi_parity/` contains the April 2026 KSIVI parity inspection
helpers. They assume local historical artifacts such as an adjacent original
`KSIVI` checkout and are not part of the normal workflow.

The old standalone ELM evaluator scripts were removed. Reusable ELM helpers now
live in `utils/elm/`; write a fresh analysis entrypoint around those helpers if
that workflow is needed again.
