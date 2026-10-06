# Canonical ConditionalGaussian KSIVI evolution

Two sequential rounds, each running three jobs in parallel:

1. Seeds 42, 43, and 44 with the canonical learning rate of 0.001.
2. Initial learning rates 5e-4, 2e-3, and 2e-4, all with seed 43.

The canonical architecture, batch size, annealing, and StepLR decay remain in use.
Toy-target binding makes the effective input noise dimension 2. The VI model has
two width-128 SiLU hidden layers and noise-dependent softplus variance. Metrics
are disabled by run overrides, leaving canonical metric defaults unchanged.
`train.vi.var_lr` is inactive for this model, which has no global `var_raw` parameter.

Each run saves 10,000 samples and a native contour plot every 5,000 updates.
The reports use the exact native plots in a 3x10 grid, with columns 5,000 through
50,000 and rows in the order above. Both PNG and PDF grids are generated.
Every panel's sample tensors and the final model/optimizer checkpoint are checked.

Run on the GPU host inside tmux:

```bash
/root/miniconda3/envs/ruivi/bin/python -u campaigns/ksivi_x_shaped_evolution_20261006/run_campaign.py \
  --results-root /root/ruivi/results/ksivi_x_shaped_evolution_20261006 \
  --tb-root /root/ruivi/tb_logs/ksivi_x_shaped_evolution_20261006
```

Campaign status is saved under the results root in `runtime/state.json`.
Reports are written under this campaign's `generated_reports/` directory.

## Conditional versus global variance investigation

The `investigation/` directory contains the author-facing mathematical report
and remote evidence for the variance comparison. `evidence_figures.pdf` includes
the proofs, complete long-run comparison table, limitations, and six scientific
figures. `report.tex` is the fuller standalone manuscript, including per-seed
gradient diagnostics and all screening results. The PDF is generated directly;
native LaTeX compilation was unavailable because of a platform-directory error.

The investigation audits 334 historical checkpoints and adds 53 remote training
runs: 23 one-seed screening runs at 10,000 updates, 21 long comparisons, three
bandwidth-detachment controls, and six fixed-bandwidth controls. Long runs have
50,000 updates and seeds 42, 43, and 44. Mean functions and initial variances
are matched, and training noise is paired. These are exploratory research
contrasts, not a preregistered significance test. Failed interventions and
counterexamples are included.

Raw artifacts are retained on the GPU server under
`/root/ruivi/results/ksivi_variance_investigation_20261006`, with TensorBoard
files under `/root/ruivi/tb_logs/ksivi_variance_investigation_20261006`. Each run
records its source commit and exact specification. The generated JSON and CSV
files retain all final metrics, evaluation errors, density integration checks,
and probe repetitions. `manifest.json` records generation revision and counts.

After pushing local code and fetching this branch on the remote server, use
the ruivi Python in tmux. Stage defaults reproduce the 53 specifications:

```bash
PY=/root/miniconda3/envs/ruivi/bin/python
$PY campaigns/ksivi_x_shaped_evolution_20261006/investigate.py validate
$PY -u campaigns/ksivi_x_shaped_evolution_20261006/investigate.py campaign --stage screen
$PY -u campaigns/ksivi_x_shaped_evolution_20261006/investigate.py campaign --stage confirm
$PY -u campaigns/ksivi_x_shaped_evolution_20261006/investigate.py campaign --stage contrast
$PY -u campaigns/ksivi_x_shaped_evolution_20261006/investigate.py campaign --stage fixed
$PY -u campaigns/ksivi_x_shaped_evolution_20261006/evidence.py evaluate \
  --output campaigns/ksivi_x_shaped_evolution_20261006/investigation/new_metrics.json
$PY -u campaigns/ksivi_x_shaped_evolution_20261006/evidence.py probes \
  --output campaigns/ksivi_x_shaped_evolution_20261006/investigation/mechanism_probes.json
$PY campaigns/ksivi_x_shaped_evolution_20261006/build_investigation.py
```

Stages skip finished runs; use `--root` and `--tbroot` pointing to new subfolders
of the existing results/log directories for a fresh campaign. Saved snapshots
contain model state and samples, not a resumable optimizer/RNG checkpoint.
The historical audit uses `investigate.py audit --root /root/ruivi/results`;
`evidence.py evaluate-old` takes the generated audit JSON as its `--root` and
a new metrics JSON path as `--output`. It evaluates one historical canonical
model per family/seed/initialization. The field `floor_fraction` uses a common
threshold of 0.00010001, including for the larger-floor controls.
Generate report artifacts remotely from committed code and return them through
Git, as required by `AGENTS.md`.
