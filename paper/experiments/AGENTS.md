# AGENTS.md

Guidance for Codex and other agents working in `paper/experiments/`.

## Purpose

This directory is a section-only LaTeX scaffold for drafting the paper's
experiment section and appendix in isolation. The durable source files are:

- `experiments.tex`: main experiment section body (results, figures, tables).
- `experiment_appendix.tex`: appendix with shared setup, metrics, and per-experiment method configurations.

`main.tex` is only a local compile wrapper that inputs both files with
`\appendix` between them.

## Workflow

- Keep this directory focused on the experiment section and its appendix.
- Do not invent final experimental claims, numbers, captions, or conclusions
  before the corresponding generated results are available.
- Do not create placeholder macro, figure, or table files unless explicitly
  requested. `neurips_2026.sty` and `ref.bib` are real local paper assets and
  should stay aligned with the remote paper source.
- Prefer stable relative references to generated campaign artifacts, especially:

```text
../../campaigns/default_config_grid/generated_reports/finalization/figures/
../../campaigns/default_config_grid/generated_reports/finalization/tables/
```

- Do not copy the remote full paper source into this scaffold unless explicitly
  requested. Once remote assets are available, wire them into `main.tex` or merge
  the source files into the remote source through the project git workflow.
- After LaTeX edits, compile locally with:

```powershell
latexmk -pdf -outdir=build main.tex
```

- Inspect `build/main.pdf` after compilation and confirm the section renders
  cleanly. Treat `build/` outputs as generated verification artifacts.
