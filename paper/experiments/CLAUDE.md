# CLAUDE.md — paper/experiments/

Guidance for Claude Code sessions working in this directory.

## Purpose

This directory is a standalone LaTeX scaffold for drafting the paper's experiment
section and its appendix in isolation. The two content files are:

- `experiments.tex`: main experiment section body (Sec. 5 in the paper). Contains
  subsections for toy 2D targets, conditioned diffusion, and BNN regression, with
  figures, tables, and high-level discussion.
- `experiment_appendix.tex`: appendix (App. A). Contains shared setup (variational
  family, training protocol), metric definitions, and per-experiment detailed
  method configurations.

`main.tex` is a local compile wrapper that inputs both files with `\appendix`
between them.

## Key Conventions

- **Do not invent results.** Never fabricate experimental numbers, captions, or
  conclusions. Only reference generated artifacts that actually exist.
- **Reference campaign outputs by path.** Figures and tables come from:
  ```
  ../../campaigns/default_config_grid/generated_reports/finalization/figures/
  ../../campaigns/default_config_grid/generated_reports/finalization/tables/
  ```
  Use the `\finalizationfigdir` and `\finalizationtabledir` macros defined in
  `main.tex`.
- **Keep the split clean.** Main experimental narrative goes in `experiments.tex`;
  detailed configurations, hyperparameter tables, and metric derivations go in
  `experiment_appendix.tex`. Cross-references between the two use `\cref`.
- **Style files are real.** `neurips_2026.sty` and `ref.bib` are shared with the
  remote paper source. Keep them aligned; do not modify without explicit request.
- **No placeholder files.** Do not create stub figure/table files. The build may
  warn about missing graphics — that is expected until campaign artifacts are
  generated.

## Compilation

From this directory:

```bash
latexmk -pdf -outdir=build main.tex
```

Output: `build/main.pdf`. Treat `build/` contents as disposable verification
artifacts.

## Editing Guidelines

1. When adding a new experiment subsection, put the narrative + figure + table
   reference in `experiments.tex`, and the detailed method config paragraph in
   the corresponding `\subsection` of `experiment_appendix.tex`.
2. When updating hyperparameters or method details, update
   `experiment_appendix.tex`; keep `experiments.tex` at a summary level.
3. Maintain consistent label conventions:
   - Main sections: `\label{subsec:*-experiments}`
   - Appendix sections: `\label{app:*-details}`
4. All `\input` paths for tables use `\finalizationtabledir` macro.
5. All `\includegraphics` paths rely on `\graphicspath` set in `main.tex`.
