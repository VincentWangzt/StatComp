# KSIVI Debugging Observations

This document tracks active hypotheses, code-reading notes, repair decisions, and verification notes.

Rules for this document:

- Every conclusion must cite a fresh run, direct code inspection, or both.
- Keep hypotheses separate from verified findings.
- When a hypothesis is invalidated, leave it here and mark it as rejected.

## Verified Findings

- Fresh banana baseline mismatch:
  - current repo short run at 2000 epochs remains at loss `7.1259`, KL `2.2191`, and average sample norm around `2.54`
  - official short run at the same 2000 inner iterations reaches loss `0.0554`
  - therefore the current KSIVI implementation is materially different from the official one before considering target-specific wrappers
- Code-reading comparison already shows a major architectural drift:
  - official `SIMINet` uses low-dimensional latent noise (`train.z_dim`) mapped to target dimension (`train.out_dim`)
  - current repo ties `epsilon_dim` to target dimension through the default VI configs
  - this drift is present for banana (`3 -> 2` mismatch), LRwaveform (`10 -> 22` mismatch), and BNN Boston (`3 -> 751` mismatch)
- An earlier controlled probe in this worktree showed that making the current repo behave more like the official latent-width/variance regime on banana immediately moved the short loss curve into the official scale.

## Active Hypotheses

- The primary KSIVI breakage is not just config values; it includes a VI-family mismatch between the unified repo and the official KSIVI implementation.
- High-dimensional KSIVI failures likely combine multiple factors:
  - wrong latent width
  - wrong variance initialization/optimization regime
  - missing BNN warm-start path
  - mismatched data-batching or stochastic-scaling semantics
  - possible objective/regularization scheduling drift

## Rejected Hypotheses

## Patch Notes

- No code patch yet. Baseline and code-comparison evidence currently points first toward KSIVI-specific VI-path and training-path repairs.
