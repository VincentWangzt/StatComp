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
- Verified banana repair:
  - after introducing KSIVI-specific latent width, activation, variance controls, and official-like optimizer settings, the current repo banana loss at step 2000 dropped from `7.1259` to `0.0362`
  - kernel bandwidth moved from roughly `0.52` to `0.24`, which is close to the official short-run regime
- Verified LRwaveform data mismatch:
  - current prepared waveform loader returned `4000` training examples
  - official KSIVI waveform `.mat` file uses `400` training examples
  - this dataset mismatch materially changes KSIVI behavior and must be controlled for parity runs
- Verified paired-batch bug on data-dependent targets:
  - the original current repo implementation sampled separate minibatches for `target_score1` and `target_score2` inside one KSIVI step
  - the repaired KSIVI runner now reuses the same sampled batch for both paired score evaluations and paired log-p regularization
- Verified LR improvement after official-data switch:
  - with the official waveform `.mat` split, current KSIVI no longer explodes immediately
  - KSD reached `1.45` by step `1000`, though longer-run stability still needs work
- Verified Boston improvement after warm start:
  - at `1000` steps, repaired Boston reached RMSE `10.52` and NLL `5.53`
  - at `5000` steps, repaired Boston improved to RMSE `4.20` and NLL `2.75`
  - Boston is trending toward convergence rather than showing structural failure
- Verified official Boston baseline and preprocessing path:
  - after installing `scikit-learn`, the official short Boston baseline ran locally and reported RMSE `2.6416` and test log-likelihood `-2.5108` at epoch `10/10`
  - the first current-repo `official_raw` Boston loader was still wrong because it standardized before removing the dev split
  - after moving dev splitting ahead of standardization, `verify_boston_official_split_v2.txt` shows exact equality for train, dev, test, and target-scaling statistics between the current repo and the official script

## Active Hypotheses

- The primary KSIVI breakage is not just config values; it includes a VI-family mismatch between the unified repo and the official KSIVI implementation.
- High-dimensional KSIVI failures likely combine multiple factors:
  - wrong latent width
  - wrong variance initialization/optimization regime
  - missing or incomplete target-specific warm-start paths
  - mismatched data-batching, stochastic-scaling, or data-source semantics
  - possible objective/regularization scheduling drift
- Remaining LR instability after step `1000` may come from:
  - optimizer settings that are still not well matched to the official LR training path
  - kernel-width dynamics as sample norms increase
  - or an LR-specific sensitivity to long-run variance growth in the current VI implementation
- Boston may still benefit from:
  - fresh reruns now that the official split path is exact
  - longer training budgets before additional code changes
  - or target-specific optimizer tuning after more long-run evidence is collected

## Rejected Hypotheses

- `LRwaveform` failure was not solely caused by missing `scale_sto=10` on the current prepared dataset.
  - forcing that scale on the prepared 4000-sample split made the run much worse.

## Patch Notes

- Commit `fbac592`: aligned KSIVI VI path with official latent-width and variance controls, added official-like optimizer knobs, fixed gradient-norm logging, and fixed paired-batch handling for data-dependent KSIVI target scores.
- Commit `b61fb9d`: added LR official-data loading support and Boston KSIVI warm-start support.
- Pending commit: adjust the current repo Boston `official_raw` path so dev splitting happens before normalization, matching the official script exactly.
