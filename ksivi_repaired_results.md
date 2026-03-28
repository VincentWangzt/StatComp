# KSIVI Repaired Results

This document records the repaired KSIVI results after code changes are verified.

## Final Patch Set

Current repair stages:

- `fbac592` `Align KSIVI VI path and paired target batching`
- `b61fb9d` `Match KSIVI LR data source and add BNN warm start`
- pending commit: exact official Boston split/dev/normalization path for `target.data.source=official_raw`

## Converged Runs

- Banana short repaired baseline:
  - step `2000` loss `0.0362`
  - KSD `0.0360`
  - qualitatively back in the official short-run regime

## Structurally Non-Viable Runs

None confirmed yet in the repaired path.

## Final Notes

- `LRwaveform` is repaired enough to avoid immediate divergence on the official dataset split, but longer-run stability is not yet locked.
- `Bnn_boston` is not yet at final convergence, but the repaired path now shows a strong monotonic improvement trend through `5000` steps.
- Earlier Boston repaired runs in this document remain useful as trend evidence, but they predate the exact official split fix and should not be used as the final parity comparison.
