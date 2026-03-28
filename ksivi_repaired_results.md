# KSIVI Repaired Results

This document records the repaired KSIVI results after code changes are verified.

## Final Patch Set

Current repair stages:

- `fbac592` `Align KSIVI VI path and paired target batching`
- `b61fb9d` `Match KSIVI LR data source and add BNN warm start`
- `d0d8925` `Match official Boston KSIVI split path`
- `5a9e723` `Restore KSIVI kernel bandwidth gradients`

## Converged Runs

- Banana short repaired baseline:
  - step `2000` loss `0.0362`
  - KSD `0.0360`
  - qualitatively back in the official short-run regime
- Banana rerun after the kernel-gradient fix:
  - step `2000` loss `0.0581`
  - KL `0.4083`
  - KSD `0.0507`
  - still in the repaired regime
- X-shaped rerun after the kernel-gradient fix:
  - step `2000` KL approximately `0.0`
  - W2 `0.0708`
  - KSD approximately `0.0`
- LRwaveform repaired long reruns:
  - step `5000` KSD `0.3910`
  - step `10000` KSD `0.2157`
  - step `15000` KSD `0.1464`
  - step `20000` KSD `0.0653`
  - kernel bandwidth remains bounded near `0.86`
- Bnn_boston repaired reruns on the exact official split:
  - step `1000` RMSE `2.94`, NLL `2.56`
  - step `5000` RMSE `2.61`, NLL `2.47`
  - the last three checkpoints flatten, so this path now looks near-converged

## Structurally Non-Viable Runs

- `student_uc` is not yet confirmed converged on the repaired path.
  - at `10000` steps, KL improves to `5.33` and W2 to `4.29`, but KSD worsens to `0.2080`
  - this target still needs more comparison against the specialized official runner

## Final Notes

- `LRwaveform` now has strong fresh evidence of convergence after the bandwidth-gradient fix.
- `Bnn_boston` is now in a strong repaired regime and is close to the official short baseline by `1000` steps, then improves slightly beyond it by `5000` steps.
- Earlier Boston repaired runs in this document remain useful as trend evidence, but they predate the exact official split fix and should not be used as the final parity comparison.
- `multimodal` improves steadily with more budget and looks more like a long-horizon convergence case than a structural bug.
- Fresh ablations support keeping the repaired patch set: every tested removal caused a measurable regression, and most removals caused a severe regression.
