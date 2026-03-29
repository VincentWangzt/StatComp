# KSIVI Ablation Reference

This document records broad KSIVI ablations run after a working repair is established.

These ablations are for reference. They help explain which changes mattered and guide simplification of the final code path.

## Ablation Matrix

| ID | Target | Change | Baseline Reference | Outcome | Notes |
|---|---|---|---|---|---|
| A001 | LRwaveform | Remove differentiable kernel-bandwidth fitting when `detach_kernel=false` | repaired LR rerun `project_lrwaveform_kernelgrad_5000.log` and `project_lrwaveform_kernelgrad_20000.log` | essential | Removal branch `6058339` pushed KSD to `3.0494` at `5000` and bandwidth to about `12.8`, versus repaired KSD `0.3910` at `5000` and `0.0653` at `20000` |
| A002 | LRwaveform | Remove the official waveform `.mat` data source and fall back to the prepared split | repaired LR rerun on the official split | essential | Removal branch `3879cc7` gave KSD `62.9552` at `1000`, versus repaired KSD about `1.91` at `1000`; the prepared `4000`-sample split is a different problem |
| A003 | LRwaveform | Remove the paired-minibatch KSIVI score/logp handling for data-dependent targets | repaired LR rerun `project_lrwaveform_kernelgrad_5000.log` | useful | Removal branch `8ac96e4` degraded KSD from `0.3910` to `0.4497` at `5000`; effect is smaller than A001/A002 but still consistently worse |
| A004 | Bnn_boston | Remove the official Boston raw split/dev/normalization path and fall back to the prepared dataset | repaired Boston rerun on the exact official split | essential | Removal branch `41e141b` gave RMSE `9.5431`, NLL `5.2621`, KSD `419.9280` at `1000`, versus repaired RMSE `2.9311`, NLL `2.5350` |
| A005 | Bnn_boston | Disable KSIVI warm-start on the dev split | repaired Boston rerun on the exact official split | essential | Removal branch `e782d4a` gave RMSE `4.2753`, NLL `2.8456`, KSD `74.0377` at `1000`, versus repaired RMSE `2.9311`, NLL `2.5350` |
| A006 | banana | Remove the KSIVI-specific low-dimensional latent and variance config overrides | repaired banana rerun `project_banana_kernelgrad_2000.log` | essential | Removal branch `4e655ea` gave loss `6.1373`, KL `1.7483`, KSD `5.2592` at `2000`, versus repaired loss `0.0581`, KL `0.4083`, KSD `0.0507` |

## Simplification Decisions

- Keep the kernel-bandwidth-gradient fix. It is directly supported by the LR before/after runs and should stay in the final code.
- Keep the official waveform source and the official Boston preprocessing path. Both are parity-critical, and their removals cause severe regressions immediately.
- Keep the Boston warm-start path. Its ablation is materially worse at the same `1000`-step budget.
- Keep the low-dimensional banana KSIVI overrides. Removing them restores the original bad banana regime.
- Keep the paired-minibatch handling for correctness. Its measured effect is smaller than the other removals, but it still degrades LR and it matches the intended KSIVI estimator more closely.
- The final code can stay simple without dropping any of the kept repaired patches, because every tested removal caused a measurable regression and most were severe.

## Additional Reference Notes

- These ablations are for reference. They explain which fixes mattered, but the final code path is justified by the repaired runs rather than by ablations alone.
- All ablation commands were run on short-lived committed branches, with outputs redirected to `results/ksivi_debug/logs/` and TensorBoard logs under `tb_logs/ksivi_ablation/`.
- `student_uc` still needs a more targeted comparison against the official specialized runner before any further simplification is safe.
