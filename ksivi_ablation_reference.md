# KSIVI Ablation Reference

This document records broad KSIVI ablations run after a working repair is established.

These ablations are for reference. They help explain which changes mattered and guide simplification of the final code path.

## Ablation Matrix

| ID | Target | Change | Baseline Reference | Outcome | Notes |
|---|---|---|---|---|---|
| A001 | LRwaveform | Restore differentiable kernel-bandwidth fitting when `detach_kernel=false` | pre-patch repaired LR run `results/ksivi_debug/logs/project_lrwaveform_long.log` | essential | Pre-patch KSD degraded back toward `2.96` by `5000`; post-patch KSD reached `0.3910` at `5000` and `0.0653` at `20000` |
| A002 | Bnn_boston | Match official raw split, dev split, and normalization order | pre-fix Boston trend runs `project_bnn_boston_repaired_v2.log` / `project_bnn_boston_long.log` | essential | Post-fix short rerun moved to RMSE `2.94`, NLL `2.56` at `1000`, close to the official short baseline |
| A003 | Bnn_boston | Keep KSIVI warm-start on the dev split | official short baseline plus repaired Boston reruns | useful | The warm-started repaired path reaches the official regime quickly; no fresh removal ablation has been run yet |
| A004 | LRwaveform | Switch from prepared waveform split to official `.mat` split | pre-fix LR prepared-data probes | essential | The prepared `4000`-sample split was not the same problem as the official `400`-sample training split |

## Simplification Decisions

- Keep the kernel-bandwidth-gradient fix. It is directly supported by the LR before/after runs and should stay in the final code.
- Keep the official Boston preprocessing path. It is required for parity and is a narrow target-specific patch.
- Keep the final code focused on the fixes with direct experimental support; do not retain speculative changes that were not needed for the repaired runs.

## Additional Reference Notes

- These ablations are for reference. They explain which fixes mattered, but the final code path is justified by the repaired runs rather than by ablations alone.
- `student_uc` still needs a more targeted comparison against the official specialized runner before any further simplification is safe.
