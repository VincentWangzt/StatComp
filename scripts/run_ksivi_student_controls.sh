#!/usr/bin/env bash
# Matched seed-43 runs for annealing and persistent log-density regularization.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
for condition in annealing logp; do
  campaign="toy_scatter_ksivi_${condition}"
  overrides=(train.annealing.enabled=true train.ksivi.log_p_reg_mode=warmup_only)
  if [ "$condition" = logp ]; then
    overrides=(train.annealing.enabled=false train.ksivi.log_p_reg_mode=always)
  fi
  mkdir -p "results/$campaign" "tb_logs/$campaign"
  python scripts/run_default_config_grid_sweep.py \
    --campaign-slug "$campaign" \
    --results-dir "results/$campaign" --tb-dir "tb_logs/$campaign" \
    --seeds 43 --methods ksivi --targets student_uc --gpus 0 \
    --extra-override "${overrides[0]}" --extra-override "${overrides[1]}" \
    --extra-override train.ksivi.log_p_reg=0.05 \
    --extra-override train.log.metric_log_freq=0 \
    --extra-override train.plot.freq=1000000000 --retry-failed
done
python scripts/compare_ksivi_student_controls.py
