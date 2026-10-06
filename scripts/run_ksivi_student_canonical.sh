#!/usr/bin/env bash
# Five canonical Student-t seeds; resume the three matching completed controls.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
campaign=toy_scatter_ksivi_detached_annealing
mkdir -p "results/$campaign" "tb_logs/$campaign" "campaigns/$campaign/runtime"
python scripts/run_default_config_grid_sweep.py \
  --campaign-slug "$campaign" \
  --results-dir "results/$campaign" --tb-dir "tb_logs/$campaign" \
  --seeds 42 43 44 45 46 --methods ksivi --targets student_uc --gpus 0 \
  --extra-override train.annealing.enabled=true \
  --extra-override train.ksivi.log_p_reg_mode=warmup_only \
  --extra-override train.ksivi.log_p_reg=0.05 \
  --extra-override train.ksivi.detach_kernel=false \
  --extra-override train.ksivi.detach_bandwidth=true \
  --extra-override train.log.metric_log_freq=0 \
  --extra-override train.plot.freq=1000000000 --retry-failed \
  > "campaigns/$campaign/runtime/canonical_controller.log" 2>&1
