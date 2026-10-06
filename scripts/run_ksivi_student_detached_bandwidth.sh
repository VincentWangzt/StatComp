#!/usr/bin/env bash
# Nine Student-t runs: detach only the median bandwidth, three seeds per setup.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
controller_pids=()
for condition in annealing logp plain; do
  campaign="toy_scatter_ksivi_detached_${condition}"
  overrides=(train.annealing.enabled=true train.ksivi.log_p_reg_mode=warmup_only train.ksivi.log_p_reg=0.05)
  if [ "$condition" = logp ]; then
    overrides=(train.annealing.enabled=false train.ksivi.log_p_reg_mode=always train.ksivi.log_p_reg=0.05)
  elif [ "$condition" = plain ]; then
    overrides=(train.annealing.enabled=false train.ksivi.log_p_reg_mode=warmup_only train.ksivi.log_p_reg=0.0)
  fi
  mkdir -p "results/$campaign" "tb_logs/$campaign" "campaigns/$campaign/runtime"
  python scripts/run_default_config_grid_sweep.py \
    --campaign-slug "$campaign" \
    --results-dir "results/$campaign" --tb-dir "tb_logs/$campaign" \
    --seeds 42 43 44 --methods ksivi --targets student_uc --gpus 0 \
    --extra-override "${overrides[0]}" --extra-override "${overrides[1]}" \
    --extra-override "${overrides[2]}" \
    --extra-override train.ksivi.detach_kernel=false \
    --extra-override train.ksivi.detach_bandwidth=true \
    --extra-override train.log.metric_log_freq=0 \
    --extra-override train.plot.freq=1000000000 --retry-failed \
    > "campaigns/$campaign/runtime/controller.log" 2>&1 &
  controller_pids+=("$!")
done
controller_status=0
for controller_pid in "${controller_pids[@]}"; do
  if ! wait "$controller_pid"; then
    controller_status=1
  fi
done
if [ "$controller_status" -ne 0 ]; then
  exit "$controller_status"
fi
python scripts/compare_ksivi_student_detached_bandwidth.py
