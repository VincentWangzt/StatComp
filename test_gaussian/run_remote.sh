#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p test_gaussian/logs

declare -A CONFIGS=(
  [DIVI]="test_gaussian/configs/divi_flat_gaussian.yaml"
  [UIVI]="test_gaussian/configs/uivi_flat_gaussian.yaml"
  [KDVI]="test_gaussian/configs/kdvi_flat_gaussian.yaml"
)

for method in DIVI UIVI KDVI; do
  config="${CONFIGS[$method]}"
  log_path="test_gaussian/logs/${method}.log"
  echo "[$(date -Is)] Running ${method}: ${config}" | tee "${log_path}"
  python src.py --config "${config}" 2>&1 | tee -a "${log_path}"
done

python test_gaussian/plot_curves.py --results-root results/test_gaussian --output-dir test_gaussian

