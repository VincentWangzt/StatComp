#!/usr/bin/env bash

# KDVI 8-Gaussians learning-rate / StepLR-gamma sweep.
#
# The caller is responsible for activating the intended Python environment.
# This script invokes `python` directly, schedules one run per visible GPU,
# resumes completed work, retries failures once, and aggregates final KL/W2
# metrics across the three seeds in each W&B recipe group.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CAMPAIGN_SLUG="kdvi_8gaussian_lr_sweep"
CAMPAIGN_ROOT="${REPO_ROOT}/campaigns/${CAMPAIGN_SLUG}"
RUNTIME_ROOT="${CAMPAIGN_ROOT}/runtime"
MANIFEST_PATH="${RUNTIME_ROOT}/manifest.tsv"
LOG_DIR="${RUNTIME_ROOT}/logs"
ATTEMPT_DIR="${RUNTIME_ROOT}/attempts"
DONE_DIR="${RUNTIME_ROOT}/done"
FAILED_DIR="${RUNTIME_ROOT}/failed"
RESULT_MAP_DIR="${RUNTIME_ROOT}/result_paths"
SUMMARY_CSV="${CAMPAIGN_ROOT}/summary.csv"
SUMMARY_MD="${CAMPAIGN_ROOT}/summary.md"
CONFIG_PATH="configs/kdvi_8_gaussians.yaml"

MAX_GPUS=10
GPU_IDS_ARG=""
DRY_RUN=0
SUMMARIZE_ONLY=0

usage() {
    cat <<'EOF'
Usage: bash scripts/run_kdvi_8gaussian_lr_sweep.sh [options]

Options:
  --dry-run           Generate and validate the 90-run manifest only.
  --max-gpus N        Use at most N visible GPUs (default: 10, hard cap: 10).
  --gpu-ids CSV       Use these numeric GPU IDs instead of auto-discovery.
  --summarize-only    Rebuild summary.csv and summary.md without launching jobs.
  -h, --help          Show this help.

GPU discovery first honors --gpu-ids, then CUDA_VISIBLE_DEVICES, then
nvidia-smi. The active environment's `python` executable is used directly.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

while (($#)); do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --summarize-only)
            SUMMARIZE_ONLY=1
            shift
            ;;
        --max-gpus)
            (($# >= 2)) || die "--max-gpus requires a value"
            MAX_GPUS="$2"
            shift 2
            ;;
        --gpu-ids)
            (($# >= 2)) || die "--gpu-ids requires a comma-separated value"
            GPU_IDS_ARG="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

[[ "${MAX_GPUS}" =~ ^[1-9][0-9]*$ ]] || die "--max-gpus must be a positive integer"
if ((MAX_GPUS > 10)); then
    echo "Capping --max-gpus at 10."
    MAX_GPUS=10
fi
if ((DRY_RUN && SUMMARIZE_ONLY)); then
    die "--dry-run and --summarize-only are mutually exclusive"
fi

mkdir -p \
    "${LOG_DIR}" \
    "${ATTEMPT_DIR}" \
    "${DONE_DIR}" \
    "${FAILED_DIR}" \
    "${RESULT_MAP_DIR}"

lr_label() {
    local value="$1"
    value="${value/./p}"
    value="${value//-/m}"
    printf '%s\n' "${value}"
}

gamma_label() {
    local value="$1"
    value="${value/./p}"
    printf '%s\n' "${value}"
}

generate_manifest() {
    local tmp_path="${MANIFEST_PATH}.tmp"
    local lr gamma seed recipe run_id
    local lr_slug gamma_slug

    printf 'run_id\trecipe_id\tseed\tlr\tscheduler_step_size\tscheduler_gamma\n' >"${tmp_path}"

    for lr in 1e-4 2e-4 5e-4 1e-3 2e-3; do
        lr_slug="$(lr_label "${lr}")"
        for gamma in 0.5 0.75 0.85 0.9 0.95 1.0; do
            gamma_slug="$(gamma_label "${gamma}")"
            recipe="lr${lr_slug}-steplr5000-gamma${gamma_slug}"
            for seed in 0 1 7; do
                run_id="${recipe}-seed${seed}"
                printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "${run_id}" "${recipe}" "${seed}" "${lr}" "5000" "${gamma}" \
                    >>"${tmp_path}"
            done
        done
    done

    mv "${tmp_path}" "${MANIFEST_PATH}"
}

validate_manifest() {
    python - "${MANIFEST_PATH}" <<'PY'
import csv
import sys
from collections import Counter, defaultdict

path = sys.argv[1]
with open(path, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))

run_ids = [row["run_id"] for row in rows]
groups = defaultdict(list)
lr_values = set()
gamma_values = set()
for row in rows:
    groups[row["recipe_id"]].append(int(row["seed"]))
    lr_values.add(row["lr"])
    gamma_values.add(row["scheduler_gamma"])

errors = []
if len(rows) != 90:
    errors.append(f"expected 90 runs, found {len(rows)}")
if len(set(run_ids)) != 90:
    errors.append("run IDs are not unique")
if len(groups) != 30:
    errors.append(f"expected 30 recipe groups, found {len(groups)}")
bad_groups = {
    recipe: seeds for recipe, seeds in groups.items()
    if sorted(seeds) != [0, 1, 7]
}
if bad_groups:
    errors.append(f"groups without exactly seeds 0,1,7: {len(bad_groups)}")

expected_lrs = {"1e-4", "2e-4", "5e-4", "1e-3", "2e-3"}
expected_gammas = {"0.5", "0.75", "0.85", "0.9", "0.95", "1.0"}
if lr_values != expected_lrs:
    errors.append(f"unexpected lr values: {sorted(lr_values)}")
if gamma_values != expected_gammas:
    errors.append(f"unexpected gamma values: {sorted(gamma_values)}")

step_counts = Counter(row["scheduler_step_size"] for row in rows)
if step_counts != {"5000": 90}:
    errors.append(f"unexpected scheduler step sizes: {dict(step_counts)}")

if errors:
    raise SystemExit("Manifest validation failed:\n- " + "\n- ".join(errors))

print(
    "Manifest validated: "
    f"{len(rows)} runs, {len(groups)} groups, "
    f"{len(lr_values)} learning rates, {len(gamma_values)} gammas."
)
PY
}

if ! generate_manifest; then
    exit 1
fi
if ! validate_manifest; then
    exit 1
fi

build_command() {
    local -n output_ref=$1
    local gpu_id="$2"
    local run_name="$3"
    local recipe="$4"
    local seed="$5"
    local lr="$6"
    local scheduler_step_size="$7"
    local scheduler_gamma="$8"

    output_ref=(
        python src.py --config "${CONFIG_PATH}"
        "cuda_visible_devices=${gpu_id}"
        "seed=${seed}"
        "output.results_dir=results/${CAMPAIGN_SLUG}/${run_name}"
        "train.vi.lr=${lr}"
        "train.vi.scheduler.step_size=${scheduler_step_size}"
        "train.vi.scheduler.gamma=${scheduler_gamma}"
        "tracking.campaign=${CAMPAIGN_SLUG}"
        "tracking.group=${recipe}"
        "tracking.run_name=${run_name}"
    )
}

if ((DRY_RUN)); then
    echo "Dry run complete; no experiments were launched."
    echo "Manifest: ${MANIFEST_PATH}"
    echo "First five jobs:"
    tail -n +2 "${MANIFEST_PATH}" | head -n 5
    echo "First command:"
    first_job="$(tail -n +2 "${MANIFEST_PATH}" | head -n 1)"
    IFS=$'\t' read -r run_id recipe seed lr scheduler_step_size scheduler_gamma <<<"${first_job}"
    cmd=()
    build_command cmd 0 "${run_id}" "${recipe}" "${seed}" "${lr}" "${scheduler_step_size}" "${scheduler_gamma}"
    printf '  %q' "${cmd[@]}"
    printf '\n'
    exit 0
fi

summarize_results() {
    python - \
        "${REPO_ROOT}" \
        "${MANIFEST_PATH}" \
        "${RESULT_MAP_DIR}" \
        "${SUMMARY_CSV}" \
        "${SUMMARY_MD}" <<'PY'
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

repo_root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
result_map_dir = Path(sys.argv[3])
summary_csv = Path(sys.argv[4])
summary_md = Path(sys.argv[5])

KL_TAG = "metric/vi_model/kl_ite"
W2_TAG = "metric/vi_model/w2"
FINAL_STEP = 100000

with manifest_path.open(newline="", encoding="utf-8") as handle:
    manifest = list(csv.DictReader(handle, delimiter="\t"))

by_recipe = defaultdict(list)
recipe_meta = {}

def final_metrics(metrics_path):
    values = {}
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("tag") not in {KL_TAG, W2_TAG}:
                continue
            try:
                step = int(row["step"])
                value = float(row["value"])
            except (KeyError, TypeError, ValueError):
                continue
            if step == FINAL_STEP and math.isfinite(value):
                values[row["tag"]] = value
    if KL_TAG not in values or W2_TAG not in values:
        return None
    return values[KL_TAG], values[W2_TAG]

for item in manifest:
    recipe = item["recipe_id"]
    recipe_meta[recipe] = item
    mapping = result_map_dir / f"{item['run_id']}.path"
    if not mapping.is_file():
        continue
    raw_path = mapping.read_text(encoding="utf-8").strip()
    result_path = Path(raw_path)
    if not result_path.is_absolute():
        result_path = repo_root / result_path
    metrics_path = result_path / "metrics.csv"
    if not metrics_path.is_file():
        continue
    metrics = final_metrics(metrics_path)
    if metrics is None:
        continue
    by_recipe[recipe].append((int(item["seed"]), *metrics))

rows = []
for recipe in sorted(recipe_meta):
    meta = recipe_meta[recipe]
    samples = sorted(by_recipe.get(recipe, []))
    seeds = [sample[0] for sample in samples]
    complete = seeds == [0, 1, 7]
    kl_values = [sample[1] for sample in samples]
    w2_values = [sample[2] for sample in samples]
    rows.append({
        "recipe_id": recipe,
        "lr": meta["lr"],
        "scheduler_step_size": int(meta["scheduler_step_size"]),
        "scheduler_gamma": meta["scheduler_gamma"],
        "seeds_complete": ",".join(str(seed) for seed in seeds),
        "n_seeds": len(seeds),
        "kl_mean": statistics.mean(kl_values) if kl_values else None,
        "kl_std": statistics.stdev(kl_values) if len(kl_values) >= 2 else None,
        "w2_mean": statistics.mean(w2_values) if w2_values else None,
        "w2_std": statistics.stdev(w2_values) if len(w2_values) >= 2 else None,
        "pareto": False,
        "status": "complete" if complete else "incomplete",
    })

complete_rows = [row for row in rows if row["status"] == "complete"]
for candidate in complete_rows:
    candidate["pareto"] = not any(
        other is not candidate
        and other["kl_mean"] <= candidate["kl_mean"]
        and other["w2_mean"] <= candidate["w2_mean"]
        and (
            other["kl_mean"] < candidate["kl_mean"]
            or other["w2_mean"] < candidate["w2_mean"]
        )
        for other in complete_rows
    )

fieldnames = [
    "recipe_id", "lr", "scheduler_step_size", "scheduler_gamma",
    "seeds_complete", "n_seeds", "kl_mean", "kl_std",
    "w2_mean", "w2_std", "pareto", "status",
]
summary_csv.parent.mkdir(parents=True, exist_ok=True)
with summary_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

def fmt(value):
    return "-" if value is None else f"{value:.6f}"

lines = [
    "# KDVI 8-Gaussians LR Sweep Summary",
    "",
    f"Complete recipe groups: **{len(complete_rows)} / {len(rows)}**.",
    "Metrics are final epoch-100K means and sample standard deviations "
    "across seeds 0, 1, and 7.",
    "",
]

if complete_rows:
    kl_winner = min(complete_rows, key=lambda row: row["kl_mean"])
    w2_winner = min(complete_rows, key=lambda row: row["w2_mean"])
    lines.extend([
        "## Winners",
        "",
        f"- **KL-ITE:** `{kl_winner['recipe_id']}` - "
        f"{fmt(kl_winner['kl_mean'])} +/- {fmt(kl_winner['kl_std'])}",
        f"- **W2:** `{w2_winner['recipe_id']}` - "
        f"{fmt(w2_winner['w2_mean'])} +/- {fmt(w2_winner['w2_std'])}",
        "",
        "## KL/W2 Pareto front",
        "",
        "| Recipe | KL-ITE mean +/- std | W2 mean +/- std |",
        "|---|---:|---:|",
    ])
    pareto_rows = sorted(
        (row for row in complete_rows if row["pareto"]),
        key=lambda row: (row["kl_mean"], row["w2_mean"]),
    )
    for row in pareto_rows:
        lines.append(
            f"| `{row['recipe_id']}` | {fmt(row['kl_mean'])} +/- "
            f"{fmt(row['kl_std'])} | {fmt(row['w2_mean'])} +/- "
            f"{fmt(row['w2_std'])} |"
        )
    lines.append("")
else:
    lines.extend(["No recipe currently has all three final metrics.", ""])

incomplete = [row for row in rows if row["status"] != "complete"]
lines.extend(["## Incomplete recipe groups", ""])
if incomplete:
    lines.extend([
        "| Recipe | Complete seeds |",
        "|---|---|",
    ])
    for row in incomplete:
        lines.append(
            f"| `{row['recipe_id']}` | {row['seeds_complete'] or 'none'} |"
        )
else:
    lines.append("None.")
lines.append("")

summary_md.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {summary_csv}")
print(f"Wrote {summary_md}")
PY
}

if ((SUMMARIZE_ONLY)); then
    summarize_results
    exit 0
fi

command -v python >/dev/null 2>&1 || die "python was not found in the active environment"
if ((BASH_VERSINFO[0] < 5)); then
    die "Bash 5 or newer is required for dynamic GPU scheduling"
fi

discover_gpus() {
    local raw_ids=""
    local candidate
    local -a parsed=()
    local -A seen=()

    if [[ -n "${GPU_IDS_ARG}" ]]; then
        raw_ids="${GPU_IDS_ARG}"
    elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        raw_ids="${CUDA_VISIBLE_DEVICES}"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        raw_ids="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
    fi

    [[ -n "${raw_ids}" ]] || die "no GPUs found; use --gpu-ids or set CUDA_VISIBLE_DEVICES"
    IFS=',' read -r -a candidates <<<"${raw_ids}"
    for candidate in "${candidates[@]}"; do
        candidate="${candidate//[[:space:]]/}"
        [[ "${candidate}" =~ ^[0-9]+$ ]] || die "GPU IDs must be numeric; got '${candidate}'"
        if [[ -z "${seen[${candidate}]+x}" ]]; then
            parsed+=("${candidate}")
            seen["${candidate}"]=1
        fi
    done
    ((${#parsed[@]} > 0)) || die "no valid GPU IDs found"
    GPU_IDS=("${parsed[@]:0:${MAX_GPUS}}")
}

declare -a GPU_IDS=()
discover_gpus
echo "Using GPUs: ${GPU_IDS[*]}"

extract_result_path() {
    local log_path="$1"
    local result_path
    result_path="$(sed -n 's/.*Artifacts will be saved to: //p' "${log_path}" | tail -n 1 | tr -d '\r')"
    [[ -n "${result_path}" ]] || return 1
    printf '%s\n' "${result_path}"
}

run_job() {
    local gpu_id="$1"
    local run_id="$2"
    local recipe="$3"
    local seed="$4"
    local lr="$5"
    local scheduler_step_size="$6"
    local scheduler_gamma="$7"
    local attempt_file="${ATTEMPT_DIR}/${run_id}.txt"
    local done_file="${DONE_DIR}/${run_id}.done"
    local failed_file="${FAILED_DIR}/${run_id}.failed"
    local result_map_file="${RESULT_MAP_DIR}/${run_id}.path"
    local attempt=0

    if [[ -f "${done_file}" && -f "${result_map_file}" ]]; then
        return 0
    fi
    rm -f "${done_file}"
    if [[ -f "${attempt_file}" ]]; then
        attempt="$(<"${attempt_file}")"
    fi

    while ((attempt < 2)); do
        attempt=$((attempt + 1))
        printf '%s\n' "${attempt}" >"${attempt_file}"

        local run_name="${run_id}"
        if ((attempt > 1)); then
            run_name="${run_id}-retry1"
        fi
        local log_path="${LOG_DIR}/${run_id}-attempt${attempt}.log"
        local -a cmd=()
        build_command cmd "${gpu_id}" "${run_name}" "${recipe}" "${seed}" \
            "${lr}" "${scheduler_step_size}" "${scheduler_gamma}"

        echo "[$(date -Is)] GPU ${gpu_id}: ${run_name}"
        (
            cd "${REPO_ROOT}" || exit 2
            exec "${cmd[@]}"
        ) >"${log_path}" 2>&1 &
        local child_pid=$!
        trap 'kill -TERM "${child_pid}" 2>/dev/null || true; wait "${child_pid}" 2>/dev/null || true; exit 130' TERM INT
        wait "${child_pid}"
        local rc=$?
        trap - TERM INT

        if ((rc == 0)); then
            local result_path
            if result_path="$(extract_result_path "${log_path}")"; then
                printf '%s\n' "${result_path}" >"${result_map_file}"
                rm -f "${failed_file}"
                touch "${done_file}"
                echo "[$(date -Is)] GPU ${gpu_id}: completed ${run_name}"
                return 0
            fi
            echo "[$(date -Is)] GPU ${gpu_id}: ${run_name} exited successfully but its result path was not found" >&2
        else
            echo "[$(date -Is)] GPU ${gpu_id}: ${run_name} failed with exit code ${rc}" >&2
        fi
    done

    touch "${failed_file}"
    return 1
}

declare -a JOBS=()
while IFS=$'\t' read -r run_id recipe seed lr scheduler_step_size scheduler_gamma; do
    [[ "${run_id}" == "run_id" ]] && continue
    if [[ -f "${DONE_DIR}/${run_id}.done" && -f "${RESULT_MAP_DIR}/${run_id}.path" ]]; then
        continue
    fi
    JOBS+=("${run_id}"$'\t'"${recipe}"$'\t'"${seed}"$'\t'"${lr}"$'\t'"${scheduler_step_size}"$'\t'"${scheduler_gamma}")
done <"${MANIFEST_PATH}"

echo "Pending runs: ${#JOBS[@]} / 90"

declare -A ACTIVE_GPU=()
declare -A ACTIVE_RUN=()
declare -a FREE_GPUS=("${GPU_IDS[@]}")
NEXT_JOB=0
FAILURES=0
INTERRUPTED=0

terminate_children() {
    INTERRUPTED=1
    echo "Stopping active sweep jobs..." >&2
    local pid
    for pid in "${!ACTIVE_GPU[@]}"; do
        kill -TERM "${pid}" 2>/dev/null || true
    done
    for pid in "${!ACTIVE_GPU[@]}"; do
        wait "${pid}" 2>/dev/null || true
    done
}
trap terminate_children INT TERM

launch_available_jobs() {
    while ((${#FREE_GPUS[@]} > 0 && NEXT_JOB < ${#JOBS[@]})); do
        local gpu_index=$((${#FREE_GPUS[@]} - 1))
        local gpu_id="${FREE_GPUS[${gpu_index}]}"
        unset 'FREE_GPUS[gpu_index]'
        FREE_GPUS=("${FREE_GPUS[@]}")

        local job="${JOBS[${NEXT_JOB}]}"
        NEXT_JOB=$((NEXT_JOB + 1))
        local run_id recipe seed lr scheduler_step_size scheduler_gamma
        IFS=$'\t' read -r run_id recipe seed lr scheduler_step_size scheduler_gamma <<<"${job}"

        run_job "${gpu_id}" "${run_id}" "${recipe}" "${seed}" \
            "${lr}" "${scheduler_step_size}" "${scheduler_gamma}" &
        local pid=$!
        ACTIVE_GPU["${pid}"]="${gpu_id}"
        ACTIVE_RUN["${pid}"]="${run_id}"
    done
}

launch_available_jobs
while ((${#ACTIVE_GPU[@]} > 0)); do
    finished_pid=""
    wait -n -p finished_pid "${!ACTIVE_GPU[@]}"
    rc=$?
    ((INTERRUPTED)) && break
    [[ -n "${finished_pid}" ]] || continue
    gpu_id="${ACTIVE_GPU[${finished_pid}]}"
    run_id="${ACTIVE_RUN[${finished_pid}]}"
    unset 'ACTIVE_GPU['"${finished_pid}"']'
    unset 'ACTIVE_RUN['"${finished_pid}"']'
    FREE_GPUS+=("${gpu_id}")
    if ((rc != 0)); then
        FAILURES=$((FAILURES + 1))
        echo "Run exhausted its retries: ${run_id}" >&2
    fi
    launch_available_jobs
done

trap - INT TERM
if ((INTERRUPTED)); then
    exit 130
fi

summarize_results

if ((FAILURES > 0)); then
    echo "Sweep finished with ${FAILURES} run(s) still failed after retry." >&2
    exit 1
fi

echo "Sweep completed successfully."
