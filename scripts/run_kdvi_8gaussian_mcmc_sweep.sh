#!/usr/bin/env bash

# Exhaustive KDVI 8-Gaussians MCMC sweep.
#
# The caller is responsible for activating the intended Python environment.
# This script invokes `python` directly, schedules one run per visible GPU,
# resumes completed work, retries failures once, and aggregates final KL/W2
# metrics across the three seeds in each W&B recipe group.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CAMPAIGN_SLUG="kdvi_8gaussian_mcmc_sweep"
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
Usage: bash scripts/run_kdvi_8gaussian_mcmc_sweep.sh [options]

Options:
  --dry-run           Generate and validate the 360-run manifest only.
  --max-gpus N        Use at most N visible GPUs (default: 10, hard cap: 10).
  --gpu-ids CSV       Use these numeric GPU IDs instead of auto-discovery.
  --summarize-only    Rebuild summary.csv and summary.md without launching jobs.
  -h, --help          Show this help.

GPU discovery first honors CUDA_VISIBLE_DEVICES and otherwise queries
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

generate_manifest() {
    local tmp_path="${MANIFEST_PATH}.tmp"
    local mcmc_type nominal_step k_mode anneal_steps seed recipe run_id

    printf 'run_id\trecipe_id\tseed\tmcmc_type\tstep_schedule\tnominal_step\tk_mode\tannealing_steps\n' >"${tmp_path}"

    for mcmc_type in mala sgld; do
        for nominal_step in 0.50 0.20 0.10 0.05 0.02; do
            local step_label="${nominal_step/./p}"
            for k_mode in k1 k2 k5 k10 k1to20; do
                for anneal_steps in 50000 100000; do
                    local ann_label="$((anneal_steps / 1000))k"
                    recipe="${mcmc_type}-coupled-step${step_label}-${k_mode}-ann${ann_label}"
                    for seed in 0 1 7; do
                        run_id="${recipe}-seed${seed}"
                        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                            "${run_id}" "${recipe}" "${seed}" "${mcmc_type}" \
                            coupled "${nominal_step}" "${k_mode}" "${anneal_steps}" \
                            >>"${tmp_path}"
                    done
                done
            done
        done
    done

    for mcmc_type in mala sgld; do
        for k_mode in k1 k2 k5 k10 k1to20; do
            for anneal_steps in 50000 100000; do
                local ann_label="$((anneal_steps / 1000))k"
                recipe="${mcmc_type}-cos1to0p01d${ann_label}-${k_mode}-ann${ann_label}"
                for seed in 0 1 7; do
                    run_id="${recipe}-seed${seed}"
                    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                        "${run_id}" "${recipe}" "${seed}" "${mcmc_type}" \
                        cosine NA "${k_mode}" "${anneal_steps}" \
                        >>"${tmp_path}"
                done
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
for row in rows:
    groups[row["recipe_id"]].append(int(row["seed"]))

errors = []
if len(rows) != 360:
    errors.append(f"expected 360 runs, found {len(rows)}")
if len(set(run_ids)) != 360:
    errors.append("run IDs are not unique")
if len(groups) != 120:
    errors.append(f"expected 120 recipe groups, found {len(groups)}")
bad_groups = {
    recipe: seeds for recipe, seeds in groups.items()
    if sorted(seeds) != [0, 1, 7]
}
if bad_groups:
    errors.append(f"groups without exactly seeds 0,1,7: {len(bad_groups)}")

schedule_counts = Counter(row["step_schedule"] for row in rows)
if schedule_counts != {"coupled": 300, "cosine": 60}:
    errors.append(f"unexpected schedule counts: {dict(schedule_counts)}")

anchor = [
    row for row in rows
    if row["mcmc_type"] == "mala"
    and row["step_schedule"] == "coupled"
    and row["nominal_step"] == "0.50"
    and row["k_mode"] == "k1"
    and row["annealing_steps"] == "50000"
]
if sorted(int(row["seed"]) for row in anchor) != [0, 1, 7]:
    errors.append("the current-default anchor is missing or malformed")

if errors:
    raise SystemExit("Manifest validation failed:\n- " + "\n- ".join(errors))

print(
    "Manifest validated: "
    f"{len(rows)} runs, {len(groups)} groups, "
    f"{schedule_counts['coupled']} coupled, {schedule_counts['cosine']} cosine."
)
PY
}

generate_manifest
validate_manifest

build_command() {
    local -n output_ref=$1
    local gpu_id="$2"
    local run_name="$3"
    local recipe="$4"
    local seed="$5"
    local mcmc_type="$6"
    local step_schedule="$7"
    local nominal_step="$8"
    local k_mode="$9"
    local anneal_steps="${10}"

    output_ref=(
        python src.py --config "${CONFIG_PATH}"
        "use_cuda=true"
        "cuda_visible_devices=${gpu_id}"
        "seed=${seed}"
        "output.results_dir=results/${CAMPAIGN_SLUG}/${run_name}"
        "train.epochs=100000"
        "train.annealing.enabled=true"
        "train.annealing.scheme=linear"
        "train.annealing.steps=${anneal_steps}"
        "train.kdvi.mcmc_type=${mcmc_type}"
        "metric.kl_ite.enabled=true"
        "metric.kl_ite.num_samples=5000"
        "metric.w2.enabled=true"
        "metric.w2.num_samples=5000"
        "metric.w2.num_projections=100"
        "train.log.metric_log_freq=5000"
        "tracking.campaign=${CAMPAIGN_SLUG}"
        "tracking.group=${recipe}"
        "tracking.run_name=${run_name}"
    )

    if [[ "${k_mode}" == "k1to20" ]]; then
        output_ref+=(
            "train.kdvi.mcmc_steps=1"
            "train.kdvi.mcmc_steps_schedule.enabled=true"
            "train.kdvi.mcmc_steps_schedule.min_steps=1"
            "train.kdvi.mcmc_steps_schedule.max_steps=20"
            "train.kdvi.mcmc_steps_schedule.warmup_epochs=50000"
        )
    else
        output_ref+=(
            "train.kdvi.mcmc_steps=${k_mode#k}"
            "train.kdvi.mcmc_steps_schedule.enabled=false"
        )
    fi

    if [[ "${step_schedule}" == "coupled" ]]; then
        output_ref+=(
            "train.kdvi.mcmc_step_size=${nominal_step}"
            "train.kdvi.step_size_schedule.type=coupled"
        )
    else
        output_ref+=(
            "train.kdvi.step_size_schedule.type=cosine"
            "train.kdvi.step_size_schedule.start=1.0"
            "train.kdvi.step_size_schedule.end=0.01"
            "train.kdvi.step_size_schedule.steps=${anneal_steps}"
        )
    fi
}

if ((DRY_RUN)); then
    echo "Dry run complete; no experiments were launched."
    echo "Manifest: ${MANIFEST_PATH}"
    echo "First five jobs:"
    tail -n +2 "${MANIFEST_PATH}" | head -n 5
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
        "mcmc_type": meta["mcmc_type"],
        "step_schedule": meta["step_schedule"],
        "nominal_step": meta["nominal_step"],
        "k_mode": meta["k_mode"],
        "annealing_steps": int(meta["annealing_steps"]),
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
    "recipe_id", "mcmc_type", "step_schedule", "nominal_step",
    "k_mode", "annealing_steps", "seeds_complete", "n_seeds",
    "kl_mean", "kl_std", "w2_mean", "w2_std", "pareto", "status",
]
summary_csv.parent.mkdir(parents=True, exist_ok=True)
with summary_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

def fmt(value):
    return "—" if value is None else f"{value:.6f}"

lines = [
    "# KDVI 8-Gaussians MCMC Sweep — Summary",
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
        f"- **KL-ITE:** `{kl_winner['recipe_id']}` — "
        f"{fmt(kl_winner['kl_mean'])} ± {fmt(kl_winner['kl_std'])}",
        f"- **W2:** `{w2_winner['recipe_id']}` — "
        f"{fmt(w2_winner['w2_mean'])} ± {fmt(w2_winner['w2_std'])}",
        "",
        "## KL/W2 Pareto front",
        "",
        "| Recipe | KL-ITE mean ± std | W2 mean ± std |",
        "|---|---:|---:|",
    ])
    pareto_rows = sorted(
        (row for row in complete_rows if row["pareto"]),
        key=lambda row: (row["kl_mean"], row["w2_mean"]),
    )
    for row in pareto_rows:
        lines.append(
            f"| `{row['recipe_id']}` | {fmt(row['kl_mean'])} ± "
            f"{fmt(row['kl_std'])} | {fmt(row['w2_mean'])} ± "
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
    local mcmc_type="$5"
    local step_schedule="$6"
    local nominal_step="$7"
    local k_mode="$8"
    local anneal_steps="$9"
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
            "${mcmc_type}" "${step_schedule}" "${nominal_step}" \
            "${k_mode}" "${anneal_steps}"

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
while IFS=$'\t' read -r run_id recipe seed mcmc_type step_schedule nominal_step k_mode anneal_steps; do
    [[ "${run_id}" == "run_id" ]] && continue
    if [[ -f "${DONE_DIR}/${run_id}.done" && -f "${RESULT_MAP_DIR}/${run_id}.path" ]]; then
        continue
    fi
    JOBS+=("${run_id}"$'\t'"${recipe}"$'\t'"${seed}"$'\t'"${mcmc_type}"$'\t'"${step_schedule}"$'\t'"${nominal_step}"$'\t'"${k_mode}"$'\t'"${anneal_steps}")
done <"${MANIFEST_PATH}"

echo "Pending runs: ${#JOBS[@]} / 360"

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
        local run_id recipe seed mcmc_type step_schedule nominal_step k_mode anneal_steps
        IFS=$'\t' read -r run_id recipe seed mcmc_type step_schedule nominal_step k_mode anneal_steps <<<"${job}"

        run_job "${gpu_id}" "${run_id}" "${recipe}" "${seed}" \
            "${mcmc_type}" "${step_schedule}" "${nominal_step}" \
            "${k_mode}" "${anneal_steps}" &
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
