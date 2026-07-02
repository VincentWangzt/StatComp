#!/usr/bin/env bash

# KDVI student_uc MCMC step-size sweep for SGLD and MALA.
#
# The caller is responsible for activating the intended Python environment.
# This script invokes `python` directly, schedules one run per visible GPU,
# resumes completed work, retries failures once, and aggregates final KL/W2
# metrics across three seeds for each MCMC-type/step-size recipe.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CAMPAIGN_SLUG="kdvi_student_uc_mcmc_step_sweep"
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
CONFIG_PATH="configs/kdvi_student_uc.yaml"

MAX_GPUS=10
GPU_IDS_ARG=""
DRY_RUN=0
SUMMARIZE_ONLY=0

usage() {
    cat <<'EOF'
Usage: bash scripts/run_kdvi_student_uc_mcmc_step_sweep.sh [options]

Options:
  --dry-run           Generate and validate the 42-run manifest only.
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

step_label() {
    local value="$1"
    value="${value/./p}"
    value="${value//-/m}"
    value="${value//+/p}"
    printf '%s\n' "${value}"
}

generate_manifest() {
    local tmp_path="${MANIFEST_PATH}.tmp"
    local mcmc_type seed step recipe run_id step_slug

    printf 'run_id\trecipe_id\ttarget\tseed\tconfig_path\tmcmc_type\tmcmc_step_size\tmetric_family\n' >"${tmp_path}"

    for mcmc_type in sgld mala; do
        for step in 1e-1 5e-2 2e-2 1e-2 5e-3 2e-3 1e-3; do
            step_slug="$(step_label "${step}")"
            recipe="KDVI-student_uc-mcmcstep${step_slug}-${mcmc_type}"
            for seed in 0 1 7; do
                run_id="${recipe}-seed${seed}"
                printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "${run_id}" "${recipe}" student_uc "${seed}" \
                    "${CONFIG_PATH}" "${mcmc_type}" "${step}" kl_w2 >>"${tmp_path}"
            done
        done
    done

    mv "${tmp_path}" "${MANIFEST_PATH}"
}

validate_manifest() {
    python - "${REPO_ROOT}" "${MANIFEST_PATH}" <<'PY'
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

repo_root = Path(sys.argv[1])
path = Path(sys.argv[2])
with path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))

expected_steps = {"1e-1", "5e-2", "2e-2", "1e-2", "5e-3", "2e-3", "1e-3"}
run_ids = [row["run_id"] for row in rows]
groups = defaultdict(list)
type_steps = defaultdict(set)
missing = []

for row in rows:
    groups[row["recipe_id"]].append(int(row["seed"]))
    type_steps[row["mcmc_type"]].add(row["mcmc_step_size"])
    if row["target"] != "student_uc":
        missing.append(f"unexpected target {row['target']}")
    if row["metric_family"] != "kl_w2":
        missing.append(f"unexpected metric_family {row['metric_family']}")
    if not (repo_root / row["config_path"]).is_file():
        missing.append(str(repo_root / row["config_path"]))

errors = []
if len(rows) != 42:
    errors.append(f"expected 42 runs, found {len(rows)}")
if len(set(run_ids)) != 42:
    errors.append("run IDs are not unique")
if len(groups) != 14:
    errors.append(f"expected 14 MCMC-type/step groups, found {len(groups)}")
bad_groups = {
    recipe: seeds for recipe, seeds in groups.items()
    if sorted(seeds) != [0, 1, 7]
}
if bad_groups:
    errors.append(f"groups without exactly seeds 0,1,7: {len(bad_groups)}")

type_counts = Counter(row["mcmc_type"] for row in rows)
if type_counts != {"sgld": 21, "mala": 21}:
    errors.append(f"unexpected MCMC type counts: {dict(type_counts)}")
for mcmc_type in ("sgld", "mala"):
    if type_steps[mcmc_type] != expected_steps:
        errors.append(
            f"{mcmc_type} has unexpected step sizes: "
            f"{sorted(type_steps[mcmc_type])}"
        )
if missing:
    errors.append("manifest path/value errors:\n  " + "\n  ".join(sorted(set(missing))))

if errors:
    raise SystemExit("Manifest validation failed:\n- " + "\n- ".join(errors))

print(
    "Manifest validated: "
    f"{len(rows)} runs, {len(groups)} groups, "
    f"{type_counts['sgld']} SGLD, {type_counts['mala']} MALA."
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
    local mcmc_type="$6"
    local mcmc_step_size="$7"

    output_ref=(
        python src.py --config "${CONFIG_PATH}"
        "cuda_visible_devices=${gpu_id}"
        "seed=${seed}"
        "output.results_dir=results/${CAMPAIGN_SLUG}/${run_name}"
        "train.kdvi.mcmc_type=${mcmc_type}"
        "train.kdvi.mcmc_step_size=${mcmc_step_size}"
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
    IFS=$'\t' read -r run_id recipe target seed config_path mcmc_type mcmc_step_size metric_family <<<"${first_job}"
    cmd=()
    build_command cmd 0 "${run_id}" "${recipe}" "${seed}" "${mcmc_type}" "${mcmc_step_size}"
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
EXPECTED_SEEDS = [0, 1, 7]
MCMC_ORDER = ["sgld", "mala"]
STEP_ORDER = ["1e-1", "5e-2", "2e-2", "1e-2", "5e-3", "2e-3", "1e-3"]

with manifest_path.open(newline="", encoding="utf-8") as handle:
    manifest = list(csv.DictReader(handle, delimiter="\t"))

recipe_meta = {}
by_recipe = defaultdict(list)

def final_metrics(metrics_path):
    points = {KL_TAG: [], W2_TAG: []}
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            tag = row.get("tag")
            if tag not in points:
                continue
            try:
                step = int(float(row["step"]))
                value = float(row["value"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                points[tag].append((step, value))
    if any(not values for values in points.values()):
        return None
    kl_iter, kl_value = max(points[KL_TAG], key=lambda item: item[0])
    w2_iter, w2_value = max(points[W2_TAG], key=lambda item: item[0])
    return {
        "kl_ite": kl_value,
        "kl_ite_iter": kl_iter,
        "w2": w2_value,
        "w2_iter": w2_iter,
    }

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
    if metrics:
        by_recipe[recipe].append({"seed": int(item["seed"]), **metrics})

def mean(values):
    return statistics.mean(values) if values else None

def stdev(values):
    return statistics.stdev(values) if len(values) >= 2 else None

def sort_key(recipe):
    meta = recipe_meta[recipe]
    return (
        MCMC_ORDER.index(meta["mcmc_type"]),
        STEP_ORDER.index(meta["mcmc_step_size"]),
    )

rows = []
for recipe in sorted(recipe_meta, key=sort_key):
    meta = recipe_meta[recipe]
    samples = sorted(by_recipe.get(recipe, []), key=lambda item: item["seed"])
    seeds = [sample["seed"] for sample in samples]
    complete = seeds == EXPECTED_SEEDS
    row = {
        "recipe_id": recipe,
        "target": meta["target"],
        "mcmc_type": meta["mcmc_type"],
        "mcmc_step_size": meta["mcmc_step_size"],
        "metric_family": meta["metric_family"],
        "seeds_complete": ",".join(str(seed) for seed in seeds),
        "n_seeds": len(seeds),
        "status": "complete" if complete else "incomplete",
        "pareto": False,
    }
    for slug in ("kl_ite", "w2"):
        values = [sample[slug] for sample in samples if slug in sample]
        iters = [sample[f"{slug}_iter"] for sample in samples if f"{slug}_iter" in sample]
        row[f"{slug}_mean"] = mean(values)
        row[f"{slug}_std"] = stdev(values)
        row[f"{slug}_count"] = len(values)
        row[f"{slug}_final_iter_min"] = min(iters) if iters else None
        row[f"{slug}_final_iter_max"] = max(iters) if iters else None
    rows.append(row)

complete_rows = [row for row in rows if row["status"] == "complete"]
for candidate in complete_rows:
    candidate["pareto"] = not any(
        other is not candidate
        and other["kl_ite_mean"] <= candidate["kl_ite_mean"]
        and other["w2_mean"] <= candidate["w2_mean"]
        and (
            other["kl_ite_mean"] < candidate["kl_ite_mean"]
            or other["w2_mean"] < candidate["w2_mean"]
        )
        for other in complete_rows
    )

fieldnames = [
    "recipe_id",
    "target",
    "mcmc_type",
    "mcmc_step_size",
    "metric_family",
    "seeds_complete",
    "n_seeds",
    "status",
    "pareto",
]
for slug in ("kl_ite", "w2"):
    fieldnames.extend([
        f"{slug}_mean",
        f"{slug}_std",
        f"{slug}_count",
        f"{slug}_final_iter_min",
        f"{slug}_final_iter_max",
    ])

summary_csv.parent.mkdir(parents=True, exist_ok=True)
with summary_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

def fmt(value):
    return "-" if value is None else f"{value:.6f}"

lines = [
    "# KDVI Student UC MCMC Step-Size Sweep Summary",
    "",
    f"Complete MCMC-type/step groups: **{len(complete_rows)} / {len(rows)}**.",
    "Metrics are final logged values summarized as means and sample standard deviations across seeds 0, 1, and 7.",
    "",
]

if complete_rows:
    kl_winner = min(complete_rows, key=lambda row: row["kl_ite_mean"])
    w2_winner = min(complete_rows, key=lambda row: row["w2_mean"])
    lines.extend([
        "## Overall Winners",
        "",
        f"- **KL-ITE:** `{kl_winner['recipe_id']}` - "
        f"{fmt(kl_winner['kl_ite_mean'])} +/- {fmt(kl_winner['kl_ite_std'])}",
        f"- **W2:** `{w2_winner['recipe_id']}` - "
        f"{fmt(w2_winner['w2_mean'])} +/- {fmt(w2_winner['w2_std'])}",
        "",
        "## Overall KL/W2 Pareto Front",
        "",
        "| Recipe | KL-ITE mean +/- std | W2 mean +/- std |",
        "|---|---:|---:|",
    ])
    pareto_rows = sorted(
        (row for row in complete_rows if row["pareto"]),
        key=lambda row: (row["kl_ite_mean"], row["w2_mean"]),
    )
    for row in pareto_rows:
        lines.append(
            f"| `{row['recipe_id']}` | "
            f"{fmt(row['kl_ite_mean'])} +/- {fmt(row['kl_ite_std'])} | "
            f"{fmt(row['w2_mean'])} +/- {fmt(row['w2_std'])} |"
        )
    lines.append("")

for mcmc_type in MCMC_ORDER:
    type_rows = [row for row in rows if row["mcmc_type"] == mcmc_type]
    complete_type_rows = [row for row in type_rows if row["status"] == "complete"]
    lines.extend([f"## {mcmc_type.upper()}", ""])
    if complete_type_rows:
        kl_winner = min(complete_type_rows, key=lambda row: row["kl_ite_mean"])
        w2_winner = min(complete_type_rows, key=lambda row: row["w2_mean"])
        lines.extend([
            "### Winners",
            "",
            f"- **KL-ITE:** `{kl_winner['recipe_id']}` - "
            f"{fmt(kl_winner['kl_ite_mean'])} +/- {fmt(kl_winner['kl_ite_std'])}",
            f"- **W2:** `{w2_winner['recipe_id']}` - "
            f"{fmt(w2_winner['w2_mean'])} +/- {fmt(w2_winner['w2_std'])}",
            "",
        ])
    else:
        lines.extend(["No complete step-size groups yet.", ""])

    lines.extend([
        "### All Step Sizes",
        "",
        "| Step size | Status | Seeds | KL-ITE | W2 |",
        "|---:|---|---|---:|---:|",
    ])
    for row in type_rows:
        lines.append(
            f"| `{row['mcmc_step_size']}` | {row['status']} | "
            f"{row['seeds_complete'] or 'none'} | "
            f"{fmt(row['kl_ite_mean'])} +/- {fmt(row['kl_ite_std'])} | "
            f"{fmt(row['w2_mean'])} +/- {fmt(row['w2_std'])} |"
        )
    lines.append("")

incomplete = [row for row in rows if row["status"] != "complete"]
lines.extend(["## Incomplete Groups", ""])
if incomplete:
    lines.extend(["| Recipe | Complete seeds |", "|---|---|"])
    for row in incomplete:
        lines.append(f"| `{row['recipe_id']}` | {row['seeds_complete'] or 'none'} |")
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
    local -a candidates=()
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
    local mcmc_step_size="$6"
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
        build_command cmd "${gpu_id}" "${run_name}" "${recipe}" \
            "${seed}" "${mcmc_type}" "${mcmc_step_size}"

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
while IFS=$'\t' read -r run_id recipe target seed config_path mcmc_type mcmc_step_size metric_family; do
    [[ "${run_id}" == "run_id" ]] && continue
    if [[ -f "${DONE_DIR}/${run_id}.done" && -f "${RESULT_MAP_DIR}/${run_id}.path" ]]; then
        continue
    fi
    JOBS+=("${run_id}"$'\t'"${recipe}"$'\t'"${seed}"$'\t'"${mcmc_type}"$'\t'"${mcmc_step_size}")
done <"${MANIFEST_PATH}"

echo "Pending runs: ${#JOBS[@]} / 42"

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
        local run_id recipe seed mcmc_type mcmc_step_size
        IFS=$'\t' read -r run_id recipe seed mcmc_type mcmc_step_size <<<"${job}"

        run_job "${gpu_id}" "${run_id}" "${recipe}" "${seed}" \
            "${mcmc_type}" "${mcmc_step_size}" &
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
