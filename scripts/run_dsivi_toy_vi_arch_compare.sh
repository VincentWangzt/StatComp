#!/usr/bin/env bash

# DSIVI 2D toy-target VI architecture comparison.
#
# Compares each target's default DSIVI VI architecture against
# configs/vi_models/ConditionalGaussian-Eps32-ELU-LogStd.yaml across seeds
# 0, 1, and 7. This is 6 targets x 2 architectures x 3 seeds = 36 runs.
#
# The caller is responsible for activating the intended Python environment.
# The script invokes `python` directly, schedules one run per visible GPU,
# resumes completed work, retries failures once, and writes summary.csv and
# summary.md with final ELBO, KL-ITE, and W2 means across seeds.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CAMPAIGN_SLUG="dsivi_toy_vi_arch_compare"
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
ALT_VI_CONFIG="configs/vi_models/ConditionalGaussian-Eps32-ELU-LogStd.yaml"

MAX_GPUS=10
GPU_IDS_ARG=""
DRY_RUN=0
SUMMARIZE_ONLY=0

usage() {
    cat <<'EOF'
Usage: bash scripts/run_dsivi_toy_vi_arch_compare.sh [options]

Options:
  --dry-run           Generate and validate the 36-run manifest only.
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

target_config_path() {
    local target="$1"
    printf 'configs/targets/%s.yaml\n' "${target}"
}

dsivi_config_path() {
    local target="$1"
    if [[ "${target}" == "8_gaussians_small" ]]; then
        printf 'configs/dsivi_8_gaussians.yaml\n'
    else
        printf 'configs/dsivi_%s.yaml\n' "${target}"
    fi
}

generate_manifest() {
    local tmp_path="${MANIFEST_PATH}.tmp"
    local target arch seed run_id recipe config_path target_override target_config_override
    local -a targets=(banana x_shaped multimodal 8_gaussians 8_gaussians_small student_uc)
    local -a arches=(default eps32_elu_logstd)
    local -a seeds=(0 1 7)

    printf 'run_id\trecipe_id\ttarget\tarch\tseed\tconfig_path\ttarget_override\ttarget_config_override\n' >"${tmp_path}"

    for target in "${targets[@]}"; do
        config_path="$(dsivi_config_path "${target}")"
        target_override="-"
        target_config_override="-"
        if [[ "${target}" == "8_gaussians_small" ]]; then
            target_override="8_gaussians_small"
            target_config_override="$(target_config_path "${target}")"
        fi

        for arch in "${arches[@]}"; do
            recipe="DSIVI-${target}-${arch}"
            for seed in "${seeds[@]}"; do
                run_id="${recipe}-seed${seed}"
                printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "${run_id}" "${recipe}" "${target}" "${arch}" "${seed}" \
                    "${config_path}" "${target_override}" "${target_config_override}" \
                    >>"${tmp_path}"
            done
        done
    done

    mv "${tmp_path}" "${MANIFEST_PATH}"
}

validate_manifest() {
    python - "${REPO_ROOT}" "${MANIFEST_PATH}" "${ALT_VI_CONFIG}" <<'PY'
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

repo_root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
alt_vi_config = Path(sys.argv[3])

with manifest_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))

targets = {
    "banana",
    "x_shaped",
    "multimodal",
    "8_gaussians",
    "8_gaussians_small",
    "student_uc",
}
arches = {"default", "eps32_elu_logstd"}
seeds = [0, 1, 7]

groups = defaultdict(list)
run_ids = [row["run_id"] for row in rows]
missing = []
for row in rows:
    groups[(row["target"], row["arch"])].append(int(row["seed"]))
    for key in ("config_path", "target_config_override"):
        value = row[key]
        if value != "-" and not (repo_root / value).is_file():
            missing.append(str(repo_root / value))

errors = []
if len(rows) != 36:
    errors.append(f"expected 36 runs, found {len(rows)}")
if len(set(run_ids)) != 36:
    errors.append("run IDs are not unique")
if set(row["target"] for row in rows) != targets:
    errors.append(f"unexpected targets: {sorted(set(row['target'] for row in rows))}")
if set(row["arch"] for row in rows) != arches:
    errors.append(f"unexpected architectures: {sorted(set(row['arch'] for row in rows))}")
if len(groups) != 12:
    errors.append(f"expected 12 target-architecture groups, found {len(groups)}")
bad_groups = {
    group: group_seeds
    for group, group_seeds in groups.items()
    if sorted(group_seeds) != seeds
}
if bad_groups:
    errors.append(f"groups without exactly seeds 0,1,7: {len(bad_groups)}")
if Counter(row["arch"] for row in rows) != {"default": 18, "eps32_elu_logstd": 18}:
    errors.append("unexpected per-architecture run counts")
if not (repo_root / alt_vi_config).is_file():
    missing.append(str(repo_root / alt_vi_config))

small_rows = [row for row in rows if row["target"] == "8_gaussians_small"]
if len(small_rows) != 6 or any(
    row["config_path"] != "configs/dsivi_8_gaussians.yaml"
    or row["target_override"] != "8_gaussians_small"
    or row["target_config_override"] != "configs/targets/8_gaussians_small.yaml"
    for row in small_rows
):
    errors.append("8_gaussians_small override rows are malformed")
if missing:
    errors.append("missing config files:\n  " + "\n  ".join(sorted(set(missing))))

if errors:
    raise SystemExit("Manifest validation failed:\n- " + "\n- ".join(errors))

print("Manifest validated: 36 runs, 6 targets, 2 architectures, 3 seeds.")
PY
}

generate_manifest
validate_manifest

build_command() {
    local -n output_ref=$1
    local gpu_id="$2"
    local run_name="$3"
    local recipe="$4"
    local target="$5"
    local arch="$6"
    local seed="$7"
    local config_path="$8"
    local target_override="$9"
    local target_config_override="${10}"

    output_ref=(
        python src.py --config "${config_path}"
        "use_cuda=true"
        "cuda_visible_devices=${gpu_id}"
        "seed=${seed}"
        "output.results_dir=results/${CAMPAIGN_SLUG}/${run_name}"
        "tracking.campaign=${CAMPAIGN_SLUG}"
        "tracking.group=${recipe}"
        "tracking.run_name=${run_name}"
    )

    if [[ "${target_override}" != "-" ]]; then
        output_ref+=("target_type=${target_override}")
    fi
    if [[ "${target_config_override}" != "-" ]]; then
        output_ref+=("target_config_path=${target_config_override}")
    fi
    if [[ "${arch}" == "eps32_elu_logstd" ]]; then
        output_ref+=(
            "vi_model_type=ConditionalGaussian"
            "vi_model_config_path=${ALT_VI_CONFIG}"
        )
    fi
}

if ((DRY_RUN)); then
    echo "Dry run complete; no experiments were launched."
    echo "Manifest: ${MANIFEST_PATH}"
    echo "First five jobs:"
    tail -n +2 "${MANIFEST_PATH}" | head -n 5
    echo "First command:"
    first_job="$(tail -n +2 "${MANIFEST_PATH}" | head -n 1)"
    IFS=$'\t' read -r run_id recipe target arch seed config_path target_override target_config_override <<<"${first_job}"
    cmd=()
    build_command cmd 0 "${run_id}" "${recipe}" "${target}" "${arch}" \
        "${seed}" "${config_path}" "${target_override}" "${target_config_override}"
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

METRICS = {
    "metric/vi_model/elbo": ("elbo", "max"),
    "metric/vi_model/kl_ite": ("kl_ite", "min"),
    "metric/vi_model/w2": ("w2", "min"),
}
EXPECTED_SEEDS = [0, 1, 7]
TARGET_ORDER = [
    "banana",
    "x_shaped",
    "multimodal",
    "8_gaussians",
    "8_gaussians_small",
    "student_uc",
]
ARCH_ORDER = ["default", "eps32_elu_logstd"]

with manifest_path.open(newline="", encoding="utf-8") as handle:
    manifest = list(csv.DictReader(handle, delimiter="\t"))

by_recipe = defaultdict(list)
recipe_meta = {}

def read_metrics(metrics_path):
    points = {tag: [] for tag in METRICS}
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
    out = {}
    for tag, tag_points in points.items():
        if not tag_points:
            continue
        slug, mode = METRICS[tag]
        final_step, final_value = max(tag_points, key=lambda item: item[0])
        best_step, best_value = (
            min(tag_points, key=lambda item: item[1])
            if mode == "min"
            else max(tag_points, key=lambda item: item[1])
        )
        out[f"{slug}_final"] = final_value
        out[f"{slug}_final_iter"] = final_step
        out[f"{slug}_best"] = best_value
        out[f"{slug}_best_iter"] = best_step
    return out

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
    metrics = read_metrics(metrics_path)
    if metrics:
        by_recipe[recipe].append({"seed": int(item["seed"]), **metrics})

def mean(values):
    return statistics.mean(values) if values else None

def stdev(values):
    return statistics.stdev(values) if len(values) >= 2 else None

def sort_key(recipe):
    meta = recipe_meta[recipe]
    return (TARGET_ORDER.index(meta["target"]), ARCH_ORDER.index(meta["arch"]))

rows = []
for recipe in sorted(recipe_meta, key=sort_key):
    meta = recipe_meta[recipe]
    samples = sorted(by_recipe.get(recipe, []), key=lambda item: item["seed"])
    seeds = [sample["seed"] for sample in samples]
    complete = seeds == EXPECTED_SEEDS
    row = {
        "recipe_id": recipe,
        "target": meta["target"],
        "arch": meta["arch"],
        "config_path": meta["config_path"],
        "vi_model_config_path": "default"
        if meta["arch"] == "default"
        else "configs/vi_models/ConditionalGaussian-Eps32-ELU-LogStd.yaml",
        "seeds_complete": ",".join(str(seed) for seed in seeds),
        "n_seeds": len(seeds),
        "status": "complete" if complete else "incomplete",
    }
    for slug in ("elbo", "kl_ite", "w2"):
        for kind in ("final", "best"):
            values = [sample[f"{slug}_{kind}"] for sample in samples if f"{slug}_{kind}" in sample]
            row[f"{slug}_{kind}_mean"] = mean(values)
            row[f"{slug}_{kind}_std"] = stdev(values)
            row[f"{slug}_{kind}_count"] = len(values)
    rows.append(row)

fieldnames = [
    "recipe_id",
    "target",
    "arch",
    "config_path",
    "vi_model_config_path",
    "seeds_complete",
    "n_seeds",
    "status",
]
for slug in ("elbo", "kl_ite", "w2"):
    for kind in ("final", "best"):
        fieldnames.extend([
            f"{slug}_{kind}_mean",
            f"{slug}_{kind}_std",
            f"{slug}_{kind}_count",
        ])

summary_csv.parent.mkdir(parents=True, exist_ok=True)
with summary_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

def fmt(value):
    return "-" if value is None else f"{value:.6f}"

complete_rows = [row for row in rows if row["status"] == "complete"]
lines = [
    "# DSIVI Toy VI Architecture Comparison",
    "",
    f"Complete target-architecture groups: **{len(complete_rows)} / {len(rows)}**.",
    "Metrics are final-value means and sample standard deviations across seeds 0, 1, and 7.",
    "",
    "| Target | VI architecture | ELBO | KL-ITE | W2 | Seeds |",
    "|---|---|---:|---:|---:|---:|",
]
for target in TARGET_ORDER:
    for arch in ARCH_ORDER:
        row = next((item for item in rows if item["target"] == target and item["arch"] == arch), None)
        if row is None:
            continue
        lines.append(
            f"| `{target}` | `{arch}` | "
            f"{fmt(row['elbo_final_mean'])} +/- {fmt(row['elbo_final_std'])} | "
            f"{fmt(row['kl_ite_final_mean'])} +/- {fmt(row['kl_ite_final_std'])} | "
            f"{fmt(row['w2_final_mean'])} +/- {fmt(row['w2_final_std'])} | "
            f"{row['n_seeds']} |"
        )

incomplete = [row for row in rows if row["status"] != "complete"]
lines.extend(["", "## Incomplete Groups", ""])
if incomplete:
    lines.extend(["| Group | Complete seeds |", "|---|---|"])
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
    local target="$4"
    local arch="$5"
    local seed="$6"
    local config_path="$7"
    local target_override="$8"
    local target_config_override="$9"
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
        build_command cmd "${gpu_id}" "${run_name}" "${recipe}" "${target}" \
            "${arch}" "${seed}" "${config_path}" "${target_override}" \
            "${target_config_override}"

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
while IFS=$'\t' read -r run_id recipe target arch seed config_path target_override target_config_override; do
    [[ "${run_id}" == "run_id" ]] && continue
    if [[ -f "${DONE_DIR}/${run_id}.done" && -f "${RESULT_MAP_DIR}/${run_id}.path" ]]; then
        continue
    fi
    JOBS+=("${run_id}"$'\t'"${recipe}"$'\t'"${target}"$'\t'"${arch}"$'\t'"${seed}"$'\t'"${config_path}"$'\t'"${target_override}"$'\t'"${target_config_override}")
done <"${MANIFEST_PATH}"

echo "Pending runs: ${#JOBS[@]} / 36"

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
        local run_id recipe target arch seed config_path target_override target_config_override
        IFS=$'\t' read -r run_id recipe target arch seed config_path target_override target_config_override <<<"${job}"

        run_job "${gpu_id}" "${run_id}" "${recipe}" "${target}" "${arch}" \
            "${seed}" "${config_path}" "${target_override}" \
            "${target_config_override}" &
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
