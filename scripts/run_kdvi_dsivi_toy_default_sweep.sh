#!/usr/bin/env bash

# KDVI + DSIVI default toy-target sweep.
#
# The caller is responsible for activating the intended Python environment.
# This script invokes `python` directly, schedules one run per visible GPU,
# resumes completed work, retries failures once, and aggregates available
# final/best VI metrics across ten seeds for each method/target pair.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CAMPAIGN_SLUG="kdvi_dsivi_toy_default_sweep_10seed"
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

MAX_GPUS=10
GPU_IDS_ARG=""
DRY_RUN=0
SUMMARIZE_ONLY=0

usage() {
    cat <<'EOF'
Usage: bash scripts/run_kdvi_dsivi_toy_default_sweep.sh [options]

Options:
  --dry-run           Generate and validate the 140-run manifest only.
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

generate_manifest() {
    local tmp_path="${MANIFEST_PATH}.tmp"
    local method target seed prefix config_path target_override target_config_override
    local recipe run_id metric_source

    printf 'run_id\trecipe_id\tmethod\ttarget\tseed\tconfig_path\ttarget_override\ttarget_config_override\tmetric_source\n' >"${tmp_path}"

    for method in KDVI DSIVI; do
        prefix="$(printf '%s' "${method}" | tr '[:upper:]' '[:lower:]')"
        for target in banana x_shaped multimodal 8_gaussians 8_gaussians_small student_uc Langevin_post; do
            target_override="-"
            target_config_override="-"
            metric_source="${target}"

            if [[ "${method}" == "DSIVI" && "${target}" == "8_gaussians_small" ]]; then
                config_path="configs/dsivi_8_gaussians.yaml"
                target_override="8_gaussians_small"
                target_config_override="configs/targets/8_gaussians_small.yaml"
            else
                config_path="configs/${prefix}_${target}.yaml"
            fi

            if [[ "${method}" == "KDVI" && "${target}" == "8_gaussians_small" ]]; then
                metric_source="8_gaussians"
            fi

            recipe="${method}-${target}"
            for seed in 0 1 7 42 43 44 45 46 47 48; do
                run_id="${recipe}-seed${seed}"
                printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "${run_id}" "${recipe}" "${method}" "${target}" "${seed}" \
                    "${config_path}" "${target_override}" "${target_config_override}" \
                    "${metric_source}" >>"${tmp_path}"
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

run_ids = [row["run_id"] for row in rows]
groups = defaultdict(list)
methods = set()
targets = set()
missing = []
for row in rows:
    groups[row["recipe_id"]].append(int(row["seed"]))
    methods.add(row["method"])
    targets.add(row["target"])
    config_path = repo_root / row["config_path"]
    if not config_path.is_file():
        missing.append(str(config_path))
    target_config = row["target_config_override"]
    if target_config != "-" and not (repo_root / target_config).is_file():
        missing.append(str(repo_root / target_config))

errors = []
if len(rows) != 140:
    errors.append(f"expected 140 runs, found {len(rows)}")
if len(set(run_ids)) != 140:
    errors.append("run IDs are not unique")
if len(groups) != 14:
    errors.append(f"expected 14 method-target groups, found {len(groups)}")
bad_groups = {
    recipe: seeds for recipe, seeds in groups.items()
    if sorted(seeds) != [0, 1, 7, 42, 43, 44, 45, 46, 47, 48]
}
if bad_groups:
    errors.append(f"groups without exactly seeds 0,1,7,42,43,44,45,46,47,48: {len(bad_groups)}")
if methods != {"KDVI", "DSIVI"}:
    errors.append(f"unexpected methods: {sorted(methods)}")
expected_targets = {
    "banana", "x_shaped", "multimodal", "8_gaussians",
    "8_gaussians_small", "student_uc", "Langevin_post",
}
if targets != expected_targets:
    errors.append(f"unexpected targets: {sorted(targets)}")
method_counts = Counter(row["method"] for row in rows)
if method_counts != {"KDVI": 70, "DSIVI": 70}:
    errors.append(f"unexpected per-method counts: {dict(method_counts)}")
if missing:
    errors.append("missing config files:\n  " + "\n  ".join(sorted(set(missing))))

small_dsivi = [
    row for row in rows
    if row["method"] == "DSIVI" and row["target"] == "8_gaussians_small"
]
if len(small_dsivi) != 10 or any(
    row["config_path"] != "configs/dsivi_8_gaussians.yaml"
    or row["target_override"] != "8_gaussians_small"
    or row["target_config_override"] != "configs/targets/8_gaussians_small.yaml"
    for row in small_dsivi
):
    errors.append("DSIVI 8_gaussians_small override rows are malformed")

if errors:
    raise SystemExit("Manifest validation failed:\n- " + "\n- ".join(errors))

print(
    "Manifest validated: "
    f"{len(rows)} runs, {len(groups)} method-target groups, "
    f"{len(methods)} methods, {len(targets)} targets."
)
PY
}

generate_manifest
validate_manifest

append_dsivi_metric_overrides() {
    local -n output_ref=$1
    local metric_source="$2"

    output_ref+=(
        "metric.kl_ite.num_samples=10000"
        "metric.w2.enabled=true"
        "metric.w2.num_samples=10000"
        "metric.w2.num_projections=1000"
        "metric.elbo.enabled=true"
        "metric.elbo.batch_size=512"
        "metric.elbo.num_batches=10"
        "metric.elbo.num_z_samples=5000"
        "metric.fisher.enabled=false"
        "metric.fisher.num_samples=1000"
        "metric.fisher.num_is_samples=512"
        "metric.ksd.enabled=false"
        "metric.ksd.num_samples=2000"
        "metric.mmd.enabled=false"
        "metric.mmd.num_samples=1000"
        "metric.bnn.enabled=false"
        "metric.bnn.num_samples=500"
        "metric.expected_log_marginal.num_ref_samples=1000"
        "metric.expected_log_marginal.num_model_samples=50000"
        "metric.expected_log_marginal.sample_batch_size=50000"
        "metric.expected_log_marginal.dim_chunk=50"
        "metric.expected_log_marginal.ref_chunk=500"
        "metric.expected_log_marginal.model_chunk=5000"
        "metric.expected_log_marginal.min_bandwidth=1.0e-6"
        "metric.expected_log_marginal.dtype=float32"
    )

    if [[ "${metric_source}" == "Langevin_post" ]]; then
        output_ref+=(
            "metric.kl_ite.enabled=false"
            "metric.expected_log_marginal.enabled=true"
        )
    else
        output_ref+=(
            "metric.kl_ite.enabled=true"
            "metric.expected_log_marginal.enabled=false"
        )
    fi
}

build_command() {
    local output_array_name="$1"
    local -n output_ref="${output_array_name}"
    local gpu_id="$2"
    local run_name="$3"
    local recipe="$4"
    local method="$5"
    local target="$6"
    local seed="$7"
    local config_path="$8"
    local target_override="$9"
    local target_config_override="${10}"
    local metric_source="${11}"

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
    if [[ "${method}" == "KDVI" ]]; then
        append_dsivi_metric_overrides "${output_array_name}" "${metric_source}"
    fi
}

if ((DRY_RUN)); then
    echo "Dry run complete; no experiments were launched."
    echo "Manifest: ${MANIFEST_PATH}"
    echo "First five jobs:"
    tail -n +2 "${MANIFEST_PATH}" | head -n 5
    echo "First command:"
    first_job="$(tail -n +2 "${MANIFEST_PATH}" | head -n 1)"
    IFS=$'\t' read -r run_id recipe method target seed config_path target_override target_config_override metric_source <<<"${first_job}"
    cmd=()
    build_command cmd 0 "${run_id}" "${recipe}" "${method}" "${target}" \
        "${seed}" "${config_path}" "${target_override}" \
        "${target_config_override}" "${metric_source}"
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
    "metric/vi_model/kl_ite": ("kl_ite", "min"),
    "metric/vi_model/w2": ("w2", "min"),
    "metric/vi_model/elbo": ("elbo", "max"),
    "metric/vi_model/kde_expected_log_marginal": ("kde_expected_log_marginal", "max"),
}
EXPECTED_SEEDS = [0, 1, 7, 42, 43, 44, 45, 46, 47, 48]

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

rows = []
for recipe in sorted(recipe_meta, key=lambda key: (recipe_meta[key]["target"], recipe_meta[key]["method"])):
    meta = recipe_meta[recipe]
    samples = sorted(by_recipe.get(recipe, []), key=lambda item: item["seed"])
    seeds = [sample["seed"] for sample in samples]
    complete = seeds == EXPECTED_SEEDS
    row = {
        "recipe_id": recipe,
        "method": meta["method"],
        "target": meta["target"],
        "config_path": meta["config_path"],
        "seeds_complete": ",".join(str(seed) for seed in seeds),
        "n_seeds": len(seeds),
        "status": "complete" if complete else "incomplete",
    }
    for slug in ("kl_ite", "w2", "elbo", "kde_expected_log_marginal"):
        for kind in ("final", "best"):
            values = [sample[f"{slug}_{kind}"] for sample in samples if f"{slug}_{kind}" in sample]
            row[f"{slug}_{kind}_mean"] = mean(values)
            row[f"{slug}_{kind}_std"] = stdev(values)
            row[f"{slug}_{kind}_count"] = len(values)
    rows.append(row)

fieldnames = [
    "recipe_id", "method", "target", "config_path", "seeds_complete",
    "n_seeds", "status",
]
for slug in ("kl_ite", "w2", "elbo", "kde_expected_log_marginal"):
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
    "# KDVI + DSIVI Toy Default Sweep Summary",
    "",
    f"Complete method-target groups: **{len(complete_rows)} / {len(rows)}**.",
    "Metrics are means and sample standard deviations across seeds 0, 1, 7, 42, 43, 44, 45, 46, 47, and 48.",
    "",
    "## Final Metrics",
    "",
    "| Target | Method | KL-ITE | W2 | ELBO | KDE ELM |",
    "|---|---|---:|---:|---:|---:|",
]

target_order = [
    "banana", "x_shaped", "multimodal", "8_gaussians",
    "8_gaussians_small", "student_uc", "Langevin_post",
]
method_order = ["KDVI", "DSIVI"]
row_map = {(row["target"], row["method"]): row for row in rows}
for target in target_order:
    for method in method_order:
        row = row_map.get((target, method))
        if row is None:
            continue
        lines.append(
            f"| `{target}` | `{method}` | "
            f"{fmt(row['kl_ite_final_mean'])} +/- {fmt(row['kl_ite_final_std'])} | "
            f"{fmt(row['w2_final_mean'])} +/- {fmt(row['w2_final_std'])} | "
            f"{fmt(row['elbo_final_mean'])} +/- {fmt(row['elbo_final_std'])} | "
            f"{fmt(row['kde_expected_log_marginal_final_mean'])} +/- "
            f"{fmt(row['kde_expected_log_marginal_final_std'])} |"
        )
lines.append("")

incomplete = [row for row in rows if row["status"] != "complete"]
lines.extend(["## Incomplete Method-Target Groups", ""])
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
    local method="$4"
    local target="$5"
    local seed="$6"
    local config_path="$7"
    local target_override="$8"
    local target_config_override="$9"
    local metric_source="${10}"
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
        build_command cmd "${gpu_id}" "${run_name}" "${recipe}" "${method}" \
            "${target}" "${seed}" "${config_path}" "${target_override}" \
            "${target_config_override}" "${metric_source}"

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
while IFS=$'\t' read -r run_id recipe method target seed config_path target_override target_config_override metric_source; do
    [[ "${run_id}" == "run_id" ]] && continue
    if [[ -f "${DONE_DIR}/${run_id}.done" && -f "${RESULT_MAP_DIR}/${run_id}.path" ]]; then
        continue
    fi
    JOBS+=("${run_id}"$'\t'"${recipe}"$'\t'"${method}"$'\t'"${target}"$'\t'"${seed}"$'\t'"${config_path}"$'\t'"${target_override}"$'\t'"${target_config_override}"$'\t'"${metric_source}")
done <"${MANIFEST_PATH}"

echo "Pending runs: ${#JOBS[@]} / 140"

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
        local run_id recipe method target seed config_path target_override target_config_override metric_source
        IFS=$'\t' read -r run_id recipe method target seed config_path target_override target_config_override metric_source <<<"${job}"

        run_job "${gpu_id}" "${run_id}" "${recipe}" "${method}" "${target}" \
            "${seed}" "${config_path}" "${target_override}" \
            "${target_config_override}" "${metric_source}" &
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
