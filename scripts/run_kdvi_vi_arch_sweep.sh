#!/usr/bin/env bash

# KDVI VI-model architecture sweep on the big 8-Gaussians target.
#
# The caller is responsible for activating the intended Python environment.
# This script invokes `python` directly, schedules one run per visible GPU,
# resumes completed work, retries failures once, and aggregates final/best
# KL/W2 metrics across the three seeds in each architecture recipe.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CAMPAIGN_SLUG="kdvi_vi_arch_sweep"
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
Usage: bash scripts/run_kdvi_vi_arch_sweep.sh [options]

Options:
  --dry-run           Generate and validate the 162-run manifest only.
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

value_label() {
    local value="$1"
    value="${value/./p}"
    value="${value//-/m}"
    printf '%s\n' "${value}"
}

generate_manifest() {
    local tmp_path="${MANIFEST_PATH}.tmp"
    local epsilon_dim hidden_dim num_layers activation seed recipe run_id
    local eps_slug

    printf 'run_id\trecipe_id\tseed\tepsilon_dim\thidden_dim\tnum_layers\tactivation\n' >"${tmp_path}"

    for epsilon_dim in 16 64; do
        eps_slug="$(value_label "${epsilon_dim}")"
        for hidden_dim in 128 256 512; do
            for num_layers in 2 3 4; do
                for activation in silu elu relu; do
                    recipe="eps${eps_slug}-h${hidden_dim}-l${num_layers}-${activation}"
                    for seed in 0 1 7; do
                        run_id="${recipe}-seed${seed}"
                        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                            "${run_id}" "${recipe}" "${seed}" "${epsilon_dim}" \
                            "${hidden_dim}" "${num_layers}" "${activation}" \
                            >>"${tmp_path}"
                    done
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
eps = set()
hidden = set()
layers = set()
activations = set()
for row in rows:
    groups[row["recipe_id"]].append(int(row["seed"]))
    eps.add(row["epsilon_dim"])
    hidden.add(row["hidden_dim"])
    layers.add(row["num_layers"])
    activations.add(row["activation"])

errors = []
if len(rows) != 162:
    errors.append(f"expected 162 runs, found {len(rows)}")
if len(set(run_ids)) != 162:
    errors.append("run IDs are not unique")
if len(groups) != 54:
    errors.append(f"expected 54 architecture groups, found {len(groups)}")
bad_groups = {
    recipe: seeds for recipe, seeds in groups.items()
    if sorted(seeds) != [0, 1, 7]
}
if bad_groups:
    errors.append(f"groups without exactly seeds 0,1,7: {len(bad_groups)}")
if eps != {"16", "64"}:
    errors.append(f"unexpected epsilon_dim values: {sorted(eps)}")
if hidden != {"128", "256", "512"}:
    errors.append(f"unexpected hidden_dim values: {sorted(hidden)}")
if layers != {"2", "3", "4"}:
    errors.append(f"unexpected num_layers values: {sorted(layers)}")
if activations != {"silu", "elu", "relu"}:
    errors.append(f"unexpected activation values: {sorted(activations)}")

run_count_by_seed = Counter(row["seed"] for row in rows)
if run_count_by_seed != {"0": 54, "1": 54, "7": 54}:
    errors.append(f"unexpected per-seed counts: {dict(run_count_by_seed)}")

if errors:
    raise SystemExit("Manifest validation failed:\n- " + "\n- ".join(errors))

print(
    "Manifest validated: "
    f"{len(rows)} runs, {len(groups)} groups, "
    f"{len(eps)} epsilon dims, {len(hidden)} hidden dims, "
    f"{len(layers)} layer counts, {len(activations)} activations."
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
    local epsilon_dim="$6"
    local hidden_dim="$7"
    local num_layers="$8"
    local activation="$9"

    output_ref=(
        python src.py --config "${CONFIG_PATH}"
        "use_cuda=true"
        "cuda_visible_devices=${gpu_id}"
        "seed=${seed}"
        "output.results_dir=results/${CAMPAIGN_SLUG}/${run_name}"
        "vi_model.epsilon_dim=${epsilon_dim}"
        "vi_model.hidden_dim=${hidden_dim}"
        "vi_model.num_layers=${num_layers}"
        "vi_model.activation=${activation}"
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
    IFS=$'\t' read -r run_id recipe seed epsilon_dim hidden_dim num_layers activation <<<"${first_job}"
    cmd=()
    build_command cmd 0 "${run_id}" "${recipe}" "${seed}" \
        "${epsilon_dim}" "${hidden_dim}" "${num_layers}" "${activation}"
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
    "metric/vi_model/kl_ite": ("kl", "min"),
    "metric/vi_model/w2": ("w2", "min"),
}
EXPECTED_SEEDS = [0, 1, 7]

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
for recipe in sorted(recipe_meta):
    meta = recipe_meta[recipe]
    samples = sorted(by_recipe.get(recipe, []), key=lambda item: item["seed"])
    seeds = [sample["seed"] for sample in samples]
    complete = seeds == EXPECTED_SEEDS
    row = {
        "recipe_id": recipe,
        "epsilon_dim": int(meta["epsilon_dim"]),
        "hidden_dim": int(meta["hidden_dim"]),
        "num_layers": int(meta["num_layers"]),
        "activation": meta["activation"],
        "seeds_complete": ",".join(str(seed) for seed in seeds),
        "n_seeds": len(seeds),
        "pareto": False,
        "status": "complete" if complete else "incomplete",
    }
    for slug in ("kl", "w2"):
        for kind in ("final", "best"):
            values = [sample[f"{slug}_{kind}"] for sample in samples if f"{slug}_{kind}" in sample]
            row[f"{slug}_{kind}_mean"] = mean(values)
            row[f"{slug}_{kind}_std"] = stdev(values)
            row[f"{slug}_{kind}_count"] = len(values)
    rows.append(row)

complete_rows = [
    row for row in rows
    if row["status"] == "complete"
    and row["kl_final_mean"] is not None
    and row["w2_final_mean"] is not None
]
for candidate in complete_rows:
    candidate["pareto"] = not any(
        other is not candidate
        and other["kl_final_mean"] <= candidate["kl_final_mean"]
        and other["w2_final_mean"] <= candidate["w2_final_mean"]
        and (
            other["kl_final_mean"] < candidate["kl_final_mean"]
            or other["w2_final_mean"] < candidate["w2_final_mean"]
        )
        for other in complete_rows
    )

fieldnames = [
    "recipe_id", "epsilon_dim", "hidden_dim", "num_layers", "activation",
    "seeds_complete", "n_seeds",
    "kl_final_mean", "kl_final_std", "kl_final_count",
    "kl_best_mean", "kl_best_std", "kl_best_count",
    "w2_final_mean", "w2_final_std", "w2_final_count",
    "w2_best_mean", "w2_best_std", "w2_best_count",
    "pareto", "status",
]
summary_csv.parent.mkdir(parents=True, exist_ok=True)
with summary_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

def fmt(value):
    return "-" if value is None else f"{value:.6f}"

lines = [
    "# KDVI VI Architecture Sweep Summary",
    "",
    f"Complete architecture groups: **{len(complete_rows)} / {len(rows)}**.",
    "Metrics are means and sample standard deviations across seeds 0, 1, and 7.",
    "",
]

if complete_rows:
    kl_winner = min(complete_rows, key=lambda row: row["kl_final_mean"])
    w2_winner = min(complete_rows, key=lambda row: row["w2_final_mean"])
    lines.extend([
        "## Winners",
        "",
        f"- **Final KL-ITE:** `{kl_winner['recipe_id']}` - "
        f"{fmt(kl_winner['kl_final_mean'])} +/- {fmt(kl_winner['kl_final_std'])}",
        f"- **Final W2:** `{w2_winner['recipe_id']}` - "
        f"{fmt(w2_winner['w2_final_mean'])} +/- {fmt(w2_winner['w2_final_std'])}",
        "",
        "## Final KL/W2 Pareto Front",
        "",
        "| Recipe | KL-ITE mean +/- std | W2 mean +/- std |",
        "|---|---:|---:|",
    ])
    pareto_rows = sorted(
        (row for row in complete_rows if row["pareto"]),
        key=lambda row: (row["kl_final_mean"], row["w2_final_mean"]),
    )
    for row in pareto_rows:
        lines.append(
            f"| `{row['recipe_id']}` | {fmt(row['kl_final_mean'])} +/- "
            f"{fmt(row['kl_final_std'])} | {fmt(row['w2_final_mean'])} +/- "
            f"{fmt(row['w2_final_std'])} |"
        )
    lines.append("")
else:
    lines.extend(["No architecture group currently has all final KL/W2 metrics.", ""])

incomplete = [row for row in rows if row["status"] != "complete"]
lines.extend(["## Incomplete Architecture Groups", ""])
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
    local epsilon_dim="$5"
    local hidden_dim="$6"
    local num_layers="$7"
    local activation="$8"
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
            "${epsilon_dim}" "${hidden_dim}" "${num_layers}" "${activation}"

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
while IFS=$'\t' read -r run_id recipe seed epsilon_dim hidden_dim num_layers activation; do
    [[ "${run_id}" == "run_id" ]] && continue
    if [[ -f "${DONE_DIR}/${run_id}.done" && -f "${RESULT_MAP_DIR}/${run_id}.path" ]]; then
        continue
    fi
    JOBS+=("${run_id}"$'\t'"${recipe}"$'\t'"${seed}"$'\t'"${epsilon_dim}"$'\t'"${hidden_dim}"$'\t'"${num_layers}"$'\t'"${activation}")
done <"${MANIFEST_PATH}"

echo "Pending runs: ${#JOBS[@]} / 162"

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
        local run_id recipe seed epsilon_dim hidden_dim num_layers activation
        IFS=$'\t' read -r run_id recipe seed epsilon_dim hidden_dim num_layers activation <<<"${job}"

        run_job "${gpu_id}" "${run_id}" "${recipe}" "${seed}" \
            "${epsilon_dim}" "${hidden_dim}" "${num_layers}" "${activation}" &
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
