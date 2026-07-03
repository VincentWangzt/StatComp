#!/usr/bin/env bash

# KDVI toy-target MCMC step-size sweep.
#
# The caller is responsible for activating the intended Python environment.
# This script invokes `python` directly, schedules one run per visible GPU,
# resumes completed work, retries failures once, and aggregates final metrics
# across three seeds for each target/step-size recipe.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CAMPAIGN_SLUG="kdvi_toy_mcmc_step_sweep_mala"
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
Usage: bash scripts/run_kdvi_toy_mcmc_step_sweep.sh [options]

Options:
  --dry-run           Generate and validate the 147-run manifest only.
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
    local target seed step config_path metric_family recipe run_id step_slug

    printf 'run_id\trecipe_id\ttarget\tseed\tconfig_path\tmcmc_step_size\tmetric_family\n' >"${tmp_path}"

    for target in banana x_shaped multimodal 8_gaussians 8_gaussians_small student_uc Langevin_post; do
        config_path="configs/kdvi_${target}.yaml"
        metric_family="kl_w2"
        if [[ "${target}" == "Langevin_post" ]]; then
            metric_family="elm_w2"
        fi

        for step in 1e-1 5e-2 2e-2 1e-2 5e-3 2e-3 1e-3; do
            if [[ "${target}" == "Langevin_post" ]]; then
                case "${step}" in
                    1e-1) step="1e-2" ;;
                    5e-2) step="5e-3" ;;
                    2e-2) step="2e-3" ;;
                    1e-2) step="1e-3" ;;
                    5e-3) step="5e-4" ;;
                    2e-3) step="2e-4" ;;
                    1e-3) step="1e-4" ;;
                esac
            fi
            step_slug="$(step_label "${step}")"
            recipe="KDVI-${target}-mcmcstep${step_slug}-mala"
            for seed in 0 1 7; do
                run_id="${recipe}-seed${seed}"
                printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "${run_id}" "${recipe}" "${target}" "${seed}" \
                    "${config_path}" "${step}" "${metric_family}" >>"${tmp_path}"
            done
        done
    done

    mv "${tmp_path}" "${MANIFEST_PATH}"
}

validate_manifest() {
    python - "${REPO_ROOT}" "${MANIFEST_PATH}" <<'PY'
import csv
import sys
from collections import defaultdict
from pathlib import Path

repo_root = Path(sys.argv[1])
path = Path(sys.argv[2])
with path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))

expected_targets = {
    "banana",
    "x_shaped",
    "multimodal",
    "8_gaussians",
    "8_gaussians_small",
    "student_uc",
    "Langevin_post",
}
expected_2d_steps = {"1e-1", "5e-2", "2e-2", "1e-2", "5e-3", "2e-3", "1e-3"}
expected_langevin_steps = {"1e-2", "5e-3", "2e-3", "1e-3", "5e-4", "2e-4", "1e-4"}

run_ids = [row["run_id"] for row in rows]
groups = defaultdict(list)
target_steps = defaultdict(set)
targets = set()
missing = []
for row in rows:
    groups[row["recipe_id"]].append(int(row["seed"]))
    targets.add(row["target"])
    target_steps[row["target"]].add(row["mcmc_step_size"])
    if not (repo_root / row["config_path"]).is_file():
        missing.append(str(repo_root / row["config_path"]))

errors = []
if len(rows) != 147:
    errors.append(f"expected 147 runs, found {len(rows)}")
if len(set(run_ids)) != 147:
    errors.append("run IDs are not unique")
if len(groups) != 49:
    errors.append(f"expected 49 target-step recipe groups, found {len(groups)}")
bad_groups = {
    recipe: seeds for recipe, seeds in groups.items()
    if sorted(seeds) != [0, 1, 7]
}
if bad_groups:
    errors.append(f"groups without exactly seeds 0,1,7: {len(bad_groups)}")
if targets != expected_targets:
    errors.append(f"unexpected targets: {sorted(targets)}")
for target in expected_targets - {"Langevin_post"}:
    if target_steps[target] != expected_2d_steps:
        errors.append(f"{target} has unexpected step sizes: {sorted(target_steps[target])}")
if target_steps["Langevin_post"] != expected_langevin_steps:
    errors.append(
        "Langevin_post has unexpected step sizes: "
        f"{sorted(target_steps['Langevin_post'])}"
    )
metric_families = {row["target"]: row["metric_family"] for row in rows}
if metric_families.get("Langevin_post") != "elm_w2":
    errors.append("Langevin_post rows must use metric_family=elm_w2")
for target in expected_targets - {"Langevin_post"}:
    if metric_families.get(target) != "kl_w2":
        errors.append(f"{target} rows must use metric_family=kl_w2")
if missing:
    errors.append("missing config files:\n  " + "\n  ".join(sorted(set(missing))))

if errors:
    raise SystemExit("Manifest validation failed:\n- " + "\n- ".join(errors))

print(
    "Manifest validated: "
    f"{len(rows)} runs, {len(groups)} target-step groups, "
    f"{len(targets)} targets."
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
    local target="$5"
    local seed="$6"
    local config_path="$7"
    local mcmc_step_size="$8"
    local metric_family="$9"

    output_ref=(
        python src.py --config "${config_path}"
        "cuda_visible_devices=${gpu_id}"
        "seed=${seed}"
        "output.results_dir=results/${CAMPAIGN_SLUG}/${run_name}"
        "train.kdvi.mcmc_step_size=${mcmc_step_size}"
        "tracking.campaign=${CAMPAIGN_SLUG}"
        "tracking.group=${recipe}"
        "tracking.run_name=${run_name}"
        "train.kdvi.mcmc_type=mala"
    )

    if [[ "${metric_family}" == "elm_w2" ]]; then
        output_ref+=(
            "metric.kl_ite.enabled=false"
            "metric.expected_log_marginal.enabled=true"
            "metric.w2.enabled=true"
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
    IFS=$'\t' read -r run_id recipe target seed config_path mcmc_step_size metric_family <<<"${first_job}"
    cmd=()
    build_command cmd 0 "${run_id}" "${recipe}" "${target}" "${seed}" \
        "${config_path}" "${mcmc_step_size}" "${metric_family}"
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
ELM_TAG = "metric/vi_model/kde_expected_log_marginal"
EXPECTED_SEEDS = [0, 1, 7]
TARGET_ORDER = [
    "banana",
    "x_shaped",
    "multimodal",
    "8_gaussians",
    "8_gaussians_small",
    "student_uc",
    "Langevin_post",
]
STEP_ORDER = ["1e-1", "5e-2", "2e-2", "1e-2", "5e-3", "2e-3", "1e-3"]
LANGEVIN_STEP_ORDER = ["1e-2", "5e-3", "2e-3", "1e-3", "5e-4", "2e-4", "1e-4"]

with manifest_path.open(newline="", encoding="utf-8") as handle:
    manifest = list(csv.DictReader(handle, delimiter="\t"))

recipe_meta = {}
by_recipe = defaultdict(list)

def final_metrics(metrics_path, metric_family):
    required = [W2_TAG, ELM_TAG] if metric_family == "elm_w2" else [KL_TAG, W2_TAG]
    points = {tag: [] for tag in required}
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
    out = {}
    for tag, values in points.items():
        step, value = max(values, key=lambda item: item[0])
        if tag == KL_TAG:
            out["kl_ite"] = value
            out["kl_ite_iter"] = step
        elif tag == W2_TAG:
            out["w2"] = value
            out["w2_iter"] = step
        elif tag == ELM_TAG:
            out["kde_expected_log_marginal"] = value
            out["kde_expected_log_marginal_iter"] = step
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
    metrics = final_metrics(metrics_path, item["metric_family"])
    if metrics:
        by_recipe[recipe].append({"seed": int(item["seed"]), **metrics})

def mean(values):
    return statistics.mean(values) if values else None

def stdev(values):
    return statistics.stdev(values) if len(values) >= 2 else None

def step_rank(row):
    order = LANGEVIN_STEP_ORDER if row["target"] == "Langevin_post" else STEP_ORDER
    try:
        return order.index(row["mcmc_step_size"])
    except ValueError:
        return len(order)

rows = []
for recipe in sorted(recipe_meta, key=lambda key: (
    TARGET_ORDER.index(recipe_meta[key]["target"]),
    step_rank(recipe_meta[key]),
)):
    meta = recipe_meta[recipe]
    samples = sorted(by_recipe.get(recipe, []), key=lambda item: item["seed"])
    seeds = [sample["seed"] for sample in samples]
    complete = seeds == EXPECTED_SEEDS
    row = {
        "recipe_id": recipe,
        "target": meta["target"],
        "mcmc_step_size": meta["mcmc_step_size"],
        "metric_family": meta["metric_family"],
        "seeds_complete": ",".join(str(seed) for seed in seeds),
        "n_seeds": len(seeds),
        "status": "complete" if complete else "incomplete",
        "pareto": False,
    }
    for slug in ("kl_ite", "w2", "kde_expected_log_marginal"):
        values = [sample[slug] for sample in samples if slug in sample]
        iters = [sample[f"{slug}_iter"] for sample in samples if f"{slug}_iter" in sample]
        row[f"{slug}_mean"] = mean(values)
        row[f"{slug}_std"] = stdev(values)
        row[f"{slug}_count"] = len(values)
        row[f"{slug}_final_iter_min"] = min(iters) if iters else None
        row[f"{slug}_final_iter_max"] = max(iters) if iters else None
    rows.append(row)

complete_rows = [row for row in rows if row["status"] == "complete"]
for target in TARGET_ORDER:
    target_rows = [row for row in complete_rows if row["target"] == target]
    for candidate in target_rows:
        if candidate["metric_family"] == "elm_w2":
            candidate["pareto"] = not any(
                other is not candidate
                and other["kde_expected_log_marginal_mean"] >= candidate["kde_expected_log_marginal_mean"]
                and other["w2_mean"] <= candidate["w2_mean"]
                and (
                    other["kde_expected_log_marginal_mean"] > candidate["kde_expected_log_marginal_mean"]
                    or other["w2_mean"] < candidate["w2_mean"]
                )
                for other in target_rows
            )
        else:
            candidate["pareto"] = not any(
                other is not candidate
                and other["kl_ite_mean"] <= candidate["kl_ite_mean"]
                and other["w2_mean"] <= candidate["w2_mean"]
                and (
                    other["kl_ite_mean"] < candidate["kl_ite_mean"]
                    or other["w2_mean"] < candidate["w2_mean"]
                )
                for other in target_rows
            )

fieldnames = [
    "recipe_id",
    "target",
    "mcmc_step_size",
    "metric_family",
    "seeds_complete",
    "n_seeds",
    "status",
    "pareto",
]
for slug in ("kl_ite", "w2", "kde_expected_log_marginal"):
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
    "# KDVI Toy MCMC Step-Size MALA Sweep Summary",
    "",
    f"Complete target-step groups: **{len(complete_rows)} / {len(rows)}**.",
    "Metrics are final logged values summarized as means and sample standard deviations across seeds 0, 1, and 7.",
    "",
]

for target in TARGET_ORDER:
    target_rows = [row for row in rows if row["target"] == target]
    complete_target_rows = [row for row in target_rows if row["status"] == "complete"]
    lines.extend([f"## {target}", ""])
    if not complete_target_rows:
        lines.extend(["No complete step-size groups yet.", ""])
    elif target == "Langevin_post":
        elm_winner = max(
            complete_target_rows,
            key=lambda row: row["kde_expected_log_marginal_mean"],
        )
        w2_winner = min(complete_target_rows, key=lambda row: row["w2_mean"])
        lines.extend([
            "### Winners",
            "",
            f"- **KDE ELM:** `{elm_winner['recipe_id']}` - "
            f"{fmt(elm_winner['kde_expected_log_marginal_mean'])} +/- "
            f"{fmt(elm_winner['kde_expected_log_marginal_std'])}",
            f"- **W2:** `{w2_winner['recipe_id']}` - "
            f"{fmt(w2_winner['w2_mean'])} +/- {fmt(w2_winner['w2_std'])}",
            "",
            "### ELM/W2 Pareto Front",
            "",
            "| Step size | KDE ELM mean +/- std | W2 mean +/- std |",
            "|---:|---:|---:|",
        ])
        pareto_rows = sorted(
            (row for row in complete_target_rows if row["pareto"]),
            key=lambda row: (-row["kde_expected_log_marginal_mean"], row["w2_mean"]),
        )
        for row in pareto_rows:
            lines.append(
                f"| `{row['mcmc_step_size']}` | "
                f"{fmt(row['kde_expected_log_marginal_mean'])} +/- "
                f"{fmt(row['kde_expected_log_marginal_std'])} | "
                f"{fmt(row['w2_mean'])} +/- {fmt(row['w2_std'])} |"
            )
        lines.append("")
    else:
        kl_winner = min(complete_target_rows, key=lambda row: row["kl_ite_mean"])
        w2_winner = min(complete_target_rows, key=lambda row: row["w2_mean"])
        lines.extend([
            "### Winners",
            "",
            f"- **KL-ITE:** `{kl_winner['recipe_id']}` - "
            f"{fmt(kl_winner['kl_ite_mean'])} +/- {fmt(kl_winner['kl_ite_std'])}",
            f"- **W2:** `{w2_winner['recipe_id']}` - "
            f"{fmt(w2_winner['w2_mean'])} +/- {fmt(w2_winner['w2_std'])}",
            "",
            "### KL/W2 Pareto Front",
            "",
            "| Step size | KL-ITE mean +/- std | W2 mean +/- std |",
            "|---:|---:|---:|",
        ])
        pareto_rows = sorted(
            (row for row in complete_target_rows if row["pareto"]),
            key=lambda row: (row["kl_ite_mean"], row["w2_mean"]),
        )
        for row in pareto_rows:
            lines.append(
                f"| `{row['mcmc_step_size']}` | "
                f"{fmt(row['kl_ite_mean'])} +/- {fmt(row['kl_ite_std'])} | "
                f"{fmt(row['w2_mean'])} +/- {fmt(row['w2_std'])} |"
            )
        lines.append("")

    lines.extend([
        "### All Step Sizes",
        "",
    ])
    if target == "Langevin_post":
        lines.extend([
            "| Step size | Status | Seeds | KDE ELM | W2 |",
            "|---:|---|---|---:|---:|",
        ])
        for row in target_rows:
            lines.append(
                f"| `{row['mcmc_step_size']}` | {row['status']} | "
                f"{row['seeds_complete'] or 'none'} | "
                f"{fmt(row['kde_expected_log_marginal_mean'])} +/- "
                f"{fmt(row['kde_expected_log_marginal_std'])} | "
                f"{fmt(row['w2_mean'])} +/- {fmt(row['w2_std'])} |"
            )
    else:
        lines.extend([
            "| Step size | Status | Seeds | KL-ITE | W2 |",
            "|---:|---|---|---:|---:|",
        ])
        for row in target_rows:
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
    local target="$4"
    local seed="$5"
    local config_path="$6"
    local mcmc_step_size="$7"
    local metric_family="$8"
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
            "${seed}" "${config_path}" "${mcmc_step_size}" "${metric_family}"

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
while IFS=$'\t' read -r run_id recipe target seed config_path mcmc_step_size metric_family; do
    [[ "${run_id}" == "run_id" ]] && continue
    if [[ -f "${DONE_DIR}/${run_id}.done" && -f "${RESULT_MAP_DIR}/${run_id}.path" ]]; then
        continue
    fi
    JOBS+=("${run_id}"$'\t'"${recipe}"$'\t'"${target}"$'\t'"${seed}"$'\t'"${config_path}"$'\t'"${mcmc_step_size}"$'\t'"${metric_family}")
done <"${MANIFEST_PATH}"

echo "Pending runs: ${#JOBS[@]} / 147"

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
        local run_id recipe target seed config_path mcmc_step_size metric_family
        IFS=$'\t' read -r run_id recipe target seed config_path mcmc_step_size metric_family <<<"${job}"

        run_job "${gpu_id}" "${run_id}" "${recipe}" "${target}" "${seed}" \
            "${config_path}" "${mcmc_step_size}" "${metric_family}" &
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
