#!/usr/bin/env bash
#
# KDVI Comprehensive Benchmark
# 2 schedules × 5 seeds × 6 targets = 60 runs
#
# Campaign 1 (kdvi_bench_long):  KSIVI-like schedule (50k/100k epochs)
# Campaign 2 (kdvi_bench_short): DSIVI-like schedule (10k epochs)
#
# Usage:
#   bash scripts/run_kdvi_benchmark.sh [OPTIONS]
#
# Options:
#   --seeds "42 43 44 45 46"   Seeds (default: "42 43 44 45 46")
#   --targets "banana ..."     Restrict to specific targets (default: all 6 toy)
#   --gpus "0 1 2 3"           GPU indices (default: auto-discover)
#   --dry-run                  Preview without running
#   --only-long                Run only Campaign 1 (KSIVI-like)
#   --only-short               Run only Campaign 2 (DSIVI-like)
#   --skip-finalization        Skip per-run finalization
#   --retry-failed             Retry previously failed runs
#   --finalize-workers N       Async finalization workers (default: 2)
#   --help                     Print help and exit
#
set -euo pipefail

# ─── Resolve repo root ────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ─── Defaults ─────────────────────────────────────────────────────────────────
SEEDS="42 43 44 45 46"
TARGETS=""
GPUS=""
DRY_RUN=false
ONLY_LONG=false
ONLY_SHORT=false
SKIP_FINALIZATION=false
RETRY_FAILED=false
FINALIZE_WORKERS=2

# ─── Parse arguments ──────────────────────────────────────────────────────────
print_help() {
    sed -n '/^# Usage:/,/^set -euo/p' "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \?//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seeds)
            SEEDS="$2"; shift 2 ;;
        --targets)
            TARGETS="$2"; shift 2 ;;
        --gpus)
            GPUS="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        --only-long)
            ONLY_LONG=true; shift ;;
        --only-short)
            ONLY_SHORT=true; shift ;;
        --skip-finalization)
            SKIP_FINALIZATION=true; shift ;;
        --retry-failed)
            RETRY_FAILED=true; shift ;;
        --finalize-workers)
            FINALIZE_WORKERS="$2"; shift 2 ;;
        --help|-h)
            print_help ;;
        *)
            echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ─── Detect Python ────────────────────────────────────────────────────────────
PYTHON="${PYTHON:-python}"
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: Python not found. Set PYTHON env var or activate your venv." >&2
    exit 1
fi

echo "═══════════════════════════════════════════════════════════════════════"
echo " KDVI Comprehensive Benchmark"
echo " 2 schedules × 5 seeds × 6 targets = 60 runs"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "  Python:     $PYTHON"
echo "  Seeds:      $SEEDS"
echo "  Targets:    ${TARGETS:-all toy (banana, 8_gaussians, multimodal, x_shaped, student_uc, Langevin_post)}"
echo "  Dry run:    $DRY_RUN"
echo ""

# ─── Build common args ────────────────────────────────────────────────────────
build_common_args() {
    local campaign_slug="$1"
    local -a args=()

    args+=(--campaign-slug "$campaign_slug")
    args+=(--results-dir "results/$campaign_slug")
    args+=(--tb-dir "tb_logs/$campaign_slug")
    # shellcheck disable=SC2086
    args+=(--seeds $SEEDS)
    args+=(--methods kdvi)

    if [[ -n "$TARGETS" ]]; then
        # shellcheck disable=SC2086
        args+=(--targets $TARGETS)
    fi
    if [[ -n "$GPUS" ]]; then
        # shellcheck disable=SC2086
        args+=(--gpus $GPUS)
    fi
    if [[ "$DRY_RUN" == true ]]; then
        args+=(--dry-run)
    fi
    if [[ "$RETRY_FAILED" == true ]]; then
        args+=(--retry-failed)
    fi
    if [[ "$SKIP_FINALIZATION" != true ]]; then
        args+=(--finalize-mode async)
        args+=(--finalize-workers "$FINALIZE_WORKERS")
    fi

    echo "${args[@]}"
}

# ─── Campaign 1: KSIVI-like schedule (long) ──────────────────────────────────
if [[ "$ONLY_SHORT" == false ]]; then
    echo "───────────────────────────────────────────────────────────────────────"
    echo " Campaign 1/2: KSIVI-like schedule (kdvi_bench_long)"
    echo "   50k epochs (100k Langevin_post), eval/plot every 5k (10k LP)"
    echo "───────────────────────────────────────────────────────────────────────"
    echo ""

    # shellcheck disable=SC2046,SC2086
    COMMON_ARGS=($(build_common_args "kdvi_bench_long"))

    echo "  Command: $PYTHON scripts/run_default_config_grid_sweep.py ${COMMON_ARGS[*]}"
    echo ""

    "$PYTHON" scripts/run_default_config_grid_sweep.py "${COMMON_ARGS[@]}"
    EXIT_CODE=$?

    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "ERROR: Campaign 1 exited with code $EXIT_CODE" >&2
        exit $EXIT_CODE
    fi

    if [[ "$DRY_RUN" == true ]]; then
        echo ""
        echo "(Dry run — Campaign 1 preview complete)"
        echo ""
    else
        echo ""
        echo "  Campaign 1 complete."
        echo ""
    fi
fi

# ─── Campaign 2: DSIVI-like schedule (short) ─────────────────────────────────
if [[ "$ONLY_LONG" == false ]]; then
    echo "───────────────────────────────────────────────────────────────────────"
    echo " Campaign 2/2: DSIVI-like schedule (kdvi_bench_short)"
    echo "   10k epochs, eval/plot every 1k, gamma=0.7"
    echo "───────────────────────────────────────────────────────────────────────"
    echo ""

    # shellcheck disable=SC2046,SC2086
    COMMON_ARGS=($(build_common_args "kdvi_bench_short"))

    # DSIVI-like overrides
    COMMON_ARGS+=(--extra-override "train.epochs=10000")
    COMMON_ARGS+=(--extra-override "train.checkpoint.freq=1000")
    COMMON_ARGS+=(--extra-override "train.sample.freq=1000")
    COMMON_ARGS+=(--extra-override "train.plot.freq=1000")
    COMMON_ARGS+=(--extra-override "train.log.metric_log_freq=100")
    COMMON_ARGS+=(--extra-override "train.annealing.steps=5000")
    COMMON_ARGS+=(--extra-override "train.vi.scheduler.gamma=0.7")

    echo "  Command: $PYTHON scripts/run_default_config_grid_sweep.py ${COMMON_ARGS[*]}"
    echo ""

    "$PYTHON" scripts/run_default_config_grid_sweep.py "${COMMON_ARGS[@]}"
    EXIT_CODE=$?

    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "ERROR: Campaign 2 exited with code $EXIT_CODE" >&2
        exit $EXIT_CODE
    fi

    if [[ "$DRY_RUN" == true ]]; then
        echo ""
        echo "(Dry run — Campaign 2 preview complete)"
        echo ""
    else
        echo ""
        echo "  Campaign 2 complete."
        echo ""
    fi
fi

# ─── Done ─────────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════════════"
echo " KDVI Benchmark complete."
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "  Campaign 1 (long):   campaigns/kdvi_bench_long/"
echo "  Campaign 2 (short):  campaigns/kdvi_bench_short/"
echo "  Results (long):      results/kdvi_bench_long/"
echo "  Results (short):     results/kdvi_bench_short/"
echo "  TB logs (long):      tb_logs/kdvi_bench_long/"
echo "  TB logs (short):     tb_logs/kdvi_bench_short/"
echo ""
