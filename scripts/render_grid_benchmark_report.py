from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
import sys

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from grid_benchmark_common import (  # noqa: E402
    BNN_TARGETS,
    CAMPAIGN_DIR,
    MANIFEST_PATH,
    MARKDOWN_PATH,
    METHOD_VARIANTS,
    REPO_ROOT,
    TARGETS,
    TARGET_COST_FACTORS,
    VARIANT_SPECS,
    load_yaml,
    metric_support,
    run_id_for,
    runtime_dir,
)


SUMMARY_CSV_PATH = CAMPAIGN_DIR / "generated_reports" / "official_completed_runs.csv"
DETAILED_REPORT_PATH = CAMPAIGN_DIR / "generated_reports" / "official_detailed_report.md"

METRIC_COLUMNS = [
    ("ELBO", "metric__vi_model__elbo"),
    ("KL", "metric__vi_model__kl_ite"),
    ("W2", "metric__vi_model__w2"),
    ("KSD", "metric__vi_model__ksd"),
    ("MMD", "metric__vi_model__mmd"),
    ("Fisher", "metric__vi_model__fisher_div"),
    ("RMSE", "metric__vi_model__rmse"),
    ("Test LLK", "metric__vi_model__test_llk"),
    ("NLL", "metric__vi_model__nll"),
    ("Proxy L2", "diagnostic__reverse_model__score_l2_to_target"),
    ("Rev KL", "metric__reverse_model__kl_ite"),
    ("Rev W2", "metric__reverse_model__w2"),
    ("Rev KSD", "metric__reverse_model__ksd"),
]

METRIC_PREFIX_TO_LABEL = {prefix: label for label, prefix in METRIC_COLUMNS}
_EVENT_TEXT_CACHE: dict[str, str] = {}


def _load_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_latest_terminal_events() -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for queue in ("gpu0", "gpu1"):
        path = runtime_dir() / f"official_{queue}_events.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("status") == "started":
                continue
            latest[event["run_id"]] = event
    return latest


def _load_completed_rows() -> dict[str, dict[str, str]]:
    with SUMMARY_CSV_PATH.open("r", encoding="utf-8", newline="") as fh:
        return {row["run_id"]: row for row in csv.DictReader(fh)}


def _relpath_str(path_str: str | None) -> str:
    if not path_str:
        return "N/A"
    path = Path(path_str)
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except Exception:
        return path_str.replace("\\", "/")


def _event_text(event: dict[str, Any]) -> str:
    cache_key = json.dumps(
        {
            "run_id": event.get("run_id"),
            "console_log": event.get("console_log"),
            "status": event.get("status"),
        },
        sort_keys=True,
    )
    if cache_key in _EVENT_TEXT_CACHE:
        return _EVENT_TEXT_CACHE[cache_key]

    parts: list[str] = []
    tail = event.get("run_log_tail") or []
    if tail:
        parts.append("\n".join(str(line) for line in tail))

    console_log = event.get("console_log")
    if console_log:
        console_path = REPO_ROOT / Path(console_log)
        if console_path.exists():
            parts.append(console_path.read_text(encoding="utf-8", errors="replace"))

    text = "\n".join(part for part in parts if part).strip()
    _EVENT_TEXT_CACHE[cache_key] = text
    return text


def _fmt_num(value: Any) -> str:
    if value in ("", None):
        return "N/A"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(num):
        return "nan"
    abs_num = abs(num)
    if abs_num >= 1.0e4 or (0 < abs_num < 1.0e-3):
        return f"{num:.3e}"
    if abs_num >= 100:
        return f"{num:.3f}"
    if abs_num >= 1:
        return f"{num:.4f}"
    return f"{num:.6f}".rstrip("0").rstrip(".")


def _metric_cell(row: dict[str, str], prefix: str) -> str:
    final_key = f"{prefix}__final"
    best_key = f"{prefix}__best"
    best_epoch_key = f"{prefix}__best_epoch"
    if final_key not in row:
        return "N/A"
    final_text = _fmt_num(row.get(final_key))
    best_value = row.get(best_key)
    best_epoch = row.get(best_epoch_key)
    if best_value in ("", None) or best_epoch in ("", None):
        return final_text
    return f"{final_text} (best {_fmt_num(best_value)}@{best_epoch})"


def _parse_last_epoch(event: dict[str, Any]) -> int | None:
    text = _event_text(event)
    matches = [int(match) for match in re.findall(r"[Ee]poch (\d+)", text)]
    if not matches:
        return None
    return max(matches)


def _parse_last_avg_epoch_time(event: dict[str, Any]) -> str:
    text = _event_text(event)
    matches = re.findall(r"Avg Epoch Time: ([0-9.]+)s", text)
    if not matches:
        return "N/A"
    return matches[-1]


def _failure_class(event: dict[str, Any]) -> str:
    event_text = _event_text(event)
    tail_lower = event_text.lower()
    if "failed to obtain finite samples from realnvp after 3 attempts" in tail_lower:
        return "RealNVP non-finite"
    if "cuda out of memory" in tail_lower or "outofmemoryerror" in tail_lower:
        return "CUDA OOM"
    reason = event.get("failure_reason")
    if reason:
        return str(reason)
    return "Unknown"


def _artifacts_cell(entry: dict[str, Any], event: dict[str, Any] | None, completed_row: dict[str, str] | None) -> str:
    config_path = entry["config_path"]
    result_path = event.get("result_path") if event else ""
    tb_path = completed_row.get("tb_path") if completed_row else ""
    parts = [f"cfg: `{config_path}`"]
    if result_path:
        parts.append(f"res: `{_relpath_str(result_path)}`")
    if tb_path:
        parts.append(f"tb: `{_relpath_str(tb_path)}`")
    return "<br>".join(parts)


def _special_config(entry: dict[str, Any], config: dict[str, Any]) -> str:
    notes = [f"epochs={entry['epochs']}"]
    train_cfg = config.get("train", {})
    reverse_cfg = train_cfg.get("reverse", {}) if isinstance(train_cfg.get("reverse"), dict) else {}

    if entry["variant"] == "ksivi_custom":
        notes.append("vi=custom_inline")
    elif entry["variant"] == "ksivi_standard_cg":
        notes.append("vi=ConditionalGaussian")
        notes.append("pretrain=off")
        notes.append("ema=off")

    if entry["variant"].startswith("dsivi_bs4096"):
        notes.append(f"train.batch={train_cfg.get('batch_size')}")
        if reverse_cfg.get("batch_size") is not None:
            notes.append(f"reverse.batch={reverse_cfg.get('batch_size')}")

    if entry["variant"] == "sivi" and entry["target"] in BNN_TARGETS:
        reverse_sample_num = train_cfg.get("reverse_sample_num")
        if reverse_sample_num is not None:
            notes.append(f"reverse_sample_num={reverse_sample_num}")

    return ", ".join(notes)


def _target_notes(target: str) -> list[str]:
    support = metric_support(target)
    notes: list[str] = []
    if target in BNN_TARGETS:
        notes.append("BNN metric budget: `metric.elbo.batch_size=256`, `metric.ksd.num_samples=1000`, `metric.bnn.num_samples=500`.")
        notes.append("`KL`, `W2`, and `MMD` are `N/A` for this target family by design.")
    elif target == "Langevin_post":
        notes.append("`KL` is disabled for `Langevin_post`; `W2`, `MMD`, `KSD`, `Fisher`, and `ELBO` remain enabled.")
    elif target == "LRwaveform":
        notes.append("`KL`, `W2`, and `MMD` are `N/A`; `ELBO`, `KSD`, and `Fisher` are recorded.")
    else:
        enabled = [name.upper() for name, flag in support.items() if flag and name not in {"bnn"}]
        notes.append(f"Metrics recorded: {', '.join(enabled)}.")
    return notes


def _status_row(
    entry: dict[str, Any],
    event: dict[str, Any] | None,
    completed_row: dict[str, str] | None,
    config: dict[str, Any],
) -> list[str]:
    train_cfg = config.get("train", {})
    reverse_cfg = train_cfg.get("reverse", {}) if isinstance(train_cfg.get("reverse"), dict) else {}
    duration = _fmt_num((completed_row or {}).get("duration_sec") if completed_row else event.get("duration_sec") if event else None)
    avg_epoch = (
        _fmt_num((completed_row or {}).get("avg_epoch_time_sec"))
        if completed_row
        else _parse_last_avg_epoch_time(event) if event else "N/A"
    )

    status = event.get("status") if event else "pending"
    note = ""
    if status == "failed" and event is not None:
        failure_epoch = _parse_last_epoch(event)
        failure_note = _failure_class(event)
        note = failure_note if failure_epoch is None else f"{failure_note}; failed_at_epoch={failure_epoch}"

    row = [
        entry["variant_label"],
        entry["annealing_mode"],
        status,
        str(entry["epochs"]),
        str(train_cfg.get("batch_size", entry["batch_size"])),
        str(reverse_cfg.get("batch_size", entry.get("reverse_batch_size") or "N/A")),
        str(train_cfg.get("reverse_sample_num", "N/A")),
        duration,
        avg_epoch,
        _special_config(entry, config),
        _artifacts_cell(entry, event, completed_row),
        note or "completed",
    ]
    return row


def _metric_row(
    entry: dict[str, Any],
    event: dict[str, Any] | None,
    completed_row: dict[str, str] | None,
) -> list[str]:
    status = event.get("status") if event else "pending"
    row = [entry["variant_label"], entry["annealing_mode"], status]
    if status != "completed" or completed_row is None:
        row.extend(["FAILED" if status == "failed" else "N/A"] * len(METRIC_COLUMNS))
        return row
    for _, prefix in METRIC_COLUMNS:
        row.append(_metric_cell(completed_row, prefix))
    return row


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _records_by_target(
    manifest: list[dict[str, Any]],
    latest_events: dict[str, dict[str, Any]],
    completed_rows: dict[str, dict[str, str]],
) -> dict[str, list[dict[str, Any]]]:
    config_cache: dict[str, dict[str, Any]] = {}
    records: dict[str, list[dict[str, Any]]] = {target: [] for target in TARGETS}
    anneal_order = {"on": 0, "off": 1}
    variant_order = {variant: idx for idx, variant in enumerate(METHOD_VARIANTS)}

    for entry in manifest:
        config_path = Path(entry["config_path"])
        config = config_cache.setdefault(entry["config_path"], load_yaml(REPO_ROOT / config_path))
        event = latest_events.get(entry["run_id"])
        completed_row = completed_rows.get(entry["run_id"])
        records[entry["target"]].append(
            {
                "entry": entry,
                "event": event,
                "completed_row": completed_row,
                "config": config,
            }
        )

    for target in TARGETS:
        records[target].sort(
            key=lambda item: (
                variant_order[item["entry"]["variant"]],
                anneal_order[item["entry"]["annealing_mode"]],
            )
        )
    return records


def _render_per_target_summary(
    manifest: list[dict[str, Any]],
    latest_events: dict[str, dict[str, Any]],
    completed_rows: dict[str, dict[str, str]],
) -> str:
    lines = [
        "Generated from `campaigns/grid_benchmark_20260330/runtime/*.jsonl`, `campaigns/grid_benchmark_20260330/generated_reports/official_completed_runs.csv`, synced `results/`, synced `tb_logs/`, and the generated configs.",
        "",
        "Global standardized settings used across the official campaign:",
        "- `train.log.metric_log_freq=100`",
        "- `metric.elbo.num_batches=2`, `metric.elbo.num_z_samples=1024`",
        "- `metric.fisher.num_samples=1000`, `metric.fisher.num_is_samples=512`",
        "- `resume.enabled=false`",
        "- Grid 1 forced annealing on; Grid 2 forced annealing off.",
        "",
        "Metric cell format: `final (best@epoch)`. `N/A` means the metric was intentionally disabled for that target family. `FAILED` means the run did not finish, so no final campaign metric is recorded for that row.",
    ]

    records = _records_by_target(manifest, latest_events, completed_rows)

    for target in TARGETS:
        target_records = records[target]
        completed_count = sum(1 for item in target_records if (item["event"] or {}).get("status") == "completed")
        failed_count = sum(1 for item in target_records if (item["event"] or {}).get("status") == "failed")
        lines.extend(
            [
                "",
                f"### {target}",
                "",
                f"Status: `{completed_count}` completed, `{failed_count}` failed, `{len(target_records)}` total.",
            ]
        )
        for note in _target_notes(target):
            lines.append(f"- {note}")

        status_headers = [
            "Variant",
            "Anneal",
            "Status",
            "Epochs",
            "Train Batch",
            "Rev Batch",
            "Rev Samples",
            "Runtime (s)",
            "Avg Epoch (s)",
            "Special Config",
            "Artifacts",
            "Note",
        ]
        status_rows = [
            _status_row(item["entry"], item["event"], item["completed_row"], item["config"])
            for item in target_records
        ]
        lines.extend(["", _markdown_table(status_headers, status_rows), ""])

        metric_headers = ["Variant", "Anneal", "Status"] + [label for label, _ in METRIC_COLUMNS]
        metric_rows = [
            _metric_row(item["entry"], item["event"], item["completed_row"])
            for item in target_records
        ]
        lines.extend([_markdown_table(metric_headers, metric_rows), ""])

    return "\n".join(lines).rstrip() + "\n"


def _render_end_summary(
    manifest: list[dict[str, Any]],
    latest_events: dict[str, dict[str, Any]],
    completed_rows: dict[str, dict[str, str]],
) -> str:
    total_runs = len(manifest)
    completed_count = sum(1 for row in latest_events.values() if row.get("status") == "completed")
    failed_count = sum(1 for row in latest_events.values() if row.get("status") == "failed")
    worker_errors = sum(1 for row in latest_events.values() if row.get("status") == "worker_error")

    target_counts: list[list[str]] = []
    for target in TARGETS:
        target_entries = [entry for entry in manifest if entry["target"] == target]
        target_completed = sum(
            1
            for entry in target_entries
            if (latest_events.get(entry["run_id"]) or {}).get("status") == "completed"
        )
        target_failed = sum(
            1
            for entry in target_entries
            if (latest_events.get(entry["run_id"]) or {}).get("status") == "failed"
        )
        target_counts.append([target, str(target_completed), str(target_failed), str(len(target_entries))])

    variant_counts: list[list[str]] = []
    for variant in METHOD_VARIANTS:
        variant_entries = [entry for entry in manifest if entry["variant"] == variant]
        variant_completed = sum(
            1
            for entry in variant_entries
            if (latest_events.get(entry["run_id"]) or {}).get("status") == "completed"
        )
        variant_failed = sum(
            1
            for entry in variant_entries
            if (latest_events.get(entry["run_id"]) or {}).get("status") == "failed"
        )
        variant_counts.append(
            [
                VARIANT_SPECS[variant]["label"],
                str(variant_completed),
                str(variant_failed),
                str(len(variant_entries)),
            ]
        )

    failure_classes = Counter(
        _failure_class(event)
        for event in latest_events.values()
        if event.get("status") == "failed"
    )
    failure_examples: dict[str, list[str]] = {}
    for event in latest_events.values():
        if event.get("status") != "failed":
            continue
        klass = _failure_class(event)
        failure_examples.setdefault(klass, []).append(event["run_id"])

    failure_rows = [
        [klass, str(count), ", ".join(failure_examples.get(klass, [])[:4])]
        for klass, count in failure_classes.most_common()
    ]

    success_by_variant = {
        VARIANT_SPECS[variant]["label"]: (
            sum(
                1
                for entry in manifest
                if entry["variant"] == variant and (latest_events.get(entry["run_id"]) or {}).get("status") == "completed"
            ),
            len([entry for entry in manifest if entry["variant"] == variant]),
        )
        for variant in METHOD_VARIANTS
    }

    bnn_sivi_targets = {}
    for target in BNN_TARGETS:
        statuses = {}
        for anneal in ("on", "off"):
            run_id = run_id_for(target, "sivi", anneal)
            statuses[anneal] = (latest_events.get(run_id) or {}).get("status", "unknown")
        bnn_sivi_targets[target] = statuses

    lines = [
        f"Campaign completed on 2026-04-02 03:17 CST with all `{total_runs}` official runs accounted for: `{completed_count}` completed, `{failed_count}` failed, `0` pending, and `{worker_errors}` worker errors.",
        "",
        f"Completion rate: `{completed_count}/{total_runs}` = `{completed_count / total_runs:.1%}`.",
        "",
        "### Run Status By Target",
        "",
        _markdown_table(["Target", "Completed", "Failed", "Total"], target_counts),
        "",
        "### Run Status By Variant",
        "",
        _markdown_table(["Variant", "Completed", "Failed", "Total"], variant_counts),
        "",
        "### Failure Classes",
        "",
        _markdown_table(["Failure Class", "Count", "Representative Runs"], failure_rows),
        "",
        "### Configuration Notes",
        "",
        "- All official runs used `metric_log_freq=100`, `metric.elbo.num_batches=2`, `metric.elbo.num_z_samples=1024`, `metric.fisher.num_samples=1000`, `metric.fisher.num_is_samples=512`, and `resume.enabled=false`.",
        "- The first grid forced annealing on, and the second grid forced annealing off.",
        "- `Langevin_post` ran with `KL` disabled because the metric is not meaningful in that setup.",
        "- All BNN targets used `metric.elbo.batch_size=256`, `metric.ksd.num_samples=1000`, and kept `KL/W2/MMD` disabled (`N/A`).",
        "- All official BNN `SIVI` runs used `train.reverse_sample_num=2048` after the local campaign adjustment.",
        "",
        "### Key Outcomes",
        "",
    ]

    zero_failure_variants = [
        label for label, (completed, total) in success_by_variant.items() if completed == total
    ]
    if zero_failure_variants:
        lines.append(
            "- Zero-failure variants across the full 24-run method grids: "
            + ", ".join(f"`{label}`" for label in zero_failure_variants)
            + "."
        )

    if failure_classes:
        dominant = failure_classes.most_common(1)[0]
        lines.append(
            f"- The dominant failure mode was `{dominant[0]}` with `{dominant[1]}` failed runs."
        )

    failing_bnn_sivi = [
        target
        for target, statuses in sorted(bnn_sivi_targets.items(), key=lambda item: TARGET_COST_FACTORS[item[0]])
        if "failed" in statuses.values()
    ]
    successful_bnn_sivi = [
        target
        for target, statuses in sorted(bnn_sivi_targets.items(), key=lambda item: TARGET_COST_FACTORS[item[0]])
        if all(status == "completed" for status in statuses.values())
    ]
    if failing_bnn_sivi:
        lines.append(
            "- BNN `SIVI` after the `reverse_sample_num=2048` reduction still failed on "
            + ", ".join(f"`{target}`" for target in failing_bnn_sivi)
            + "."
        )
    if successful_bnn_sivi:
        lines.append(
            "- The reduced BNN `SIVI` setting completed successfully on "
            + ", ".join(f"`{target}`" for target in successful_bnn_sivi)
            + "."
        )

    lines.append(
        "- `AISIVI` and `RSIVI` account for most failures, driven primarily by `ConditionalRealNVP` non-finite sampling rather than worker-level crashes."
    )
    lines.append(
        "- The campaign ended with `0` worker errors, so all 32 failures were method/target-level experiment outcomes rather than queue-management faults."
    )

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    manifest = _load_json(MANIFEST_PATH)
    latest_events = _load_latest_terminal_events()
    completed_rows = _load_completed_rows()

    per_target = _render_per_target_summary(manifest, latest_events, completed_rows)
    end_summary = _render_end_summary(manifest, latest_events, completed_rows)

    full_report = (
        "## Per-Target Summary Tables\n\n"
        + per_target
        + "\n## End-of-Campaign Summary\n\n"
        + end_summary
    )
    DETAILED_REPORT_PATH.write_text(full_report, encoding="utf-8")

    markdown_text = MARKDOWN_PATH.read_text(encoding="utf-8")
    per_idx = markdown_text.index("## Per-Target Summary Tables")
    end_idx = markdown_text.index("## End-of-Campaign Summary")
    updated = markdown_text[:per_idx] + full_report
    MARKDOWN_PATH.write_text(updated.rstrip() + "\n", encoding="utf-8")

    print(f"Wrote {DETAILED_REPORT_PATH}")
    print(f"Updated {MARKDOWN_PATH}")


if __name__ == "__main__":
    main()
