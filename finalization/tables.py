from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .config import repo_path


LOWER_IS_BETTER = {
    "w2",
    "w2_edge_5",
    "w2_edge_8",
    "w2_edge_10",
    "duration_sec",
    "wall_clock_sec",
    "rmse",
    "nll",
}
HIGHER_IS_BETTER = {"elbo", "kde_elm"}
NO_BOLD_METRICS = {"training_iterations"}

METRIC_LABELS = {
    "elbo": "ELBO",
    "w2": "W2",
    "w2_edge_5": "edge width 5",
    "w2_edge_8": "edge width 8",
    "w2_edge_10": "edge width 10",
    "duration_sec": "wall-clock (s)",
    "wall_clock_sec": "wall-clock (s)",
    "training_time_sec": "training time (s)",
    "training_iterations": "iterations",
    "kde_elm": "KDE ELM",
    "rmse": "RMSE",
    "nll": "NLL",
}


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _fmt(value: float | None, precision: int) -> str:
    if value is None:
        return "--"
    if abs(value) >= 1.0e4 or (0 < abs(value) < 1.0e-3):
        return f"{value:.{precision}e}"
    return f"{value:.{precision}f}"


def _mean_key(metric: str) -> str:
    if metric == "duration_sec":
        return "wall_clock_sec_mean"
    if metric == "training_iterations":
        return "training_iterations_mean"
    return f"{metric}_mean"


def _se_key(metric: str) -> str | None:
    if metric == "duration_sec":
        return "wall_clock_sec_se"
    if metric == "training_iterations":
        return None
    return f"{metric}_se"


def _metric_value(row: dict[str, Any], metric: str) -> float | None:
    value = _float(row.get(_mean_key(metric)))
    if value is None and metric == "duration_sec":
        value = _float(row.get("duration_sec_mean"))
    return value


def _metric_se(row: dict[str, Any], metric: str) -> float | None:
    se_key = _se_key(metric)
    if se_key is None:
        return None
    value = _float(row.get(se_key))
    if value is None and metric == "duration_sec":
        value = _float(row.get("duration_sec_se"))
    return value


def _cell(row: dict[str, Any], metric: str, *, bold: bool, value_precision: int, se_precision: int) -> str:
    mean = _metric_value(row, metric)
    se = _metric_se(row, metric)
    if mean is None:
        return "--"
    text = _fmt(mean, value_precision)
    if se is not None:
        text = f"{text} $\\pm$ {_fmt(se, se_precision)}"
    return f"\\textbf{{{text}}}" if bold and metric not in NO_BOLD_METRICS else text


def _best_methods(rows: list[dict[str, Any]], metric: str) -> set[str]:
    if metric in NO_BOLD_METRICS:
        return set()
    values: list[tuple[str, float]] = []
    for row in rows:
        value = _metric_value(row, metric)
        if value is not None:
            values.append((str(row["method"]), value))
    if not values:
        return set()
    best = max(value for _, value in values) if metric in HIGHER_IS_BETTER else min(value for _, value in values)
    return {method for method, value in values if math.isclose(value, best) or value == best}


def _ordered_methods(
    methods: list[str],
    *,
    drop: set[str] | None = None,
    last: list[str] | None = None,
) -> list[str]:
    drop_upper = {method.upper() for method in (drop or set())}
    last_upper = [method.upper() for method in (last or [])]
    selected = [method for method in methods if method.upper() not in drop_upper]
    base = [method for method in selected if method.upper() not in set(last_upper)]
    tail = [method for target in last_upper for method in selected if method.upper() == target]
    return base + tail


def _label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " "))


def _target_metric_table(
    rows: list[dict[str, Any]],
    *,
    targets: list[str],
    methods: list[str],
    metrics: list[str],
    caption: str,
    label: str,
    value_precision: int,
    se_precision: int,
) -> str:
    colspec = "l" + "c" * (len(methods) * len(metrics))
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
    ]
    metric_header = " & ".join(f"\\multicolumn{{{len(methods)}}}{{c}}{{{_label(metric)}}}" for metric in metrics)
    lines.append(f"Target & {metric_header} \\\\")
    method_header = " & ".join(method for _metric in metrics for method in methods)
    lines.append(f" & {method_header} \\\\")
    lines.append("\\midrule")

    by_target_method = {(row["target"], row["method"]): row for row in rows}
    by_target = {target: [row for row in rows if row["target"] == target and str(row["method"]) in methods] for target in targets}
    for target in targets:
        cells = [target]
        best_by_metric = {metric: _best_methods(by_target.get(target, []), metric) for metric in metrics}
        for metric in metrics:
            for method in methods:
                row = by_target_method.get((target, method))
                cells.append(
                    "--"
                    if row is None
                    else _cell(
                        row,
                        metric,
                        bold=method in best_by_metric[metric],
                        value_precision=value_precision,
                        se_precision=se_precision,
                    )
                )
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def _method_metric_table(
    rows: list[dict[str, Any]],
    *,
    target: str,
    methods: list[str],
    metrics: list[str],
    caption: str,
    label: str,
    value_precision: int,
    se_precision: int,
) -> str:
    target_rows = [row for row in rows if row["target"] == target and str(row["method"]) in methods]
    by_method = {row["method"]: row for row in target_rows}
    best_by_metric = {metric: _best_methods(target_rows, metric) for metric in metrics}
    colspec = "l" + "c" * len(metrics)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        "Method & " + " & ".join(_label(metric) for metric in metrics) + " \\\\",
        "\\midrule",
    ]
    for method in methods:
        row = by_method.get(method)
        cells = [method]
        for metric in metrics:
            cells.append(
                "--"
                if row is None
                else _cell(
                    row,
                    metric,
                    bold=method in best_by_metric[metric],
                    value_precision=value_precision,
                    se_precision=se_precision,
                )
            )
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def render_tables(summary_rows: list[dict[str, Any]], cfg: Any) -> dict[str, Path]:
    root = repo_path(str(cfg.campaign.output_dir))
    assert root is not None
    out_dir = root / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [str(method) for method in cfg.selection.methods]
    vp = int(cfg.tables.value_precision)
    sp = int(cfg.tables.se_precision)
    outputs: dict[str, Path] = {}

    if bool(cfg.modules.get("toy_tables", False)):
        toy_targets = [str(target) for target in cfg.selection.evaluation_targets if str(target) != "Langevin_post" and not str(target).startswith("Bnn_")]
        toy_methods = _ordered_methods(methods, drop={"SIVI"}, last=["DSIVI"])
        toy_text = _target_metric_table(
            summary_rows,
            targets=toy_targets,
            methods=toy_methods,
            metrics=["elbo", "w2", "duration_sec"],
            caption="Toy target final evaluation metrics.",
            label="tab:toy-final-eval",
            value_precision=vp,
            se_precision=sp,
        )
        outputs["toy"] = out_dir / "toy_metrics.tex"
        outputs["toy"].write_text(toy_text, encoding="utf-8")

    if bool(cfg.modules.get("langevin_table", False)):
        langevin_methods = _ordered_methods(methods, last=["DSIVI"])
        langevin_text = _method_metric_table(
            summary_rows,
            target="Langevin_post",
            methods=langevin_methods,
            metrics=["kde_elm", "duration_sec", "training_iterations"],
            caption="Langevin_post final evaluation metrics.",
            label="tab:langevin-final-eval",
            value_precision=vp,
            se_precision=sp,
        )
        outputs["langevin"] = out_dir / "langevin_metrics.tex"
        outputs["langevin"].write_text(langevin_text, encoding="utf-8")

    if bool(cfg.modules.get("student_edge_table", False)):
        student_methods = _ordered_methods(methods, drop={"KSIVI"}, last=["DSIVI"])
        edge_text = _method_metric_table(
            summary_rows,
            target="student_uc",
            methods=student_methods,
            metrics=["w2_edge_5", "w2_edge_8", "w2_edge_10"],
            caption="student_uc constrained W2 by edge width.",
            label="tab:student-edge-w2",
            value_precision=vp,
            se_precision=sp,
        )
        outputs["student_edge"] = out_dir / "student_edge_w2.tex"
        outputs["student_edge"].write_text(edge_text, encoding="utf-8")

    if bool(cfg.modules.get("bnn_table", False)):
        bnn_targets = [str(target) for target in cfg.selection.bnn_targets]
        bnn_methods = _ordered_methods(methods, last=["DSIVI"])
        bnn_text = _target_metric_table(
            summary_rows,
            targets=bnn_targets,
            methods=bnn_methods,
            metrics=["rmse", "nll"],
            caption="BNN test RMSE and test NLL.",
            label="tab:bnn-rmse-nll",
            value_precision=vp,
            se_precision=sp,
        )
        outputs["bnn"] = out_dir / "bnn_rmse_nll.tex"
        outputs["bnn"].write_text(bnn_text, encoding="utf-8")

    return outputs
