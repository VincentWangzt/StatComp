from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .config import repo_path


LOWER_IS_BETTER = {
    "d_kl",
    "w2",
    "w2_trunc_abs_6",
    "w2_trunc_abs_8",
    "w2_edge_5",
    "w2_edge_8",
    "w2_edge_10",
    "duration_sec",
    "wall_clock_sec",
    "rmse",
    "nll",
}
HIGHER_IS_BETTER = {"elbo", "kde_elm"}
NO_BOLD_METRICS = {"training_iterations", "duration_sec", "wall_clock_sec", "training_time_sec"}

METRIC_LABELS = {
    "elbo": "ELBO",
    "d_kl": "$D_{\\mathrm{KL}}$",
    "w2": "W2",
    "w2_trunc_abs_6": "W2 $|x|<6$",
    "w2_trunc_abs_8": "W2 $|x|<8$",
    "w2_edge_5": "edge width 5",
    "w2_edge_8": "edge width 8",
    "w2_edge_10": "edge width 10",
    "duration_sec": "wall-clock (s)",
    "wall_clock_sec": "wall-clock (s)",
    "training_time_sec": "training time (s)",
    "training_iterations": "iterations",
    "kde_elm": "KDE ELM",
    "rmse": "Test RMSE",
    "nll": "Test NLL",
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


def _fmt_compact_count(value: float | None) -> str:
    if value is None:
        return "--"
    rounded = int(round(value))
    if rounded >= 1000 and rounded % 1000 == 0:
        return f"{rounded // 1000}K"
    return f"{rounded}"


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
    if row is None:
        return None
    if metric == "d_kl":
        value = _metric_value(row, "elbo")
        return -value if value is not None else None
    value = _float(row.get(_mean_key(metric)))
    if value is None and metric == "duration_sec":
        value = _float(row.get("duration_sec_mean"))
    return value


def _metric_se(row: dict[str, Any], metric: str) -> float | None:
    if row is None:
        return None
    if metric == "d_kl":
        return _metric_se(row, "elbo")
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
    if metric == "training_iterations":
        return _fmt_compact_count(mean)
    text = _fmt(mean, value_precision)
    if se is not None:
        text = f"{text} $\\pm$ {_fmt(se, se_precision)}"
    return f"\\textbf{{{text}}}" if bold and metric not in NO_BOLD_METRICS else text


def _value_se_text(
    mean: float | None,
    se: float | None,
    *,
    value_precision: int,
    se_precision: int,
    bold: bool = False,
    se_footnotesize: bool = False,
) -> str:
    if mean is None:
        return "--"
    value_text = _fmt(mean, value_precision)
    if bold:
        value_text = f"\\textbf{{{value_text}}}"
    if se is None:
        return value_text
    se_text = _fmt(se, se_precision)
    if bold:
        se_text = f"\\textbf{{{se_text}}}"
    if se_footnotesize:
        se_text = f"{{\\footnotesize {se_text}}}"
    return f"{value_text} $\\pm$ {se_text}"


def _cell_small_se(
    row: dict[str, Any],
    metric: str,
    *,
    bold: bool,
    value_precision: int,
    se_precision: int,
) -> str:
    return _value_se_text(
        _metric_value(row, metric),
        _metric_se(row, metric),
        value_precision=value_precision,
        se_precision=se_precision,
        bold=bold and metric not in NO_BOLD_METRICS,
        se_footnotesize=True,
    )


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


TOY_METHOD_GRID_METHODS = ["UIVI", "AISIVI", "DSIVI"]
TOY_METHOD_GRID_TARGETS = ["x_shaped", "student_uc", "8_gaussians"]
TOY_METHOD_GRID_W2_METRICS = {
    "student_uc": "w2_trunc_abs_8",
    "8_gaussians": "w2_trunc_abs_6",
    "x_shaped": "w2",
}


def _mean_and_se(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance) / math.sqrt(len(values))


def _value_se_cell(
    mean: float | None,
    se: float | None,
    *,
    value_precision: int,
    se_precision: int,
    bold: bool = False,
) -> str:
    if mean is None:
        return "--"
    return _value_se_text(
        mean,
        se,
        value_precision=value_precision,
        se_precision=se_precision,
        bold=bold,
        se_footnotesize=True,
    )


def _display_target(target: str) -> str:
    return target.replace("_", "\\_")


def _display_bnn_dataset(target: str) -> str:
    name = target.removeprefix("Bnn_").replace("_", " ")
    return name[:1].upper() + name[1:]


def _pooled_mean_se_from_summaries(rows: list[dict[str, Any]], metric: str) -> tuple[float | None, float | None]:
    parts: list[tuple[int, float, float]] = []
    for row in rows:
        mean = _metric_value(row, metric)
        if mean is None:
            continue
        count = int(_float(row.get(f"{metric}_count")) or 1)
        se = _metric_se(row, metric) or 0.0
        std = se * math.sqrt(count)
        parts.append((count, mean, std))
    total_count = sum(count for count, _, _ in parts)
    if total_count == 0:
        return None, None
    mean = sum(count * value for count, value, _ in parts) / total_count
    if total_count == 1:
        return mean, 0.0
    ss = 0.0
    for count, value, std in parts:
        ss += (count - 1) * std * std
        ss += count * (value - mean) ** 2
    variance = ss / (total_count - 1)
    return mean, math.sqrt(variance) / math.sqrt(total_count)


def render_langevin_table(summary_rows: list[dict[str, Any]], methods: list[str], cfg: Any) -> str:
    vp = int(cfg.tables.value_precision)
    sp = int(cfg.tables.se_precision)
    target_rows = [
        row
        for row in summary_rows
        if row["target"] == "Langevin_post" and str(row["method"]) in {*methods, "SGLD"}
    ]
    by_method = {str(row["method"]): row for row in target_rows}
    learned_rows = [row for row in target_rows if str(row["method"]).upper() != "SGLD"]
    best_by_metric = {metric: _best_methods(learned_rows, metric) for metric in ["kde_elm", "duration_sec"]}
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Langevin_post final evaluation metrics.}",
        "\\label{tab:langevin-final-eval}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Method & KDE ELM & wall-clock (s) & iterations \\\\",
        "\\midrule",
    ]
    sgld_row = by_method.get("SGLD")
    if sgld_row is not None:
        lines.append(
            "SGLD & "
            + _cell(sgld_row, "kde_elm", bold=False, value_precision=vp, se_precision=sp)
            + " & -- & -- \\\\"
        )
        lines.append("\\midrule")
    for method in methods:
        row = by_method.get(method)
        if row is None:
            cells = [method, "--", "--", "--"]
        else:
            cells = [
                method,
                _cell(row, "kde_elm", bold=method in best_by_metric["kde_elm"], value_precision=vp, se_precision=sp),
                _cell(row, "duration_sec", bold=method in best_by_metric["duration_sec"], value_precision=vp, se_precision=sp),
                _cell(row, "training_iterations", bold=False, value_precision=vp, se_precision=sp),
            ]
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def render_bnn_table(summary_rows: list[dict[str, Any]], targets: list[str], methods: list[str], cfg: Any) -> str:
    vp = int(cfg.tables.value_precision)
    sp = int(cfg.tables.se_precision)
    by_target_method = {(row["target"], row["method"]): row for row in summary_rows}
    colspec = "l" + "c" * (len(methods) * 2)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{BNN test RMSE and test NLL.}",
        "\\label{tab:bnn-rmse-nll}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        f"\\multirow{{2}}{{*}}{{Dataset}} & \\multicolumn{{{len(methods)}}}{{c}}{{Test RMSE}} & \\multicolumn{{{len(methods)}}}{{c}}{{Test NLL}} \\\\",
        f"\\cmidrule(lr){{2-{len(methods) + 1}}}\\cmidrule(lr){{{len(methods) + 2}-{len(methods) * 2 + 1}}}",
        " & " + " & ".join(methods + methods) + " \\\\",
        "\\midrule",
    ]
    for target in targets:
        target_rows = [row for row in summary_rows if row["target"] == target and str(row["method"]) in methods]
        best_by_metric = {metric: _best_methods(target_rows, metric) for metric in ["rmse", "nll"]}
        cells = [_display_bnn_dataset(target)]
        for metric in ["rmse", "nll"]:
            for method in methods:
                row = by_target_method.get((target, method))
                cells.append(
                    "--"
                    if row is None
                    else _cell_small_se(
                        row,
                        metric,
                        bold=method in best_by_metric[metric],
                        value_precision=vp,
                        se_precision=sp,
                    )
                )
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def render_toy_method_grid(summary_rows: list[dict[str, Any]], cfg: Any) -> str:
    vp = int(cfg.tables.value_precision)
    sp = int(cfg.tables.se_precision)
    by_target_method = {(row["target"], row["method"]): row for row in summary_rows}
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Selected toy target final evaluation metrics by method.}",
        "\\label{tab:toy-method-grid}",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Target & Metric & " + " & ".join(TOY_METHOD_GRID_METHODS) + " \\\\",
        "\\midrule",
    ]
    best_d_kl: dict[str, set[str]] = {}
    best_w2: dict[str, set[str]] = {}
    for target in TOY_METHOD_GRID_TARGETS:
        target_rows = [
            row
            for row in summary_rows
            if row["target"] == target and str(row["method"]) in TOY_METHOD_GRID_METHODS
        ]
        best_d_kl[target] = _best_methods(target_rows, "d_kl")
        best_w2[target] = _best_methods(target_rows, TOY_METHOD_GRID_W2_METRICS[target])

    for target in TOY_METHOD_GRID_TARGETS:
        target_label = f"\\multirow{{2}}{{*}}{{{_display_target(target)}}}"
        for row_idx, (metric_label, metric_key, best_by_target) in enumerate(
            [
                ("$D_{\\mathrm{KL}}$", "d_kl", best_d_kl),
                (_label(TOY_METHOD_GRID_W2_METRICS[target]), TOY_METHOD_GRID_W2_METRICS[target], best_w2),
            ]
        ):
            cells = [target_label if row_idx == 0 else "", metric_label]
            for method in TOY_METHOD_GRID_METHODS:
                row = by_target_method.get((target, method))
                cells.append(
                    "--"
                    if row is None
                    else _cell_small_se(
                        row,
                        metric_key,
                        bold=method in best_by_target[target],
                        value_precision=vp,
                        se_precision=sp,
                    )
                )
            lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    cells = ["\\multicolumn{2}{l}{Training time (s)}"]
    for method in TOY_METHOD_GRID_METHODS:
        method_rows = [
            row
            for row in summary_rows
            if row["target"] in TOY_METHOD_GRID_TARGETS and str(row["method"]) == method
        ]
        mean, se = _pooled_mean_se_from_summaries(method_rows, "training_time_sec")
        cells.append(_value_se_cell(mean, se, value_precision=vp, se_precision=sp))
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

    if bool(cfg.modules.get("toy_method_grid", False)):
        toy_method_grid_text = render_toy_method_grid(summary_rows, cfg)
        outputs["toy_method_grid"] = out_dir / "toy_method_grid.tex"
        outputs["toy_method_grid"].write_text(toy_method_grid_text, encoding="utf-8")

    if bool(cfg.modules.get("langevin_table", False)):
        langevin_methods = _ordered_methods(methods, last=["DSIVI"])
        langevin_text = render_langevin_table(summary_rows, langevin_methods, cfg)
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
        bnn_text = render_bnn_table(summary_rows, bnn_targets, bnn_methods, cfg)
        outputs["bnn"] = out_dir / "bnn_rmse_nll.tex"
        outputs["bnn"].write_text(bnn_text, encoding="utf-8")

    return outputs
