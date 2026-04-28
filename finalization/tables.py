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
    "rmse",
    "nll",
}
HIGHER_IS_BETTER = {"elbo", "kde_elm"}


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


def _cell(row: dict[str, Any], metric: str, *, bold: bool, value_precision: int, se_precision: int) -> str:
    if metric == "duration_sec":
        mean = _float(row.get("duration_sec_mean"))
        se = _float(row.get("duration_sec_se"))
    elif metric == "training_iterations":
        mean = _float(row.get("training_iterations_mean"))
        se = None
    else:
        mean = _float(row.get(f"{metric}_mean"))
        se = _float(row.get(f"{metric}_se"))
    if mean is None:
        return "--"
    text = _fmt(mean, value_precision)
    if se is not None:
        text = f"{text} $\\pm$ {_fmt(se, se_precision)}"
    return f"\\textbf{{{text}}}" if bold else text


def _best_methods(rows: list[dict[str, Any]], metric: str) -> set[str]:
    values: list[tuple[str, float]] = []
    for row in rows:
        key = "duration_sec_mean" if metric == "duration_sec" else f"{metric}_mean"
        value = _float(row.get(key))
        if value is not None:
            values.append((str(row["method"]), value))
    if not values:
        return set()
    if metric in HIGHER_IS_BETTER:
        best = max(value for _, value in values)
    else:
        best = min(value for _, value in values)
    return {method for method, value in values if math.isclose(value, best) or value == best}


def _latex_table(
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
    metric_header = " & ".join(
        f"\\multicolumn{{{len(methods)}}}{{c}}{{{metric.replace('_', ' ').upper()}}}"
        for metric in metrics
    )
    lines.append(f"Target & {metric_header} \\\\")
    method_header = " & ".join(method for _metric in metrics for method in methods)
    lines.append(f" & {method_header} \\\\")
    lines.append("\\midrule")

    by_target_method = {(row["target"], row["method"]): row for row in rows}
    by_target = {target: [row for row in rows if row["target"] == target] for target in targets}
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
        toy_targets = [str(target) for target in cfg.selection.scatter_targets]
        toy_text = _latex_table(
            summary_rows,
            targets=toy_targets,
            methods=methods,
            metrics=["elbo", "w2", "duration_sec"],
            caption="Toy target final evaluation metrics.",
            label="tab:toy-final-eval",
            value_precision=vp,
            se_precision=sp,
        )
        outputs["toy"] = out_dir / "toy_metrics.tex"
        outputs["toy"].write_text(toy_text, encoding="utf-8")

    if bool(cfg.modules.get("langevin_table", False)):
        langevin_text = _latex_table(
            summary_rows,
            targets=["Langevin_post"],
            methods=methods,
            metrics=["kde_elm", "duration_sec", "training_iterations"],
            caption="Langevin posterior final evaluation metrics.",
            label="tab:langevin-final-eval",
            value_precision=vp,
            se_precision=sp,
        )
        outputs["langevin"] = out_dir / "langevin_metrics.tex"
        outputs["langevin"].write_text(langevin_text, encoding="utf-8")

    if bool(cfg.modules.get("student_edge_table", False)):
        edge_text = _latex_table(
            summary_rows,
            targets=["student_uc"],
            methods=methods,
            metrics=["w2_edge_5", "w2_edge_8", "w2_edge_10"],
            caption="Student-t constrained sliced Wasserstein distances by edge width.",
            label="tab:student-edge-w2",
            value_precision=vp,
            se_precision=sp,
        )
        outputs["student_edge"] = out_dir / "student_edge_w2.tex"
        outputs["student_edge"].write_text(edge_text, encoding="utf-8")

    if bool(cfg.modules.get("bnn_table", False)):
        bnn_targets = [str(target) for target in cfg.selection.bnn_targets]
        bnn_text = _latex_table(
            summary_rows,
            targets=bnn_targets,
            methods=methods,
            metrics=["rmse", "nll"],
            caption="BNN test RMSE and test NLL.",
            label="tab:bnn-rmse-nll",
            value_precision=vp,
            se_precision=sp,
        )
        outputs["bnn"] = out_dir / "bnn_rmse_nll.tex"
        outputs["bnn"].write_text(bnn_text, encoding="utf-8")

    return outputs
