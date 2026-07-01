from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = [
    ("DSIVI", "DIVI"),
    ("UIVI", "UIVI"),
    ("KDVI", "KDVI"),
]

TAGS = {
    "kl": "metric/vi_model/kl_ite",
    "w2": "metric/vi_model/w2",
}


def latest_run(results_root: Path, runner: str) -> Path:
    target_root = results_root / runner / "flat_gaussian"
    if not target_root.exists():
        raise FileNotFoundError(f"Missing run directory: {target_root}")
    runs = [path for path in target_root.iterdir() if path.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No runs found under {target_root}")
    return sorted(runs, key=lambda path: path.name)[-1]


def read_metric_rows(metrics_path: Path, tag: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with metrics_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("tag") == tag:
                rows.append(row)
    return rows


def copy_run_artifacts(run_dir: Path, output_dir: Path, display: str) -> None:
    dest = output_dir / "runs" / display
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("metrics.csv", "full_config.yaml", "run.log", "wandb_run.json"):
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, dest / name)
    plots_dir = run_dir / "plots"
    if plots_dir.exists():
        dest_plots = dest / "plots"
        if dest_plots.exists():
            shutil.rmtree(dest_plots)
        shutil.copytree(plots_dir, dest_plots)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(curves: dict[str, list[tuple[int, float]]], ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    colors = {"DIVI": "#2ca02c", "UIVI": "#1f77b4", "KDVI": "#d62728"}
    for method, series in curves.items():
        if not series:
            continue
        steps = [step for step, _ in series]
        values = [value for _, value in series]
        ax.plot(
            steps,
            values,
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=method,
            color=colors.get(method),
        )
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/test_gaussian")
    parser.add_argument("--output-dir", default="test_gaussian")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    curve_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    curves = {
        "kl": {},
        "w2": {},
    }

    for runner, display in METHODS:
        run_dir = latest_run(results_root, runner)
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics.csv: {metrics_path}")
        copy_run_artifacts(run_dir, output_dir, display)

        final_row: dict[str, object] = {
            "method": display,
            "runner": runner,
            "run_dir": str(run_dir),
        }
        for metric_name, tag in TAGS.items():
            rows = read_metric_rows(metrics_path, tag)
            if not rows:
                raise RuntimeError(f"No rows for {tag} in {metrics_path}")
            series: list[tuple[int, float]] = []
            for row in rows:
                step = int(float(row["step"]))
                value = float(row["value"])
                series.append((step, value))
                curve_rows.append(
                    {
                        "method": display,
                        "runner": runner,
                        "metric": metric_name,
                        "tag": tag,
                        "step": step,
                        "value": value,
                        "run_dir": str(run_dir),
                    }
                )
            curves[metric_name][display] = series
            final_row[f"final_{metric_name}"] = series[-1][1]
            final_row[f"final_{metric_name}_step"] = series[-1][0]
            final_row[f"num_{metric_name}_points"] = len(series)
        final_rows.append(final_row)

    write_csv(
        output_dir / "metric_curves.csv",
        curve_rows,
        ["method", "runner", "metric", "tag", "step", "value", "run_dir"],
    )
    write_csv(
        output_dir / "final_metrics.csv",
        final_rows,
        [
            "method",
            "runner",
            "run_dir",
            "final_kl",
            "final_kl_step",
            "num_kl_points",
            "final_w2",
            "final_w2_step",
            "num_w2_points",
        ],
    )

    plot_metric(curves["kl"], "KL-ITE to exact samples", output_dir / "kl_curve.png")
    plot_metric(curves["w2"], "Sliced W2 to exact samples", output_dir / "w2_curve.png")


if __name__ == "__main__":
    main()

