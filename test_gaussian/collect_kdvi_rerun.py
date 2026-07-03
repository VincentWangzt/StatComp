from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOSS_RE = re.compile(r"Epoch\s+(?P<step>\d+):\s+Avg Loss:\s+(?P<loss>[-+0-9.eE]+)")
METRIC_TAGS = {
    "kl": "metric/vi_model/kl_ite",
    "w2": "metric/vi_model/w2",
}


def latest_kdvi_run(results_root: Path) -> Path:
    root = results_root / "KDVI" / "flat_gaussian"
    if not root.exists():
        raise FileNotFoundError(f"Missing KDVI flat Gaussian result root: {root}")
    runs = [path for path in root.iterdir() if path.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No KDVI flat Gaussian runs under {root}")
    return sorted(runs, key=lambda path: path.name)[-1]


def parse_loss(run_log: Path) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    for line in run_log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LOSS_RE.search(line)
        if match:
            rows.append((int(match.group("step")), float(match.group("loss"))))
    if not rows:
        raise RuntimeError(f"No Avg Loss lines found in {run_log}")
    return rows


def read_metric(metrics_path: Path, tag: str) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    with metrics_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("tag") == tag:
                rows.append((int(float(row["step"])), float(row["value"])))
    return rows


def write_series(path: Path, header: tuple[str, str], rows: list[tuple[int, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def plot_series(path: Path, rows: list[tuple[int, float]], ylabel: str, color: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(
        [step for step, _ in rows],
        [value for _, value in rows],
        color=color,
        linewidth=1.8,
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def latest_contour(plots_dir: Path) -> Path:
    contours = []
    for path in plots_dir.glob("contour_epoch_*.png"):
        try:
            step = int(path.stem.rsplit("_", 1)[1])
        except ValueError:
            continue
        contours.append((step, path))
    if not contours:
        raise FileNotFoundError(f"No contour_epoch_*.png found under {plots_dir}")
    return sorted(contours)[-1][1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/test_gaussian")
    parser.add_argument("--output-dir", default="test_gaussian/kdvi_rerun")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dir = latest_kdvi_run(results_root)
    run_out = output_dir / "run"
    run_out.mkdir(parents=True, exist_ok=True)

    for name in ("metrics.csv", "full_config.yaml", "run.log", "wandb_run.json"):
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, run_out / name)

    plots_src = run_dir / "plots"
    plots_out = run_out / "plots"
    if plots_src.exists():
        if plots_out.exists():
            shutil.rmtree(plots_out)
        shutil.copytree(plots_src, plots_out)

    loss_rows = parse_loss(run_dir / "run.log")
    write_series(output_dir / "loss_curve.csv", ("step", "avg_loss"), loss_rows)
    plot_series(output_dir / "loss_curve.png", loss_rows, "KDVI Avg Loss", "#d62728")

    metrics_path = run_dir / "metrics.csv"
    summary_rows = []
    for name, tag in METRIC_TAGS.items():
        rows = read_metric(metrics_path, tag)
        if rows:
            write_series(output_dir / f"{name}_curve.csv", ("step", name), rows)
            plot_series(output_dir / f"{name}_curve.png", rows, name.upper(), "#1f77b4")
            summary_rows.append({"metric": name, "final_step": rows[-1][0], "final_value": rows[-1][1], "num_points": len(rows)})

    sample_src = latest_contour(run_dir / "plots")
    sample_name = f"sample_{sample_src.name}"
    shutil.copy2(sample_src, output_dir / sample_name)
    groundtruth_src = run_dir / "plots" / "groundtruth_contour.png"
    if groundtruth_src.exists():
        shutil.copy2(groundtruth_src, output_dir / "groundtruth_contour.png")

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["metric", "final_step", "final_value", "num_points"])
        writer.writeheader()
        writer.writerows(summary_rows)

    (output_dir / "run_dir.txt").write_text(str(run_dir) + "\n", encoding="utf-8")
    print(f"Collected KDVI rerun from {run_dir} into {output_dir}")
    print(f"Final sample contour: {output_dir / sample_name}")


if __name__ == "__main__":
    main()