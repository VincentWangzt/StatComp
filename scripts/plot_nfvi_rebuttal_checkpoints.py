"""Plot representative DIVI/NFVI checkpoints for the 8-Gaussians rebuttal.

The representative run for each method is selected by median sliced-Wasserstein
distance across the benchmark seeds.  This makes the one-checkpoint-per-setting
visualization deterministic and avoids choosing unusually good or bad runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from omegaconf import OmegaConf
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runner.runners import Runners  # noqa: E402


DEFAULT_METHODS = ("DIVI", "NFVI-4", "NFVI-8", "NFVI-16")
METHOD_LABELS = {
    "DIVI": "DIVI",
    "NFVI-4": "RealNVP-4",
    "NFVI-8": "RealNVP-8",
    "NFVI-16": "RealNVP-16",
}
TARGET_CONTOUR_COLOR = "#4C78A8"
SAMPLE_COLOR = "#D55E00"


@dataclass(frozen=True)
class RunRecord:
    method: str
    seed: int
    elbo: float
    w2: float
    run_dir: Path

    @property
    def checkpoint_path(self) -> Path:
        return self.run_dir / "final_vi_model.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render one representative checkpoint plot for each DIVI/NFVI "
            "setting in the 8-Gaussians benchmark."
        )
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("analysis/nfvi_rebuttal_20260726"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <report-dir>/plots.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=DEFAULT_METHODS,
        default=DEFAULT_METHODS,
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--grid-size", type=int, default=240)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected runs without loading checkpoints.",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[RunRecord]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        RunRecord(
            method=row["method"],
            seed=int(row["seed"]),
            elbo=float(row["elbo"]),
            w2=float(row["w2"]),
            run_dir=Path(row["run_dir"]),
        )
        for row in rows
    ]


def select_median_w2_records(
    records: list[RunRecord],
    methods: tuple[str, ...] | list[str],
) -> list[RunRecord]:
    """Select the observed median-W2 seed for each requested method."""
    selected: list[RunRecord] = []
    for method in methods:
        candidates = sorted(
            (record for record in records if record.method == method),
            key=lambda record: (record.w2, record.seed),
        )
        if not candidates:
            raise ValueError(f"No benchmark records found for {method}")
        selected.append(candidates[(len(candidates) - 1) // 2])
    return selected


def build_runner(
    record: RunRecord,
    device: torch.device,
    runtime_dir: Path,
) -> Any:
    is_flow = record.method.startswith("NFVI-")
    config_path = PROJECT_ROOT / "configs" / (
        "nfvi_8_gaussians.yaml" if is_flow else "dsivi_8_gaussians.yaml"
    )
    config = OmegaConf.load(config_path)
    config.config_path = str(config_path)
    config.seed = record.seed
    config.device = str(device)
    config.use_cuda = device.type == "cuda"
    config.cuda_visible_devices = "0"
    config.train.checkpoint.enabled = False
    config.output = {
        "results_dir": str(runtime_dir / "results"),
        "tb_dir": str(runtime_dir / "tb_logs"),
    }
    if is_flow:
        config.setdefault("vi_model", {})
        config.vi_model.num_flow_layers = int(record.method.rsplit("-", 1)[1])

    runner = Runners[str(config.runner_type)](config=config)
    state = torch.load(
        record.checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    runner.vi_model.load_state_dict(state)
    runner.vi_model.eval()
    return runner


def sample_checkpoint(
    record: RunRecord,
    device: torch.device,
    runtime_dir: Path,
    num_samples: int,
) -> tuple[np.ndarray, Any]:
    runner = build_runner(record, device, runtime_dir)
    torch.manual_seed(2_000_000 + record.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(2_000_000 + record.seed)
    with torch.no_grad():
        _, samples = runner.vi_model.sampling(num=num_samples)
    points = samples[:, :2].detach().cpu().numpy()
    runner.writer.close()
    return points, runner.target_model


def target_surface(
    target_model: Any,
    bbox: tuple[float, float, float, float],
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(bbox[0], bbox[1], grid_size)
    y = np.linspace(bbox[2], bbox[3], grid_size)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    positions = torch.as_tensor(
        np.column_stack((xx.ravel(), yy.ravel())),
        dtype=torch.float32,
        device=target_model.device,
    )
    with torch.no_grad():
        logp = target_model.logp(positions).reshape(grid_size, grid_size)
    logp = logp.detach().cpu().numpy()
    density = np.exp(logp - np.nanmax(logp))
    return xx, yy, density


def draw_panel(
    ax: plt.Axes,
    record: RunRecord,
    points: np.ndarray,
    surface: tuple[np.ndarray, np.ndarray, np.ndarray],
    bbox: tuple[float, float, float, float],
) -> None:
    xx, yy, density = surface
    contour_levels = np.linspace(0.08, 1.0, 9)
    ax.contour(
        xx,
        yy,
        density,
        levels=contour_levels,
        colors=TARGET_CONTOUR_COLOR,
        linewidths=0.8,
        alpha=0.9,
        zorder=1,
    )
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=4,
        c=SAMPLE_COLOR,
        marker=".",
        linewidths=0,
        alpha=0.34,
        rasterized=True,
        zorder=2,
    )
    ax.set(
        xlim=(bbox[0], bbox[1]),
        ylim=(bbox[2], bbox[3]),
        xticks=(-4, 0, 4),
        yticks=(-4, 0, 4),
        xlabel=r"$z_1$",
        ylabel=r"$z_2$",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        (
            f"{METHOD_LABELS[record.method]}\n"
            f"seed {record.seed} · ELBO {record.elbo:.3f} · SW₂ {record.w2:.3f}"
        ),
        fontsize=10,
    )
    ax.tick_params(labelsize=8, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)


def save_individual_plots(
    output_dir: Path,
    records: list[RunRecord],
    points_by_method: dict[str, np.ndarray],
    surface: tuple[np.ndarray, np.ndarray, np.ndarray],
    bbox: tuple[float, float, float, float],
    dpi: int,
) -> list[Path]:
    paths: list[Path] = []
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=TARGET_CONTOUR_COLOR,
            linewidth=1.2,
            label="Target density",
        ),
        Line2D(
            [0],
            [0],
            color=SAMPLE_COLOR,
            marker=".",
            linestyle="none",
            markersize=5,
            label="Variational samples",
        ),
    ]
    for record in records:
        fig, ax = plt.subplots(figsize=(4.25, 4.1))
        draw_panel(ax, record, points_by_method[record.method], surface, bbox)
        ax.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.26),
            ncol=2,
            frameon=False,
            fontsize=8,
        )
        fig.tight_layout()
        stem = record.method.lower().replace("-", "_")
        png_path = output_dir / f"{stem}.png"
        pdf_path = output_dir / f"{stem}.pdf"
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        paths.extend((png_path, pdf_path))
    return paths


def save_combined_plot(
    output_dir: Path,
    records: list[RunRecord],
    points_by_method: dict[str, np.ndarray],
    surface: tuple[np.ndarray, np.ndarray, np.ndarray],
    bbox: tuple[float, float, float, float],
    dpi: int,
) -> list[Path]:
    fig, axes = plt.subplots(
        1,
        len(records),
        figsize=(3.0 * len(records), 3.65),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for index, record in enumerate(records):
        ax = axes[0, index]
        draw_panel(ax, record, points_by_method[record.method], surface, bbox)
        if index:
            ax.set_ylabel("")
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=TARGET_CONTOUR_COLOR,
            linewidth=1.2,
            label="Target density",
        ),
        Line2D(
            [0],
            [0],
            color=SAMPLE_COLOR,
            marker=".",
            linestyle="none",
            markersize=5,
            label="Variational samples",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=0.8)
    png_path = output_dir / "checkpoint_comparison.png"
    pdf_path = output_dir / "checkpoint_comparison.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def write_selection_metadata(
    output_dir: Path,
    records: list[RunRecord],
    num_samples: int,
) -> Path:
    path = output_dir / "selection.json"
    payload = {
        "selection_rule": "observed median sliced-W2 run within each method",
        "num_plot_samples": num_samples,
        "runs": [
            {
                "method": record.method,
                "display_name": METHOD_LABELS[record.method],
                "seed": record.seed,
                "elbo": record.elbo,
                "w2": record.w2,
                "checkpoint": str(record.checkpoint_path),
            }
            for record in records
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    report_dir = (PROJECT_ROOT / args.report_dir).resolve()
    output_dir = (
        (PROJECT_ROOT / args.output_dir).resolve()
        if args.output_dir is not None
        else report_dir / "plots"
    )
    records = load_records(report_dir / "run_metrics.csv")
    selected = select_median_w2_records(records, args.methods)

    for record in selected:
        print(
            f"{record.method}: seed={record.seed}, ELBO={record.elbo:.6f}, "
            f"SW2={record.w2:.6f}, checkpoint={record.checkpoint_path}"
        )
    if args.dry_run:
        return

    missing = [
        str(record.checkpoint_path)
        for record in selected
        if not record.checkpoint_path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Selected checkpoint files are missing:\n" + "\n".join(missing)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = (
        PROJECT_ROOT
        / "results"
        / "nfvi_rebuttal_8_gaussians"
        / "checkpoint_plot_loading"
    )
    device = torch.device(args.device)
    points_by_method: dict[str, np.ndarray] = {}
    target_model = None
    for record in selected:
        points, target_model = sample_checkpoint(
            record,
            device,
            runtime_dir,
            args.num_samples,
        )
        points_by_method[record.method] = points
    assert target_model is not None

    bbox = (-6.0, 6.0, -6.0, 6.0)
    surface = target_surface(target_model, bbox, args.grid_size)
    generated = save_individual_plots(
        output_dir,
        selected,
        points_by_method,
        surface,
        bbox,
        args.dpi,
    )
    generated.extend(
        save_combined_plot(
            output_dir,
            selected,
            points_by_method,
            surface,
            bbox,
            args.dpi,
        )
    )
    generated.append(
        write_selection_metadata(output_dir, selected, args.num_samples)
    )
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
