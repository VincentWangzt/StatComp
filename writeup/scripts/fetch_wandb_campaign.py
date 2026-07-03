from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv


METRICS: dict[str, tuple[str, str]] = {
    "metric/vi_model/kl_ite": ("kl_ite", "min"),
    "metric/vi_model/w2": ("w2", "min"),
    "metric/vi_model/elbo": ("elbo", "max"),
    "metric/vi_model/kde_expected_log_marginal": (
        "kde_expected_log_marginal",
        "max",
    ),
}
TARGET_ORDER = [
    "banana",
    "x_shaped",
    "multimodal",
    "8_gaussians",
    "8_gaussians_small",
    "student_uc",
    "Langevin_post",
]
METHOD_ORDER = ["KDVI", "DSIVI"]
PLOT_STEP_SCALE = {"DSIVI": 10}
VISUAL_SEED = 7
PLOTLY_FONT_FAMILY = "Times New Roman, Times, serif"
PLOTLY_SOURCE_WIDTH = 1000
PLOTLY_SOURCE_HEIGHT = 410
PLOTLY_PNG_SCALE = 2


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch W&B campaign data and media for the Quarto writeup."
    )
    parser.add_argument("campaign", help="Value of config.tracking.campaign")
    parser.add_argument(
        "--project",
        default=None,
        help="W&B project. Defaults to WANDB_PROJECT or KDVI.",
    )
    parser.add_argument(
        "--entity",
        default=None,
        help="W&B entity. Defaults to WANDB_ENTITY when set.",
    )
    parser.add_argument(
        "--visual-seed",
        type=int,
        default=VISUAL_SEED,
        help="Seed used for target posterior-image sliders.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1200,
        help="Approximate W&B history sample count per run.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing cached data/assets for the campaign first.",
    )
    return parser.parse_args()


def wandb_path(project: str, entity: str | None) -> str:
    return f"{entity}/{project}" if entity else project


def method_target_from_group(group: str) -> tuple[str, str]:
    method, _, target = group.partition("-")
    return method, target


def seed_from_run(run: Any) -> int | None:
    config = getattr(run, "config", {}) or {}
    seed = config.get("seed")
    if seed is not None:
        try:
            return int(seed)
        except (TypeError, ValueError):
            pass
    match = re.search(r"seed(\d+)", run.name or "")
    return int(match.group(1)) if match else None


def step_from_row(row: pd.Series) -> int | None:
    for key in ("epoch", "_step", "step"):
        value = row.get(key)
        if pd.notna(value):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                continue
    return None


def finite(value: Any) -> bool:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(converted)


def clean_dir(path: Path, force: bool) -> None:
    if force and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def collect_run_rows(runs: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        method, target = method_target_from_group(run.group or "")
        rows.append(
            {
                "run_id": run.id,
                "name": run.name,
                "group": run.group,
                "method": method,
                "target": target,
                "seed": seed_from_run(run),
                "state": run.state,
                "created_at": str(getattr(run, "created_at", "")),
                "url": run.url,
            }
        )
    return rows


def dedupe_finished_runs(runs: list[Any]) -> list[Any]:
    best: dict[tuple[str, int | None], Any] = {}
    for run in runs:
        if run.state != "finished":
            continue
        key = (run.group or "", seed_from_run(run))
        prev = best.get(key)
        if prev is None or str(run.created_at) > str(prev.created_at):
            best[key] = run
    return sorted(
        best.values(),
        key=lambda run: (
            TARGET_ORDER.index(method_target_from_group(run.group or "")[1])
            if method_target_from_group(run.group or "")[1] in TARGET_ORDER
            else 999,
            METHOD_ORDER.index(method_target_from_group(run.group or "")[0])
            if method_target_from_group(run.group or "")[0] in METHOD_ORDER
            else 999,
            seed_from_run(run) if seed_from_run(run) is not None else 9999,
        ),
    )


def collect_metric_points(run: Any, samples: int) -> list[dict[str, Any]]:
    method, target = method_target_from_group(run.group or "")
    step_scale = PLOT_STEP_SCALE.get(method, 1)
    seed = seed_from_run(run)
    rows: list[dict[str, Any]] = []
    for wandb_key, (metric, _) in METRICS.items():
        try:
            frame = run.history(
                keys=["epoch", wandb_key],
                samples=samples,
                pandas=True,
            )
        except Exception:
            continue
        if frame.empty or wandb_key not in frame.columns:
            continue
        for _, item in frame.iterrows():
            raw_step = step_from_row(item)
            if raw_step is None:
                continue
            value = item.get(wandb_key)
            if finite(value):
                rows.append(
                    {
                        "run_id": run.id,
                        "run_name": run.name,
                        "group": run.group,
                        "method": method,
                        "target": target,
                        "seed": seed,
                        "step": raw_step,
                        "plot_step": raw_step * step_scale,
                        "metric": metric,
                        "value": float(value),
                    }
                )
    return rows


def collect_final_metric_points(runs: list[Any]) -> pd.DataFrame:
    rows = []
    for run in runs:
        method, target = method_target_from_group(run.group or "")
        seed = seed_from_run(run)
        summary = run.summary or {}
        for wandb_key, (metric, _) in METRICS.items():
            value = summary.get(wandb_key)
            if finite(value):
                rows.append(
                    {
                        "run_id": run.id,
                        "run_name": run.name,
                        "group": run.group,
                        "method": method,
                        "target": target,
                        "seed": seed,
                        "metric": metric,
                        "value": float(value),
                    }
                )
    return pd.DataFrame(rows)


def summarize_final_metrics(final_points: pd.DataFrame, runs: list[Any]) -> pd.DataFrame:
    run_counts = defaultdict(int)
    for run in runs:
        run_counts[run.group or ""] += 1

    groups = sorted(
        run_counts,
        key=lambda group: (
            TARGET_ORDER.index(method_target_from_group(group)[1])
            if method_target_from_group(group)[1] in TARGET_ORDER
            else 999,
            METHOD_ORDER.index(method_target_from_group(group)[0])
            if method_target_from_group(group)[0] in METHOD_ORDER
            else 999,
        ),
    )
    rows = []
    for group in groups:
        method, target = method_target_from_group(group)
        row: dict[str, Any] = {
            "group": group,
            "method": method,
            "target": target,
            "run_count": run_counts[group],
        }
        if not final_points.empty:
            for metric in {slug for slug, _ in METRICS.values()}:
                values = final_points[
                    (final_points["group"] == group)
                    & (final_points["metric"] == metric)
                ]["value"]
                row[f"{metric}_final_mean"] = (
                    float(values.mean()) if len(values) else math.nan
                )
                row[f"{metric}_final_std"] = (
                    float(values.std(ddof=1)) if len(values) > 1 else math.nan
                )
                row[f"{metric}_final_count"] = int(len(values))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_metric_history(points: pd.DataFrame) -> pd.DataFrame:
    if points.empty:
        return pd.DataFrame(
            columns=[
                "method",
                "target",
                "metric",
                "plot_step",
                "mean",
                "std",
                "count",
            ]
        )
    points = points.copy()
    if "plot_step" not in points.columns:
        points["plot_step"] = points.apply(
            lambda row: row["step"] * PLOT_STEP_SCALE.get(row["method"], 1),
            axis=1,
        )
    grouped = points.groupby(["method", "target", "metric", "plot_step"])["value"]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    return out.sort_values(["target", "metric", "method", "plot_step"])


def write_plotly_plots(aggregates: pd.DataFrame, plot_dir: Path) -> None:
    import plotly.graph_objects as go

    plot_dir.mkdir(parents=True, exist_ok=True)
    png_export_warning_shown = False
    method_styles = {
        "KDVI": {
            "line": "#1f77b4",
            "fill": "rgba(31, 119, 180, 0.16)",
        },
        "DSIVI": {
            "line": "#d62728",
            "fill": "rgba(214, 39, 40, 0.14)",
        },
    }
    metric_labels = {
        "kl_ite": "KL-ITE",
        "w2": "Sliced W2",
        "elbo": "ELBO",
        "kde_expected_log_marginal": "KDE expected log marginal",
    }
    for target in TARGET_ORDER:
        for metric, label in metric_labels.items():
            subset = aggregates[
                (aggregates["target"] == target)
                & (aggregates["metric"] == metric)
            ].copy()
            if subset.empty:
                continue
            fig = go.Figure()
            for method in METHOD_ORDER:
                method_subset = subset[subset["method"] == method].sort_values(
                    "plot_step"
                )
                if method_subset.empty:
                    continue
                style = method_styles.get(
                    method,
                    {"line": "#444", "fill": "rgba(68, 68, 68, 0.12)"},
                )
                x = method_subset["plot_step"]
                mean = method_subset["mean"]
                std = method_subset["std"].fillna(0.0)
                upper = mean + std
                lower = mean - std
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=upper,
                        mode="lines",
                        line={"width": 0},
                        hoverinfo="skip",
                        showlegend=False,
                        legendgroup=method,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=lower,
                        mode="lines",
                        line={"width": 0},
                        fill="tonexty",
                        fillcolor=style["fill"],
                        hoverinfo="skip",
                        showlegend=False,
                        legendgroup=method,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=mean,
                        mode="lines+markers",
                        name=method,
                        legendgroup=method,
                        line={"color": style["line"], "width": 2.4},
                        marker={"size": 5},
                        customdata=method_subset[["count", "std"]],
                        hovertemplate=(
                            "plot epoch=%{x}<br>"
                            "mean=%{y:.5g}<br>"
                            "std=%{customdata[1]:.5g}<br>"
                            "band=mean +/- 1 std<br>"
                            "run count=%{customdata[0]}"
                            f"<extra>{method}</extra>"
                        ),
                    )
                )
            fig.update_layout(
                title={
                    "text": (
                        f"{target}: {label}"
                        "<br><sup>shaded band: mean +/- 1 std across runs</sup>"
                    ),
                    "font": {"family": PLOTLY_FONT_FAMILY, "size": 19},
                },
                template="plotly_white",
                width=PLOTLY_SOURCE_WIDTH,
                height=PLOTLY_SOURCE_HEIGHT,
                margin={"l": 104, "r": 130, "t": 64, "b": 86},
                font={"family": PLOTLY_FONT_FAMILY, "size": 16},
                xaxis_title="plot epoch (DSIVI x10)",
                yaxis_title=label,
                legend={
                    "orientation": "v",
                    "x": 1.02,
                    "xanchor": "left",
                    "y": 0.98,
                    "yanchor": "top",
                    "traceorder": "normal",
                    "itemsizing": "constant",
                    "font": {"family": PLOTLY_FONT_FAMILY, "size": 16},
                },
                autosize=False,
            )
            fig.update_xaxes(
                automargin=False,
                title_font={"size": 17},
                title_standoff=18,
                tickfont={"size": 15},
            )
            fig.update_yaxes(
                automargin=False,
                title_font={"size": 17},
                title_standoff=18,
                tickfont={"size": 15},
            )
            output = plot_dir / f"{target}_{metric}.html"
            png_output = output.with_suffix(".png")
            try:
                fig.write_image(
                    png_output,
                    width=PLOTLY_SOURCE_WIDTH,
                    height=PLOTLY_SOURCE_HEIGHT,
                    scale=PLOTLY_PNG_SCALE,
                )
            except ValueError as exc:
                if not png_export_warning_shown:
                    print(
                        "WARNING: skipping Plotly PNG export; install kaleido "
                        f"to enable fig.write_image(). First error: {exc}",
                        file=sys.stderr,
                    )
                    png_export_warning_shown = True
            fig.write_html(
                output,
                full_html=True,
                include_plotlyjs="cdn",
                config={"displaylogo": False, "responsive": False},
            )


def file_step(name: str) -> int:
    match = re.search(r"posterior_(\d+)_", name)
    return int(match.group(1)) if match else 0


def int_value(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def media_paths(value: Any) -> list[str]:
    if isinstance(value, dict):
        out = []
        for key in ("path", "name"):
            item = value.get(key)
            if isinstance(item, str):
                out.append(item)
        return out
    if isinstance(value, list):
        return [
            path
            for item in value
            for path in media_paths(item)
        ]
    if isinstance(value, str):
        return [value]
    return []


def collect_media_epochs(run: Any, media_key: str) -> dict[str, int]:
    epochs: dict[str, int] = {}

    def add_row(row: dict[str, Any]) -> None:
        epoch = int_value(row.get("epoch"))
        if epoch is None:
            return
        for path in media_paths(row.get(media_key)):
            normalized = path.replace("\\", "/")
            epochs[normalized] = epoch
            epochs[normalized.rsplit("/", 1)[-1]] = epoch

    try:
        for row in run.scan_history(keys=["epoch", media_key]):
            add_row(row)
    except Exception:
        try:
            frame = run.history(keys=["epoch", media_key], pandas=True)
        except Exception:
            return epochs
        for _, item in frame.iterrows():
            add_row(item.to_dict())
    return epochs


def download_file(run: Any, wandb_file: Any, output_path: Path) -> None:
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        downloaded = Path(wandb_file.download(root=tmpdir, replace=True).name)
        shutil.copyfile(downloaded, output_path)


def collect_visuals(
    runs: list[Any],
    assets_dir: Path,
    campaign: str,
    visual_seed: int,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    by_target = {}
    for run in runs:
        method, target = method_target_from_group(run.group or "")
        if method == "KDVI" and seed_from_run(run) == visual_seed:
            by_target[target] = run

    for target, run in sorted(by_target.items()):
        target_dir = assets_dir / target
        files = list(run.files())
        groundtruth_files = [
            item for item in files
            if item.name.startswith("media/images/plots/groundtruth_samples")
            and item.name.endswith(".png")
        ]
        posterior_files = sorted(
            [
                item for item in files
                if item.name.startswith("media/images/plots/posterior_")
                and item.name.endswith(".png")
            ],
            key=lambda item: file_step(item.name),
        )
        if not groundtruth_files or not posterior_files:
            continue

        groundtruth_out = target_dir / "groundtruth.png"
        download_file(run, groundtruth_files[0], groundtruth_out)
        media_epochs = collect_media_epochs(run, "plots/posterior")
        frames = []
        for item in posterior_files:
            wandb_step = file_step(item.name)
            epoch = (
                media_epochs.get(item.name)
                or media_epochs.get(item.name.rsplit("/", 1)[-1])
                or wandb_step
            )
            out = target_dir / f"posterior_epoch_{epoch}.png"
            download_file(run, item, out)
            frame = {
                "epoch": epoch,
                "path": (
                    Path("assets")
                    / "wandb"
                    / campaign
                    / target
                    / out.name
                ).as_posix(),
            }
            if wandb_step and wandb_step != epoch:
                frame["wandb_step"] = wandb_step
            frames.append(frame)
        frames.sort(key=lambda frame: frame["epoch"])
        manifest[target] = {
            "run_name": run.name,
            "seed": visual_seed,
            "groundtruth": (
                Path("assets")
                / "wandb"
                / campaign
                / target
                / groundtruth_out.name
            ).as_posix(),
            "posterior": frames,
        }
    return manifest


def main() -> int:
    args = parse_args()
    root = repo_root()
    load_dotenv(root / ".env", override=False)
    project = args.project or os.getenv("WANDB_PROJECT") or "KDVI"
    entity = args.entity or os.getenv("WANDB_ENTITY") or None

    import wandb

    data_dir = root / "writeup" / "data" / args.campaign
    assets_dir = root / "writeup" / "assets" / "wandb" / args.campaign
    plot_dir = root / "writeup" / "assets" / "plotly" / args.campaign
    clean_dir(data_dir, args.force)
    clean_dir(assets_dir, args.force)
    clean_dir(plot_dir, args.force)

    api = wandb.Api(timeout=60)
    runs = list(
        api.runs(
            wandb_path(project, entity),
            filters={"config.tracking.campaign": args.campaign},
            per_page=200,
        )
    )
    if not runs:
        raise RuntimeError(
            f"No W&B runs found for campaign {args.campaign!r} "
            f"in project {wandb_path(project, entity)!r}."
        )

    finished_runs = dedupe_finished_runs(runs)
    pd.DataFrame(collect_run_rows(runs)).to_csv(
        data_dir / "runs.csv", index=False
    )

    points: list[dict[str, Any]] = []
    for idx, run in enumerate(finished_runs, start=1):
        print(
            f"[{idx}/{len(finished_runs)}] history: {run.group} / {run.name}",
            flush=True,
        )
        points.extend(collect_metric_points(run, samples=args.samples))
    points_frame = pd.DataFrame(points)
    points_frame.to_csv(data_dir / "metric_history.csv", index=False)

    final_points = collect_final_metric_points(finished_runs)
    final_points.to_csv(data_dir / "final_metric_points.csv", index=False)
    final_metrics = summarize_final_metrics(final_points, finished_runs)
    final_metrics.to_csv(data_dir / "final_metrics.csv", index=False)

    aggregates = aggregate_metric_history(points_frame)
    aggregates.to_csv(data_dir / "metric_aggregates.csv", index=False)
    write_plotly_plots(aggregates, plot_dir)

    visual_manifest = collect_visuals(
        finished_runs, assets_dir, args.campaign, args.visual_seed
    )
    (data_dir / "visual_manifest.json").write_text(
        json.dumps(visual_manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    print(
        f"Fetched {len(runs)} runs, cached {len(finished_runs)} finished "
        f"runs, wrote {len(points_frame)} metric points."
    )
    print(f"Data: {data_dir}")
    print(f"W&B media: {assets_dir}")
    print(f"Plotly: {plot_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
