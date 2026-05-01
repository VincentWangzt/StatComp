from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import csv
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ROOT = REPO_ROOT / "analysis" / "ksivi_parity_20260414"
DEFAULT_RESULTS_ROOT = DEFAULT_RUN_ROOT / "scratch_results"
DEFAULT_TB_ROOT = DEFAULT_RUN_ROOT / "scratch_tb"
ORACLE_CSV = DEFAULT_RUN_ROOT / "oracle" / "oracle_final_metrics.csv"

BNN_TARGETS = (
    "Bnn_boston",
    "Bnn_concrete",
    "Bnn_power",
    "Bnn_protein",
    "Bnn_winered",
    "Bnn_yacht",
)


def _load_oracle() -> dict[str, dict[str, float]]:
    if not ORACLE_CSV.exists():
        return {}
    with ORACLE_CSV.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["target"]: {"nll": float(row["nll"]), "rmse": float(row["rmse"])}
        for row in rows
    }


def _extract_metrics(tb_run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    acc = event_accumulator.EventAccumulator(str(tb_run_dir))
    acc.Reload()
    out: dict[str, list[tuple[int, float]]] = {}
    for tag in ("metric/vi_model/nll", "metric/vi_model/rmse", "train/vi_model/loss"):
        if tag in acc.Tags().get("scalars", []):
            out[tag] = [(int(ev.step), float(ev.value)) for ev in acc.Scalars(tag)]
    return out


def _latest_run_dir(root: Path, target: str) -> Path | None:
    target_dir = root / "KSIVI" / target
    if not target_dir.exists():
        return None
    candidates = [path for path in target_dir.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize KSIVI BNN parity runs.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--tb-root", type=Path, default=DEFAULT_TB_ROOT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_RUN_ROOT / "run_summary",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    oracle = _load_oracle()
    curve_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []

    for target in BNN_TARGETS:
        tb_run_dir = _latest_run_dir(args.tb_root.resolve(), target)
        result_run_dir = _latest_run_dir(args.results_root.resolve(), target)
        if tb_run_dir is None or result_run_dir is None:
            continue
        metrics = _extract_metrics(tb_run_dir)
        nll_points = metrics.get("metric/vi_model/nll", [])
        rmse_points = metrics.get("metric/vi_model/rmse", [])
        for (step_nll, nll), (step_rmse, rmse) in zip(nll_points, rmse_points, strict=False):
            if step_nll != step_rmse:
                continue
            curve_rows.append(
                {
                    "target": target,
                    "run_dir": str(result_run_dir),
                    "epoch": step_nll,
                    "optimizer_updates": step_nll,
                    "nll": nll,
                    "rmse": rmse,
                }
            )
        if not nll_points or not rmse_points:
            continue
        final_nll = nll_points[-1][1]
        final_rmse = rmse_points[-1][1]
        oracle_row = oracle.get(target, {})
        oracle_nll = oracle_row.get("nll", float("nan"))
        oracle_rmse = oracle_row.get("rmse", float("nan"))
        final_rows.append(
            {
                "target": target,
                "run_dir": str(result_run_dir),
                "epoch": nll_points[-1][0],
                "optimizer_updates": nll_points[-1][0],
                "nll": final_nll,
                "rmse": final_rmse,
                "oracle_nll": oracle_nll,
                "oracle_rmse": oracle_rmse,
                "delta_nll": final_nll - oracle_nll if oracle_nll == oracle_nll else float("nan"),
                "delta_rmse": final_rmse - oracle_rmse if oracle_rmse == oracle_rmse else float("nan"),
                "delta_rmse_pct": (
                    ((final_rmse - oracle_rmse) / oracle_rmse) * 100.0
                    if oracle_rmse == oracle_rmse and oracle_rmse != 0.0
                    else float("nan")
                ),
            }
        )

    with (out_dir / "curve_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["target", "run_dir", "epoch", "optimizer_updates", "nll", "rmse"],
        )
        writer.writeheader()
        writer.writerows(curve_rows)
    with (out_dir / "final_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target",
                "run_dir",
                "epoch",
                "optimizer_updates",
                "nll",
                "rmse",
                "oracle_nll",
                "oracle_rmse",
                "delta_nll",
                "delta_rmse",
                "delta_rmse_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(final_rows)

    md_lines = [
        "# KSIVI BNN Parity Run Summary",
        "",
        "| Target | NLL | RMSE | Oracle NLL | Oracle RMSE | dNLL | dRMSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in final_rows:
        def _fmt(value: float) -> str:
            return "nan" if value != value else f"{value:.4f}"

        md_lines.append(
            f"| {row['target']} | {_fmt(float(row['nll']))} | {_fmt(float(row['rmse']))} | "
            f"{_fmt(float(row['oracle_nll']))} | {_fmt(float(row['oracle_rmse']))} | "
            f"{_fmt(float(row['delta_nll']))} | {_fmt(float(row['delta_rmse']))} |"
        )
    (out_dir / "final_metrics.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote parity summaries to {out_dir}")


if __name__ == "__main__":
    main()
