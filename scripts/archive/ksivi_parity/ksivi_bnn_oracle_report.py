from __future__ import annotations

import _bootstrap  # noqa: F401

import csv
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
ORIGINAL_ROOT = REPO_ROOT.parent / "KSIVI"
OUT_DIR = REPO_ROOT / "analysis" / "ksivi_parity_20260414" / "oracle"

ORACLE_FINAL = ORIGINAL_ROOT / "generated_reports" / "bnn_metrics_remote" / "final_metrics.csv"
ORACLE_CURVE = ORIGINAL_ROOT / "generated_reports" / "bnn_metrics_remote" / "curve_metrics.csv"
PROTEIN_FINAL = ORIGINAL_ROOT / "generated_reports" / "protein_custom_20260414" / "protein_custom_only.csv"
PROTEIN_CURVE = ORIGINAL_ROOT / "generated_reports" / "protein_custom_20260414" / "curve_metrics.csv"
CURRENT_SUMMARY = (
    REPO_ROOT
    / "campaigns"
    / "grid_benchmark_20260330"
    / "generated_reports"
    / "official_reevaluation_summary.csv"
)

BNN_TARGETS = (
    "Bnn_boston",
    "Bnn_concrete",
    "Bnn_power",
    "Bnn_protein",
    "Bnn_winered",
    "Bnn_yacht",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def _current_bnn_rows() -> dict[str, dict[str, str]]:
    rows = _read_csv(CURRENT_SUMMARY)
    picked: dict[str, dict[str, str]] = {}
    for row in rows:
        if row.get("variant_label") != "KSIVI-custom":
            continue
        target = row.get("target")
        if target not in BNN_TARGETS:
            continue
        if row.get("annealing_mode") != "off":
            continue
        picked[target] = row
    return picked


def _load_oracle_final_rows() -> dict[str, dict[str, str]]:
    rows = {row["target"]: row for row in _read_csv(ORACLE_FINAL)}
    rows["Bnn_protein"] = _read_csv(PROTEIN_FINAL)[0]
    return rows


def _load_oracle_curve_rows() -> list[dict[str, str]]:
    rows = [row for row in _read_csv(ORACLE_CURVE) if row.get("target") != "Bnn_protein"]
    rows.extend(_read_csv(PROTEIN_CURVE))
    return rows


def _config_name_for(target: str) -> str:
    return "kernel_sivi_boston.yml" if target == "Bnn_boston" else f"kernel_sivi_{target[4:].lower()}.yml"


def _load_config_snapshot(target: str) -> dict[str, object]:
    config_path = ORIGINAL_ROOT / "configs" / _config_name_for(target)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    oracle_rows = _load_oracle_final_rows()
    current_rows = _current_bnn_rows()
    curve_lookup = _load_oracle_curve_rows()

    oracle_summary: list[dict[str, object]] = []
    delta_summary: list[dict[str, object]] = []
    config_snapshot_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []

    for target in BNN_TARGETS:
        oracle = oracle_rows[target]
        current = current_rows.get(target)
        oracle_nll = _float(oracle.get("nll"))
        oracle_rmse = _float(oracle.get("rmse"))
        current_nll = _float(current.get("nll_mean") if current else None)
        current_rmse = _float(current.get("rmse_mean") if current else None)
        oracle_summary.append(
            {
                "target": target,
                "run_dir": oracle.get("run_dir", ""),
                "optimizer_updates": int(oracle.get("optimizer_updates", "0")),
                "nll": oracle_nll,
                "rmse": oracle_rmse,
                "source_bundle": (
                    "protein_custom_20260414" if target == "Bnn_protein" else "bnn_metrics_remote"
                ),
            }
        )
        delta_summary.append(
            {
                "target": target,
                "oracle_nll": oracle_nll,
                "current_nll": current_nll,
                "delta_nll": current_nll - oracle_nll,
                "oracle_rmse": oracle_rmse,
                "current_rmse": current_rmse,
                "delta_rmse": current_rmse - oracle_rmse,
                "delta_rmse_pct": (
                    ((current_rmse - oracle_rmse) / oracle_rmse) * 100.0
                    if oracle_rmse == oracle_rmse and oracle_rmse != 0.0 and current_rmse == current_rmse
                    else float("nan")
                ),
                "current_run_id": current.get("run_id", "") if current else "",
                "current_result_path": current.get("result_path", "") if current else "",
            }
        )
        config_snapshot_rows.append(
            {
                "target": target,
                "config_path": str(ORIGINAL_ROOT / "configs" / _config_name_for(target)),
                "config_yaml": yaml.safe_dump(
                    _load_config_snapshot(target),
                    sort_keys=False,
                    allow_unicode=False,
                ).strip(),
            }
        )
        for row in curve_lookup:
            if row.get("target") != target:
                continue
            curve_rows.append(
                {
                    "target": target,
                    "run_dir": row["run_dir"],
                    "epoch": int(row["epoch"]),
                    "optimizer_updates": int(row["optimizer_updates"]),
                    "nll": float(row["nll"]),
                    "rmse": float(row["rmse"]),
                }
            )

    _write_csv(
        OUT_DIR / "oracle_final_metrics.csv",
        oracle_summary,
        ["target", "run_dir", "optimizer_updates", "nll", "rmse", "source_bundle"],
    )
    _write_csv(
        OUT_DIR / "current_vs_oracle.csv",
        delta_summary,
        [
            "target",
            "oracle_nll",
            "current_nll",
            "delta_nll",
            "oracle_rmse",
            "current_rmse",
            "delta_rmse",
            "delta_rmse_pct",
            "current_run_id",
            "current_result_path",
        ],
    )
    _write_csv(
        OUT_DIR / "oracle_curve_metrics.csv",
        curve_rows,
        ["target", "run_dir", "epoch", "optimizer_updates", "nll", "rmse"],
    )
    _write_csv(
        OUT_DIR / "oracle_config_snapshots.csv",
        config_snapshot_rows,
        ["target", "config_path", "config_yaml"],
    )

    md_lines = [
        "# KSIVI BNN Oracle Summary",
        "",
        "| Target | Oracle NLL | Oracle RMSE | Current NLL | Current RMSE | dNLL | dRMSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in delta_summary:
        def _fmt(value: object) -> str:
            if isinstance(value, float):
                return "nan" if value != value else f"{value:.4f}"
            return str(value)

        md_lines.append(
            f"| {row['target']} | {_fmt(row['oracle_nll'])} | {_fmt(row['oracle_rmse'])} | "
            f"{_fmt(row['current_nll'])} | {_fmt(row['current_rmse'])} | "
            f"{_fmt(row['delta_nll'])} | {_fmt(row['delta_rmse'])} |"
        )
    (OUT_DIR / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote oracle report bundle to {OUT_DIR}")


if __name__ == "__main__":
    main()
