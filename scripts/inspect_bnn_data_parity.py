from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_ROOT = REPO_ROOT.parent / "KSIVI"
PROTEIN_SOURCE = Path(
    r"D:\FileStorage\xwechat_files\wxid_9fnzvu9n04eh22_ffa1\msg\file\2026-04\protein.csv"
)

DATASETS: dict[str, dict[str, object]] = {
    "Bnn_boston": {
        "source": REPO_ROOT / "data" / "boston" / "boston_housing.txt",
        "reference": ORIGINAL_ROOT / "datasets" / "boston_housing.txt",
        "prepared": REPO_ROOT / "data" / "boston",
        "delimiter": None,
    },
    "Bnn_concrete": {
        "source": REPO_ROOT / "data" / "Concrete_Data.csv",
        "reference": ORIGINAL_ROOT / "datasets" / "Concrete_Data.csv",
        "prepared": REPO_ROOT / "data" / "concrete",
        "delimiter": ",",
    },
    "Bnn_power": {
        "source": REPO_ROOT / "data" / "power.csv",
        "reference": ORIGINAL_ROOT / "datasets" / "power.csv",
        "prepared": REPO_ROOT / "data" / "power",
        "delimiter": ",",
    },
    "Bnn_protein": {
        "source": REPO_ROOT / "data" / "protein.csv",
        "reference": PROTEIN_SOURCE,
        "prepared": REPO_ROOT / "data" / "protein",
        "delimiter": ",",
    },
    "Bnn_winered": {
        "source": REPO_ROOT / "data" / "winered.csv",
        "reference": ORIGINAL_ROOT / "datasets" / "winered.csv",
        "prepared": REPO_ROOT / "data" / "winered",
        "delimiter": ";",
    },
    "Bnn_yacht": {
        "source": REPO_ROOT / "data" / "yacht.csv",
        "reference": ORIGINAL_ROOT / "datasets" / "yacht.csv",
        "prepared": REPO_ROOT / "data" / "yacht",
        "delimiter": None,
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_raw_array(path: Path, delimiter: str | None) -> np.ndarray:
    return np.loadtxt(path, delimiter=delimiter)


def _prep_semantics(data: np.ndarray) -> dict[str, object]:
    x = data[:, :-1]
    y = data[:, -1]
    x_train_all, x_test, y_train_all, y_test = train_test_split(
        x,
        y,
        test_size=0.1,
        random_state=42,
    )
    y_train_all = y_train_all[:, None]
    y_test = y_test[:, None]
    dev_size = min(int(np.round(0.1 * x_train_all.shape[0])), 500)
    x_dev = x_train_all[-dev_size:]
    y_dev = y_train_all[-dev_size:]
    x_train = x_train_all[:-dev_size]
    y_train = y_train_all[:-dev_size]
    mu_x = x_train.mean(axis=0)
    std_x = x_train.std(axis=0) + 1e-8
    split_fingerprint = hashlib.sha256(
        np.ascontiguousarray(x_train_all[: min(32, x_train_all.shape[0])]).tobytes()
    ).hexdigest()[:16]
    return {
        "train_shape": list(x_train.shape),
        "dev_shape": list(x_dev.shape),
        "test_shape": list(x_test.shape),
        "x_train_mean_checksum": hashlib.sha256(np.ascontiguousarray(mu_x).tobytes()).hexdigest()[:16],
        "x_train_std_checksum": hashlib.sha256(np.ascontiguousarray(std_x).tobytes()).hexdigest()[:16],
        "y_train_mean": float(y_train.mean()),
        "y_train_std": float(y_train.std()) + 1e-8,
        "split_fingerprint": split_fingerprint,
        "dev_label_mean": float(y_dev.mean()),
        "test_label_mean": float(y_test.mean()),
    }


def _pt_stats(directory: Path) -> dict[str, object]:
    # These prepared payloads are local trusted artifacts that may use an older
    # pickle protocol, so we intentionally allow the full loader here.
    train = torch.load(directory / "train.pt", map_location="cpu", weights_only=False)
    test = torch.load(directory / "test.pt", map_location="cpu", weights_only=False)
    x_train = train["X"].cpu().numpy()
    y_train = train["y"].cpu().numpy()
    x_dev = train["X_dev"].cpu().numpy()
    y_dev = train["y_dev"].cpu().numpy()
    x_test = test["X"].cpu().numpy()
    y_test = test["y"].cpu().numpy()
    return {
        "train_shape": list(x_train.shape),
        "dev_shape": list(x_dev.shape),
        "test_shape": list(x_test.shape),
        "pt_x_train_mean_checksum": hashlib.sha256(np.ascontiguousarray(x_train.mean(axis=0)).tobytes()).hexdigest()[:16],
        "pt_x_train_std_checksum": hashlib.sha256(np.ascontiguousarray(x_train.std(axis=0)).tobytes()).hexdigest()[:16],
        "pt_y_train_mean": float(train["mean_y"].item()),
        "pt_y_train_std": float(train["std_y"].item()),
        "pt_dev_label_mean": float(y_dev.mean()),
        "pt_test_label_mean": float(y_test.mean()),
        "pt_train_target_norm_mean": float(y_train.mean()),
        "pt_train_target_norm_std": float(y_train.std()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect KSIVI BNN source/prepared data parity.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "analysis" / "ksivi_parity_20260414" / "data_diagnostics",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    details: dict[str, dict[str, object]] = {}
    for target, spec in DATASETS.items():
        source = Path(spec["source"])
        reference = Path(spec["reference"])
        prepared = Path(spec["prepared"])
        delimiter = spec["delimiter"]
        raw = _load_raw_array(source, delimiter)
        raw_stats = _prep_semantics(raw)
        pt_stats = _pt_stats(prepared)
        row = {
            "target": target,
            "source_path": str(source),
            "reference_path": str(reference),
            "source_sha256": _sha256(source),
            "reference_sha256": _sha256(reference),
            "source_matches_reference": _sha256(source) == _sha256(reference),
            "source_shape": str(list(raw.shape)),
            "prepared_train_shape": str(pt_stats["train_shape"]),
            "prepared_dev_shape": str(pt_stats["dev_shape"]),
            "prepared_test_shape": str(pt_stats["test_shape"]),
            "expected_train_shape": str(raw_stats["train_shape"]),
            "expected_dev_shape": str(raw_stats["dev_shape"]),
            "expected_test_shape": str(raw_stats["test_shape"]),
            "expected_x_train_mean_checksum": raw_stats["x_train_mean_checksum"],
            "expected_x_train_std_checksum": raw_stats["x_train_std_checksum"],
            "pt_x_train_mean_checksum": pt_stats["pt_x_train_mean_checksum"],
            "pt_x_train_std_checksum": pt_stats["pt_x_train_std_checksum"],
            "expected_y_train_mean": raw_stats["y_train_mean"],
            "expected_y_train_std": raw_stats["y_train_std"],
            "pt_y_train_mean": pt_stats["pt_y_train_mean"],
            "pt_y_train_std": pt_stats["pt_y_train_std"],
            "split_fingerprint": raw_stats["split_fingerprint"],
            "pt_train_target_norm_mean": pt_stats["pt_train_target_norm_mean"],
            "pt_train_target_norm_std": pt_stats["pt_train_target_norm_std"],
        }
        rows.append(row)
        details[target] = {
            "row": row,
            "raw_stats": raw_stats,
            "pt_stats": pt_stats,
        }

    with (out_dir / "bnn_data_diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "bnn_data_diagnostics.json").write_text(
        json.dumps(details, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# BNN Data Diagnostics",
        "",
        "| Target | Source matches reference | Source SHA | Reference SHA | Train | Dev | Test |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['target']} | {row['source_matches_reference']} | "
            f"`{str(row['source_sha256'])[:12]}` | `{str(row['reference_sha256'])[:12]}` | "
            f"{row['prepared_train_shape']} | {row['prepared_dev_shape']} | {row['prepared_test_shape']} |"
        )
    (out_dir / "bnn_data_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote data diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
