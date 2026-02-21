from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class LoadedSecom:
    frame: pd.DataFrame
    feature_columns: list[str]


def load_raw_secom(input_dir: Path) -> LoadedSecom:
    data_path = input_dir / "secom.data"
    labels_path = input_dir / "secom_labels.data"

    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")

    x = pd.read_csv(data_path, sep=r"\s+", header=None, engine="python")
    labels = pd.read_csv(labels_path, sep=r"\s+", header=None, names=["y_raw", "ts_raw"])
    labels["ts_raw"] = labels["ts_raw"].astype(str).str.replace('"', "", regex=False)

    if len(x) != len(labels):
        raise ValueError(
            f"Row count mismatch between features ({len(x)}) and labels ({len(labels)})"
        )

    feature_columns = [f"x{i}" for i in range(x.shape[1])]
    x.columns = feature_columns
    df = x.copy()
    df["y_raw"] = labels["y_raw"].astype(int)
    df["timestamp_raw"] = labels["ts_raw"]
    df["raw_row_id"] = pd.RangeIndex(start=0, stop=len(df), step=1, dtype="int64")
    return LoadedSecom(frame=df, feature_columns=feature_columns)


def parse_sort_and_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(
        out["timestamp_raw"], dayfirst=True, errors="coerce", format="%d/%m/%Y %H:%M:%S"
    )
    out = out.dropna(subset=["timestamp"]).copy()
    out["y_bin"] = (out["y_raw"] == 1).astype(int)

    # Stable deterministic sort contract.
    out = out.sort_values(["timestamp", "raw_row_id"], kind="mergesort").reset_index(drop=True)
    out["sorted_row_id"] = pd.RangeIndex(start=0, stop=len(out), step=1, dtype="int64")
    return out


def write_dataframe_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

