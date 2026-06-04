"""Raw SECOM file loading and canonical row ordering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SECOM_LABEL_VALUES = {-1, 1}


@dataclass(frozen=True)
class LoadedSecom:
    """Raw feature frame with the generated feature-column names."""

    frame: pd.DataFrame
    feature_columns: list[str]


def _validate_feature_row_widths(data_path: Path) -> None:
    """Reject ragged raw feature files before Pandas turns short rows into missing cells."""
    expected_width: int | None = None
    for row_number, line in enumerate(data_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{data_path.name}: empty feature row at line {row_number}")
        width = len(line.split())
        if expected_width is None:
            expected_width = width
        elif width != expected_width:
            raise ValueError(
                f"{data_path.name}: inconsistent feature count at line {row_number}: "
                f"expected {expected_width}, got {width}"
            )
    if expected_width is None:
        raise ValueError(f"{data_path.name}: no feature rows found")


def _validated_labels(raw_labels: pd.Series) -> pd.Series:
    """Return integer SECOM labels after enforcing the documented pass/fail values."""
    labels = pd.to_numeric(raw_labels, errors="raise")
    invalid = sorted(set(labels.dropna().astype(object)) - SECOM_LABEL_VALUES)
    has_nan = labels.isna().any()
    if has_nan:
        raise ValueError("y_raw must contain only SECOM labels -1 or 1; contains NaN values")
    if invalid:
        raise ValueError(f"y_raw must contain only SECOM labels -1 or 1; got invalid values {invalid}")
    return labels.astype(int)


def load_raw_secom(input_dir: Path) -> LoadedSecom:
    """Load SECOM feature and label files from an input directory."""
    data_path = input_dir / "secom.data"
    labels_path = input_dir / "secom_labels.data"

    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")

    _validate_feature_row_widths(data_path)
    x = pd.read_csv(data_path, sep=r"\s+", header=None, engine="python")
    labels = pd.read_csv(labels_path, sep=r"\s+", header=None, names=["y_raw", "ts_raw"])
    labels["ts_raw"] = labels["ts_raw"].astype(str).str.replace('"', "", regex=False)

    if len(x) != len(labels):
        raise ValueError(f"Row count mismatch between features ({len(x)}) and labels ({len(labels)})")

    feature_columns = [f"x{i}" for i in range(x.shape[1])]
    x.columns = feature_columns
    df = x.copy()
    df["y_raw"] = _validated_labels(labels["y_raw"])
    df["timestamp_raw"] = labels["ts_raw"]
    df["raw_row_id"] = pd.RangeIndex(start=0, stop=len(df), step=1, dtype="int64")
    return LoadedSecom(frame=df, feature_columns=feature_columns)


def parse_sort_and_label(df: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamps, convert labels to fail/pass binary values, and sort stably."""
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp_raw"], dayfirst=True, errors="coerce", format="%d/%m/%Y %H:%M:%S")
    missing_timestamp_count = int(out["timestamp"].isna().sum())
    if missing_timestamp_count:
        raise ValueError(f"timestamp_raw contains {missing_timestamp_count} unparseable timestamp value(s)")
    labels = _validated_labels(out["y_raw"])
    out["y_raw"] = labels
    out["y_bin"] = (labels == 1).astype(int)

    # Stable deterministic sort contract used by chronological DEV/LOCKBOX splitting.
    out = out.sort_values(["timestamp", "raw_row_id"], kind="mergesort").reset_index(drop=True)
    out["sorted_row_id"] = pd.RangeIndex(start=0, stop=len(out), step=1, dtype="int64")
    return out
