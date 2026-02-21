from __future__ import annotations

from pathlib import Path
from uuid import uuid4
import shutil

import numpy as np
import pandas as pd
import pytest


def _make_synthetic_secom(n_rows: int = 260, n_features: int = 12, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_rows, n_features))
    # Add small signal into first features.
    y = (rng.random(n_rows) < 0.20).astype(int)
    x[:, 0] += 0.8 * y
    x[:, 1] -= 0.4 * y

    start = pd.Timestamp("2008-01-01 00:00:00")
    timestamps = [start + pd.Timedelta(hours=12 * i) for i in range(n_rows)]
    ts_str = [t.strftime("%d/%m/%Y %H:%M:%S") for t in timestamps]

    x_df = pd.DataFrame(x)
    labels = pd.DataFrame(
        {
            "y_raw": np.where(y == 1, 1, -1),
            "ts_raw": [f'"{s}"' for s in ts_str],
        }
    )
    return x_df, labels


@pytest.fixture()
def workspace_tmp_dir() -> Path:
    root = Path(".test_tmp") / str(uuid4())
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture()
def synthetic_input_dir(workspace_tmp_dir: Path) -> Path:
    input_dir = workspace_tmp_dir / "data" / "raw"
    input_dir.mkdir(parents=True, exist_ok=True)
    x_df, labels = _make_synthetic_secom()
    x_df.to_csv(input_dir / "secom.data", sep=" ", header=False, index=False, na_rep="NaN")
    labels.to_csv(input_dir / "secom_labels.data", sep=" ", header=False, index=False)
    return input_dir
