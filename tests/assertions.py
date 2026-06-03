from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


def assert_artifacts_exist(reports_dir: Path, artifact_names: Iterable[str]) -> None:
    for name in artifact_names:
        assert (reports_dir / name).exists(), name


def assert_columns_include(frame: pd.DataFrame, expected_columns: Iterable[str]) -> None:
    missing = set(expected_columns) - set(frame.columns)
    assert not missing, f"missing columns: {sorted(missing)}"


def assert_text_contains_all(text: str, expected_fragments: Iterable[str]) -> None:
    for fragment in expected_fragments:
        assert fragment in text, f"missing text fragment: {fragment!r}"


def assert_text_excludes_all(text: str, forbidden_fragments: Iterable[str]) -> None:
    for fragment in forbidden_fragments:
        assert fragment not in text, f"forbidden text fragment present: {fragment!r}"


def threshold_equal(a: float, b: float) -> bool:
    """Compare threshold sentinels and finite float thresholds."""
    if np.isnan(a) and np.isnan(b):
        return True
    if np.isinf(a) and np.isinf(b):
        return bool(np.sign(a) == np.sign(b))
    return bool(np.isclose(float(a), float(b), atol=1e-12))
