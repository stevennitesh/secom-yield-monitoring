from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd


def assert_artifacts_exist(reports_dir: Path, artifact_names: Iterable[str]) -> None:
    for name in artifact_names:
        assert (reports_dir / name).exists(), name


def assert_columns_include(frame: pd.DataFrame, expected_columns: Iterable[str]) -> None:
    assert set(expected_columns).issubset(frame.columns)


def assert_text_contains_all(text: str, expected_fragments: Iterable[str]) -> None:
    for fragment in expected_fragments:
        assert fragment in text


def assert_text_excludes_all(text: str, forbidden_fragments: Iterable[str]) -> None:
    for fragment in forbidden_fragments:
        assert fragment not in text
