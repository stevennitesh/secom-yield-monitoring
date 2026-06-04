"""Tests for parsing the original SECOM raw feature and label files."""

from __future__ import annotations

from pathlib import Path

import pytest

from secom.io import load_raw_secom, parse_sort_and_label


def _write_raw_input(input_dir: Path, feature_rows: list[str], label_rows: list[str]) -> None:
    """Write raw SECOM-style feature and label files for parser tests."""
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "secom.data").write_text("\n".join(feature_rows) + "\n", encoding="utf-8")
    (input_dir / "secom_labels.data").write_text("\n".join(label_rows) + "\n", encoding="utf-8")


def test_load_raw_secom_rejects_ragged_feature_rows(workspace_tmp_dir: Path) -> None:
    """Raw feature loading should fail when row widths are inconsistent."""
    input_dir = workspace_tmp_dir / "ragged"
    _write_raw_input(
        input_dir,
        feature_rows=["1.0 2.0 3.0", "4.0 5.0"],
        label_rows=['-1 "19/07/2008 11:55:00"', '1 "19/07/2008 12:55:00"'],
    )

    with pytest.raises(ValueError, match="inconsistent feature count"):
        load_raw_secom(input_dir)


def test_load_raw_secom_rejects_undocumented_label_values(workspace_tmp_dir: Path) -> None:
    """Raw label loading should accept only the documented SECOM labels."""
    input_dir = workspace_tmp_dir / "bad_label"
    _write_raw_input(
        input_dir,
        feature_rows=["1.0 2.0", "3.0 4.0"],
        label_rows=['-1 "19/07/2008 11:55:00"', '0 "19/07/2008 12:55:00"'],
    )

    with pytest.raises(ValueError, match="only SECOM labels -1 or 1"):
        load_raw_secom(input_dir)


def test_load_raw_secom_reports_nan_label_values(workspace_tmp_dir: Path) -> None:
    """Raw label loading should distinguish NaN labels from invalid values."""
    input_dir = workspace_tmp_dir / "nan_label"
    _write_raw_input(
        input_dir,
        feature_rows=["1.0 2.0", "3.0 4.0"],
        label_rows=['-1 "19/07/2008 11:55:00"', 'NaN "19/07/2008 12:55:00"'],
    )

    with pytest.raises(ValueError, match="contains NaN values"):
        load_raw_secom(input_dir)


def test_parse_sort_and_label_rejects_unparseable_timestamps(workspace_tmp_dir: Path) -> None:
    """Timestamp parsing failures should be reported before study sorting."""
    input_dir = workspace_tmp_dir / "bad_timestamp"
    _write_raw_input(
        input_dir,
        feature_rows=["1.0 2.0", "3.0 4.0"],
        label_rows=['-1 "19/07/2008 11:55:00"', '1 "not-a-date"'],
    )
    loaded = load_raw_secom(input_dir)

    with pytest.raises(ValueError, match="unparseable timestamp"):
        parse_sort_and_label(loaded.frame)


def test_parse_sort_and_label_preserves_duplicate_timestamp_rows_with_raw_row_tiebreak(
    workspace_tmp_dir: Path,
) -> None:
    """Duplicate timestamps should sort deterministically by raw row order."""
    input_dir = workspace_tmp_dir / "duplicate_timestamps"
    _write_raw_input(
        input_dir,
        feature_rows=["1.0 2.0", "3.0 4.0", "5.0 6.0"],
        label_rows=[
            '1 "19/07/2008 12:55:00"',
            '-1 "19/07/2008 11:55:00"',
            '1 "19/07/2008 11:55:00"',
        ],
    )

    parsed = parse_sort_and_label(load_raw_secom(input_dir).frame)

    assert parsed["raw_row_id"].tolist() == [1, 2, 0]
    assert parsed["y_bin"].tolist() == [0, 1, 1]
