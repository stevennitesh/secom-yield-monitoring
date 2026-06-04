"""Test helpers for writing artifact CSV rows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from secom.artifacts import write_csv


def write_artifact_rows(reports_dir: Path, artifact_name: str, rows: list[dict[str, object]]) -> None:
    """Write multiple artifact rows through the production CSV writer."""
    write_csv(pd.DataFrame(rows), reports_dir / artifact_name)


def write_artifact_row(reports_dir: Path, artifact_name: str, row: dict[str, object]) -> None:
    """Write one artifact row through the production CSV writer."""
    write_artifact_rows(reports_dir, artifact_name, [row])
