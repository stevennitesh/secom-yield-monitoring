"""Tests for report figure rendering edge cases."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from secom.report_figures import write_lockbox_vs_mspc_figure, write_temporal_drift_figure
from tests.assertions import assert_renderable_png


def test_temporal_drift_missing_artifact_writes_placeholder_png(workspace_tmp_dir: Path) -> None:
    """Missing temporal drift input should still produce a renderable placeholder."""
    output_path = workspace_tmp_dir / "temporal_drift.png"

    write_temporal_drift_figure(None, output_path)

    assert_renderable_png(output_path)


def test_lockbox_vs_mspc_all_nan_values_write_placeholder_png(workspace_tmp_dir: Path) -> None:
    """All-missing matched-TNR values should not crash final report figure rendering."""
    output_path = workspace_tmp_dir / "lockbox_vs_mspc.png"
    lockbox = pd.DataFrame(
        [
            {
                "role": "primary",
                "threshold_policy": "scientific",
                "TPR_at_TNR90": np.nan,
            }
        ]
    )
    mspc = pd.DataFrame(
        [
            {
                "eval_scope": "lockbox",
                "best_MSPC_TPR_at_TNR90": np.nan,
            }
        ]
    )

    write_lockbox_vs_mspc_figure(lockbox, mspc, output_path)

    assert_renderable_png(output_path)
