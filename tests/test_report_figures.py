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


def test_feature_plot_scopes_configuration_and_uses_frequency_only() -> None:
    """Other families and large coefficients must not distort a stability chart."""
    from secom.report_figures import feature_stability_plot_rows

    summary = pd.DataFrame(
        [
            {"selector": "ReliefF", "classifier": "krr", "replication_mode": "strict", "mean_BER": 0.30},
            {"selector": "ReliefF", "classifier": "logreg", "replication_mode": "strict", "mean_BER": 0.40},
        ]
    )
    report = pd.DataFrame(
        [
            {
                "selector": "ReliefF",
                "classifier": "krr",
                "replication_mode": "strict",
                "feature_name_or_source_col": "X1",
                "selection_frequency": 0.8,
                "expected_contribution": 1.0,
            },
            {
                "selector": "ReliefF",
                "classifier": "krr",
                "replication_mode": "strict",
                "feature_name_or_source_col": "X2",
                "selection_frequency": 0.9,
                "expected_contribution": 0.0,
            },
            {
                "selector": "ReliefF",
                "classifier": "logreg",
                "replication_mode": "strict",
                "feature_name_or_source_col": "X3",
                "selection_frequency": 1.0,
                "expected_contribution": 999.0,
            },
        ]
    )
    rows = feature_stability_plot_rows(report, summary)
    assert rows["feature_name_or_source_col"].tolist() == ["X2", "X1"]


def test_benchmark_figure_categories_include_replication_mode(workspace_tmp_dir: Path, monkeypatch) -> None:
    """Two modes of the same family must occupy distinct categorical bars."""
    from matplotlib.axes import Axes
    from secom.report_figures import write_benchmark_comparison_figure

    labels = []
    original_barh = Axes.barh

    def capture(self, y, *args, **kwargs):
        labels.extend(list(y))
        return original_barh(self, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "barh", capture)
    summary = pd.DataFrame(
        [
            {
                "selector": "ReliefF",
                "classifier": "krr",
                "replication_mode": mode,
                "mean_BER": 0.30,
                "CI_lower_BER": 0.25,
                "CI_upper_BER": 0.35,
            }
            for mode in ("strict", "with_missing_indicators")
        ]
    )
    write_benchmark_comparison_figure(summary, summary, workspace_tmp_dir / "benchmark.png")
    assert len(labels) == len(set(labels)) == 4
    assert sum("with_missing_indicators" in label for label in labels) == 2
