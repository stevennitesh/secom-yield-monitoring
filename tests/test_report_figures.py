"""Tests for report figure rendering edge cases."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from secom.report_figures import write_lockbox_vs_mspc_figure, write_temporal_drift_figure
from tests.assertions import assert_renderable_png


def test_temporal_drift_figure_uses_primary_not_first_row(workspace_tmp_dir: Path, monkeypatch) -> None:
    from matplotlib.axes import Axes

    heights = []
    original_bar = Axes.bar

    def capture(self, x, height, *args, **kwargs):
        heights.append(list(height))
        return original_bar(self, x, height, *args, **kwargs)

    monkeypatch.setattr(Axes, "bar", capture)
    drift = pd.DataFrame(
        [
            {"model_scope": "challenger", "abs_prevalence_shift": 0.7, "max_PSI": 9.0, "median_PSI": 4.0},
            {"model_scope": "primary", "abs_prevalence_shift": 0.02, "max_PSI": 0.4, "median_PSI": 0.2},
        ]
    )
    write_temporal_drift_figure(drift, workspace_tmp_dir / "drift.png")
    assert heights == [[2.0], [0.4, 0.2]]


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
                "calibration_selected_MSPC_TPR_at_TNR90": np.nan,
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
                "stability_weighted_coefficient": 1.0,
            },
            {
                "selector": "ReliefF",
                "classifier": "krr",
                "replication_mode": "strict",
                "feature_name_or_source_col": "X2",
                "selection_frequency": 0.9,
                "stability_weighted_coefficient": 0.0,
            },
            {
                "selector": "ReliefF",
                "classifier": "logreg",
                "replication_mode": "strict",
                "feature_name_or_source_col": "X3",
                "selection_frequency": 1.0,
                "stability_weighted_coefficient": 999.0,
            },
        ]
    )
    rows = feature_stability_plot_rows(report, summary)
    assert rows["feature_name_or_source_col"].tolist() == ["X2", "X1"]


def test_primary_benchmark_chart_excludes_hindsight_family_minima(workspace_tmp_dir: Path, monkeypatch) -> None:
    """A secondary zero-error row cannot dominate the complete-procedure headline."""
    from matplotlib.axes import Axes
    from secom.report_figures import write_benchmark_comparison_figure

    widths = []
    original_barh = Axes.barh

    def capture(self, y, width, *args, **kwargs):
        widths.append(list(width))
        return original_barh(self, y, width, *args, **kwargs)

    monkeypatch.setattr(Axes, "barh", capture)

    def summary(error, caught, alerts):
        return pd.DataFrame(
            [
                {
                    "procedure": "joint",
                    "mean_BER": error,
                    "min_BER": 0.2,
                    "max_BER": 0.4,
                    "pooled_TP": caught,
                    "pooled_FP": alerts,
                    "pooled_FN": 104 - caught,
                    "pooled_TN": 1463 - alerts,
                },
                {
                    "procedure": "values_only",
                    "mean_BER": 0,
                    "min_BER": 0,
                    "max_BER": 0,
                    "pooled_TP": 104,
                    "pooled_FP": 0,
                    "pooled_FN": 0,
                    "pooled_TN": 1463,
                },
            ]
        )

    output = workspace_tmp_dir / "benchmark.png"
    write_benchmark_comparison_figure(summary(0.32, 70, 445), summary(0.30, 66, 371), output)
    assert widths == [[32, 30], [70, 66], [445, 371]]
    assert_renderable_png(output)


def test_primary_benchmark_chart_requires_both_complete_procedures(workspace_tmp_dir, monkeypatch):
    from secom import report_figures

    messages = []
    monkeypatch.setattr(
        report_figures, "_save_placeholder_figure", lambda path, title, message: messages.append(message)
    )
    family = pd.DataFrame([{"procedure": "values_only", "mean_BER": 0.1}])
    joint = pd.DataFrame([{"procedure": "joint", "mean_BER": 0.3}])
    report_figures.write_benchmark_comparison_figure(joint, family, workspace_tmp_dir / "benchmark.png")
    assert messages == ["Complete selected procedures are unavailable."]


def test_delta_figure_positive_means_lower_tuned_error(workspace_tmp_dir: Path, monkeypatch) -> None:
    from matplotlib.axes import Axes
    from secom.report_figures import write_tuned_delta_figure

    widths = []
    original_barh = Axes.barh

    def capture(self, y, width, *args, **kwargs):
        widths.extend(width)
        return original_barh(self, y, width, *args, **kwargs)

    monkeypatch.setattr(Axes, "barh", capture)
    original = pd.DataFrame([{"procedure": "joint", "mean_BER": 0.32}])
    tuned = pd.DataFrame([{"procedure": "joint", "mean_BER": 0.30}])
    write_tuned_delta_figure(original, tuned, workspace_tmp_dir / "delta.png")
    assert np.allclose(widths, [2.0])


def test_later_chart_uses_frozen_counts_not_retrospective_roc(workspace_tmp_dir: Path, monkeypatch) -> None:
    from matplotlib.axes import Axes

    heights = []
    original_bar = Axes.bar

    def capture(self, x, height, *args, **kwargs):
        heights.append(list(height))
        return original_bar(self, x, height, *args, **kwargs)

    monkeypatch.setattr(Axes, "bar", capture)
    lr = pd.DataFrame(
        [
            {
                "role": "primary",
                "threshold_policy": "scientific",
                "TP": 7,
                "FN": 2,
                "FP": 168,
                "TN": 58,
                "TPR_at_TNR90": 0.333,
            }
        ]
    )
    mspc = pd.DataFrame(
        [
            {
                "eval_scope": "lockbox",
                "TP": 0,
                "FN": 9,
                "FP": 3,
                "TN": 223,
                "calibration_selected_MSPC_TPR_at_TNR90": 0.111,
            }
        ]
    )
    write_lockbox_vs_mspc_figure(lr, mspc, workspace_tmp_dir / "later.png")
    assert np.allclose(heights[0], [100 * 7 / 9, 0])
    assert np.allclose(heights[1], [100 * 168 / 226, 100 * 3 / 226])


def test_figure_captions_derive_folds_and_later_counts(workspace_tmp_dir, monkeypatch):
    from secom import report_figures

    notes = []

    def capture(fig, output_path, note):
        notes.append(note)
        report_figures.plt.close(fig)

    monkeypatch.setattr(report_figures, "_finish", capture)
    summary = pd.DataFrame([{"procedure": "joint", "n_folds": 4, "mean_BER": 0.3, "min_BER": 0.2, "max_BER": 0.4}])
    report_figures.write_benchmark_comparison_figure(summary, summary, workspace_tmp_dir / "benchmark.png")
    report_figures.write_tuned_delta_figure(summary, summary, workspace_tmp_dir / "delta.png")
    lr = pd.DataFrame([{"role": "primary", "threshold_policy": "scientific", "TP": 8, "FN": 12, "FP": 17, "TN": 63}])
    mspc = pd.DataFrame([{"eval_scope": "lockbox", "TP": 0, "FN": 20, "FP": 3, "TN": 77}])
    report_figures.write_lockbox_vs_mspc_figure(lr, mspc, workspace_tmp_dir / "later.png")
    assert all("reference 4; tuned 4" in note for note in notes[:2])
    assert "20 failures and 80 passes" in notes[2]
    assert not any("nine" in note or "ten" in note.lower() for note in notes)
