"""Figure generation helpers for final reports."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def _configure_matplotlib_cache() -> None:
    """Use a writable Matplotlib config directory when callers have not set one."""
    if "MPLCONFIGDIR" in os.environ:
        return
    cache_dir = Path(tempfile.gettempdir()) / "secom-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)


_configure_matplotlib_cache()

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

FIGURE_DPI = 180


def _save_placeholder_figure(output_path: Path, title: str, message: str) -> None:
    """Write a deterministic placeholder figure when source artifacts are unavailable."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def write_benchmark_comparison_figure(
    benchmark_summary: pd.DataFrame | None,
    benchmark_tuned_summary: pd.DataFrame | None,
    output_path: Path,
) -> None:
    """Plot the best original and tuned benchmark rows by mean BER."""
    if (
        benchmark_summary is None
        or benchmark_summary.empty
        or benchmark_tuned_summary is None
        or benchmark_tuned_summary.empty
    ):
        _save_placeholder_figure(
            output_path,
            "Benchmark Comparison",
            "Benchmark summary artifacts are missing.",
        )
        return

    original = benchmark_summary.nsmallest(5, "mean_BER").copy()
    tuned = benchmark_tuned_summary.nsmallest(5, "mean_BER").copy()
    original["study"] = "original"
    tuned["study"] = "tuned"
    frame = pd.concat([original, tuned], ignore_index=True)
    frame["label"] = (
        frame["study"] + ": " + frame["selector"] + " / " + frame["classifier"] + " / " + frame["replication_mode"]
    )
    frame = frame.sort_values("mean_BER", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = frame["study"].map({"original": "#355070", "tuned": "#b56576"}).tolist()
    lower = (frame["mean_BER"] - frame["CI_lower_BER"]).clip(lower=0.0)
    upper = (frame["CI_upper_BER"] - frame["mean_BER"]).clip(lower=0.0)
    ax.barh(frame["label"], frame["mean_BER"], color=colors, xerr=np.vstack([lower, upper]))
    ax.set_title("Best Benchmark Rows by Mean BER")
    ax.set_xlabel("Mean BER")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def write_tuned_delta_figure(
    benchmark_summary: pd.DataFrame | None,
    benchmark_tuned_summary: pd.DataFrame | None,
    output_path: Path,
) -> None:
    """Plot tuned-minus-original BER deltas for shared benchmark rows."""
    if (
        benchmark_summary is None
        or benchmark_summary.empty
        or benchmark_tuned_summary is None
        or benchmark_tuned_summary.empty
    ):
        _save_placeholder_figure(
            output_path,
            "Tuned vs Original BER Delta",
            "Benchmark comparison artifacts are missing.",
        )
        return

    merged = benchmark_summary.merge(
        benchmark_tuned_summary,
        on=["selector", "classifier", "replication_mode"],
        suffixes=("_original", "_tuned"),
    )
    if merged.empty:
        _save_placeholder_figure(
            output_path,
            "Tuned vs Original BER Delta",
            "No shared selector/classifier/mode rows were found.",
        )
        return

    merged["delta_BER"] = merged["mean_BER_tuned"] - merged["mean_BER_original"]
    merged["label"] = merged["selector"] + " / " + merged["classifier"] + " / " + merged["replication_mode"]
    merged = merged.sort_values("delta_BER", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, 0.35 * len(merged) + 1.5)))
    colors = ["#6d597a" if value <= 0 else "#e56b6f" for value in merged["delta_BER"]]
    ax.barh(merged["label"], merged["delta_BER"], color=colors)
    ax.axvline(0.0, color="#444444", linewidth=1)
    ax.set_title("Tuned vs Original Mean BER Delta")
    ax.set_xlabel("delta BER (tuned - original)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def feature_stability_plot_rows(feature_report: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Choose a single leading configuration; rank only by cross-fold selection frequency."""
    best = summary.sort_values(["mean_BER", "selector", "classifier", "replication_mode"]).iloc[0]
    rows = feature_report[
        (feature_report["selector"] == best["selector"])
        & (feature_report["classifier"] == best["classifier"])
        & (feature_report["replication_mode"] == best["replication_mode"])
    ].copy()
    return rows.sort_values(["selection_frequency", "feature_name_or_source_col"], ascending=[False, True]).head(10)


def write_feature_stability_figure(
    feature_report: pd.DataFrame | None,
    tuned_feature_report: pd.DataFrame | None,
    output_path: Path,
    *,
    benchmark_summary: pd.DataFrame | None = None,
    benchmark_tuned_summary: pd.DataFrame | None = None,
) -> None:
    """Show original and tuned stability in separate, explicitly scoped panels."""
    inputs = (feature_report, tuned_feature_report, benchmark_summary, benchmark_tuned_summary)
    if any(frame is None or frame.empty for frame in inputs):
        _save_placeholder_figure(
            output_path, "Feature Stability", "Feature reports or benchmark summaries are missing."
        )
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    for ax, study, report, summary in zip(
        axes,
        ("Original", "Tuned"),
        (feature_report, tuned_feature_report),
        (benchmark_summary, benchmark_tuned_summary),
        strict=True,
    ):
        rows = feature_stability_plot_rows(report, summary)
        best = summary.sort_values(["mean_BER", "selector", "classifier", "replication_mode"]).iloc[0]
        colors = rows["feature_type"].map({"value": "#457b9d", "missing_indicator": "#e76f51"})
        ax.barh(rows["feature_name_or_source_col"], rows["selection_frequency"] * 100, color=colors)
        ax.set_title(f"{study}: {best['selector']} / {best['classifier']}\n{best['replication_mode']}")
        ax.set_xlabel("Selected in outer folds (%)")
        ax.set_xlim(0, 105)
        ax.invert_yaxis()
    fig.suptitle("Feature selection stability (blue: value; orange: missing indicator)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def write_temporal_drift_figure(
    temporal_drift: pd.DataFrame | None,
    output_path: Path,
) -> None:
    """Plot primary temporal drift quantities."""
    if temporal_drift is None or temporal_drift.empty:
        _save_placeholder_figure(
            output_path,
            "Temporal Drift Summary",
            "Temporal drift summary artifact is missing.",
        )
        return

    row = temporal_drift.iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(["Absolute prevalence shift"], [100 * float(row.get("abs_prevalence_shift", np.nan))], color="#2a9d8f")
    axes[0].set_ylabel("percentage points")
    psi = pd.Series(
        {"max PSI": float(row.get("max_PSI", np.nan)), "median PSI": float(row.get("median_PSI", np.nan))}
    ).dropna()
    axes[1].bar(psi.index, psi.values, color=["#e76f51", "#e9c46a"])
    axes[1].set_ylabel("Population Stability Index")
    fig.suptitle(f"Temporal drift ({row.get('drift_gate_status', 'unknown')})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def write_lockbox_vs_mspc_figure(
    temporal_lockbox: pd.DataFrame | None,
    temporal_mspc: pd.DataFrame | None,
    output_path: Path,
) -> None:
    """Compare supervised lockbox TPR with the MSPC lockbox baseline at TNR90."""
    if temporal_lockbox is None or temporal_lockbox.empty or temporal_mspc is None or temporal_mspc.empty:
        _save_placeholder_figure(
            output_path,
            "Lockbox Supervised vs MSPC",
            "Lockbox or MSPC artifact is missing.",
        )
        return

    supervised = temporal_lockbox[
        (temporal_lockbox["role"] == "primary") & (temporal_lockbox["threshold_policy"] == "scientific")
    ]
    mspc = temporal_mspc[temporal_mspc["eval_scope"] == "lockbox"]
    if supervised.empty or mspc.empty:
        _save_placeholder_figure(
            output_path,
            "Lockbox Supervised vs MSPC",
            "Primary scientific lockbox row or MSPC lockbox row is missing.",
        )
        return

    supervised_row = supervised.iloc[0]
    mspc_row = mspc.iloc[0]
    labels = ["Supervised", "MSPC"]
    values = [
        float(supervised_row.get("TPR_at_TNR90", np.nan)),
        float(mspc_row.get("best_MSPC_TPR_at_TNR90", np.nan)),
    ]
    finite_values = [value for value in values if np.isfinite(value)]
    if not finite_values:
        _save_placeholder_figure(
            output_path,
            "Lockbox Supervised vs MSPC",
            "Matched-TNR90 TPR values are missing.",
        )
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, values, color=["#264653", "#f4a261"])
    ax.set_ylim(0, max(1.0, max(finite_values) + 0.1))
    ax.set_title("Retrospective Lockbox ROC Comparison (TNR ≥ 90%)")
    ax.set_ylabel("TPR_at_TNR90")
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def write_workload_cost_figure(
    temporal_manager: pd.DataFrame | None,
    temporal_cost: pd.DataFrame | None,
    output_path: Path,
) -> None:
    """Plot workload framing and illustrative temporal cost curves."""
    if temporal_manager is None or temporal_manager.empty or temporal_cost is None or temporal_cost.empty:
        _save_placeholder_figure(
            output_path,
            "Workload and Cost Framing",
            "Temporal manager or cost artifact is missing.",
        )
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    manager = temporal_manager.copy()
    manager["label"] = manager["role"] + " / " + manager["threshold_policy"]
    axes[0].bar(manager["label"], manager["mean_weekly_flagged_samples"], color="#577590")
    axes[0].set_title("Weekly Flagged Samples")
    axes[0].set_ylabel("mean flagged samples")
    axes[0].tick_params(axis="x", rotation=20)

    for column in ["primary_scientific", "primary_operational", "all_pass_baseline", "all_flag_baseline"]:
        if column in temporal_cost.columns:
            axes[1].plot(temporal_cost["cost_ratio"], temporal_cost[column], marker="o", label=column)
    axes[1].set_title("Illustrative Cost Curves")
    axes[1].set_xlabel("cost ratio")
    axes[1].set_ylabel("normalized cost")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
