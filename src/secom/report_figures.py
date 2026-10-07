"""Six reader-facing figures from existing audited study artifacts."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from secom.report_language import (
    CLASSIFIERS,
    family_label,
    fold_count_label,
    later_sample_scope,
    procedure_label,
    role_label,
)


def _configure_matplotlib_cache() -> None:
    if "MPLCONFIGDIR" not in os.environ:
        cache_dir = Path(tempfile.gettempdir()) / "secom-matplotlib"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache_dir)


_configure_matplotlib_cache()
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

FIGURE_DPI = 180
BLUE, TEAL, ORANGE, GRAY = "#355c7d", "#148578", "#c86b35", "#717b85"
plt.rcParams.update(
    {"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False}
)


def _finish(fig, output_path: Path, note: str) -> None:
    fig.text(0.02, 0.02, note, ha="left", va="bottom", fontsize=10, color="#444444")
    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def _save_placeholder_figure(output_path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    _finish(fig, output_path, "No numerical comparison is available from these artifacts.")


def _labels(frame: pd.DataFrame) -> pd.Series:
    return frame.procedure.map(procedure_label) if "procedure" in frame else frame.apply(family_label, axis=1)


def write_benchmark_comparison_figure(benchmark_summary, benchmark_tuned_summary, output_path: Path) -> None:
    """Keep the joint procedure primary and separate fold-mean error from pooled counts."""
    rows = []
    for frame in (benchmark_summary, benchmark_tuned_summary):
        if frame is None or frame.empty or "procedure" not in frame:
            break
        joint = frame[frame.procedure.eq("joint")]
        if len(joint) != 1 or not np.isfinite(joint.iloc[0].mean_BER):
            break
        rows.append(joint.iloc[0])
    if len(rows) != 2:
        _save_placeholder_figure(
            output_path, "Reference and tuned benchmark", "Complete selected procedures are unavailable."
        )
        return
    frame = pd.DataFrame(rows).reset_index(drop=True)
    counts_available = all(c in frame and frame[c].notna().all() for c in ("pooled_TP", "pooled_FP"))
    fig, axes = plt.subplots(1, 3 if counts_available else 1, figsize=(12, 5), squeeze=False)
    axes = axes[0]
    colors = [BLUE, TEAL]
    means = 100 * frame.mean_BER
    lower = 100 * (frame.mean_BER - frame.min_BER).clip(lower=0)
    upper = 100 * (frame.max_BER - frame.mean_BER).clip(lower=0)
    axes[0].barh([0, 1], means, color=colors, xerr=np.vstack([lower, upper]), capsize=3)
    axes[0].set_xlim(0, max(60, 100 * frame.max_BER.max() + 12))
    axes[0].xaxis.set_major_formatter(PercentFormatter(100))
    axes[0].set_title("Balanced error\nLower is better")
    for i, row in frame.iterrows():
        axes[0].text(100 * row.max_BER + 2, i, f"{100 * row.mean_BER:.2f}%", va="center", fontsize=11)
    if counts_available:
        for ax, field, title in zip(
            axes[1:], ("pooled_TP", "pooled_FP"), ("Failures caught", "False alerts on passes")
        ):
            values = frame[field].astype(int)
            ax.barh([0, 1], values, color=colors)
            ax.set_xlim(0, max(1, values.max()) * 1.25)
            ax.set_title(title)
            ax.set_xlabel("Pooled held-out count")
            for i, value in enumerate(values):
                ax.text(value + max(1, values.max()) * 0.03, i, str(value), va="center", fontsize=12)
    for ax in axes:
        ax.set_yticks([0, 1], ["Reference", "Tuned"])
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.15)
    scope = "Pooled class totals unavailable."
    if all(c in frame for c in ("pooled_TP", "pooled_FN", "pooled_FP", "pooled_TN")):
        failures = (frame.pooled_TP + frame.pooled_FN).unique()
        passes = (frame.pooled_FP + frame.pooled_TN).unique()
        if len(failures) == len(passes) == 1:
            scope = f"Counts pool {int(failures[0]):,} failures and {int(passes[0]):,} passes."
    fig.suptitle("Reference vs tuned: the complete selected procedure", fontsize=15)
    _finish(
        fig,
        output_path,
        f"Recorded test-fold counts: reference {fold_count_label(benchmark_summary)}; tuned {fold_count_label(benchmark_tuned_summary)}. BER bars: fold means; whiskers: observed ranges.\n{scope} Fold spread is descriptive; model and threshold selection use training data only.",
    )


def write_tuned_delta_figure(benchmark_summary, benchmark_tuned_summary, output_path: Path) -> None:
    """Positive reductions have the same improvement sign as the report prose."""
    if any(f is None or f.empty for f in (benchmark_summary, benchmark_tuned_summary)):
        _save_placeholder_figure(
            output_path, "Balanced-error reduction after tuning", "Benchmark summaries are unavailable."
        )
        return
    keys = ["procedure"] if "procedure" in benchmark_summary else ["selector", "classifier", "replication_mode"]
    frame = benchmark_summary.merge(benchmark_tuned_summary, on=keys, suffixes=("_reference", "_tuned"))
    if frame.empty:
        _save_placeholder_figure(
            output_path, "Balanced-error reduction after tuning", "No paired procedures are available."
        )
        return
    delta = 100 * (frame.mean_BER_reference - frame.mean_BER_tuned)
    fig, ax = plt.subplots(figsize=(11, max(5, 0.42 * len(frame) + 2)))
    ax.barh(_labels(frame), delta, color=[TEAL if v >= 0 else ORANGE for v in delta])
    padding = max(1, delta.abs().max()) * 0.025
    for i, value in enumerate(delta):
        label = f"{value:+.2f} pp" if abs(value) > 1e-12 else "0.00 pp"
        ax.text(
            value + (padding if value >= 0 else -padding), i, label, ha="left" if value >= 0 else "right", va="center"
        )
    ax.margins(x=0.18)
    ax.invert_yaxis()
    ax.axvline(0, color=GRAY, linewidth=1)
    ax.set_xlabel("Reference minus tuned balanced error (percentage points)")
    ax.set_title("Positive = lower error after tuning; negative = higher error")
    ax.grid(axis="x", alpha=0.15)
    _finish(
        fig,
        output_path,
        f"Recorded test-fold counts: reference {fold_count_label(benchmark_summary)}; tuned {fold_count_label(benchmark_tuned_summary)}. Mean differences are descriptive.\nPaired predefined procedures, not significance tests. Input-mode contrasts do not replace the complete selected headline.",
    )


def feature_stability_plot_rows(feature_report: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Choose one exploratory family and rank only by selection frequency."""
    best = summary.sort_values(["mean_BER", "selector", "classifier", "replication_mode"]).iloc[0]
    rows = feature_report[
        (feature_report.selector == best.selector)
        & (feature_report.classifier == best.classifier)
        & (feature_report.replication_mode == best.replication_mode)
    ].copy()
    return rows.sort_values(["selection_frequency", "feature_name_or_source_col"], ascending=[False, True]).head(10)


def write_feature_stability_figure(
    feature_report, tuned_feature_report, output_path: Path, *, benchmark_summary=None, benchmark_tuned_summary=None
) -> None:
    if any(
        f is None or f.empty for f in (feature_report, tuned_feature_report, benchmark_summary, benchmark_tuned_summary)
    ):
        _save_placeholder_figure(
            output_path, "Which anonymous inputs recur?", "Feature reports or summaries are unavailable."
        )
        return
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharex=True)
    for ax, study, report, summary in zip(
        axes,
        ("Reference", "Tuned"),
        (feature_report, tuned_feature_report),
        (benchmark_summary, benchmark_tuned_summary),
        strict=True,
    ):
        rows = feature_stability_plot_rows(report, summary)
        best = summary.sort_values(["mean_BER", "selector", "classifier", "replication_mode"]).iloc[0]
        labels = rows.apply(
            lambda r: (
                f"Column {str(r.feature_name_or_source_col)[1:]}: {'missing flag' if r.feature_type == 'missing_indicator' else 'value'}"
            ),
            axis=1,
        )
        ax.barh(
            labels,
            rows.selection_frequency * 100,
            color=rows.feature_type.map({"value": BLUE, "missing_indicator": ORANGE}),
        )
        ax.set_title(
            f"{study}: {best.selector} / {CLASSIFIERS.get(best.classifier, best.classifier)}\n{procedure_label(best.replication_mode)}"
        )
        ax.set_xlim(0, 105)
        ax.set_xlabel("Selected across outer training folds (%)")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.15)
    fig.suptitle("Which anonymous inputs recur across training folds?")
    fig.legend(
        handles=[Patch(color=BLUE, label="Measurement value"), Patch(color=ORANGE, label="Missing-measurement flag")],
        loc="lower right",
        bbox_to_anchor=(0.98, 0.03),
        ncol=2,
    )
    _finish(
        fig,
        output_path,
        f"Anonymous file columns; recorded outer folds: reference {fold_count_label(benchmark_summary)}, tuned {fold_count_label(benchmark_tuned_summary)}. Exploratory family selection.\nA family combines selector, model and inputs. Frequency orders bars; association is not causal or independent validation.",
    )


def write_temporal_drift_figure(temporal_drift, output_path: Path) -> None:
    if temporal_drift is None or temporal_drift.empty:
        _save_placeholder_figure(
            output_path, "What changed in later samples?", "Temporal drift summary is unavailable."
        )
        return
    primary = temporal_drift[temporal_drift.model_scope == "primary"]
    if primary.empty:
        _save_placeholder_figure(
            output_path, "What changed in later samples?", "Primary model reference is unavailable."
        )
        return
    row = primary.iloc[0]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5))
    axes[0].bar(["Failure prevalence"], [100 * float(row.get("abs_prevalence_shift", np.nan))], color=TEAL)
    axes[0].set_ylabel("Absolute change (percentage points)")
    axes[0].set_title("Label frequency")
    if pd.notna(row.get("dev_fail_rate")) and pd.notna(row.get("lockbox_fail_rate")):
        axes[0].text(
            0.5,
            0.93,
            f"{row.dev_fail_rate:.2%} → {row.lockbox_fail_rate:.2%}\nFull development → later block",
            transform=axes[0].transAxes,
            ha="center",
            va="top",
            fontsize=10,
        )
        axes[0].set_ylim(0, max(1, 100 * float(row.abs_prevalence_shift)) * 1.35)
    psi = pd.Series({"Maximum": row.get("max_PSI", np.nan), "Median": row.get("median_PSI", np.nan)}).dropna()
    axes[1].bar(psi.index, psi.values, color=[ORANGE, BLUE])
    axes[1].set_ylabel("Population Stability Index (PSI)")
    count = row.get("psi_feature_count")
    axes[1].set_title("Raw-measurement shift" + (f"\n{int(count)} selected value features" if pd.notna(count) else ""))
    axes[2].axis("off")
    pvalue = float(row.get("ks_pvalue_scores", np.nan))
    axes[2].set_title("Model-score shift")
    axes[2].text(
        0.5,
        0.55,
        f"Distribution-test p-value\n{pvalue:.3g}\n\nReference: held-out calibration\nscores from this same model",
        ha="center",
        va="center",
    )
    fig.suptitle("Primary logistic regression: earlier references versus the retrospective later block")
    _finish(
        fig,
        output_path,
        "Prevalence: full development versus later block. PSI: selected value features versus earlier fitting samples; larger means more change.\nScore test: same-model held-out calibration reference. Descriptive diagnostics do not establish causes or operational superiority.",
    )


def write_lockbox_vs_mspc_figure(temporal_lockbox, temporal_mspc, output_path: Path) -> None:
    """Use frozen operating counts, never thresholds selected from later evaluation labels."""
    if any(f is None or f.empty for f in (temporal_lockbox, temporal_mspc)):
        _save_placeholder_figure(
            output_path, "Later samples at frozen thresholds", "Later-block artifacts are unavailable."
        )
        return
    lr = temporal_lockbox[temporal_lockbox.role == "primary"]
    mspc = temporal_mspc[temporal_mspc.eval_scope == "lockbox"]
    required = {"TP", "FN", "FP", "TN"}
    if lr.empty or mspc.empty or not all(required <= set(f) for f in (lr, mspc)):
        _save_placeholder_figure(
            output_path, "Later samples at frozen thresholds", "Frozen-threshold confusion counts are unavailable."
        )
        return
    labels = [
        "LR: balanced-error\nthreshold" if p == "scientific" else "LR: workload-limited\nthreshold"
        for p in lr.threshold_policy
    ] + ["Statistical process\ncontrol (MSPC)"]
    frame = pd.concat([lr, mspc.iloc[[0]]], ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, count, denominator, title in (
        (axes[0], "TP", frame.TP + frame.FN, "Failures caught"),
        (axes[1], "FP", frame.FP + frame.TN, "False alerts on passes"),
    ):
        rates = 100 * frame[count] / denominator
        bars = ax.bar(labels, rates, color=[BLUE, TEAL, GRAY][: len(frame)])
        for bar, n, total in zip(bars, frame[count], denominator, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{int(n)} / {int(total)}", ha="center")
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_formatter(PercentFormatter(100))
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle("Retrospective later-block results · thresholds frozen on earlier calibration")
    _finish(
        fig,
        output_path,
        later_sample_scope(frame)
        + " No later-label threshold choice in this chart.\nIntervals and retrospective 90%-specificity diagnostics remain in the appendix; no superiority claim.",
    )


def write_workload_cost_figure(temporal_manager, temporal_cost, output_path: Path, *, temporal_lockbox=None) -> None:
    if any(f is None or f.empty for f in (temporal_manager, temporal_cost)):
        _save_placeholder_figure(
            output_path, "Hypothetical workload and cost", "Calibration workload or cost artifact is unavailable."
        )
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    manager = temporal_manager.copy()
    labels = manager.apply(lambda r: role_label(r.role, r.threshold_policy).replace(": ", ":\n"), axis=1)
    axes[0].barh(labels, manager.mean_weekly_flagged_samples, color=BLUE)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Mean flagged samples per calibration week")
    axes[0].set_title("Calibration alert workload")
    labels = {
        "primary_scientific": "Primary LR: balanced-error threshold",
        "primary_operational": "Primary LR: workload-limited threshold",
        "all_pass_baseline": "Always predict pass",
        "all_flag_baseline": "Flag every sample",
    }
    for column, color, linestyle, marker in zip(
        labels, (BLUE, TEAL, GRAY, ORANGE), ("-", "--", ":", "-."), ("o", "s", "^", "D"), strict=True
    ):
        if column in temporal_cost:
            axes[1].plot(
                temporal_cost.cost_ratio,
                temporal_cost[column],
                marker=marker,
                linestyle=linestyle,
                label=labels[column],
                color=color,
            )
    axes[1].set_xlabel("Missed-failure cost / false-alert cost")
    axes[1].set_ylabel("Hypothetical cost per retrospective later-block sample\n(false-alert cost = 1)")
    n = None
    if (
        temporal_lockbox is not None
        and not temporal_lockbox.empty
        and {"TP", "FN", "FP", "TN"} <= set(temporal_lockbox)
    ):
        n = int(temporal_lockbox.iloc[0][["TP", "FN", "FP", "TN"]].sum())
    axes[1].set_title("Retrospective costs" + (f" (N={n})" if n is not None else ""))
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.15)
    _finish(
        fig,
        output_path,
        "Workload: held-out calibration used to choose thresholds. Costs: frozen rules on the retrospective later block; assumptions are hypothetical.\nThe 10% rule limits mean weekly flagged fraction, not each week's hard cap. Neither panel validates future capacity or realized savings.",
    )
