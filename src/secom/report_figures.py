"""Six reader-facing figures from existing audited study artifacts."""

from __future__ import annotations

import os
import tempfile
from textwrap import wrap
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
from matplotlib.ticker import PercentFormatter  # noqa: E402

FIGURE_DPI = 180
INK, MUTED = "#183246", "#526575"
BLUE, TEAL, ORANGE, GRAY = "#61788e", "#087f78", "#b56638", "#929ea8"
plt.rcParams.update(
    {
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "axes.edgecolor": "#c4cfd7",
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "grid.color": "#dde5ea",
        "grid.linewidth": 0.7,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def _finish(fig, output_path: Path, note: str) -> None:
    width = max(75, int(fig.get_figwidth() * 12))
    lines = [line for paragraph in note.splitlines() for line in wrap(paragraph, width=width)]
    fig.text(0.02, 0.02, "\n".join(lines), ha="left", va="bottom", fontsize=9, color=MUTED, linespacing=1.45)
    footer_height = 0.055 + len(lines) * 0.034
    fig.tight_layout(rect=(0.005, footer_height, 0.995, 0.93))
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def _grid(ax, axis: str = "x") -> None:
    ax.grid(axis=axis)
    ax.tick_params(axis="both", length=0, pad=7)
    ax.spines["left" if axis == "x" else "bottom"].set_visible(False)


def _save_placeholder_figure(output_path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.8))
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
    fig, axes = plt.subplots(1, 3 if counts_available else 1, figsize=(12, 4.4), squeeze=False)
    axes = axes[0]
    colors = [BLUE, TEAL]
    means = 100 * frame.mean_BER
    lower = 100 * (frame.mean_BER - frame.min_BER).clip(lower=0)
    upper = 100 * (frame.max_BER - frame.mean_BER).clip(lower=0)
    axes[0].barh(
        [0, 1],
        means,
        color=colors,
        height=0.45,
        xerr=np.vstack([lower, upper]),
        capsize=4,
        error_kw={"ecolor": INK, "elinewidth": 1.1},
    )
    axes[0].set_xlim(0, max(60, 100 * frame.max_BER.max() + 12))
    axes[0].xaxis.set_major_formatter(PercentFormatter(100))
    axes[0].set_title("Balanced error\nLower is better", loc="left")
    for i, row in frame.iterrows():
        axes[0].text(
            100 * row.max_BER + 2, i, f"{100 * row.mean_BER:.2f}%", va="center", fontsize=11, weight="semibold"
        )
    if counts_available:
        for ax, field, title in zip(
            axes[1:], ("pooled_TP", "pooled_FP"), ("Failures caught", "False alerts on passes")
        ):
            values = frame[field].astype(int)
            complement = "pooled_FN" if field == "pooled_TP" else "pooled_TN"
            totals = frame[field] + frame[complement] if complement in frame else None
            total = int(totals.iloc[0]) if totals is not None and totals.nunique() == 1 else None
            ax.barh([0, 1], values, color=colors, height=0.45)
            limit = total if total is not None else max(1, values.max()) * 1.25
            ax.set_xlim(0, limit)
            class_name = "failures" if field == "pooled_TP" else "passes"
            ax.set_title(title + (f"\n{total:,} {class_name} evaluated" if total is not None else ""), loc="left")
            ax.set_xlabel("Pooled held-out count")
            for i, value in enumerate(values):
                label = f"{value:,} / {total:,}" if total is not None else f"{value:,}"
                inside = value > 0.74 * limit
                ax.text(
                    value + limit * (-0.025 if inside else 0.025),
                    i,
                    label,
                    va="center",
                    ha="right" if inside else "left",
                    fontsize=11,
                    color="white" if inside else INK,
                    weight="semibold",
                )
    for ax in axes:
        ax.set_yticks([0, 1], ["Reference", "Tuned"])
        ax.set_ylim(-0.55, 1.55)
        ax.invert_yaxis()
        _grid(ax)
    scope = "Pooled class totals unavailable."
    if all(c in frame for c in ("pooled_TP", "pooled_FN", "pooled_FP", "pooled_TN")):
        failures = (frame.pooled_TP + frame.pooled_FN).unique()
        passes = (frame.pooled_FP + frame.pooled_TN).unique()
        if len(failures) == len(passes) == 1:
            scope = f"Counts pool {int(failures[0]):,} failures and {int(passes[0]):,} passes."
    fig.suptitle("Reference and tuned: the complete selected procedure", fontsize=14, weight="semibold")
    _finish(
        fig,
        output_path,
        f"Recorded test-fold counts: reference {fold_count_label(benchmark_summary)}; tuned {fold_count_label(benchmark_tuned_summary)}. BER means and observed fold ranges are descriptive.\n{scope} Model and threshold selection use training data only.",
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
    fig, ax = plt.subplots(figsize=(11, max(3.9, 0.43 * len(frame) + 2.2)))
    ax.barh(_labels(frame), delta, color=[TEAL if v >= 0 else ORANGE for v in delta], height=0.52)
    padding = max(1, delta.abs().max()) * 0.025
    for i, value in enumerate(delta):
        label = f"{value:+.2f} pp" if abs(value) > 1e-12 else "0.00 pp"
        ax.text(
            value + (padding if value >= 0 else -padding),
            i,
            label,
            ha="left" if value >= 0 else "right",
            va="center",
            weight="semibold",
            fontsize=10,
        )
    ax.margins(x=0.18)
    ax.invert_yaxis()
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_xlabel("Reference minus tuned balanced error (percentage points)")
    ax.set_title("How balanced error changed after tuning", loc="left", pad=14)
    _grid(ax)
    _finish(
        fig,
        output_path,
        f"Positive = lower error; negative = higher error. Test folds: reference {fold_count_label(benchmark_summary)}; tuned {fold_count_label(benchmark_tuned_summary)}.\nPaired mean differences are descriptive, not significance tests. Input-mode contrasts remain supporting comparisons.",
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
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True)
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
                f"Column {str(r.feature_name_or_source_col)[1:]} · {'missing flag' if r.feature_type == 'missing_indicator' else 'value'}"
            ),
            axis=1,
        )
        ax.barh(
            labels,
            rows.selection_frequency * 100,
            color=rows.feature_type.map({"value": BLUE, "missing_indicator": ORANGE}),
            height=0.60,
        )
        for i, frequency in enumerate(rows.selection_frequency * 100):
            ax.text(frequency + 1.5, i, f"{frequency:.0f}%", va="center", fontsize=10, weight="semibold")
        ax.set_title(
            f"{study}: {best.selector} / {CLASSIFIERS.get(best.classifier, best.classifier)}\n{procedure_label(best.replication_mode)}",
            loc="left",
            fontsize=11,
        )
        ax.set_xlim(0, 112)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Selected across outer training folds (%)")
        ax.invert_yaxis()
        _grid(ax)
    fig.suptitle("Which anonymous inputs recur across training folds?", fontsize=14, weight="semibold")
    _finish(
        fig,
        output_path,
        f"Anonymous columns; outer training folds: reference {fold_count_label(benchmark_summary)}, tuned {fold_count_label(benchmark_tuned_summary)}. Each panel shows one exploratory family.\nFrequency measures recurrence, not causality or independent validation. Values and missing flags are named on each bar.",
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
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.7))
    prevalence_shift = 100 * float(row.get("abs_prevalence_shift", np.nan))
    axes[0].bar(["Earlier → later"], [prevalence_shift], color=TEAL, width=0.44)
    axes[0].set_xlim(-0.7, 0.7)
    axes[0].set_ylabel("Absolute change (percentage points)")
    axes[0].set_title("Failure prevalence", loc="left")
    if np.isfinite(prevalence_shift):
        axes[0].set_ylim(0, max(1, prevalence_shift) * 1.25)
        axes[0].text(
            0,
            prevalence_shift + max(1, prevalence_shift) * 0.04,
            f"{prevalence_shift:.2f} pp",
            ha="center",
            weight="semibold",
        )
    if pd.notna(row.get("dev_fail_rate")) and pd.notna(row.get("lockbox_fail_rate")):
        axes[0].set_title(f"Failure prevalence\n{row.dev_fail_rate:.2%} → {row.lockbox_fail_rate:.2%}", loc="left")
    psi = pd.Series({"Maximum": row.get("max_PSI", np.nan), "Median": row.get("median_PSI", np.nan)}).dropna()
    axes[1].bar(
        psi.index, psi.values, color=[ORANGE if label == "Maximum" else BLUE for label in psi.index], width=0.55
    )
    if not psi.empty:
        axes[1].set_ylim(0, max(0.1, psi.max()) * 1.25)
        for i, value in enumerate(psi):
            axes[1].text(i, value + max(0.1, psi.max()) * 0.04, f"{value:.2f}", ha="center", weight="semibold")
    axes[1].set_ylabel("Population Stability Index (PSI)")
    count = row.get("psi_feature_count")
    axes[1].set_title(
        "Measurement shift" + (f"\n{int(count)} selected value features" if pd.notna(count) else ""), loc="left"
    )
    _grid(axes[0], "y")
    _grid(axes[1], "y")
    axes[2].axis("off")
    pvalue = float(row.get("ks_pvalue_scores", np.nan))
    axes[2].set_title("Model-score shift", loc="left")
    axes[2].text(0.5, 0.72, "KS distribution-test p-value", ha="center", fontsize=10, color=MUTED)
    axes[2].text(0.5, 0.50, f"{pvalue:.3g}", ha="center", fontsize=23, weight="semibold")
    axes[2].text(
        0.5,
        0.24,
        "Reference: held-out calibration\nscores from this same model",
        ha="center",
        va="center",
        fontsize=10,
        color=MUTED,
    )
    fig.suptitle("Primary logistic regression: change in the retrospective later block", fontsize=14, weight="semibold")
    _finish(
        fig,
        output_path,
        "Earlier references: full development (prevalence), fitting samples (PSI), same-model held-out calibration (scores).\nAll compare with the retrospective later block. Larger PSI means more change; diagnostics do not establish cause or operational superiority.",
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9))
    colors = [BLUE if policy == "scientific" else TEAL for policy in lr.threshold_policy] + [GRAY]
    for ax, count, denominator, title in (
        (axes[0], "TP", frame.TP + frame.FN, "Failures caught"),
        (axes[1], "FP", frame.FP + frame.TN, "False alerts on passes"),
    ):
        rates = 100 * frame[count] / denominator
        bars = ax.bar(labels, rates, color=colors, width=0.55)
        for bar, n, total, rate in zip(bars, frame[count], denominator, rates, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f"{int(n)} / {int(total)}\n{rate:.1f}%",
                ha="center",
                fontsize=10,
                weight="semibold",
            )
        ax.set_ylim(0, 116)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_major_formatter(PercentFormatter(100))
        ax.set_title(title, loc="left")
        _grid(ax, "y")
    fig.suptitle(
        "Logistic regression and process control on the retrospective later block", fontsize=14, weight="semibold"
    )
    _finish(
        fig,
        output_path,
        later_sample_scope(frame)
        + " Thresholds froze on earlier calibration; no later-label choice.\nSparse failures restrict conclusions. Intervals and retrospective 90%-specificity diagnostics are tabulated separately.",
    )


def write_workload_cost_figure(temporal_manager, temporal_cost, output_path: Path, *, temporal_lockbox=None) -> None:
    if any(f is None or f.empty for f in (temporal_manager, temporal_cost)):
        _save_placeholder_figure(
            output_path, "Hypothetical workload and cost", "Calibration workload or cost artifact is unavailable."
        )
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), gridspec_kw={"width_ratios": [1, 1.2]})
    manager = temporal_manager.copy()
    labels = manager.apply(lambda r: role_label(r.role, r.threshold_policy).replace(": ", ":\n"), axis=1)
    workload = manager.mean_weekly_flagged_samples
    axes[0].barh(
        labels,
        workload,
        color=manager.threshold_policy.map({"scientific": BLUE, "operational": TEAL}),
        height=0.52,
    )
    maximum = max(1, workload.max())
    axes[0].set_xlim(0, maximum * 1.23)
    for i, value in enumerate(workload):
        axes[0].text(value + maximum * 0.025, i, f"{value:.1f}", va="center", fontsize=10, weight="semibold")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Mean flagged samples per calibration week")
    axes[0].set_title("Calibration alert workload", loc="left")
    _grid(axes[0])
    labels = {
        "primary_scientific": "Primary: balanced-error threshold",
        "primary_operational": "Primary: workload-limited threshold",
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
    axes[1].set_ylabel("Hypothetical cost per\nretrospective later-block sample\n(false-alert cost = 1)")
    axes[1].set_xticks(sorted(temporal_cost.cost_ratio.unique()))
    axes[1].set_xlim(0, max(1, temporal_cost.cost_ratio.max()) * 1.03)
    cost_columns = [column for column in labels if column in temporal_cost]
    if cost_columns:
        axes[1].set_ylim(0, max(0.1, temporal_cost[cost_columns].max().max()) * 1.38)
    n = None
    if (
        temporal_lockbox is not None
        and not temporal_lockbox.empty
        and {"TP", "FN", "FP", "TN"} <= set(temporal_lockbox)
    ):
        n = int(temporal_lockbox.iloc[0][["TP", "FN", "FP", "TN"]].sum())
    axes[1].set_title("Retrospective costs" + (f" (N={n})" if n is not None else ""), loc="left")
    axes[1].legend(fontsize=9, frameon=False, loc="upper left", ncol=2, columnspacing=1.4, handlelength=2.2)
    _grid(axes[1], "y")
    fig.suptitle("Logistic regression: illustrative workload and retrospective costs", fontsize=14, weight="semibold")
    _finish(
        fig,
        output_path,
        "Workload: held-out calibration; the 10% rule limits the mean weekly fraction, not each week's hard cap.\nCosts: frozen rules on the retrospective later block. Costs are hypothetical; neither panel validates future capacity or realized savings.",
    )
