"""Markdown report assembly from generated study artifacts."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from secom.artifacts import read_csv_if_exists, read_manifest
from secom.config import ArtifactName, BenchmarkClassifier, ReplicationMode, StudyStatus, ThresholdPolicy
from secom.report_figures import (
    write_benchmark_comparison_figure,
    write_feature_stability_figure,
    write_lockbox_vs_mspc_figure,
    write_temporal_drift_figure,
    write_tuned_delta_figure,
    write_workload_cost_figure,
)


def _format_float(value: object) -> str:
    """Format numeric report values with a compact missing-value fallback."""
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "n/a"
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def _format_cell(value: object) -> str:
    """Format one Markdown table cell."""
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return _format_float(float(value))
    return str(value)


def _format_percent(value: object) -> str:
    """Format fractional metric values as report percentages."""
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "n/a"
        return f"{100.0 * float(value):.1f}"
    except Exception:
        return str(value)


def _markdown_table(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    headers: list[str] | None = None,
    max_rows: int | None = None,
) -> list[str]:
    """Render a small DataFrame slice as Markdown table lines."""
    table = frame.loc[:, columns].copy()
    if max_rows is not None:
        table = table.head(max_rows)
    header_row = headers if headers is not None else columns
    lines = [
        "| " + " | ".join(header_row) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    lines.extend(
        "| " + " | ".join(_format_cell(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    )
    return lines


_UCI_ORIGINAL_BASELINE_ROWS = [
    {
        "uci_method": "S2N",
        "selector": "S2N",
        "uci_BER": "34.5 +/- 2.6",
        "uci_True+": "57.8 +/- 5.3",
        "uci_True-": "73.1 +/- 2.1",
    },
    {
        "uci_method": "Ttest",
        "selector": "Ttest",
        "uci_BER": "33.7 +/- 2.1",
        "uci_True+": "59.6 +/- 4.7",
        "uci_True-": "73.0 +/- 1.8",
    },
    {
        "uci_method": "Relief",
        "selector": "ReliefF",
        "uci_BER": "40.1 +/- 2.8",
        "uci_True+": "48.3 +/- 5.9",
        "uci_True-": "71.6 +/- 3.2",
    },
    {
        "uci_method": "Pearson",
        "selector": "Pearson",
        "uci_BER": "34.1 +/- 2.0",
        "uci_True+": "57.4 +/- 4.3",
        "uci_True-": "74.4 +/- 4.9",
    },
    {
        "uci_method": "Ftest",
        "selector": "F-test",
        "uci_BER": "33.5 +/- 2.2",
        "uci_True+": "59.1 +/- 4.8",
        "uci_True-": "73.8 +/- 1.8",
    },
    {
        "uci_method": "Gram Schmidt",
        "selector": "Gram-Schmidt",
        "uci_BER": "35.6 +/- 2.4",
        "uci_True+": "51.2 +/- 11.8",
        "uci_True-": "77.5 +/- 2.3",
    },
]


def _uci_baseline_match(benchmark_summary: pd.DataFrame | None, selector: str) -> pd.Series | None:
    """Return the local strict KRR row that best matches the UCI original benchmark setup."""
    if benchmark_summary is None or benchmark_summary.empty:
        return None

    rows = benchmark_summary[benchmark_summary["selector"].astype(str) == selector].copy()
    if rows.empty:
        return None

    preferred = rows[
        (rows["classifier"].astype(str) == BenchmarkClassifier.KRR)
        & (rows["replication_mode"].astype(str) == ReplicationMode.STRICT)
    ]
    if preferred.empty:
        preferred = rows[rows["replication_mode"].astype(str) == ReplicationMode.STRICT]
    if preferred.empty:
        preferred = rows
    return preferred.sort_values(["mean_BER", "classifier", "replication_mode"]).iloc[0]


def _uci_original_baseline_table(benchmark_summary: pd.DataFrame | None) -> list[str]:
    """Compare local original replication rows with the UCI SECOM reference benchmark table."""
    rows = []
    missing_local_result = "not run"
    for baseline in _UCI_ORIGINAL_BASELINE_ROWS:
        local = _uci_baseline_match(benchmark_summary, str(baseline["selector"]))
        rows.append(
            {
                "UCI method": baseline["uci_method"],
                "local selector": baseline["selector"],
                "UCI BER %": baseline["uci_BER"],
                "UCI True+ %": baseline["uci_True+"],
                "UCI True- %": baseline["uci_True-"],
                "local BER %": _format_percent(local["mean_BER"]) if local is not None else missing_local_result,
                "local True+ %": _format_percent(local["mean_True+"]) if local is not None else missing_local_result,
                "local True- %": _format_percent(local["mean_True-"]) if local is not None else missing_local_result,
            }
        )
    return _markdown_table(
        pd.DataFrame(rows),
        [
            "UCI method",
            "local selector",
            "UCI BER %",
            "UCI True+ %",
            "UCI True- %",
            "local BER %",
            "local True+ %",
            "local True- %",
        ],
    )


def _uci_selector_definition_note() -> str:
    """Explain selector-definition differences that affect UCI/local interpretation."""
    return (
        "Interpretation note: the local Ttest row uses a pooled two-sample t statistic to align with the UCI "
        "selector label; Welch-t remains available only as an explicit local selector. Binary-label ANOVA F-test "
        "ranking and absolute Pearson correlation ranking are mathematically monotonic for non-constant features, "
        "so they can select the same 40-feature set and produce identical local rows. The UCI reference table reports "
        "separate Ftest and Pearson rows, which should be read as that benchmark's implementation/protocol "
        "definitions rather than a guarantee that the two selectors are distinct under this replication."
    )


def _feature_interpretation_claim_note() -> str:
    """Return the feature-report claim boundary used by final and scaffold reports."""
    return (
        "Feature outputs are model-prioritization evidence from resampled benchmark artifacts, not causal proof "
        "or validated process-driver identification. Stability across resamples matters more than a single "
        "full-fit ranking, and missing-indicator features are kept distinct from raw value features."
    )


def _top_benchmark_table(benchmark_summary: pd.DataFrame) -> list[str]:
    """Render primary benchmark BER/TPR/TNR evidence."""
    table = benchmark_summary.sort_values(["mean_BER", "selector", "classifier", "replication_mode"]).copy()
    return _markdown_table(
        table,
        [
            "selector",
            "classifier",
            "replication_mode",
            "mean_BER",
            "CI_lower_BER",
            "CI_upper_BER",
            "mean_True+",
            "mean_True-",
        ],
        headers=["selector", "classifier", "mode", "mean_BER", "CI_low", "CI_high", "mean_TPR", "mean_TNR"],
    )


def _supporting_benchmark_table(benchmark_summary: pd.DataFrame) -> list[str]:
    """Render threshold-independent and supporting benchmark metrics."""
    table = benchmark_summary.sort_values(["mean_BER", "selector", "classifier", "replication_mode"]).copy()
    return _markdown_table(
        table,
        [
            "selector",
            "classifier",
            "replication_mode",
            "mean_ROC_AUC",
            "mean_PR_AUC",
            "mean_MCC",
            "mean_F2",
        ],
        headers=["selector", "classifier", "mode", "mean_ROC_AUC", "mean_PR_AUC", "mean_MCC", "mean_F2"],
    )


def _search_space_count(frame: pd.DataFrame, column: str) -> int:
    """Return the number of distinct search values for one optional config column."""
    if column == "n_neighbors":
        values = frame[column].dropna().to_numpy(dtype=float) if column in frame.columns else np.array([], dtype=float)
        return int(pd.unique(values).size) if values.size else 0
    return int(frame[column].dropna().nunique()) if column in frame.columns else 0


def _search_space_table(frame: pd.DataFrame, *, evaluated_columns: list[str]) -> list[str]:
    """Summarize evaluated hyperparameter breadth by selector/classifier/mode."""
    summary_rows = []
    for (selector, classifier, mode), group in frame.groupby(
        ["selector", "classifier", "replication_mode"], sort=False
    ):
        summary_rows.append(
            {
                "selector": selector,
                "classifier": classifier,
                "mode": mode,
                "evaluated_configs": int(group[evaluated_columns].drop_duplicates().shape[0]),
                "k_values": _search_space_count(group, "k"),
                "c_values": _search_space_count(group, "C"),
                "alpha_values": _search_space_count(group, "alpha"),
                "gamma_values": _search_space_count(group, "gamma"),
                "n_neighbors_values": _search_space_count(group, "n_neighbors"),
            }
        )
    return _markdown_table(
        pd.DataFrame(summary_rows),
        [
            "selector",
            "classifier",
            "mode",
            "evaluated_configs",
            "k_values",
            "c_values",
            "alpha_values",
            "gamma_values",
            "n_neighbors_values",
        ],
    )


def _original_search_space_table(benchmark_sweep: pd.DataFrame) -> list[str]:
    """Summarize original benchmark search-space breadth by selector/classifier/mode."""
    return _search_space_table(benchmark_sweep, evaluated_columns=["k", "alpha", "gamma", "C", "n_neighbors"])


def _original_best_config_table(benchmark_best: pd.DataFrame) -> list[str]:
    """Render original benchmark selected configurations."""
    table = benchmark_best.sort_values(["mean_BER", "selector", "classifier", "replication_mode"]).copy()
    cols = ["selector", "classifier", "replication_mode", "k", "C", "alpha", "gamma", "n_neighbors", "mean_BER"]
    existing_cols = [col for col in cols if col in table.columns]
    headers_map = {
        "selector": "selector",
        "classifier": "classifier",
        "replication_mode": "mode",
        "k": "k",
        "C": "C",
        "alpha": "alpha",
        "gamma": "gamma",
        "n_neighbors": "n_neighbors",
        "mean_BER": "mean_BER",
    }
    return _markdown_table(
        table,
        existing_cols,
        headers=[headers_map[col] for col in existing_cols],
    )


def _tuned_search_space_table(benchmark_tuned_search: pd.DataFrame) -> list[str]:
    """Summarize tuned benchmark nested-search breadth by selector/classifier/mode."""
    return _search_space_table(
        benchmark_tuned_search,
        evaluated_columns=["fold", "k", "alpha", "gamma", "C", "n_neighbors"],
    )


def _temporal_selection_summary_table(temporal_selection: pd.DataFrame) -> list[str]:
    """Render temporal selector roles in primary/challenger/supporting order."""
    preferred = [
        "selector",
        "status",
        "mean_BER",
        "mean_True+",
        "mean_True-",
        "modal_k",
        "modal_C",
        "modal_scaler",
        "modal_n_neighbors",
    ]
    keep = [col for col in preferred if col in temporal_selection.columns]
    order = {"primary": 0, "challenger": 1, "supporting": 2}
    table = (
        temporal_selection.assign(_status_rank=temporal_selection["status"].map(order).fillna(99))
        .sort_values(["_status_rank", "mean_BER", "selector"])
        .drop(columns="_status_rank")
    )
    return _markdown_table(table[keep], keep)


def _tuned_best_config_table(benchmark_tuned_best: pd.DataFrame) -> list[str]:
    """Render modal tuned configurations selected across outer folds."""
    table = benchmark_tuned_best.sort_values(["mean_BER", "selector", "classifier", "replication_mode"]).copy()
    return _markdown_table(
        table,
        [
            "selector",
            "classifier",
            "replication_mode",
            "k",
            "C",
            "alpha",
            "gamma",
            "n_neighbors",
            "selection_count",
            "mean_inner_ROC_AUC",
            "mean_inner_BER",
        ],
        headers=[
            "selector",
            "classifier",
            "mode",
            "k",
            "C",
            "alpha",
            "gamma",
            "n_neighbors",
            "selected_count",
            "mean_inner_ROC_AUC",
            "mean_inner_BER",
        ],
    )


def _best_row_feature_table(
    feature_report: pd.DataFrame,
    selector: str,
    classifier: str,
    replication_mode: str,
) -> list[str]:
    """Render the top feature rows for one selector/classifier/mode configuration."""
    rows = feature_report[
        (feature_report["selector"] == selector)
        & (feature_report["classifier"] == classifier)
        & (feature_report["replication_mode"] == replication_mode)
    ].copy()
    if rows.empty:
        return ["- No feature rows available for the leading benchmark configuration."]
    rows = rows.sort_values(
        ["expected_contribution", "selection_frequency", "feature_name_or_source_col"],
        ascending=[False, False, True],
    ).head(10)
    claim_note = f"- {_feature_interpretation_claim_note()}"
    if rows["conditional_effect_magnitude"].isna().all():
        lines = [
            claim_note,
            "- Effect magnitudes are unavailable for the leading classifier, so this table is shown as a stability-first view.",
            "",
            "| feature | type | selection_frequency | cluster_id |",
            "|---|---|---:|---:|",
        ]
        lines.extend(
            f"| {row.feature_name_or_source_col} | {row.feature_type} |"
            f" {_format_float(row.selection_frequency)} | {_format_float(row.cluster_id)} |"
            for row in rows.itertuples(index=False)
        )
        return lines

    lines = [
        claim_note,
        "",
        "| feature | type | selection_frequency | effect_magnitude | expected_contribution | cluster_id |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows.itertuples(index=False):
        lines.append(
            f"| {row.feature_name_or_source_col} | {row.feature_type} | {_format_float(row.selection_frequency)} |"
            f" {_format_float(row.conditional_effect_magnitude)} | {_format_float(row.expected_contribution)} |"
            f" {_format_float(row.cluster_id)} |"
        )
    return lines


@dataclass(frozen=True)
class ReportContext:
    """All optional artifacts and preselected rows needed by final report assembly."""

    reports_dir: Path
    manifest: dict[str, object]
    benchmark_sweep: pd.DataFrame | None
    benchmark_best: pd.DataFrame | None
    benchmark_summary: pd.DataFrame | None
    benchmark_ablation: pd.DataFrame | None
    feature_report: pd.DataFrame | None
    benchmark_tuned_search: pd.DataFrame | None
    benchmark_tuned_best: pd.DataFrame | None
    benchmark_tuned_summary: pd.DataFrame | None
    benchmark_tuned_ablation: pd.DataFrame | None
    benchmark_tuned_feature_report: pd.DataFrame | None
    temporal_selection: pd.DataFrame | None
    temporal_lockbox: pd.DataFrame | None
    temporal_drift: pd.DataFrame | None
    temporal_mspc: pd.DataFrame | None
    temporal_manager: pd.DataFrame | None
    temporal_cost: pd.DataFrame | None
    best_benchmark_row: pd.Series | None
    best_tuned_benchmark_row: pd.Series | None
    modal_tuned_config_row: pd.Series | None
    primary_temporal_row: pd.Series | None
    primary_scientific_lockbox_row: pd.Series | None
    drift_row: pd.Series | None
    mspc_lockbox_row: pd.Series | None


_REPORT_CONTEXT_ARTIFACTS: dict[str, str] = {
    "benchmark_sweep": ArtifactName.BENCHMARK_SWEEP,
    "benchmark_best": ArtifactName.BENCHMARK_BEST_CONFIG,
    "benchmark_summary": ArtifactName.BENCHMARK_SUMMARY,
    "benchmark_ablation": ArtifactName.BENCHMARK_ABLATION,
    "feature_report": ArtifactName.FEATURE_REPORT,
    "benchmark_tuned_search": ArtifactName.BENCHMARK_TUNED_SEARCH,
    "benchmark_tuned_best": ArtifactName.BENCHMARK_TUNED_BEST_CONFIG,
    "benchmark_tuned_summary": ArtifactName.BENCHMARK_TUNED_SUMMARY,
    "benchmark_tuned_ablation": ArtifactName.BENCHMARK_TUNED_ABLATION,
    "benchmark_tuned_feature_report": ArtifactName.BENCHMARK_TUNED_FEATURE_REPORT,
    "temporal_selection": ArtifactName.TEMPORAL_MODEL_SELECTION,
    "temporal_lockbox": ArtifactName.TEMPORAL_LOCKBOX,
    "temporal_drift": ArtifactName.TEMPORAL_DRIFT,
    "temporal_mspc": ArtifactName.TEMPORAL_MSPC,
    "temporal_manager": ArtifactName.TEMPORAL_MANAGER_OUTPUTS,
    "temporal_cost": ArtifactName.TEMPORAL_COST_CURVES,
}


def _first_row(frame: pd.DataFrame | None, mask: pd.Series | None = None) -> pd.Series | None:
    """Return the first row from an optional artifact frame."""
    if frame is None or frame.empty:
        return None
    rows = frame if mask is None else frame[mask]
    if rows.empty:
        return None
    return rows.iloc[0]


def _load_report_context(output_dir: Path) -> ReportContext:
    """Load all report artifacts and compute the leading rows used by the narrative."""
    reports = output_dir / "reports"
    manifest_path = reports / ArtifactName.MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = read_manifest(manifest_path)
    frames = {
        field: read_csv_if_exists(reports / artifact_name) for field, artifact_name in _REPORT_CONTEXT_ARTIFACTS.items()
    }
    benchmark_summary = frames["benchmark_summary"]
    benchmark_tuned_summary = frames["benchmark_tuned_summary"]
    benchmark_tuned_best = frames["benchmark_tuned_best"]
    temporal_selection = frames["temporal_selection"]
    temporal_lockbox = frames["temporal_lockbox"]
    temporal_drift = frames["temporal_drift"]
    temporal_mspc = frames["temporal_mspc"]

    # These chosen rows are narrative anchors only; full evidence tables remain in the report.
    best_benchmark_row = None
    if benchmark_summary is not None and not benchmark_summary.empty:
        best_benchmark_row = benchmark_summary.sort_values(
            ["mean_BER", "selector", "classifier", "replication_mode"]
        ).iloc[0]

    best_tuned_benchmark_row = None
    if benchmark_tuned_summary is not None and not benchmark_tuned_summary.empty:
        best_tuned_benchmark_row = benchmark_tuned_summary.sort_values(
            ["mean_BER", "selector", "classifier", "replication_mode"]
        ).iloc[0]

    modal_tuned_config_row = None
    if benchmark_tuned_best is not None and not benchmark_tuned_best.empty:
        modal_tuned_config_row = benchmark_tuned_best.sort_values(
            ["selection_count", "mean_BER", "selector", "classifier", "replication_mode"],
            ascending=[False, True, True, True, True],
        ).iloc[0]

    primary_temporal_row = _first_row(
        temporal_selection,
        temporal_selection["is_primary"].astype(bool) if temporal_selection is not None else None,
    )

    primary_scientific_lockbox_row = _first_row(
        temporal_lockbox,
        ((temporal_lockbox["role"] == "primary") & (temporal_lockbox["threshold_policy"] == ThresholdPolicy.SCIENTIFIC))
        if temporal_lockbox is not None
        else None,
    )

    drift_row = _first_row(temporal_drift)
    mspc_lockbox_row = _first_row(
        temporal_mspc,
        temporal_mspc["eval_scope"] == "lockbox" if temporal_mspc is not None else None,
    )

    return ReportContext(
        reports_dir=reports,
        manifest=manifest,
        **frames,
        best_benchmark_row=best_benchmark_row,
        best_tuned_benchmark_row=best_tuned_benchmark_row,
        modal_tuned_config_row=modal_tuned_config_row,
        primary_temporal_row=primary_temporal_row,
        primary_scientific_lockbox_row=primary_scientific_lockbox_row,
        drift_row=drift_row,
        mspc_lockbox_row=mspc_lockbox_row,
    )


def _append_bullet_list(lines: list[str], items: list[str]) -> None:
    """Append Markdown bullet lines in-place."""
    lines.extend(f"- {item}" for item in items)


def _append_benchmark_summary_table(
    lines: list[str],
    heading: str,
    frame: pd.DataFrame | None,
) -> None:
    """Append a primary benchmark summary section."""
    lines.append(heading)
    lines.append("")
    if frame is None or frame.empty:
        lines.append("- Benchmark summary artifact missing or empty.")
    else:
        lines.extend(_top_benchmark_table(frame))
    lines.append("")


def _append_supporting_metrics_table(
    lines: list[str],
    heading: str,
    frame: pd.DataFrame | None,
) -> None:
    """Append a supporting metrics section."""
    lines.append(heading)
    lines.append("")
    if frame is None or frame.empty:
        lines.append("- Supporting benchmark metrics artifact missing or empty.")
    else:
        lines.extend(_supporting_benchmark_table(frame))
    lines.append("")


def _append_figure(lines: list[str], alt_text: str, relative_path: str, caption: str) -> None:
    """Append a Markdown image reference and caption."""
    lines.append(f"![{alt_text}]({relative_path})")
    lines.append("")
    lines.append(caption)
    lines.append("")


def _write_final_report_figures(ctx: ReportContext) -> None:
    """Write all figures referenced by the canonical final report."""
    figures_dir = ctx.reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    write_benchmark_comparison_figure(
        ctx.benchmark_summary,
        ctx.benchmark_tuned_summary,
        figures_dir / "benchmark_comparison.png",
    )
    write_tuned_delta_figure(
        ctx.benchmark_summary,
        ctx.benchmark_tuned_summary,
        figures_dir / "tuned_vs_original_delta.png",
    )
    write_feature_stability_figure(
        ctx.feature_report,
        ctx.benchmark_tuned_feature_report,
        figures_dir / "feature_stability.png",
    )
    write_temporal_drift_figure(ctx.temporal_drift, figures_dir / "temporal_drift.png")
    write_lockbox_vs_mspc_figure(
        ctx.temporal_lockbox,
        ctx.temporal_mspc,
        figures_dir / "lockbox_vs_mspc.png",
    )
    write_workload_cost_figure(
        ctx.temporal_manager,
        ctx.temporal_cost,
        figures_dir / "workload_cost_framing.png",
    )


def _raise_for_failed_report_audit(output_dir: Path) -> None:
    """Block final report generation when active artifacts fail the study audit."""
    from secom.workflows.audit import run_study_audit

    audit = run_study_audit(output_dir)
    if audit.ok:
        return
    details = "; ".join(audit.errors[:5])
    if len(audit.errors) > 5:
        details = f"{details}; ... ({len(audit.errors)} total errors)"
    raise RuntimeError(f"Cannot render final report because study audit failed: {details}")


def _write_markdown_with_optional_pdf(final_path: Path, lines: list[str], *, export_pdf: bool) -> None:
    """Write final Markdown and append PDF export status when requested."""
    final_path.write_text("\n".join(lines), encoding="utf-8")
    if not export_pdf:
        return

    pdf_path = final_path.with_suffix(".pdf")
    pandoc_path = shutil.which("pandoc")
    if pandoc_path is None:
        pdf_note = "PDF export skipped because pandoc is not available."
    else:
        try:
            subprocess.run(
                [pandoc_path, str(final_path), "-o", str(pdf_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            pdf_note = f"PDF export written to `{pdf_path.name}`."
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() or exc.stdout.strip() or "unknown error"
            pdf_note = f"PDF export skipped because pandoc failed: {detail}"

    lines.append(f"- PDF export status: {pdf_note}")
    lines.append("")
    final_path.write_text("\n".join(lines), encoding="utf-8")


def write_report_skeleton(output_dir: Path) -> Path:
    """Write a long-form report skeleton from whatever artifacts are currently present."""
    ctx = _load_report_context(output_dir)
    reports = ctx.reports_dir
    manifest = ctx.manifest
    benchmark_sweep = ctx.benchmark_sweep
    benchmark_best = ctx.benchmark_best
    benchmark_summary = ctx.benchmark_summary
    benchmark_ablation = ctx.benchmark_ablation
    feature_report = ctx.feature_report
    benchmark_tuned_search = ctx.benchmark_tuned_search
    benchmark_tuned_best = ctx.benchmark_tuned_best
    benchmark_tuned_summary = ctx.benchmark_tuned_summary
    benchmark_tuned_ablation = ctx.benchmark_tuned_ablation
    benchmark_tuned_feature_report = ctx.benchmark_tuned_feature_report
    temporal_selection = ctx.temporal_selection
    temporal_lockbox = ctx.temporal_lockbox
    temporal_drift = ctx.temporal_drift
    temporal_mspc = ctx.temporal_mspc
    temporal_manager = ctx.temporal_manager
    temporal_cost = ctx.temporal_cost
    best_benchmark_row = ctx.best_benchmark_row
    best_tuned_benchmark_row = ctx.best_tuned_benchmark_row
    modal_tuned_config_row = ctx.modal_tuned_config_row
    primary_temporal_row = ctx.primary_temporal_row
    primary_scientific_lockbox_row = ctx.primary_scientific_lockbox_row
    drift_row = ctx.drift_row
    mspc_lockbox_row = ctx.mspc_lockbox_row

    # Skeleton text intentionally keeps prompts and placeholder guidance for human completion.
    lines: list[str] = []
    lines.append("# Final Report Skeleton")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    restrictions = manifest.get("temporal_claim_restrictions", [])
    lines.append(
        "- The benchmark studies support a credible yield-prediction signal in the SECOM sensor/process measurements."
    )
    lines.append(f"- Temporal claim restrictions: `{len(restrictions)}`")
    if best_benchmark_row is not None:
        lines.append(
            "- Leading original replication configuration:"
            f" `{best_benchmark_row['selector']}` / `{best_benchmark_row['classifier']}` /"
            f" `{best_benchmark_row['replication_mode']}` with mean BER `{_format_float(best_benchmark_row['mean_BER'])}`"
        )
    if best_tuned_benchmark_row is not None:
        lines.append(
            "- Leading tuned benchmark configuration:"
            f" `{best_tuned_benchmark_row['selector']}` / `{best_tuned_benchmark_row['classifier']}` /"
            f" `{best_tuned_benchmark_row['replication_mode']}` with mean BER `{_format_float(best_tuned_benchmark_row['mean_BER'])}`"
            f" and mean ROC_AUC `{_format_float(best_tuned_benchmark_row['mean_ROC_AUC'])}`"
        )
        lines.append(
            "- The tuned benchmark uses nested CV, so it should be read as the stricter and more conservative estimate of tuned-model performance."
        )
    if modal_tuned_config_row is not None:
        lines.append(
            "- Most frequently selected tuned configuration:"
            f" `{modal_tuned_config_row['selector']}` / `{modal_tuned_config_row['classifier']}` /"
            f" `{modal_tuned_config_row['replication_mode']}` with `k={int(modal_tuned_config_row['k'])}`"
            f" and selection_count `{int(modal_tuned_config_row['selection_count'])}`"
        )
    if primary_temporal_row is not None:
        lines.append(
            "- Temporal primary selector:"
            f" `{primary_temporal_row['selector']}` with mean BER `{_format_float(primary_temporal_row['mean_BER'])}`"
        )
        lines.append(
            "- Temporal robustness should be treated as a stress test of stability under chronological shift, not as the primary source of project success."
        )
    lines.append(f"- Primary study status: `{manifest.get('primary_study_status', StudyStatus.NOT_RUN)}`")
    lines.append(f"- Original replication status: `{manifest.get('benchmark_original_status', StudyStatus.NOT_RUN)}`")
    lines.append(f"- Tuned benchmark status: `{manifest.get('benchmark_tuned_status', StudyStatus.NOT_RUN)}`")
    lines.append(f"- Temporal robustness status: `{manifest.get('temporal_robustness_status', StudyStatus.NOT_RUN)}`")
    lines.append("")
    lines.append("## Dataset and Study Scope")
    lines.append("")
    lines.append(
        "Summarize the SECOM benchmark context, the primary replication objective, and the role of temporal robustness as secondary evidence."
    )
    lines.append("")
    lines.append("## Benchmark Replication Design")
    lines.append("")
    lines.append(
        "Describe the full-dataset replication protocol, in-fold preprocessing, in-fold feature selection, and missing-indicator ablation."
    )
    lines.append("")
    lines.append("## Benchmark Replication Results")
    lines.append("")
    lines.append("### Original Replication")
    lines.append("")
    lines.append("#### Original Replication Design")
    lines.append("")
    lines.append("- Use a fixed feature budget with the literature-style selector/classifier comparison.")
    lines.append("- Perform preprocessing and feature selection inside each training fold only.")
    lines.append(
        "- Select original classifier configurations from the non-nested replication sweep; use the tuned benchmark for the stricter nested-CV estimate."
    )
    lines.append("- Treat the missing-indicator comparison as a paired benchmark condition.")
    lines.append(
        "- Report final thresholded results through `BER`, `TPR`, and `TNR`, with supporting metrics shown separately."
    )
    lines.append("")
    lines.append("#### Original Replication Search Summary")
    lines.append("")
    if benchmark_sweep is not None and not benchmark_sweep.empty:
        lines.append("##### Search Space")
        lines.append("")
        lines.extend(_original_search_space_table(benchmark_sweep))
        lines.append("")
    else:
        lines.append("- Benchmark sweep artifact missing or empty.")
        lines.append("")
    if benchmark_best is not None and not benchmark_best.empty:
        lines.append("##### Selected Configurations")
        lines.append("")
        lines.extend(_original_best_config_table(benchmark_best))
        lines.append("")
    else:
        lines.append("- Benchmark best-config artifact missing or empty.")
        lines.append("")
    lines.append("#### Original Replication Results")
    lines.append("")
    if best_benchmark_row is not None:
        lines.append("##### Primary Evidence Table")
        lines.append("")
        lines.extend(_top_benchmark_table(benchmark_summary))
        lines.append("")
        lines.append("##### UCI Original Benchmark Reference")
        lines.append("")
        lines.append(
            "The UCI SECOM reference table reports 40-feature selector results with a simple kernel-ridge classifier and 10-fold cross-validation. Local columns use the strict original-replication KRR row when available."
        )
        lines.append("")
        lines.extend(_uci_original_baseline_table(benchmark_summary))
        lines.append("")
        lines.append(_uci_selector_definition_note())
        lines.append("")
        lines.append("##### Supporting Benchmark Metrics")
        lines.append("")
        lines.extend(_supporting_benchmark_table(benchmark_summary))
        lines.append("")
        lines.append("##### Lead Configuration")
        lines.append("")
        lines.append(
            f"- Selector: `{best_benchmark_row['selector']}`\n"
            f"- Classifier: `{best_benchmark_row['classifier']}`\n"
            f"- Replication mode: `{best_benchmark_row['replication_mode']}`\n"
            f"- Mean BER: `{_format_float(best_benchmark_row['mean_BER'])}`\n"
            f"- 95% fold-bootstrap CI: `{_format_float(best_benchmark_row['CI_lower_BER'])}` to `{_format_float(best_benchmark_row['CI_upper_BER'])}`\n"
            f"- Mean ROC_AUC: `{_format_float(best_benchmark_row['mean_ROC_AUC'])}`\n"
            f"- Mean PR_AUC: `{_format_float(best_benchmark_row['mean_PR_AUC'])}`\n"
            f"- Mean MCC: `{_format_float(best_benchmark_row['mean_MCC'])}`\n"
            f"- Mean F2: `{_format_float(best_benchmark_row['mean_F2'])}`"
        )
    else:
        lines.append("- Benchmark summary artifact missing or empty.")
    if benchmark_ablation is not None and not benchmark_ablation.empty:
        lines.append("")
        lines.append("- Original missing-indicator ablation summary:")
        lines.extend(
            f"  - {row.selector} / {row.classifier}: delta_BER={_format_float(row.delta_BER)}"
            for row in benchmark_ablation.itertuples(index=False)
        )
    lines.append("")
    lines.append("### Tuned Benchmark")
    lines.append("")
    lines.append("#### Tuned Benchmark Design")
    lines.append("")
    lines.append("- Use nested cross-validation inside each outer replication fold.")
    lines.append(
        "- Tune selector parameters, including feature budget `k` and `ReliefF` neighbor count where applicable."
    )
    lines.append("- Tune classifier parameters within the same nested search.")
    lines.append(
        "- Use `ROC_AUC` as the threshold-free inner selection objective, with `BER` as the secondary tie-break metric."
    )
    lines.append(
        "- Report final tuned benchmark performance with thresholded `BER`, `TPR`, and `TNR` plus supporting metrics."
    )
    lines.append("")
    lines.append("#### Tuned Benchmark Search Summary")
    lines.append("")
    if benchmark_tuned_search is not None and not benchmark_tuned_search.empty:
        lines.append("##### Search Space")
        lines.append("")
        lines.extend(_tuned_search_space_table(benchmark_tuned_search))
        lines.append("")
    else:
        lines.append("- Tuned search artifact missing or empty.")
        lines.append("")
    if benchmark_tuned_best is not None and not benchmark_tuned_best.empty:
        lines.append("##### Modal Selected Configurations")
        lines.append("")
        lines.extend(_tuned_best_config_table(benchmark_tuned_best))
        lines.append("")
    else:
        lines.append("- Tuned best-config artifact missing or empty.")
        lines.append("")
    lines.append("#### Tuned Benchmark Results")
    lines.append("")
    if best_tuned_benchmark_row is not None:
        lines.append("##### Primary Evidence Table")
        lines.append("")
        lines.extend(_top_benchmark_table(benchmark_tuned_summary))
        lines.append("")
        lines.append("##### Supporting Benchmark Metrics")
        lines.append("")
        lines.extend(_supporting_benchmark_table(benchmark_tuned_summary))
        lines.append("")
        lines.append("##### Lead Configuration")
        lines.append("")
        lines.append(
            f"- Selector: `{best_tuned_benchmark_row['selector']}`\n"
            f"- Classifier: `{best_tuned_benchmark_row['classifier']}`\n"
            f"- Replication mode: `{best_tuned_benchmark_row['replication_mode']}`\n"
            f"- Mean BER: `{_format_float(best_tuned_benchmark_row['mean_BER'])}`\n"
            f"- 95% fold-bootstrap CI: `{_format_float(best_tuned_benchmark_row['CI_lower_BER'])}` to `{_format_float(best_tuned_benchmark_row['CI_upper_BER'])}`\n"
            f"- Mean ROC_AUC: `{_format_float(best_tuned_benchmark_row['mean_ROC_AUC'])}`\n"
            f"- Mean PR_AUC: `{_format_float(best_tuned_benchmark_row['mean_PR_AUC'])}`\n"
            f"- Mean MCC: `{_format_float(best_tuned_benchmark_row['mean_MCC'])}`\n"
            f"- Mean F2: `{_format_float(best_tuned_benchmark_row['mean_F2'])}`"
        )
    else:
        lines.append("- Tuned benchmark summary artifact missing or empty.")
    if benchmark_tuned_ablation is not None and not benchmark_tuned_ablation.empty:
        lines.append("")
        lines.append("##### Missing-Indicator Ablation Summary")
        lines.extend(
            f"  - {row.selector} / {row.classifier}: delta_BER={_format_float(row.delta_BER)}"
            for row in benchmark_tuned_ablation.itertuples(index=False)
        )
    if best_tuned_benchmark_row is not None:
        lines.append("")
        lines.append("##### Interpretation")
        lines.append(
            "- The tuned benchmark is stricter than the original replication because hyperparameters are chosen inside nested CV rather than on the same folds used for final reporting."
        )
        lines.append(
            f"- The best tuned mean-BER row is `{best_tuned_benchmark_row['selector']}` / `{best_tuned_benchmark_row['classifier']}` / `{best_tuned_benchmark_row['replication_mode']}`."
        )
        if modal_tuned_config_row is not None:
            lines.append(
                f"- The most frequently selected tuned configuration is `{modal_tuned_config_row['selector']}` / `{modal_tuned_config_row['classifier']}` / `{modal_tuned_config_row['replication_mode']}`"
                f" with `k={int(modal_tuned_config_row['k'])}` and selection_count=`{int(modal_tuned_config_row['selection_count'])}`."
            )
            if (
                str(modal_tuned_config_row["selector"]) != str(best_tuned_benchmark_row["selector"])
                or str(modal_tuned_config_row["classifier"]) != str(best_tuned_benchmark_row["classifier"])
                or str(modal_tuned_config_row["replication_mode"]) != str(best_tuned_benchmark_row["replication_mode"])
            ):
                lines.append(
                    "- The best mean-BER tuned row and the modal tuned configuration differ, so the report should distinguish peak performance from stability of selection."
                )
    if best_benchmark_row is not None and best_tuned_benchmark_row is not None:
        lines.append("")
        lines.append("### Original vs Tuned Benchmark Comparison")
        lines.append("")
        comparison_df = pd.DataFrame(
            [
                {
                    "study": "original",
                    "selector": best_benchmark_row["selector"],
                    "classifier": best_benchmark_row["classifier"],
                    "mode": best_benchmark_row["replication_mode"],
                    "mean_BER": best_benchmark_row["mean_BER"],
                    "mean_ROC_AUC": best_benchmark_row["mean_ROC_AUC"],
                    "mean_PR_AUC": best_benchmark_row["mean_PR_AUC"],
                    "mean_MCC": best_benchmark_row["mean_MCC"],
                    "mean_F2": best_benchmark_row["mean_F2"],
                },
                {
                    "study": "tuned",
                    "selector": best_tuned_benchmark_row["selector"],
                    "classifier": best_tuned_benchmark_row["classifier"],
                    "mode": best_tuned_benchmark_row["replication_mode"],
                    "mean_BER": best_tuned_benchmark_row["mean_BER"],
                    "mean_ROC_AUC": best_tuned_benchmark_row["mean_ROC_AUC"],
                    "mean_PR_AUC": best_tuned_benchmark_row["mean_PR_AUC"],
                    "mean_MCC": best_tuned_benchmark_row["mean_MCC"],
                    "mean_F2": best_tuned_benchmark_row["mean_F2"],
                },
            ]
        )
        lines.extend(
            _markdown_table(
                comparison_df,
                [
                    "study",
                    "selector",
                    "classifier",
                    "mode",
                    "mean_BER",
                    "mean_ROC_AUC",
                    "mean_PR_AUC",
                    "mean_MCC",
                    "mean_F2",
                ],
            )
        )
        lines.append("")
        ber_delta = float(best_tuned_benchmark_row["mean_BER"]) - float(best_benchmark_row["mean_BER"])
        if ber_delta > 0:
            lines.append(
                f"- The tuned benchmark is worse by `{_format_float(ber_delta)}` BER relative to the best original replication row, which is consistent with the stricter nested-CV evaluation protocol."
            )
        elif ber_delta < 0:
            lines.append(
                f"- The tuned benchmark improves on the best original replication row by `{_format_float(abs(ber_delta))}` BER."
            )
        else:
            lines.append("- The best original and tuned benchmark rows are tied on mean BER.")
        lines.append(
            "- Original replication should be read as the benchmark-facing result, while the tuned benchmark is the more conservative estimate of what a tuned procedure achieves on unseen folds."
        )
    lines.append("")
    lines.append("## Feature Stability and Interpretation")
    lines.append("")
    lines.append("### Original Replication")
    lines.append("")
    lines.append("#### Original Feature Stability and Interpretation")
    lines.append("")
    if best_benchmark_row is not None and feature_report is not None and not feature_report.empty:
        lines.extend(
            _best_row_feature_table(
                feature_report=feature_report,
                selector=str(best_benchmark_row["selector"]),
                classifier=str(best_benchmark_row["classifier"]),
                replication_mode=str(best_benchmark_row["replication_mode"]),
            )
        )
    else:
        lines.append("- Feature report artifact missing or empty.")
    lines.append("")
    lines.append("### Tuned Benchmark")
    lines.append("")
    lines.append("#### Tuned Feature Stability and Interpretation")
    lines.append("")
    if (
        best_tuned_benchmark_row is not None
        and benchmark_tuned_feature_report is not None
        and not benchmark_tuned_feature_report.empty
    ):
        lines.extend(
            _best_row_feature_table(
                feature_report=benchmark_tuned_feature_report,
                selector=str(best_tuned_benchmark_row["selector"]),
                classifier=str(best_tuned_benchmark_row["classifier"]),
                replication_mode=str(best_tuned_benchmark_row["replication_mode"]),
            )
        )
    else:
        lines.append("- Tuned feature report artifact missing or empty.")
    lines.append("")
    lines.append("## Temporal Robustness Stress Test")
    lines.append("")
    lines.append("### Temporal Robustness Design")
    lines.append("")
    lines.append(
        "- This secondary study stress-tests whether the benchmark findings remain stable under chronological validation and future-looking holdout evaluation."
    )
    lines.append(
        "- Use a chronological DEV/LOCKBOX split with time-aware folds inside DEV, followed by selector screening, temporal model selection, config freeze, threshold freeze, lockbox evaluation, drift gating, and MSPC comparison."
    )
    lines.append(
        "- Interpret this section as secondary robustness evidence rather than the primary basis for project success."
    )
    lines.append("")
    lines.append("### Temporal Model Selection Summary")
    lines.append("")
    if primary_temporal_row is not None:
        lines.append(
            "- Primary temporal selector under the temporal protocol:"
            f" `{primary_temporal_row['selector']}`"
            f" with mean_BER=`{_format_float(primary_temporal_row['mean_BER'])}`"
        )
    if temporal_selection is not None and not temporal_selection.empty:
        challenger_rows = temporal_selection[temporal_selection["is_challenger"].astype(bool)]
        if not challenger_rows.empty:
            challenger_row = challenger_rows.iloc[0]
            lines.append(
                "- Challenger selector retained for secondary comparison:"
                f" `{challenger_row['selector']}`"
                f" with mean_BER=`{_format_float(challenger_row['mean_BER'])}`"
            )
        else:
            lines.append("- No challenger met the temporal eligibility rule.")
        lines.append("")
        lines.append("#### Selector Ranking and Modal Configurations")
        lines.append("")
        lines.extend(_temporal_selection_summary_table(temporal_selection))
    else:
        lines.append("- Temporal model selection artifact missing or empty.")
    if temporal_lockbox is not None and not temporal_lockbox.empty:
        lines.append("")
        lines.append("### Temporal Lockbox Results")
        lines.append("")
        lines.extend(
            _markdown_table(
                temporal_lockbox.sort_values(["role", "threshold_policy"]),
                [
                    "role",
                    "threshold_policy",
                    "BER",
                    "True+",
                    "True-",
                    "ROC_AUC",
                    "PR_AUC",
                    "MCC",
                    "F2",
                    "threshold_at_TNR90",
                    "TNR_at_TNR90",
                    "TPR_at_TNR90",
                ],
            )
        )
    if drift_row is not None or restrictions:
        lines.append("")
        lines.append("### Drift and Claim Restrictions")
        lines.append("")
        if drift_row is not None:
            lines.append(
                "- Primary drift status:"
                f" `{drift_row['drift_gate_status']}`"
                f" with max_PSI=`{_format_float(drift_row['max_PSI'])}`"
            )
            lines.append("")
            drift_columns = [
                col
                for col in [
                    "model_scope",
                    "drift_gate_status",
                    "lockbox_claims_allowed",
                    "abs_prevalence_shift",
                    "ks_pvalue_scores",
                    "max_PSI",
                ]
                if col in temporal_drift.columns
            ]
            lines.extend(_markdown_table(temporal_drift.sort_values("model_scope"), drift_columns))
            if primary_scientific_lockbox_row is not None and "lockbox_fails" in primary_scientific_lockbox_row.index:
                lockbox_fails = int(primary_scientific_lockbox_row["lockbox_fails"])
                lines.append("")
                lines.append(
                    f"- The lockbox contains only `{lockbox_fails}` failing samples, so recall-oriented quantities such as `TPR` are inherently unstable in this holdout."
                )
                if lockbox_fails > 0:
                    lines.append(
                        f"- With `{lockbox_fails}` failing samples, each additional captured or missed fail changes `TPR` by roughly `{(1.0 / lockbox_fails):.3f}`."
                    )
                lines.append(
                    "- Even so, the primary restriction in this run is the `HIGH_SHIFT` drift result rather than sample count alone, so the temporal lockbox should be treated as descriptive stress-test evidence."
                )
        if restrictions:
            lines.append("")
            lines.append("- Temporal claim restrictions:")
            lines.extend(f"  - `{restriction}`" for restriction in restrictions)
            lines.append(
                "- Lockbox evidence remains reportable, but restricted claims should be treated as descriptive rather than confirmatory."
            )
        else:
            lines.append("- Temporal claim restrictions: `none`")
            lines.append(
                "- Temporal lockbox evidence can be interpreted directly within the limits of this secondary study."
            )
    if mspc_lockbox_row is not None:
        lines.append("")
        lines.append("### MSPC Comparison")
        lines.append("")
        lines.extend(
            _markdown_table(
                temporal_mspc[temporal_mspc["eval_scope"] == "lockbox"],
                [
                    "eval_scope",
                    "best_MSPC_source",
                    "best_MSPC_TPR_at_TNR90",
                    "T2_AUC",
                    "Q_AUC",
                    "alarm_rate",
                    "empirical_ARL0",
                ],
                headers=[
                    "scope",
                    "best_source",
                    "best_TPR_at_TNR90",
                    "T2_AUC",
                    "Q_AUC",
                    "alarm_rate",
                    "empirical_ARL0",
                ],
            )
        )
        if primary_scientific_lockbox_row is not None:
            lines.append("")
            lines.append(
                f"- The supervised primary model reaches `TPR_at_TNR90={_format_float(primary_scientific_lockbox_row['TPR_at_TNR90'])}`, versus `best_MSPC_TPR_at_TNR90={_format_float(mspc_lockbox_row['best_MSPC_TPR_at_TNR90'])}` for MSPC."
            )
            if restrictions:
                lines.append(
                    "- That numerical advantage remains descriptive only because the drift gate restricts lockbox superiority claims in this run."
                )
    if (temporal_manager is not None and not temporal_manager.empty) or (
        temporal_cost is not None and not temporal_cost.empty
    ):
        lines.append("")
        lines.append("### Illustrative Operational Framing")
        lines.append("")
        lines.append(
            "- These workload and cost summaries are illustrative consequences of the temporal thresholds, not production-validated operating recommendations."
        )
    if temporal_manager is not None and not temporal_manager.empty:
        lines.append("#### Workload Metrics")
        lines.append("")
        lines.extend(
            _markdown_table(
                temporal_manager.sort_values(["role", "threshold_policy"]),
                [
                    "role",
                    "threshold_policy",
                    "predicted_flag_fraction",
                    "mean_weekly_flagged_wafers",
                    "mean_weekly_fail_captures",
                    "mean_weekly_fail_misses",
                ],
            )
        )
    if temporal_cost is not None and not temporal_cost.empty:
        lines.append("")
        lines.append("#### Cost Curves")
        lines.append("")
        cost_columns = [col for col in temporal_cost.columns if not temporal_cost[col].isna().all()]
        lines.extend(
            _markdown_table(
                temporal_cost.sort_values("cost_ratio"),
                cost_columns,
            )
        )
    lines.append("")
    lines.append("Interpret this section as robustness evidence, not as the primary basis for project success.")
    lines.append("")
    lines.append("## Industrialization Gaps")
    lines.append("")
    lines.append("- No stable device/tool/chamber identifier for unseen-device validation.")
    lines.append("- No intervention or maintenance history.")
    lines.append("- No explicit regime-change metadata.")
    lines.append("- Anonymous features limit process interpretation.")
    lines.append("- Operational framing in this report is illustrative, not production-validated.")
    lines.append("")
    lines.append("## Conclusions and Next Data Requirements")
    lines.append("")
    if best_benchmark_row is not None:
        lines.append(
            f"- The original replication study shows that upstream process measurements contain useful signal for downstream fail detection, with the best original row at mean BER `{_format_float(best_benchmark_row['mean_BER'])}`."
        )
    if best_tuned_benchmark_row is not None:
        lines.append(
            f"- The tuned benchmark provides a stricter nested-CV estimate, with the best tuned row at mean BER `{_format_float(best_tuned_benchmark_row['mean_BER'])}`."
        )
    if restrictions:
        lines.append(
            "- The temporal stress test remains informative, but the current run is limited by both a small lockbox fail count and a `HIGH_SHIFT` drift result, so lockbox superiority claims should remain descriptive only."
        )
    else:
        lines.append(
            "- The temporal stress test provides secondary robustness evidence without active claim restrictions in this run."
        )
    lines.append(
        "- A true industrial deployment study would still require stable device or tool identifiers, intervention history, and richer process metadata."
    )
    lines.append("")

    out_path = reports / ArtifactName.REPORT_SKELETON
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def write_final_report(output_dir: Path, *, export_pdf: bool = False) -> Path:
    """Write the final Markdown report and optionally export it through pandoc."""
    _raise_for_failed_report_audit(output_dir)
    ctx = _load_report_context(output_dir)
    restrictions = list(ctx.manifest.get("temporal_claim_restrictions", []))
    _write_final_report_figures(ctx)

    # Final report text is artifact-driven and avoids asking readers to inspect raw CSVs first.
    lines: list[str] = []
    lines.append("# SECOM Benchmark-First Yield Monitoring Study")
    lines.append("")
    lines.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_")
    lines.append(f"_Source run: `{output_dir}`_")
    lines.append("")
    lines.append(
        "This report summarizes the benchmark replication, tuned benchmark, and temporal robustness outputs "
        "from the active SECOM study artifacts. The benchmark results are the primary evidence; the temporal "
        "study is a stricter stress test of robustness under chronological shift."
    )
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    _append_bullet_list(
        lines,
        [
            "The benchmark studies support a credible yield-prediction signal in the SECOM sensor and process measurements.",
            (
                f"The strongest original replication row is `{ctx.best_benchmark_row['selector']}` / "
                f"`{ctx.best_benchmark_row['classifier']}` / `{ctx.best_benchmark_row['replication_mode']}` "
                f"with mean BER `{_format_float(ctx.best_benchmark_row['mean_BER'])}`."
                if ctx.best_benchmark_row is not None
                else "Original replication evidence is unavailable."
            ),
            (
                f"The strongest tuned benchmark row is `{ctx.best_tuned_benchmark_row['selector']}` / "
                f"`{ctx.best_tuned_benchmark_row['classifier']}` / `{ctx.best_tuned_benchmark_row['replication_mode']}` "
                f"with mean BER `{_format_float(ctx.best_tuned_benchmark_row['mean_BER'])}`."
                if ctx.best_tuned_benchmark_row is not None
                else "Tuned benchmark evidence is unavailable."
            ),
            "The tuned benchmark should be read as the more conservative estimate because hyperparameters are selected inside nested cross-validation.",
            (
                f"The temporal study selected `{ctx.primary_temporal_row['selector']}` as the primary chronological candidate."
                if ctx.primary_temporal_row is not None
                else "The temporal study did not identify a primary chronological candidate."
            ),
            (
                f"There are `{len(restrictions)}` active temporal claim restriction(s); temporal lockbox findings remain descriptive rather than confirmatory."
                if restrictions
                else "There are no active temporal claim restrictions in this run."
            ),
        ],
    )
    lines.append("")
    lines.append("## What I Built")
    lines.append("")
    _append_bullet_list(
        lines,
        [
            "A reproducible original benchmark replication workflow that keeps preprocessing and feature selection strictly inside the training folds.",
            "A tuned benchmark workflow that preserves the selector family while adding nested hyperparameter search and threshold-free inner selection.",
            "A temporal robustness workflow with chronological DEV/LOCKBOX evaluation, drift gating, and explicit claim restrictions.",
            "Artifact-driven audit and reporting outputs so results can be traced back to versioned manifests, metrics tables, and study statuses.",
        ],
    )
    lines.append("")
    lines.append("## Dataset and Study Scope")
    lines.append("")
    lines.append(
        "The active study is intentionally benchmark-first. It asks whether SECOM process measurements contain usable "
        "signal for downstream fail detection under a faithful literature-style protocol, then whether a stricter tuned "
        "benchmark changes that conclusion, and finally how those findings behave under future-looking temporal stress. "
        "This ordering matters: the benchmark studies support the core claim, while the temporal study tests robustness "
        "without being allowed to erase valid benchmark evidence by default."
    )
    lines.append("")
    lines.append("## Original Replication Design")
    lines.append("")
    lines.append(
        "The original replication keeps a fixed feature budget, compares the literature-style selector and classifier families, "
        "and treats missing-indicator features as a paired ablation. The key result is not just the best row, but the fact "
        "that multiple selector/classifier combinations remain materially better than trivial failure detection."
    )
    lines.append(
        "Original classifier configurations are selected from the same non-nested replication sweep used for reporting, so tuned benchmark results remain the stricter estimate."
    )
    lines.append("")
    lines.append("## Original Replication Search Summary")
    lines.append("")
    if ctx.benchmark_sweep is not None and not ctx.benchmark_sweep.empty:
        lines.append("### Original Search Space")
        lines.append("")
        lines.extend(_original_search_space_table(ctx.benchmark_sweep))
        lines.append("")
    else:
        lines.append("- Benchmark sweep artifact missing or empty.")
        lines.append("")
    if ctx.benchmark_best is not None and not ctx.benchmark_best.empty:
        lines.append("### Original Selected Configurations")
        lines.append("")
        lines.extend(_original_best_config_table(ctx.benchmark_best))
        lines.append("")
    else:
        lines.append("- Benchmark best-config artifact missing or empty.")
        lines.append("")
    lines.append("## Original Replication Results")
    lines.append("")
    _append_benchmark_summary_table(lines, "### Primary Benchmark Evidence", ctx.benchmark_summary)
    lines.append("### UCI Original Benchmark Reference")
    lines.append("")
    lines.append(
        "The UCI SECOM reference table reports 40-feature selector results with a simple kernel-ridge classifier and 10-fold cross-validation. Local columns use the strict original-replication KRR row when available."
    )
    lines.append("")
    lines.extend(_uci_original_baseline_table(ctx.benchmark_summary))
    lines.append("")
    lines.append(_uci_selector_definition_note())
    lines.append("")
    _append_supporting_metrics_table(lines, "### Supporting Benchmark Metrics", ctx.benchmark_summary)
    _append_figure(
        lines,
        "Benchmark comparison",
        "figures/benchmark_comparison.png",
        "Figure 1 shows the strongest original and tuned benchmark rows by mean BER, with uncertainty bars where the benchmark summaries expose fold-bootstrap confidence intervals.",
    )
    if ctx.benchmark_ablation is not None and not ctx.benchmark_ablation.empty:
        lines.append("### Missing-Indicator Ablation")
        lines.append("")
        lines.extend(
            f"- `{row.selector}` / `{row.classifier}` changes mean BER by `{_format_float(row.delta_BER)}` "
            "when missing indicators are added."
            for row in ctx.benchmark_ablation.itertuples(index=False)
        )
        lines.append("")
    lines.append("## Tuned Benchmark Design")
    lines.append("")
    lines.append(
        "The tuned benchmark tightens methodology by moving model and selector choices inside nested cross-validation. "
        "That makes the tuned results a better estimate of what a disciplined tuning process achieves on unseen folds, "
        "even when the headline BER ends up slightly worse than the best original replication row."
    )
    lines.append("")
    lines.append("## Tuned Benchmark Search Summary")
    lines.append("")
    if ctx.benchmark_tuned_search is not None and not ctx.benchmark_tuned_search.empty:
        lines.append("### Tuned Search Space")
        lines.append("")
        lines.extend(_tuned_search_space_table(ctx.benchmark_tuned_search))
        lines.append("")
    else:
        lines.append("- Tuned search artifact missing or empty.")
        lines.append("")
    if ctx.benchmark_tuned_best is not None and not ctx.benchmark_tuned_best.empty:
        lines.append("### Modal Selected Configurations")
        lines.append("")
        lines.extend(_tuned_best_config_table(ctx.benchmark_tuned_best))
        lines.append("")
    else:
        lines.append("- Tuned best-config artifact missing or empty.")
        lines.append("")
    lines.append("## Tuned Benchmark Results")
    lines.append("")
    _append_benchmark_summary_table(lines, "### Primary Tuned Evidence", ctx.benchmark_tuned_summary)
    _append_supporting_metrics_table(lines, "### Supporting Tuned Metrics", ctx.benchmark_tuned_summary)
    if ctx.modal_tuned_config_row is not None:
        lines.append("### Tuned Selection Stability")
        lines.append("")
        lines.append(
            f"- The most frequently selected tuned configuration is `{ctx.modal_tuned_config_row['selector']}` / "
            f"`{ctx.modal_tuned_config_row['classifier']}` / `{ctx.modal_tuned_config_row['replication_mode']}` "
            f"with `k={int(ctx.modal_tuned_config_row['k'])}` and selection count `{int(ctx.modal_tuned_config_row['selection_count'])}`."
        )
        lines.append("")
    lines.append("## Original vs Tuned Comparison")
    lines.append("")
    if ctx.best_benchmark_row is not None and ctx.best_tuned_benchmark_row is not None:
        comparison_df = pd.DataFrame(
            [
                {
                    "study": "original",
                    "selector": ctx.best_benchmark_row["selector"],
                    "classifier": ctx.best_benchmark_row["classifier"],
                    "mode": ctx.best_benchmark_row["replication_mode"],
                    "mean_BER": ctx.best_benchmark_row["mean_BER"],
                    "mean_ROC_AUC": ctx.best_benchmark_row["mean_ROC_AUC"],
                },
                {
                    "study": "tuned",
                    "selector": ctx.best_tuned_benchmark_row["selector"],
                    "classifier": ctx.best_tuned_benchmark_row["classifier"],
                    "mode": ctx.best_tuned_benchmark_row["replication_mode"],
                    "mean_BER": ctx.best_tuned_benchmark_row["mean_BER"],
                    "mean_ROC_AUC": ctx.best_tuned_benchmark_row["mean_ROC_AUC"],
                },
            ]
        )
        lines.extend(
            _markdown_table(
                comparison_df,
                ["study", "selector", "classifier", "mode", "mean_BER", "mean_ROC_AUC"],
            )
        )
        lines.append("")
        ber_delta = float(ctx.best_tuned_benchmark_row["mean_BER"]) - float(ctx.best_benchmark_row["mean_BER"])
        if ber_delta > 0:
            lines.append(
                f"- Relative to the best original replication row, the tuned benchmark is worse by `{_format_float(ber_delta)}` BER. "
                "That is consistent with the stricter nested-CV evaluation protocol."
            )
        elif ber_delta < 0:
            lines.append(
                f"- Relative to the best original replication row, the tuned benchmark improves BER by `{_format_float(abs(ber_delta))}`."
            )
        else:
            lines.append("- The best original and tuned benchmark rows are tied on mean BER.")
    else:
        lines.append("- Benchmark comparison is unavailable because one of the benchmark summaries is missing.")
    lines.append("")
    _append_figure(
        lines,
        "Tuned vs original BER delta",
        "figures/tuned_vs_original_delta.png",
        "Figure 2 highlights how much stricter nested cross-validation changes BER for matched selector/classifier/mode configurations.",
    )
    lines.append("## Feature Stability and Interpretation")
    lines.append("")
    lines.append(_feature_interpretation_claim_note())
    lines.append("")
    lines.append("### Original Replication")
    lines.append("")
    if ctx.best_benchmark_row is not None and ctx.feature_report is not None and not ctx.feature_report.empty:
        lines.extend(
            _best_row_feature_table(
                feature_report=ctx.feature_report,
                selector=str(ctx.best_benchmark_row["selector"]),
                classifier=str(ctx.best_benchmark_row["classifier"]),
                replication_mode=str(ctx.best_benchmark_row["replication_mode"]),
            )
        )
    else:
        lines.append("- Original feature report artifact missing or empty.")
    lines.append("")
    lines.append("### Tuned Benchmark")
    lines.append("")
    if (
        ctx.best_tuned_benchmark_row is not None
        and ctx.benchmark_tuned_feature_report is not None
        and not ctx.benchmark_tuned_feature_report.empty
    ):
        lines.extend(
            _best_row_feature_table(
                feature_report=ctx.benchmark_tuned_feature_report,
                selector=str(ctx.best_tuned_benchmark_row["selector"]),
                classifier=str(ctx.best_tuned_benchmark_row["classifier"]),
                replication_mode=str(ctx.best_tuned_benchmark_row["replication_mode"]),
            )
        )
    else:
        lines.append("- Tuned feature report artifact missing or empty.")
    lines.append("")
    _append_figure(
        lines,
        "Feature stability",
        "figures/feature_stability.png",
        "Figure 3 summarizes benchmark feature-prioritization evidence across the benchmark studies, while preserving the distinction between raw value features and missing indicators.",
    )
    lines.append("## Temporal Robustness Stress Test")
    lines.append("")
    lines.append(
        "The temporal study is a deployment-like stress test rather than the source of the project’s primary success claim. "
        "It uses a chronological DEV/LOCKBOX split, time-aware model selection, threshold freeze, drift checks, and an MSPC comparison."
    )
    lines.append("")
    lines.append("### Temporal Model Selection Summary")
    lines.append("")
    if ctx.primary_temporal_row is not None:
        lines.append(
            "- Primary temporal selector under the temporal protocol:"
            f" `{ctx.primary_temporal_row['selector']}` with mean_BER=`{_format_float(ctx.primary_temporal_row['mean_BER'])}`."
        )
    if ctx.temporal_selection is not None and not ctx.temporal_selection.empty:
        challenger_rows = ctx.temporal_selection[ctx.temporal_selection["is_challenger"].astype(bool)]
        if not challenger_rows.empty:
            challenger_row = challenger_rows.iloc[0]
            lines.append(
                "- Challenger selector retained for secondary comparison:"
                f" `{challenger_row['selector']}` with mean_BER=`{_format_float(challenger_row['mean_BER'])}`."
            )
        else:
            lines.append("- No challenger met the temporal eligibility rule.")
        lines.append("")
        lines.append("#### Selector Ranking and Modal Configurations")
        lines.append("")
        lines.extend(_temporal_selection_summary_table(ctx.temporal_selection))
        lines.append("")
    else:
        lines.append("- Temporal model selection artifact missing or empty.")
        lines.append("")
    if ctx.drift_row is not None:
        lines.append(
            f"- The current temporal run is drift-gated as `{ctx.drift_row['drift_gate_status']}` with max PSI `{_format_float(ctx.drift_row['max_PSI'])}`."
        )
    if restrictions:
        lines.append("- Active temporal claim restrictions:")
        lines.extend(f"  - `{restriction}`" for restriction in restrictions)
    else:
        lines.append("- No temporal claim restrictions are active in this run.")
    lines.append("")
    if ctx.temporal_lockbox is not None and not ctx.temporal_lockbox.empty:
        lines.append("### Lockbox Metrics")
        lines.append("")
        lines.extend(
            _markdown_table(
                ctx.temporal_lockbox.sort_values(["role", "threshold_policy"]),
                [
                    "role",
                    "threshold_policy",
                    "BER",
                    "True+",
                    "True-",
                    "ROC_AUC",
                    "PR_AUC",
                    "MCC",
                    "F2",
                ],
            )
        )
        lines.append("")
    if ctx.mspc_lockbox_row is not None:
        lines.append("### Supervised vs MSPC")
        lines.append("")
        lines.extend(
            _markdown_table(
                ctx.temporal_mspc[ctx.temporal_mspc["eval_scope"] == "lockbox"],
                ["eval_scope", "best_MSPC_source", "best_MSPC_TPR_at_TNR90", "T2_AUC", "Q_AUC"],
                headers=["scope", "best_source", "best_TPR_at_TNR90", "T2_AUC", "Q_AUC"],
            )
        )
        lines.append("")
    _append_figure(
        lines,
        "Temporal drift summary",
        "figures/temporal_drift.png",
        "Figure 4 condenses the temporal drift gate into a small set of quantities that make the claim restriction visible without reading the full CSV.",
    )
    _append_figure(
        lines,
        "Lockbox supervised vs MSPC",
        "figures/lockbox_vs_mspc.png",
        "Figure 5 compares the supervised lockbox TPR at matched TNR90 against the best MSPC comparator. When claim restrictions are active, this remains descriptive evidence only.",
    )
    _append_figure(
        lines,
        "Workload and cost framing",
        "figures/workload_cost_framing.png",
        "Figure 6 combines weekly workload framing with illustrative cost curves so operational impact can be discussed without overstating production readiness.",
    )
    lines.append("## Industrialization Gaps")
    lines.append("")
    _append_bullet_list(
        lines,
        [
            "No stable device/tool/chamber identifier for unseen-device validation.",
            "No intervention or maintenance history.",
            "No explicit regime-change metadata.",
            "No downstream decision or action outcome data.",
            "Anonymous features limit process interpretation.",
            "Single-dataset evidence only.",
            "Operational framing in this report is illustrative, not production-validated.",
        ],
    )
    lines.append("")
    lines.append("## Conclusions and Next Data Requirements")
    lines.append("")
    _append_bullet_list(
        lines,
        [
            (
                f"The benchmark layer reproduces meaningful supervised signal, with the best original row at mean BER `{_format_float(ctx.best_benchmark_row['mean_BER'])}`."
                if ctx.best_benchmark_row is not None
                else "The benchmark layer is incomplete in the current artifact set."
            ),
            (
                f"The tuned benchmark gives a stricter nested-CV estimate, with the best tuned row at mean BER `{_format_float(ctx.best_tuned_benchmark_row['mean_BER'])}`."
                if ctx.best_tuned_benchmark_row is not None
                else "The tuned benchmark layer is incomplete in the current artifact set."
            ),
            (
                "The temporal study is informative but remains descriptive-only in this run because claim restrictions are active."
                if restrictions
                else "The temporal study adds secondary robustness evidence without active claim restrictions in this run."
            ),
            "Next data collection should add device- or tool-level identifiers, intervention logs, and longer-horizon cross-context validation.",
            "A production-grade study would also require deployment decision objectives and cost accounting.",
            "Stronger process claims would require additional data to support stronger causal or process claims.",
        ],
    )
    lines.append("")
    lines.append("## Provenance Appendix")
    lines.append("")
    lines.append(f"- Generated artifact: `{ArtifactName.FINAL_REPORT}`")
    lines.append(f"- Source run directory: `{output_dir}`")
    lines.append(f"- Git commit: `{ctx.manifest.get('git_commit', 'unknown')}`")
    lines.append(f"- Git dirty: `{ctx.manifest.get('git_dirty', 'unknown')}`")
    lines.append(f"- Python executable: `{ctx.manifest.get('python_executable', 'unknown')}`")
    lines.append(f"- Study spec path: `{ctx.manifest.get('study_spec_path', 'unknown')}`")
    lines.append(f"- Study spec hash: `{ctx.manifest.get('study_spec_sha256', 'unknown')}`")
    lines.append(f"- Primary study status: `{ctx.manifest.get('primary_study_status', StudyStatus.NOT_RUN)}`")
    lines.append(
        f"- Original replication status: `{ctx.manifest.get('benchmark_original_status', StudyStatus.NOT_RUN)}`"
    )
    lines.append(f"- Tuned benchmark status: `{ctx.manifest.get('benchmark_tuned_status', StudyStatus.NOT_RUN)}`")
    lines.append(
        f"- Temporal robustness status: `{ctx.manifest.get('temporal_robustness_status', StudyStatus.NOT_RUN)}`"
    )
    lines.append("- Library versions:")
    for name, version in sorted(dict(ctx.manifest.get("library_versions", {})).items(), key=lambda item: item[0]):
        lines.append(f"  - `{name}`: `{version}`")
    if restrictions:
        lines.append("- Temporal claim restrictions:")
        lines.extend(f"  - `{restriction}`" for restriction in restrictions)
    else:
        lines.append("- Temporal claim restrictions: `none`")
    lines.append("")

    final_path = ctx.reports_dir / ArtifactName.FINAL_REPORT
    _write_markdown_with_optional_pdf(final_path, lines, export_pdf=export_pdf)
    return final_path
