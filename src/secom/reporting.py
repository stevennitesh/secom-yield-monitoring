"""Markdown report assembly from generated study artifacts."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from secom.artifacts import read_csv_if_exists, read_manifest
from secom.config import ArtifactName, BenchmarkClassifier, ReplicationMode, StudyStatus, ThresholdPolicy
from secom.report_language import (
    CLASSIFIERS,
    fold_count_label,
    later_sample_scope,
    procedure_label,
    role_label,
    table_header,
    table_value,
)
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
        number = float(value)
        return f"{number:.3g}" if 0 < abs(number) < 0.001 else f"{number:.3f}"
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
    """Render reader labels and split wide tables without losing recorded cells."""
    table = frame.loc[:, columns].copy()
    if max_rows is not None:
        table = table.head(max_rows)
    header_row = headers if headers is not None else [table_header(column) for column in columns]
    display = [
        [_format_cell(table_value(column, value)) for column, value in zip(columns, row, strict=True)]
        for row in table.itertuples(index=False, name=None)
    ]
    if len(columns) <= 9:
        return [
            "| " + " | ".join(header_row) + " |",
            "|" + "|".join(["---"] * len(columns)) + "|",
            *("| " + " | ".join(row) + " |" for row in display),
        ]
    lines = []
    # Repeat the leading identity fields and a row number so panels remain joinable,
    # including tables whose leading fields are not themselves unique.
    identity = list(range(2))
    remaining = list(range(2, len(columns)))
    panels = [remaining[start : start + 6] for start in range(0, len(remaining), 6)]
    for number, panel in enumerate(panels, start=1):
        positions = identity + panel
        lines.extend(
            [
                f"**Table panel {number} of {len(panels)} — shared row numbers identify the same record.**",
                "",
                "| Row | " + " | ".join(header_row[position] for position in positions) + " |",
                "|" + "|".join(["---"] * (len(positions) + 1)) + "|",
            ]
        )
        lines.extend(
            "| " + str(index) + " | " + " | ".join(row[position] for position in positions) + " |"
            for index, row in enumerate(display, start=1)
        )
        lines.append("")
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

_INDUSTRIALIZATION_GAPS = [
    "No stable device/tool/chamber identifier for unseen-device validation.",
    "No intervention or maintenance history.",
    "No explicit regime-change metadata.",
    "No downstream decision or action outcome data.",
    "Anonymous features limit process interpretation.",
    "Single-dataset evidence only.",
    "Operational framing in this report is illustrative, not production-validated.",
]

_INDUSTRIALIZATION_NEXT_DATA = [
    "Next data collection should add device- or tool-level identifiers, intervention logs, and longer-horizon cross-context validation.",
    "A production-grade study would also require deployment decision objectives and cost accounting.",
    "Named measurements, verified pre-outcome timing and intervention outcomes are needed to test causal or process explanations.",
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
            "min_BER",
            "max_BER",
            "mean_True+",
            "mean_True-",
        ],
        headers=["selector", "classifier", "mode", "mean_BER", "fold_min", "fold_max", "mean_TPR", "mean_TNR"],
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
                "gamma_multiplier_values": _search_space_count(group, "gamma_multiplier"),
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
            "gamma_multiplier_values",
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
        evaluated_columns=["fold", "k", "alpha", "gamma_multiplier", "C", "n_neighbors"],
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
            "gamma_multiplier",
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
            "gamma_multiplier",
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
        ["stability_weighted_coefficient", "selection_frequency", "feature_name_or_source_col"],
        ascending=[False, False, True],
    ).head(10)
    claim_note = f"- {_feature_interpretation_claim_note()}"
    if rows["absolute_scaled_coefficient"].isna().all():
        lines = [
            claim_note,
            "- Scaled coefficient magnitudes are unavailable for this exploratory family; the table shows stability.",
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
        "| feature | type | selection_frequency | absolute_scaled_coefficient | stability_weighted_coefficient | cluster_id |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows.itertuples(index=False):
        lines.append(
            f"| {row.feature_name_or_source_col} | {row.feature_type} | {_format_float(row.selection_frequency)} |"
            f" {_format_float(row.absolute_scaled_coefficient)} | {_format_float(row.stability_weighted_coefficient)} |"
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

_TEMPORAL_REPORT_FIELDS = (
    "temporal_selection",
    "temporal_lockbox",
    "temporal_drift",
    "temporal_mspc",
    "temporal_manager",
    "temporal_cost",
)


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
    temporal_status = str(manifest.get("temporal_robustness_status", StudyStatus.NOT_RUN))
    if temporal_status not in {StudyStatus.PASSED, StudyStatus.WARNING}:
        for field in _TEMPORAL_REPORT_FIELDS:
            frames[field] = None

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

    drift_row = _first_row(
        temporal_drift,
        temporal_drift["model_scope"] == "primary" if temporal_drift is not None else None,
    )
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


def _manifest_industrialization_notes(manifest: dict[str, object]) -> list[str]:
    """Return non-empty run-specific industrialization notes from the manifest."""
    notes = manifest.get("industrialization_notes", [])
    if not isinstance(notes, list):
        return []
    normalized = [str(note).strip() for note in notes]
    return [note for note in normalized if note]


def _append_industrialization_section(lines: list[str], manifest: dict[str, object]) -> None:
    """Append required industrialization gaps plus run-specific manifest notes."""
    lines.append("## Industrialization Gaps")
    lines.append("")
    _append_bullet_list(lines, _INDUSTRIALIZATION_GAPS)
    lines.extend(["", "### Next Data Requirements", ""])
    _append_bullet_list(lines, _INDUSTRIALIZATION_NEXT_DATA)

    notes = _manifest_industrialization_notes(manifest)
    if notes:
        lines.append("")
        lines.append("### Run-Specific Industrialization Notes")
        lines.append("")
        _append_bullet_list(lines, notes)


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


def _write_final_report_figures(ctx: ReportContext, reports_destination: Path | None = None) -> None:
    """Write all figures referenced by the canonical final report."""
    figures_dir = (reports_destination or ctx.reports_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    original_procedures = read_csv_if_exists(ctx.reports_dir / ArtifactName.BENCHMARK_PROCEDURE_SUMMARY)
    tuned_procedures = read_csv_if_exists(ctx.reports_dir / ArtifactName.BENCHMARK_TUNED_PROCEDURE_SUMMARY)
    write_benchmark_comparison_figure(
        original_procedures,
        tuned_procedures,
        figures_dir / "benchmark_comparison.png",
    )
    write_tuned_delta_figure(
        original_procedures,
        tuned_procedures,
        figures_dir / "tuned_vs_original_delta.png",
    )
    write_feature_stability_figure(
        ctx.feature_report,
        ctx.benchmark_tuned_feature_report,
        figures_dir / "feature_stability.png",
        benchmark_summary=ctx.benchmark_summary,
        benchmark_tuned_summary=ctx.benchmark_tuned_summary,
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


def _dataset_scope_lines(manifest: dict) -> list[str]:
    """Report actual input observations and the public-source metadata discrepancy."""
    data = manifest.get("dataset", {})
    if not data:
        return ["Dataset profile is unavailable for this artifact set."]
    prevalence = data["n_fails"] / data["n_samples"]
    return [
        f"The input files contain **{data['n_samples']:,} samples Ã— {data['n_features']} measurement columns**, "
        f"with {data['n_passes']:,} passes and {data['n_fails']} failures ({prevalence:.2%}). "
        f"There are {data['missing_cells']:,} missing measurement cells ({data['missing_fraction']:.2%}). "
        f"Valid test timestamps span {data['timestamp_min']} to {data['timestamp_max']}; no rows were dropped.",
        "",
        "The [UCI metadata](https://archive.ics.uci.edu/dataset/179/secom) and original paper describe 591 features; "
        "the distributed measurement file has 590 columns. Feature names here are zero-based source positions "
        "(`X0` through `X589` and missing indicators `M0` through `M589`). A row is a production entity, "
        "with no documented physical unit or assurance that measurements precede the outcome. "
        "The target is the recorded test pass/fail label; early intervention benefit is unmeasured.",
        "",
        f"An all-pass rule has accuracy {1 - prevalence:.2%}, TPR 0, TNR 1 and BER 0.500. "
        "This is why BER and both class recalls lead the analysis.",
    ]


def _missingness_context_lines(
    manifest: dict, feature_report: pd.DataFrame | None, best: pd.Series | None
) -> list[str]:
    """Explain co-occurring indicators and month shift without claiming causes."""
    if feature_report is None or best is None:
        return []
    selected = feature_report[
        (feature_report["selector"] == best["selector"])
        & (feature_report["classifier"] == best["classifier"])
        & (feature_report["replication_mode"] == best["replication_mode"])
        & (feature_report["feature_type"] == "missing_indicator")
        & (feature_report["selection_frequency"] > 0)
    ]
    selected_names = set(selected["feature_name_or_source_col"])
    groups = [
        group
        for group in manifest.get("dataset", {}).get("shared_missingness_patterns", [])
        if selected_names.intersection(group["features"])
    ]
    if not groups:
        return []
    lines = [
        "### Missingness Context",
        "",
        "These full-sample diagnostics explain selected missing indicators; "
        "they are not additional model validation or causal evidence.",
        "",
    ]
    for group in groups[:3]:
        names = ", ".join(f"`{name}`" for name in group["features"][:12])
        suffix = f" (and {len(group['features']) - 12} others)" if len(group["features"]) > 12 else ""
        monthly = "; ".join(f"{month}: {rate:.1%}" for month, rate in group["monthly_missing_rates"].items())
        lines.append(
            f"- {names}{suffix} have identical missingness masks. Missing rate by month: {monthly}. "
            f"Overall pass/fail missing rates: {group['pass_missing_rate']:.1%} / {group['fail_missing_rate']:.1%}."
        )
    lines += [
        "",
        "Co-occurring missingness can encode acquisition regime or time as well as process state. "
        "Repeated selection does not identify independent sensor effects, root causes, or a stable deployment signal.",
    ]
    return lines


def _benchmark_models(ctx: ReportContext) -> list[str]:
    """Use execution settings when present, otherwise the saved candidate families."""
    settings = ctx.manifest.get("execution", {}).get("settings", {})
    if settings.get("classifiers"):
        return list(settings["classifiers"])
    frames = [f for f in (ctx.benchmark_sweep, ctx.benchmark_tuned_search) if f is not None and not f.empty]
    return sorted({str(value) for frame in frames for value in frame.classifier.dropna()})


def _benchmark_protocol(ctx: ReportContext) -> str:
    settings = ctx.manifest.get("execution", {}).get("settings", {})
    scope = f"Recorded outer fold counts: reference {fold_count_label(ctx.benchmark_summary)}; tuned {fold_count_label(ctx.benchmark_tuned_summary)}. "
    scope += "The protocol uses shuffled stratified outer folds and stratified inner cross-validation. "
    if settings.get("benchmark_inner_folds") is not None:
        scope += f"Recorded inner folds: {settings['benchmark_inner_folds']}. "
    if settings.get("benchmark_seed") is not None:
        scope += f"Recorded seed: {settings['benchmark_seed']}. "
    return scope


def _benchmark_search_description(ctx: ReportContext, *, tuned: bool) -> str:
    """Describe grids from recorded execution or candidate columns, including KRR-only runs."""
    settings = ctx.manifest.get("execution", {}).get("settings", {})
    search = ctx.benchmark_tuned_search if tuned else ctx.benchmark_sweep

    def values(key, column):
        if key in settings:
            value = settings[key]
            return value if isinstance(value, list) else [value]
        if search is None or search.empty or column not in search:
            return None
        return sorted(search[column].dropna().unique().tolist())

    def describe(items):
        return "/".join("automatic" if item is None else str(item) for item in items) if items else "unavailable"

    budgets = values("tuned_feature_budgets" if tuned else "original_feature_budget", "k")
    neighbors = values("relief_neighbors_tuned" if tuned else "relief_neighbors_original", "n_neighbors")
    parts = [f"Recorded feature budgets: {describe(budgets)}. Recorded ReliefF neighbor counts: {describe(neighbors)}."]
    prefix = "tuned" if tuned else "original"
    if "krr" in _benchmark_models(ctx):
        alpha = values(f"{prefix}_krr_alpha_grid", "alpha")
        gamma = values(
            "tuned_krr_gamma_multipliers" if tuned else "original_krr_gamma_grid",
            "gamma_multiplier" if tuned else "gamma",
        )
        parts.append(
            f"Recorded KRR regularization strengths: {describe(alpha)}; {'dimension-relative kernel-width multipliers' if tuned else 'kernel-width settings'}: {describe(gamma)}."
        )
        if (
            f"{prefix}_krr_alpha_grid" in settings
            and ("tuned_krr_gamma_multipliers" if tuned else "original_krr_gamma_grid") in settings
        ):
            parts.append(f"Declared KRR configurations per selector budget: {len(alpha) * len(gamma)}.")
    if "logreg" in _benchmark_models(ctx):
        parts.append(f"Recorded logistic-regression regularization settings: {describe(values('logreg_C_grid', 'C'))}.")
    parts.append(
        "BER is the primary inner objective; AUC is supporting. Exact ties prefer fewer features, then stronger regularization (larger KRR alpha, smaller logistic C), followed by deterministic remaining order."
    )
    return " ".join(parts)


def _calibration_story_table(diagnostics: pd.DataFrame) -> list[str]:
    """Compact selected LR/KRR main-window diagnostics; full procedures stay in the appendix."""
    selected = diagnostics[diagnostics.procedure.isin(["temporal_joint", "krr_cal20_joint"]) & (diagnostics.fold > 0)]
    rows = []
    for fold, group in selected.groupby("fold", sort=True):
        row = {"Later test period": int(fold)}
        counts = []
        for procedure, label in (("temporal_joint", "Logistic regression"), ("krr_cal20_joint", "Kernel ridge (20%)")):
            match = group[group.procedure == procedure]
            if match.empty:
                row[label + " flagged range"] = "unavailable"
                continue
            item = match.iloc[0]
            counts.append(
                (label, f"{int(item.calibration_n)} / {int(item.calibration_fails)} / {int(item.calibration_passes)}")
            )
            row[label + " flagged range"] = (
                f"{item.lofo_flagged_fraction_min:.1%}–{item.lofo_flagged_fraction_max:.1%}"
                if item.lofo_available
                else "undefined"
            )
        row["Calibration samples / failures / passes"] = (
            counts[0][1]
            if counts and len({value for _, value in counts}) == 1
            else "; ".join(label + ": " + value for label, value in counts)
        )
        rows.append(row)
    if not rows:
        return ["Selected earlier-period calibration diagnostics are unavailable."]
    frame = pd.DataFrame(rows)
    return _markdown_table(
        frame,
        [
            "Later test period",
            "Calibration samples / failures / passes",
            "Logistic regression flagged range",
            "Kernel ridge (20%) flagged range",
        ],
    )


def _period_error_mean(frame: pd.DataFrame) -> str:
    """Do not turn guarded class-absent values into a claimed aggregate error."""
    if frame.empty or ("BER_available" in frame and not frame.BER_available.astype(bool).all()):
        return "unavailable"
    return f"{frame.BER.mean():.2%}"


def _render_technical_details(ctx: ReportContext) -> list[str]:
    """Render each evidence tier with joint held-out procedures as benchmark headlines."""
    lines = [
        "## Technical Appendix",
        "",
        "Detailed tables preserve the recorded values with readable display labels. They do not define additional headline results. "
        "Unless a heading includes %, rates are fractions from 0 to 1. Wide tables use numbered panels with shared row numbers.",
        "",
        "Artifact vocabulary: `strict` means measurements only; `with_missing_indicators` means measurements plus missing flags; "
        "`held_out_DEV_calibration` means held-out calibration. BER is balanced error; TPR/True+ is failure recall; "
        "TNR/True- is pass specificity. LOFO means recalibration after leaving out one failed calibration example. "
        "The saved CSVs retain their original field names and categorical identifiers.",
        "",
    ]
    procedure_frames = []
    for tuned in (False, True):
        label = "Tuned Benchmark" if tuned else "Original Replication"
        prefix = "benchmark_tuned" if tuned else "benchmark"
        summary = ctx.benchmark_tuned_summary if tuned else ctx.benchmark_summary
        search = ctx.benchmark_tuned_search if tuned else ctx.benchmark_sweep
        best = ctx.benchmark_tuned_best if tuned else ctx.benchmark_best
        ablation = ctx.benchmark_tuned_ablation if tuned else ctx.benchmark_ablation
        procedures = read_csv_if_exists(ctx.reports_dir / f"{prefix}_procedure_summary.csv")
        procedure_frames.append(procedures)
        lines.extend(
            [
                f"## {label} Design",
                "",
                _benchmark_protocol(ctx)
                + "Imputation, scaling, and selection are fitted within each inner training split. "
                "Pooled inner out-of-fold scores jointly select parameters and a BER threshold; ties use deterministic simplicity/order. "
                "The threshold is frozen before the chosen pipeline is refitted on outer training data and evaluated once on outer test data. "
                "The inner-OOF to outer-refit score-distribution difference remains a calibration limitation; no third nesting is claimed.",
                "",
                _benchmark_search_description(ctx, tuned=tuned),
                "",
                f"## {label} Search Summary",
                "",
                "### Tuned Search Space" if tuned else "### Original Search Space",
                "",
            ]
        )
        if search is not None:
            lines.extend(_tuned_search_space_table(search) if tuned else _original_search_space_table(search))
        lines.extend(
            [
                "",
                "### Modal Selected Configurations" if tuned else "### Original Selected Configurations",
                "",
                "Modal configurations describe inner selections across folds and full-data interpretation fits, not a new performance estimate.",
                "",
            ]
        )
        if best is not None:
            lines.extend(_tuned_best_config_table(best))
        lines.extend(
            [
                "",
                f"## {label} Results",
                "",
                "### Joint Held-out Procedure and Baselines",
                "",
                f"Headline source: `{prefix}_procedure_summary.csv`, recomputed from `{prefix}_predictions.csv`. "
                "Fold mean, standard deviation, and range are descriptive because training samples overlap. They are not algorithm-performance confidence intervals.",
                "",
            ]
        )
        if procedures is not None:
            lines.extend(
                _markdown_table(
                    procedures,
                    [
                        "procedure",
                        "mean_BER",
                        "std_BER",
                        "min_BER",
                        "max_BER",
                        "mean_True+",
                        "mean_True-",
                        "pooled_TP",
                        "pooled_FP",
                        "pooled_TN",
                        "pooled_FN",
                    ],
                )
            )
        else:
            lines.append("Joint held-out procedure artifacts unavailable.")
        lines.extend(["", "### Exploratory Nested Family Comparisons", ""])
        if summary is not None:
            lines.extend(_top_benchmark_table(summary))
            lines.extend(["", "### Supporting Benchmark Metrics", ""])
            lines.extend(_supporting_benchmark_table(summary))
        if ablation is not None:
            lines.extend(
                [
                    "",
                    "### Paired Missing-indicator Ablation",
                    "",
                    "Positive delta_BER means values-only BER minus values-plus-indicators BER. Paired fold deltas are descriptive.",
                    "",
                ]
            )
            lines.extend(_markdown_table(ablation, list(ablation.columns)))
        if not tuned:
            lines.extend(
                [
                    "",
                    "### UCI Original Benchmark Reference",
                    "",
                    "[UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom) describes KRR; "
                    "[McCann and Johnston (2010), Table 2](https://proceedings.mlr.press/v6/mccann10a/mccann10a.pdf) labels its baseline Naive Bayes. "
                    "These are reference context, not an exact classifier/protocol replication claim.",
                    "",
                ]
            )
            lines.extend(_uci_original_baseline_table(summary))
            lines.extend(["", _uci_selector_definition_note(), ""])
    lines.extend(
        [
            "## Original vs Tuned Benchmark Comparison",
            "",
            "The procedures use identical held-out sample IDs/folds. Positive paired delta below is original BER minus tuned BER; "
            "the spread is descriptive, not a significance test.",
            "",
        ]
    )
    from secom.metrics import safe_std

    original = read_csv_if_exists(ctx.reports_dir / ArtifactName.BENCHMARK_PROCEDURE_FOLD_METRICS)
    tuned = read_csv_if_exists(ctx.reports_dir / ArtifactName.BENCHMARK_TUNED_PROCEDURE_FOLD_METRICS)
    if original is not None and tuned is not None:
        paired = original[original.procedure == "joint"].merge(
            tuned[tuned.procedure == "joint"], on=["procedure", "fold"], suffixes=("_original", "_tuned")
        )
        delta = paired.BER_original - paired.BER_tuned
        lines.append(
            f"Joint paired delta_BER mean={_format_float(delta.mean())}, std={_format_float(safe_std(delta))}, range={_format_float(delta.min())} to {_format_float(delta.max())}."
        )
    lines.extend(
        [
            "## Feature Stability and Interpretation",
            "",
            _feature_interpretation_claim_note(),
            "",
            "Selection frequency across overlapping outer training folds is descriptive. Full-data coefficients are in-sample associations. "
            "absolute_scaled_coefficient is the absolute full-fit scaled logistic coefficient; stability_weighted_coefficient is frequency times that coefficient. "
            "Neither is an expected economic contribution or causal effect. KRR coefficient fields remain unavailable.",
            "",
        ]
    )
    for label, frame, anchor in (
        ("Original", ctx.feature_report, ctx.best_benchmark_row),
        ("Tuned", ctx.benchmark_tuned_feature_report, ctx.best_tuned_benchmark_row),
    ):
        lines.extend([f"### {label} Exploratory Family Feature Interpretation", ""])
        if frame is not None and anchor is not None:
            lines.extend(
                _best_row_feature_table(
                    frame,
                    selector=str(anchor.selector),
                    classifier=str(anchor.classifier),
                    replication_mode=str(anchor.replication_mode),
                )
            )
    lines.extend(_missingness_context_lines(ctx.manifest, ctx.feature_report, ctx.best_benchmark_row))
    lines.extend(
        [
            "## Temporal Robustness Stress Test",
            "",
            f"Temporal robustness status: `{ctx.manifest.get('temporal_robustness_status')}`",
            "",
            "### Temporal Robustness Design",
            "",
            "The retained logistic-regression role study and a bounded DEV-only KRR comparator are separate secondary stress studies. The last chronological 15% is a retrospective later evaluation block "
            "already exposed through the full-dataset benchmark and earlier reports; it is not a fresh confirmatory lockbox. "
            "DEV uses fixed nonoverlapping calendar test blocks and expanding training prefixes. Inner tuning uses deterministic chronological splits. "
            "The last chronological 20% of each training region is held-out calibration. Tuning/model fitting use the earlier fit prefix; "
            "thresholds use calibration scores from that retained model, with no refit on calibration after freezing.",
            "",
        ]
    )
    temporal_joint = read_csv_if_exists(ctx.reports_dir / ArtifactName.TEMPORAL_PROCEDURE_METRICS)
    if ctx.temporal_selection is not None:
        lines.extend(
            [
                "### DEV-only KRR and Calibration Sensitivity",
                "",
                "KRR uses the shared tuned alpha grid and dimension-relative gamma multipliers with StandardScaler. "
                "Joint input mode and configuration are chosen by earlier FIT chronological inner BER. Values-only and combined procedures "
                "use the same fixed evaluation periods. The predeclared 30% calibration sensitivity tunes independently within its earlier 70% FIT. "
                "It is descriptive and cannot promote an outer-period winner or a later-block KRR champion.",
                "",
            ]
        )
        comparator = read_csv_if_exists(ctx.reports_dir / ArtifactName.TEMPORAL_KRR_METRICS)
        if comparator is not None:
            keep = [
                c
                for c in (
                    "fold",
                    "procedure",
                    "available",
                    "unavailable_reason",
                    "BER",
                    "ROC_AUC",
                    "n_inner_timestamp_ties",
                    "True+",
                    "True-",
                    "TP",
                    "TN",
                    "FP",
                    "FN",
                )
                if c in comparator
            ]
            lines.extend(_markdown_table(comparator, keep))
        lines.extend(
            [
                "",
                "### Calibration Counts and Threshold Fragility",
                "",
                "Fewer than ten calibration failures is a fragility warning, not a reason to move a chronological boundary. "
                "BER steps are 1/(2 failures) and 1/(2 passes). Leave-one-failure-out recalibration keeps fitted scores fixed; "
                "flagged-fraction ranges use the same full calibration score set. LOFO ranges are undefined with fewer than two failures or no passes; class-specific BER steps are undefined when that class is absent. These ranges describe calibration instability only, "
                "not confidence intervals, future uncertainty or algorithm uncertainty. Per-period AUC distinguishes ranking weakness from threshold weakness. Inner rows follow stable (timestamp, raw_row_id) order; equal-timestamp boundaries are flagged in search/calibration receipts. Disjoint IDs and nondecreasing timestamps preserve the fixed splits. Raw row ID does not establish physical event order or independence within tied timestamps. Outer calendar tests remain strictly later.",
                "",
            ]
        )
        diagnostics = read_csv_if_exists(ctx.reports_dir / ArtifactName.TEMPORAL_CALIBRATION_DIAGNOSTICS)
        if diagnostics is not None:
            keep = [
                c
                for c in (
                    "fold",
                    "procedure",
                    "calibration_n",
                    "calibration_fails",
                    "calibration_passes",
                    "BER_step_failure",
                    "BER_step_pass",
                    "threshold",
                    "fragile_calibration",
                    "fit_calibration_timestamp_tie",
                    "lofo_available",
                    "lofo_threshold_min",
                    "lofo_threshold_max",
                    "lofo_flagged_fraction_min",
                    "lofo_flagged_fraction_max",
                )
                if c in diagnostics
            ]
            display_diagnostics = diagnostics[keep].astype(object).where(diagnostics[keep].notna(), "undefined")
            lines.extend(_markdown_table(display_diagnostics, keep))
        lines.extend(["", "### Temporal Joint Held-out Procedure", ""])
        if temporal_joint is not None:
            lines.extend(_markdown_table(temporal_joint, list(temporal_joint.columns)))
        lines.extend(
            [
                "",
                "### Temporal Model Selection Summary",
                "",
                "Roles are chosen from chronological inner selection on the final fit prefix. Outer family ranking remains exploratory.",
                "",
                "#### Selector Ranking and Modal Configurations",
                "",
            ]
        )
        lines.extend(_temporal_selection_summary_table(ctx.temporal_selection))
        lines.extend(
            [
                "",
                "### Lockbox Metrics",
                "",
                "Frozen-threshold confusion counts accompany rates and exact binomial TPR/TNR intervals where available. "
                "When a class is absent, guarded numerical rate placeholders are marked unavailable; they are not evidence of that class recall. Intervals are conditional on a fixed model and independent Bernoulli trials; temporal dependence and model-selection uncertainty are excluded. "
                "TNR90 thresholds selected from evaluation labels remain retrospective ROC diagnostics.",
                "",
            ]
        )
        lines.extend(_markdown_table(ctx.temporal_lockbox, list(ctx.temporal_lockbox.columns)))
        lines.extend(
            [
                "",
                "### Drift and Claim Restrictions",
                "",
                "KS compares held-out calibration scores from the same retained model with future scores. Raw-feature PSI uses the fit reference descriptively. "
                "Missingness-rate changes indicate collection-regime association, not causes. Heuristic gates cannot authorize superiority or confirmatory claims.",
                "",
            ]
        )
        lines.extend(_markdown_table(ctx.temporal_drift, list(ctx.temporal_drift.columns)))
        _append_bullet_list(lines, list(ctx.manifest.get("temporal_claim_restrictions", [])))
        lines.extend(
            [
                "",
                "### Supervised vs MSPC",
                "",
                "MSPC fits PCA on pass-only fit samples. Calibration freezes T2/Q source and thresholds before evaluation. "
                "Frozen confusion counts and retrospective TNR90 diagnostics are separate. Observed mean inter-alarm spacing across all samples is not in-control ARL0.",
                "",
            ]
        )
        lines.extend(_markdown_table(ctx.temporal_mspc, list(ctx.temporal_mspc.columns)))
        lines.extend(
            [
                "",
                "### Illustrative Operational Framing",
                "",
                "Workload comes from held-out DEV calibration predictions. The operational policy constrains unweighted mean weekly flagged fraction to 10%, not every week's hard cap. "
                "Calibration workload is used to select that policy; future operating cost and production capacity remain unvalidated.",
                "",
            ]
        )
        lines.extend(_markdown_table(ctx.temporal_manager, list(ctx.temporal_manager.columns)))
        lines.extend(["", "#### Cost Curves", ""])
        lines.extend(_markdown_table(ctx.temporal_cost, list(ctx.temporal_cost.columns)))
    else:
        lines.append("Temporal model selection artifact missing or empty.")
    # Keep full machine tables accessible without making them the default reading path.
    grouped = lines[:4]
    opened = False
    for line in lines[4:]:
        if line.startswith("## "):
            if opened:
                grouped.extend(["", "</details>", ""])
            title = line.removeprefix("## ")
            if title == "Temporal Robustness Stress Test":
                grouped.extend(["### Chronological diagnostics", ""])
            grouped.extend(["<details>", f"<summary>{title}</summary>", "", f"### Appendix: {title}"])
            opened = True
        elif line.startswith("### "):
            grouped.append("#" + line)
        elif line.startswith("#### "):
            grouped.append("#" + line)
        else:
            grouped.append(line)
    if opened:
        grouped.extend(["", "</details>", ""])
    return grouped


def _reader_metric_table(frame: pd.DataFrame, *, benchmark: bool = False) -> list[str]:
    """Show existing rates and counts with external names and explicit percentage units."""
    display = pd.DataFrame()
    display["Procedure"] = frame["procedure"].map(procedure_label)
    if "fold" in frame:
        display.insert(0, "Later test period", frame["fold"])
        display["Test samples / failures"] = frame.apply(
            lambda row: f"{int(row.n_test)} / {int(row.n_test_fails)}", axis=1
        )
    prefix = "mean_" if benchmark else ""
    for key, heading in (("BER", "Balanced error"), ("True+", "Failure recall"), ("True-", "Pass specificity")):
        display[heading] = frame[prefix + key].map(lambda value: f"{value:.2%}")
        availability = {"BER": "BER_available", "True+": "TPR_available", "True-": "TNR_available"}[key]
        if not benchmark and availability in frame:
            display.loc[~frame[availability].astype(bool), heading] = "unavailable"
    if benchmark:
        display["Fold spread (SD)"] = frame["std_BER"].map(lambda value: f"{100 * value:.2f} pp")
        display["Fold range"] = frame.apply(lambda row: f"{row.min_BER:.2%}–{row.max_BER:.2%}", axis=1)
    elif "ROC_AUC" in frame:
        display["Ranking AUC"] = frame["ROC_AUC"].map(lambda value: f"{value:.3f}")
    count_prefix = "pooled_" if benchmark else ""
    display["Failures caught"] = frame[count_prefix + "TP"]
    display["False alerts on passes"] = frame[count_prefix + "FP"]
    n = (
        frame[count_prefix + "TP"]
        + frame[count_prefix + "FP"]
        + frame[count_prefix + "TN"]
        + frame[count_prefix + "FN"]
    )
    display["Samples flagged"] = ((frame[count_prefix + "TP"] + frame[count_prefix + "FP"]) / n).map(
        lambda value: f"{value:.1%}"
    )
    return _markdown_table(display, list(display.columns))


def _render_study(ctx: ReportContext) -> list[str]:
    """Lead with the manufacturing question; keep internal keys in technical appendices."""
    dataset = ctx.manifest.get("dataset", {})
    original = read_csv_if_exists(ctx.reports_dir / ArtifactName.BENCHMARK_PROCEDURE_SUMMARY)
    tuned = read_csv_if_exists(ctx.reports_dir / ArtifactName.BENCHMARK_TUNED_PROCEDURE_SUMMARY)
    original_joint = _first_row(original, original.procedure.eq("joint") if original is not None else None)
    tuned_joint = _first_row(tuned, tuned.procedure.eq("joint") if tuned is not None else None)
    models = _benchmark_models(ctx)
    settings = ctx.manifest.get("execution", {}).get("settings", {})
    selectors = settings.get("original_selectors")
    if selectors is None and ctx.benchmark_sweep is not None and not ctx.benchmark_sweep.empty:
        selectors = ctx.benchmark_sweep.selector.unique().tolist()
    method_scope = (
        f"{len(selectors)} column-selection {'method' if len(selectors) == 1 else 'methods'}"
        if selectors is not None
        else "column-selection methods whose count is unavailable"
    )
    search_scope = (
        f"We compared {method_scope} using {' and '.join(CLASSIFIERS.get(model, model) for model in models)}. "
        if models
        else "The saved evidence does not identify the model families compared. "
    )
    reference_folds, tuned_folds = fold_count_label(original), fold_count_label(tuned)
    if reference_folds == tuned_folds and reference_folds != "unavailable":
        fold_scope = f"Both benchmarks used {reference_folds} held-out test folds. "
    elif "unavailable" not in (reference_folds, tuned_folds):
        fold_scope = f"The reference used {reference_folds} held-out test folds; tuning used {tuned_folds}. "
    else:
        fold_scope = "Test-fold counts are unavailable for one or both benchmarks. "
    budgets = []
    for key, search in (
        ("original_feature_budget", ctx.benchmark_sweep),
        ("tuned_feature_budgets", ctx.benchmark_tuned_search),
    ):
        values = settings.get(key)
        if values is None and search is not None and not search.empty and "k" in search:
            values = sorted(search.k.dropna().unique().tolist())
        values = values if isinstance(values, list) else [values] if values is not None else []
        names = [str(int(value)) for value in values]
        budgets.append(", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0] if names else None)
    budget_scope = (
        f"The reference allows up to {budgets[0]} selected inputs; tuning compares budgets of {budgets[1]} inputs. "
        if all(budgets)
        else "Input-budget details are unavailable for one or both benchmarks. "
    )
    findings = []
    if original_joint is not None and tuned_joint is not None:
        findings.append(
            f"- **Benchmark:** mean balanced error {original_joint.mean_BER:.2%} → {tuned_joint.mean_BER:.2%}; "
            f"failures caught {int(original_joint.pooled_TP)} → {int(tuned_joint.pooled_TP)}; "
            f"false alerts {int(original_joint.pooled_FP)} → {int(tuned_joint.pooled_FP)}. "
            "Rates average held-out folds; counts pool their predictions."
        )
    else:
        findings.append("- **Benchmark:** complete-procedure results are unavailable.")
    comparator = read_csv_if_exists(ctx.reports_dir / ArtifactName.TEMPORAL_KRR_METRICS)
    if comparator is not None and not comparator.empty:
        main = comparator[comparator.available.astype(bool) & comparator.procedure.eq("krr_cal20_joint")]
        if not main.empty:
            findings.append(
                f"- **Chronological stress:** kernel ridge averages {_period_error_mean(main)} balanced error and flags "
                f"{(main.TP.sum() + main.FP.sum()) / main.n_test.sum():.1%} of {int(main.n_test.sum()):,} later samples. "
                "This separate procedure tests transfer; the final later block is retrospective."
            )
    lines = [
        "# Can anonymous manufacturing measurements identify recorded failures?",
        "",
        "## Executive Summary",
        "",
        "This SECOM study asks whether recorded manufacturing measurements distinguish a failed test from a passed test. "
        "When failures are rare, predicting pass for everyone can look accurate while catching no failures. "
        "The useful comparison gives failures and passes equal weight, then examines the tradeoff between catching failures and raising false alerts.",
        "",
        *findings,
        "",
        "**Read:** [Scope](#dataset-and-study-scope) · [Reference](#original-replication-design) · "
        "[Tuning](#tuned-benchmark-design) · [Tradeoff](#original-vs-tuned-benchmark-comparison) · "
        "[Inputs](#feature-stability-and-interpretation) · [Later samples](#temporal-robustness-stress-test) · "
        "[Gaps](#industrialization-gaps) · [Conclusions](#conclusions-and-next-data-requirements) · "
        "[Technical tables](#technical-appendix) · [Provenance](#provenance-appendix)",
        "",
        "## What I Built",
        "",
        "A reproducible Python study that compares a literature-inspired reference benchmark with bounded tuning, "
        "checks the complete selection procedure on unseen test samples, and then tests transfer to later samples. "
        "Training-only transformations, saved predictions and independent artifact checks make the results traceable.",
        "",
        "## Dataset and Study Scope",
        "",
    ]
    if dataset:
        lines += [
            f"The files contain **{dataset['n_samples']:,} samples, {dataset['n_features']} anonymous measurement columns, "
            f"{dataset['n_fails']} failures and {dataset['n_passes']:,} passes**. Predicting pass for everyone would give "
            f"{dataset['n_passes'] / dataset['n_samples']:.2%} accuracy and catch zero failures. "
            f"{dataset['missing_fraction']:.2%} of measurement cells are missing.",
            "",
        ]
    lines += [
        "**Failure recall** is the fraction of failures caught. **Pass specificity** is the fraction of passes correctly left unflagged. "
        "**Balanced error** is the average of the missed-failure rate and the false-alert rate on passes: "
        "½ × [(1 − recall) + (1 − specificity)]. Lower is better; an always-pass rule has 50% balanced error.",
        "",
        "A row represents a production entity whose physical unit is undocumented. Measurements are anonymous, "
        "and their availability before the outcome is not established. This study predicts recorded test labels; "
        "it does not demonstrate early warning or intervention benefit. Public metadata describe 591 features, while the reference SECOM measurement file contains 590.",
        "",
        "## Original Replication Design",
        "",
        "The reference is a literature-inspired fixed-feature-budget benchmark, not an exact replication of the published classifier and protocol. "
        + search_scope
        + "**Kernel ridge regression (KRR)** learns a nonlinear score using "
        "similarity between samples; **logistic regression (LR)** learns a weighted combination of inputs with a logistic link.",
        "",
        "**Cross-validation** rotates which samples are held aside. Shuffled, class-stratified outer splits preserve class proportions and evaluate unseen samples. "
        + fold_scope
        + "Inside each outer training portion, inner splits choose the selector, model, input mode, settings and alert threshold. "
        "Median imputation, scaling and feature selection see training data only. The chosen procedure is refitted on outer training data, "
        "then evaluated once on its untouched outer test samples. The test labels never choose the model or threshold.",
        "",
        "**Calibration** means choosing the score threshold that triggers an alert. The benchmark chooses it from pooled inner held-out scores "
        "and freezes it before the outer refit. Scores can shift after refitting; that is a remaining calibration limitation. "
        "The evaluation covers the complete selection procedure, rather than a model picked retrospectively for its best test result.",
        "",
    ]
    lines += [
        "## Original Replication Results",
        "",
        "Rates below are fold means. Counts sum the once-held-out predictions. "
        "Standard deviation (SD) and ranges describe fold variation, not algorithm-performance confidence intervals.",
        "",
    ]
    if original is not None and not original.empty:
        lines += _reader_metric_table(original, benchmark=True)
    else:
        lines += ["Reference benchmark results are unavailable."]
    lines += [
        "",
        "## Tuned Benchmark Design",
        "",
        "The paired benchmark protocol keeps test samples, folds and seeds fixed while comparing declared input budgets and model settings. "
        + budget_scope
        + "Full grids and candidate counts are in the technical appendix. For dimension-relative kernel-ridge candidates, kernel width scales "
        "with the actual selected input count. Stronger regularization wins exact balanced-error ties after fewer features.",
        "",
        "This is bounded parameter coverage, not global optimization. Model settings, input mode and threshold remain selected entirely "
        "inside training data. Separately predefined measurements-only and measurements-plus-missing-flags procedures provide contrasts; "
        "their test results do not replace the complete selected procedure as the headline.",
        "",
        "## Tuned Benchmark Results",
        "",
    ]
    if tuned is not None and not tuned.empty:
        lines += _reader_metric_table(tuned, benchmark=True)
    else:
        lines += ["Tuned benchmark results are unavailable."]
    lines += ["", "## Original vs Tuned Benchmark Comparison", ""]
    if original_joint is not None and tuned_joint is not None:
        a, b = original_joint, tuned_joint
        caught_direction = (
            "more" if b.pooled_TP > a.pooled_TP else "fewer" if b.pooled_TP < a.pooled_TP else "the same number of"
        )
        alert_direction = (
            "more" if b.pooled_FP > a.pooled_FP else "fewer" if b.pooled_FP < a.pooled_FP else "the same number of"
        )
        lines += [
            f"The complete selected procedure changes balanced error from **{a.mean_BER:.2%} to {b.mean_BER:.2%}**, "
            f"failure recall from **{a['mean_True+']:.2%} to {b['mean_True+']:.2%}**, and pass specificity from "
            f"**{a['mean_True-']:.2%} to {b['mean_True-']:.2%}**. "
            f"The pooled counts change from {int(a.pooled_TP)} failures caught / {int(a.pooled_FP)} false alerts "
            f"to {int(b.pooled_TP)} caught / {int(b.pooled_FP)} false alerts. "
            f"Tuning produces {caught_direction} failures caught and {alert_direction} false alerts. "
            "Read both rates and counts together; a balanced-error change alone does not establish uniform improvement.",
            "",
        ]
        af = read_csv_if_exists(ctx.reports_dir / ArtifactName.BENCHMARK_PROCEDURE_FOLD_METRICS)
        bf = read_csv_if_exists(ctx.reports_dir / ArtifactName.BENCHMARK_TUNED_PROCEDURE_FOLD_METRICS)
        if af is not None and bf is not None and not af.empty and not bf.empty:
            pair = af[af.procedure == "joint"].merge(
                bf[bf.procedure == "joint"], on="fold", suffixes=("_reference", "_tuned")
            )
            delta = pair.BER_reference - pair.BER_tuned
            if not pair.empty:
                lines += [
                    f"On the paired test folds, tuning lowers balanced error in {int((delta > 1e-12).sum())}, "
                    f"raises it in {int((delta < -1e-12).sum())}, and ties in {int((delta.abs() <= 1e-12).sum())}. "
                    f"Mean reduction: {100 * delta.mean():.2f} percentage points. This is descriptive, not a significance test.",
                    "",
                ]
    else:
        lines += [
            "The complete selected-procedure comparison is unavailable; both saved benchmark summaries are required.",
            "",
        ]
    _append_figure(
        lines,
        "Complete reference and tuned procedures: balanced error and pooled alert counts",
        "figures/benchmark_comparison.png",
        fold_scope
        + "Complete procedures evaluated on unseen samples. Error bars show fold means and ranges; count panels pool held-out predictions. Fold spread is descriptive.",
    )
    lines += ["<details>", "<summary>Supporting input-mode contrasts</summary>", ""]
    _append_figure(
        lines,
        "Balanced-error reduction after tuning",
        "figures/tuned_vs_original_delta.png",
        "Positive values mean lower balanced error after tuning. The same folds and held-out samples support each contrast; no significance claim is made.",
    )
    lines += ["</details>", ""]
    lines += [
        "## Feature Stability and Interpretation",
        "",
        "Most of the search selects existing columns. Imputation fills missing measurements and scaling puts inputs on comparable scales; "
        "the explicit feature-engineering contrast adds a flag recording whether each measurement was missing. "
        "Ratios, interactions, trends and process-informed features were not systematically explored. The engineering space is not exhausted.",
        "",
        "The chart shows how often anonymous inputs were selected across overlapping training folds, for each study's exploratory "
        "lowest-error family. A family is a selector/model/input combination. This family was identified from test summaries for description and is not an independently validated champion. "
        "Column numbers are zero-based file positions, not named sensors. Missing flags can encode measurement-collection changes as well as process state.",
        "",
        _feature_interpretation_claim_note(),
        "",
    ]
    groups = dataset.get("shared_missingness_patterns", [])
    if groups:
        group = groups[0]
        names = ", ".join(name.removeprefix("M") for name in group["features"][:12])
        monthly = "; ".join(f"{month}: {rate:.1%}" for month, rate in group["monthly_missing_rates"].items())
        lines += [
            f"For example, missing flags for columns {names} have identical missingness patterns. "
            f"Their full-sample missing rate by month is {monthly}. These descriptive associations do not identify root causes "
            "or prove independent sensor effects.",
            "",
        ]
    _append_figure(
        lines,
        "Anonymous-input selection frequency",
        "figures/feature_stability.png",
        "Exploratory family stability and scaled-coefficient heuristics. Bars use selection frequency only; fitted coefficients remain in the technical appendix.",
    )
    lines += [
        "## Temporal Robustness Stress Test",
        "",
        "The shuffled benchmark asks about other samples from this dataset. The chronological stress test asks whether earlier "
        "measurements and labels transfer to later calendar periods. It is secondary evidence and does not replace the benchmark result.",
        "",
        "Fixed nonoverlapping calendar test blocks follow expanding earlier training regions. Model selection uses deterministic "
        "chronological splits inside the earlier fitting portion. A held-out portion of each training region supplies calibration "
        "scores for a retained model; the model is never refitted after its threshold is frozen. Ranking AUC is a supporting measure "
        "of how well scores order failures above passes; 0.5 is chance ordering, and AUC does not set the alert threshold.",
        "",
    ]
    lr, krr, diagnostics = None, None, None
    if ctx.temporal_selection is not None:
        lr = read_csv_if_exists(ctx.reports_dir / ArtifactName.TEMPORAL_PROCEDURE_METRICS)
        krr = read_csv_if_exists(ctx.reports_dir / ArtifactName.TEMPORAL_KRR_METRICS)
        lines += ["### Later-period logistic-regression results", ""]
        if lr is not None and not lr.empty:
            human = lr.copy()
            human["procedure"] = "Complete selected logistic-regression procedure"
            lines += _reader_metric_table(human)
            lines += [
                "",
                f"Mean per-period balanced error: {_period_error_mean(lr)}. "
                "Per-period results matter: pooling samples can conceal weak transfer when failure prevalence and alert rates differ between periods.",
                "",
            ]
        lines += [
            "### Kernel-ridge comparison and calibration-window sensitivity",
            "",
            "Kernel ridge uses the same fixed later test samples, a standard scaler and the shared bounded tuning grid. "
            "Each declared calibration-window sensitivity is independently selected using only its own earlier fitting portion. "
            "It changes both training size and calibration size; it is a compound sensitivity comparison, not an isolated threshold experiment. "
            "The two models also differ in grids and preprocessing, so this is no clean single-factor algorithm comparison.",
            "",
        ]
        if krr is not None and not krr.empty:
            available = krr[krr.available.astype(bool)]
            rows = []
            for name, group in available.groupby("procedure", sort=False):
                row = {"Procedure": procedure_label(name), "Mean period balanced error": _period_error_mean(group)}
                for _, r in group.iterrows():
                    row[f"Period {int(r.fold)}: error / AUC / flagged"] = (
                        f"{f'{r.BER:.2%}' if r.BER_available else 'unavailable'} / {r.ROC_AUC:.3f} / {(r.TP + r.FP) / r.n_test:.1%}"
                    )
                rows.append(row)
            frame = pd.DataFrame(rows)
            if rows:
                lines += _markdown_table(frame, list(frame.columns))
            else:
                lines += ["Available kernel-ridge period metrics are not recorded."]
            main = available[available.procedure == "krr_cal20_joint"]
            if not main.empty:
                fraction = (main.TP.sum() + main.FP.sum()) / main.n_test.sum()
                failures = main.TP.sum() + main.FN.sum()
                passes = main.FP.sum() + main.TN.sum()
                pooled = (
                    f"{0.5 * (main.FN.sum() / failures + main.FP.sum() / passes):.2%}"
                    if failures and passes
                    else "unavailable"
                )
                lines += [
                    "",
                    f"The main kernel-ridge procedure flags {fraction:.1%} of the {int(main.n_test.sum())} later samples. "
                    f"Its mean period balanced error is {_period_error_mean(main)}; its differently weighted pooled balanced error is {pooled}. "
                    "Flagged fraction measures alert volume alongside recall. Per-period AUC separates score-ordering weakness from threshold weakness. "
                    "Compare declared window paths descriptively; no sensitivity arm is promoted as a winner.",
                    "",
                ]
        lines += [
            "### Calibration counts and threshold sensitivity",
            "",
            "Removing one failed calibration example and recalibrating the same fixed scores checks threshold fragility. "
            "The resulting flagged-fraction range is evaluated on the full original calibration score set. "
            "It describes calibration instability only, not future uncertainty or a confidence interval. "
            "Fewer than ten calibration failures triggers a warning; it does not move a split boundary.",
            "",
        ]
        diagnostics = read_csv_if_exists(ctx.reports_dir / ArtifactName.TEMPORAL_CALIBRATION_DIAGNOSTICS)
        if diagnostics is not None and not diagnostics.empty:
            lines += _calibration_story_table(diagnostics)
        else:
            lines += ["Selected calibration diagnostics are unavailable."]
        lines += [
            "",
            "The compact table shows selected logistic-regression and main-window kernel-ridge thresholds on matching later periods. "
            "[Full calibration diagnostics](#chronological-diagnostics) are in the **Temporal Robustness Stress Test** appendix section; expand it to see every predefined window/input procedure and final model. "
            "A single failure changes balanced error by 1/(2 × calibration failures). For illustration, three failures give a 16.7-percentage-point step. "
            "More calibration data also leaves less data for fitting. Calibration sensitivity and score ordering must be read together; "
            "a different threshold alone cannot establish stable transfer.",
            "",
            "### The final later block: retrospective, frozen thresholds",
            "",
            "The final 15% of samples was already exposed in full-dataset benchmarking and earlier reports. It is a retrospective check, "
            "not independent confirmation. Only logistic-regression roles and a multivariate statistical process control (MSPC) baseline "
            "are evaluated here. MSPC uses principal components fitted on earlier passing samples; its score and alert threshold are chosen "
            "on calibration data. No later-block kernel-ridge result is implied.",
            "",
        ]
        if ctx.temporal_lockbox is not None:
            frame = ctx.temporal_lockbox.copy()
            display = pd.DataFrame(
                {
                    "Frozen rule": frame.apply(lambda r: role_label(r.role, r.threshold_policy), axis=1),
                    "Failures caught": frame.TP,
                    "Failures missed": frame.FN,
                    "False alerts on passes": frame.FP,
                    "Passes left unflagged": frame.TN,
                }
            )
            lines += _markdown_table(display, list(display.columns))
        lines += [
            "",
            later_sample_scope(ctx.temporal_lockbox)
            + " Exact intervals in the appendix are conditional on a fixed model and "
            "independent Bernoulli trials; they exclude temporal dependence and model-selection uncertainty. Retrospective thresholds "
            "chosen from evaluation labels to reach 90% pass specificity remain a separate ranking diagnostic, not frozen operating performance.",
            "",
        ]
        _append_figure(
            lines,
            "Later-block frozen alerts",
            "figures/lockbox_vs_mspc.png",
            later_sample_scope(ctx.temporal_lockbox)
            + " Existing frozen rules; class counts determine rate precision. No superiority or production claim follows.",
        )
        lines += [
            "### Measurement and score distribution checks",
            "",
            "Raw-measurement distribution comparisons use the earlier model-fitting samples as their reference. Score comparisons use "
            "held-out calibration scores from the same retained model. Missing-rate and prevalence changes are descriptive. "
            "These references answer different questions; their heuristic warnings cannot establish causes or authorize superiority.",
            "",
        ]
        _append_figure(
            lines,
            "Later measurement and score shifts",
            "figures/temporal_drift.png",
            "Primary logistic-regression model: raw-feature stability index versus earlier fitting measurements; score-distribution test versus held-out calibration. Secondary descriptive evidence.",
        )
        lines += [
            "### Hypothetical workload and cost",
            "",
            "The workload-limited threshold constrains the unweighted mean weekly flagged fraction on calibration data to 10%. "
            "It does not cap each week, or establish future workload. Cost ratios compare the assumed cost of missing a failure "
            "with the assumed cost of a false alert on a pass. Costs and capacity are hypothetical, not measured production outcomes.",
            "",
        ]
        _append_figure(
            lines,
            "Calibration workload and hypothetical costs",
            "figures/workload_cost_framing.png",
            "Calibration-only summaries used to choose operating thresholds. The mean-weekly policy is not an individual-week hard cap; hypothetical costs do not validate deployment value.",
        )
    else:
        lines += [
            f"Temporal robustness status: `{ctx.manifest.get('temporal_robustness_status')}`",
            "",
            "Temporal model selection artifact missing or empty.",
            "",
        ]
    _append_industrialization_section(lines, ctx.manifest)
    if original_joint is not None and tuned_joint is not None:
        reduction = 100 * (a.mean_BER - b.mean_BER)
        error_change = (
            f"{'lowers' if reduction > 0 else 'raises'} mean balanced error by {abs(reduction):.2f} percentage points "
            f"({a.mean_BER:.2%} to {b.mean_BER:.2%})"
            if abs(reduction) > 1e-12
            else f"leaves mean balanced error unchanged at {b.mean_BER:.2%}"
        )
        conclusion = (
            f"Bounded tuning {error_change}, with {caught_direction} failures caught "
            f"({int(a.pooled_TP)} to {int(b.pooled_TP)}) and {alert_direction} false alerts "
            f"({int(a.pooled_FP)} to {int(b.pooled_FP)}). "
            "These results describe the complete selection procedure under shuffled sampling; they do not establish uniform improvement. "
        )
    else:
        conclusion = (
            "Benchmark comparison conclusions are unavailable until both complete procedure summaries are recorded. "
        )
    later_errors = []
    if lr is not None and _period_error_mean(lr) != "unavailable":
        conclusion += f"Later-period logistic-regression balanced error averages {_period_error_mean(lr)}. "
        later_errors.append(lr.BER.mean())
    if krr is not None and not krr.empty:
        main = krr[krr.available.astype(bool) & krr.procedure.eq("krr_cal20_joint")]
        if not main.empty:
            if _period_error_mean(main) != "unavailable":
                conclusion += f"The main kernel-ridge stress test averages {_period_error_mean(main)} balanced error. "
                later_errors.append(main.BER.mean())
            fraction = (main.TP.sum() + main.FP.sum()) / main.n_test.sum()
            conclusion += (
                f"It flags {'most later samples' if fraction > 0.5 else 'later samples'} ({fraction:.1%}), "
                "which describes the alert burden alongside failure recall. "
            )
    if later_errors and tuned_joint is not None and all(value > tuned_joint.mean_BER for value in later_errors):
        conclusion += (
            "The available later-period error means exceed the shuffled tuned benchmark: these stress tests do not reproduce "
            "its class-balanced performance. Their different training and calibration procedures limit direct algorithm comparisons. "
        )
    if diagnostics is not None and not diagnostics.empty:
        selected = diagnostics[
            diagnostics.procedure.isin(["temporal_joint", "krr_cal20_joint"]) & (diagnostics.fold > 0)
        ]
        if not selected.empty:
            low, high = int(selected.calibration_fails.min()), int(selected.calibration_fails.max())
            count = str(low) if low == high else f"{low}–{high}"
            conclusion += f"The selected main-window procedures have {count} failed calibration examples per period. "
            if low < 10:
                conclusion += "Sparse calibration failures trigger threshold-fragility warnings; the recalibration ranges describe instability, not future uncertainty. "
    lines += [
        "",
        "## Conclusions and Next Data Requirements",
        "",
        conclusion.rstrip()
        + "\n\n"
        + "Named measurements, pre-outcome timing, device/tool context, intervention records and independent later data are needed before "
        "claims about stable operational benefit. There is no production-readiness, causal, early-warning or fresh confirmatory superiority claim.",
        "",
    ]
    lines += _render_technical_details(ctx)
    lines += [
        "",
        "## Provenance Appendix",
        "",
        f"Executed modeling source: `{ctx.manifest.get('source_tree', {}).get('sha256', 'unavailable')}`. "
        f"Executed study-spec identity: `{ctx.manifest.get('study_spec_sha256')}`. Git dirty state: `{ctx.manifest.get('git_dirty')}`.",
        "",
        "The execution manifest describes model training. A presentation-only export records its current rendering source separately "
        "in the publication audit record; it does not relabel the executed source. Tables and figures read unchanged audited CSVs. "
        "No model fitting occurs during rendering.",
        "",
        "Method references: [selection bias](https://jmlr.org/papers/v11/cawley10a.html), "
        "[threshold tuning](https://scikit-learn.org/stable/modules/classification_threshold.html), "
        "[cross-validation variance limits](https://www.jmlr.org/papers/volume5/grandvalet04a/grandvalet04a.pdf).",
        "",
    ]
    return lines


def write_final_report(output_dir: Path, *, export_pdf: bool = False, reports_destination: Path | None = None) -> Path:
    """Audit before rendering the canonical report and six existing figures."""
    _raise_for_failed_report_audit(output_dir)
    ctx = _load_report_context(output_dir)
    target = reports_destination or ctx.reports_dir
    target.mkdir(parents=True, exist_ok=True)
    _write_final_report_figures(ctx, target)
    path = target / ArtifactName.FINAL_REPORT
    _write_markdown_with_optional_pdf(path, _render_study(ctx), export_pdf=export_pdf)
    return path


def write_report_skeleton(output_dir: Path, *, reports_destination: Path | None = None) -> Path:
    """Render the shared narrative as a debugging scaffold without audit/publication."""
    ctx = _load_report_context(output_dir)
    lines = _render_study(ctx)
    lines[0] = "# Final Report Skeleton"
    target = reports_destination or ctx.reports_dir
    target.mkdir(parents=True, exist_ok=True)
    path = target / ArtifactName.REPORT_SKELETON
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
