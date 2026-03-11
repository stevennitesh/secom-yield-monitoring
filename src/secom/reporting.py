from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from secom.config import ArtifactName, StudyStatus


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _format_float(value: object) -> str:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "n/a"
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def _top_benchmark_table(benchmark_summary: pd.DataFrame) -> list[str]:
    table = benchmark_summary.sort_values(
        ["mean_BER", "selector", "classifier", "replication_mode"]
    ).copy()
    lines = [
        "| selector | classifier | mode | mean_BER | CI_low | CI_high | mean_TPR | mean_TNR |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"| {row['selector']} | {row['classifier']} | {row['replication_mode']} |"
            f" {_format_float(row['mean_BER'])} | {_format_float(row['CI_lower_BER'])} | {_format_float(row['CI_upper_BER'])} |"
            f" {_format_float(row['mean_True+'])} | {_format_float(row['mean_True-'])} |"
        )
    return lines


def _best_row_feature_table(
    feature_report: pd.DataFrame,
    selector: str,
    classifier: str,
    replication_mode: str,
) -> list[str]:
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
    if rows["conditional_effect_magnitude"].isna().all():
        lines = [
            "- Effect magnitudes are unavailable for the leading classifier, so this table is shown as a stability-first view.",
            "",
            "| feature | type | selection_frequency | cluster_id |",
            "|---|---|---:|---:|",
        ]
        for row in rows.itertuples(index=False):
            lines.append(
                f"| {row.feature_name_or_source_col} | {row.feature_type} |"
                f" {_format_float(row.selection_frequency)} | {_format_float(row.cluster_id)} |"
            )
        return lines

    lines = [
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


def write_report_skeleton(output_dir: Path) -> Path:
    reports = output_dir / "reports"
    manifest_path = reports / ArtifactName.MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    benchmark_summary = _read_csv(reports / ArtifactName.BENCHMARK_SUMMARY)
    benchmark_ablation = _read_csv(reports / ArtifactName.BENCHMARK_ABLATION)
    feature_report = _read_csv(reports / ArtifactName.FEATURE_REPORT)
    temporal_selection = _read_csv(reports / ArtifactName.TEMPORAL_MODEL_SELECTION)
    temporal_lockbox = _read_csv(reports / ArtifactName.TEMPORAL_LOCKBOX)
    temporal_drift = _read_csv(reports / ArtifactName.TEMPORAL_DRIFT)
    temporal_mspc = _read_csv(reports / ArtifactName.TEMPORAL_MSPC)
    temporal_manager = _read_csv(reports / ArtifactName.TEMPORAL_MANAGER_OUTPUTS)

    best_benchmark_row = None
    if benchmark_summary is not None and not benchmark_summary.empty:
        best_benchmark_row = benchmark_summary.sort_values(
            ["mean_BER", "selector", "classifier", "replication_mode"]
        ).iloc[0]

    primary_temporal_row = None
    if temporal_selection is not None and not temporal_selection.empty:
        primary_rows = temporal_selection[temporal_selection["is_primary"].astype(bool)]
        if not primary_rows.empty:
            primary_temporal_row = primary_rows.iloc[0]

    scientific_lockbox_row = None
    if temporal_lockbox is not None and not temporal_lockbox.empty:
        scientific_rows = temporal_lockbox[
            temporal_lockbox["threshold_policy"] == "scientific"
        ]
        if not scientific_rows.empty:
            scientific_lockbox_row = scientific_rows.iloc[0]

    drift_row = None
    if temporal_drift is not None and not temporal_drift.empty:
        drift_row = temporal_drift.iloc[0]

    mspc_lockbox_row = None
    if temporal_mspc is not None and not temporal_mspc.empty:
        rows = temporal_mspc[temporal_mspc["eval_scope"] == "lockbox"]
        if not rows.empty:
            mspc_lockbox_row = rows.iloc[0]

    manager_scientific_row = None
    if temporal_manager is not None and not temporal_manager.empty:
        rows = temporal_manager[temporal_manager["threshold_policy"] == "scientific"]
        if not rows.empty:
            manager_scientific_row = rows.iloc[0]

    lines: list[str] = []
    lines.append("# Final Report Skeleton")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- Primary study status: `{manifest.get('primary_study_status', StudyStatus.NOT_RUN)}`")
    lines.append(f"- Temporal robustness status: `{manifest.get('temporal_robustness_status', StudyStatus.NOT_RUN)}`")
    restrictions = manifest.get("temporal_claim_restrictions", [])
    lines.append(f"- Temporal claim restrictions: `{len(restrictions)}`")
    if best_benchmark_row is not None:
        lines.append(
            "- Leading benchmark configuration:"
            f" `{best_benchmark_row['selector']}` / `{best_benchmark_row['classifier']}` /"
            f" `{best_benchmark_row['replication_mode']}` with mean BER `{_format_float(best_benchmark_row['mean_BER'])}`"
        )
    if primary_temporal_row is not None:
        lines.append(
            "- Temporal primary selector:"
            f" `{primary_temporal_row['selector']}` with mean BER `{_format_float(primary_temporal_row['mean_BER'])}`"
        )
    lines.append("")
    lines.append("## Dataset and Study Scope")
    lines.append("")
    lines.append("Summarize the SECOM benchmark context, the primary replication objective, and the role of temporal robustness as secondary evidence.")
    lines.append("")
    lines.append("## Benchmark Replication Design")
    lines.append("")
    lines.append("Describe the full-dataset replication protocol, in-fold preprocessing, in-fold feature selection, and missing-indicator ablation.")
    lines.append("")
    lines.append("## Benchmark Replication Results")
    lines.append("")
    if best_benchmark_row is not None:
        lines.append("### Primary Evidence Table")
        lines.append("")
        lines.extend(_top_benchmark_table(benchmark_summary))
        lines.append("")
        lines.append("### Lead Configuration")
        lines.append("")
        lines.append(
            f"- Selector: `{best_benchmark_row['selector']}`\n"
            f"- Classifier: `{best_benchmark_row['classifier']}`\n"
            f"- Replication mode: `{best_benchmark_row['replication_mode']}`\n"
            f"- Mean BER: `{_format_float(best_benchmark_row['mean_BER'])}`\n"
            f"- 95% CI: `{_format_float(best_benchmark_row['CI_lower_BER'])}` to `{_format_float(best_benchmark_row['CI_upper_BER'])}`"
        )
    else:
        lines.append("- Benchmark summary artifact missing or empty.")
    if benchmark_ablation is not None and not benchmark_ablation.empty:
        lines.append("")
        lines.append("- Missing-indicator ablation summary:")
        for row in benchmark_ablation.itertuples(index=False):
            lines.append(
                f"  - {row.selector} / {row.classifier}: delta_BER={_format_float(row.delta_BER)}"
            )
    lines.append("")
    lines.append("## Feature Stability and Interpretation")
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
    lines.append("## Temporal Robustness Stress Test")
    lines.append("")
    if primary_temporal_row is not None:
        lines.append(
            "- Primary temporal selector:"
            f" `{primary_temporal_row['selector']}`"
            f" with mean_BER=`{_format_float(primary_temporal_row['mean_BER'])}`"
        )
    if scientific_lockbox_row is not None:
        lines.append(
            "- Scientific lockbox row:"
            f" BER=`{_format_float(scientific_lockbox_row['BER'])}`,"
            f" TPR=`{_format_float(scientific_lockbox_row['True+'])}`,"
            f" TNR=`{_format_float(scientific_lockbox_row['True-'])}`"
        )
    if drift_row is not None:
        lines.append(
            "- Drift status:"
            f" `{drift_row['drift_gate_status']}`"
            f" with max_PSI=`{_format_float(drift_row['max_PSI'])}`"
        )
    if mspc_lockbox_row is not None:
        lines.append(
            "- Lockbox MSPC matched-TNR result:"
            f" source=`{mspc_lockbox_row['best_MSPC_source']}`"
            f" TPR_at_TNR90=`{_format_float(mspc_lockbox_row['best_MSPC_TPR_at_TNR90'])}`"
        )
    if manager_scientific_row is not None:
        lines.append(
            "- Illustrative workload framing:"
            f" predicted_flag_fraction=`{_format_float(manager_scientific_row['predicted_flag_fraction'])}`,"
            f" mean_weekly_flagged_wafers=`{_format_float(manager_scientific_row['mean_weekly_flagged_wafers'])}`"
        )
    if restrictions:
        lines.append("- Temporal claim restrictions:")
        for restriction in restrictions:
            lines.append(f"  - `{restriction}`")
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
    lines.append("- State what the benchmark study replicated successfully.")
    lines.append("- State what the temporal stress test supports or restricts.")
    lines.append("- State what data would be needed for a true industrial deployment study.")
    lines.append("")

    out_path = reports / ArtifactName.REPORT_SKELETON
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
