from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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


def _format_cell(value: object) -> str:
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return _format_float(float(value))
    return str(value)


def _markdown_table(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    headers: list[str] | None = None,
    max_rows: int | None = None,
) -> list[str]:
    table = frame.loc[:, columns].copy()
    if max_rows is not None:
        table = table.head(max_rows)
    header_row = headers if headers is not None else columns
    lines = [
        "| " + " | ".join(header_row) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in table.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format_cell(value) for value in row) + " |")
    return lines


def _top_benchmark_table(benchmark_summary: pd.DataFrame) -> list[str]:
    table = benchmark_summary.sort_values(
        ["mean_BER", "selector", "classifier", "replication_mode"]
    ).copy()
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
    table = benchmark_summary.sort_values(
        ["mean_BER", "selector", "classifier", "replication_mode"]
    ).copy()
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
    benchmark_tuned_summary = _read_csv(reports / ArtifactName.BENCHMARK_TUNED_SUMMARY)
    benchmark_tuned_ablation = _read_csv(reports / ArtifactName.BENCHMARK_TUNED_ABLATION)
    benchmark_tuned_feature_report = _read_csv(reports / ArtifactName.BENCHMARK_TUNED_FEATURE_REPORT)
    temporal_selection = _read_csv(reports / ArtifactName.TEMPORAL_MODEL_SELECTION)
    temporal_lockbox = _read_csv(reports / ArtifactName.TEMPORAL_LOCKBOX)
    temporal_drift = _read_csv(reports / ArtifactName.TEMPORAL_DRIFT)
    temporal_mspc = _read_csv(reports / ArtifactName.TEMPORAL_MSPC)
    temporal_manager = _read_csv(reports / ArtifactName.TEMPORAL_MANAGER_OUTPUTS)
    temporal_cost = _read_csv(reports / ArtifactName.TEMPORAL_COST_CURVES)

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

    primary_temporal_row = None
    if temporal_selection is not None and not temporal_selection.empty:
        primary_rows = temporal_selection[temporal_selection["is_primary"].astype(bool)]
        if not primary_rows.empty:
            primary_temporal_row = primary_rows.iloc[0]

    drift_row = None
    if temporal_drift is not None and not temporal_drift.empty:
        drift_row = temporal_drift.iloc[0]

    mspc_lockbox_row = None
    if temporal_mspc is not None and not temporal_mspc.empty:
        rows = temporal_mspc[temporal_mspc["eval_scope"] == "lockbox"]
        if not rows.empty:
            mspc_lockbox_row = rows.iloc[0]

    lines: list[str] = []
    lines.append("# Final Report Skeleton")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- Primary study status: `{manifest.get('primary_study_status', StudyStatus.NOT_RUN)}`")
    lines.append(f"- Original replication status: `{manifest.get('benchmark_original_status', StudyStatus.NOT_RUN)}`")
    lines.append(f"- Tuned benchmark status: `{manifest.get('benchmark_tuned_status', StudyStatus.NOT_RUN)}`")
    lines.append(f"- Temporal robustness status: `{manifest.get('temporal_robustness_status', StudyStatus.NOT_RUN)}`")
    restrictions = manifest.get("temporal_claim_restrictions", [])
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
    lines.append("### Original Replication")
    lines.append("")
    if best_benchmark_row is not None:
        lines.append("#### Primary Evidence Table")
        lines.append("")
        lines.extend(_top_benchmark_table(benchmark_summary))
        lines.append("")
        lines.append("#### Supporting Benchmark Metrics")
        lines.append("")
        lines.extend(_supporting_benchmark_table(benchmark_summary))
        lines.append("")
        lines.append("#### Lead Configuration")
        lines.append("")
        lines.append(
            f"- Selector: `{best_benchmark_row['selector']}`\n"
            f"- Classifier: `{best_benchmark_row['classifier']}`\n"
            f"- Replication mode: `{best_benchmark_row['replication_mode']}`\n"
            f"- Mean BER: `{_format_float(best_benchmark_row['mean_BER'])}`\n"
            f"- 95% CI: `{_format_float(best_benchmark_row['CI_lower_BER'])}` to `{_format_float(best_benchmark_row['CI_upper_BER'])}`\n"
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
        for row in benchmark_ablation.itertuples(index=False):
            lines.append(
                f"  - {row.selector} / {row.classifier}: delta_BER={_format_float(row.delta_BER)}"
            )
    lines.append("")
    lines.append("### Tuned Benchmark")
    lines.append("")
    if best_tuned_benchmark_row is not None:
        lines.append("#### Primary Evidence Table")
        lines.append("")
        lines.extend(_top_benchmark_table(benchmark_tuned_summary))
        lines.append("")
        lines.append("#### Supporting Benchmark Metrics")
        lines.append("")
        lines.extend(_supporting_benchmark_table(benchmark_tuned_summary))
        lines.append("")
        lines.append("#### Lead Configuration")
        lines.append("")
        lines.append(
            f"- Selector: `{best_tuned_benchmark_row['selector']}`\n"
            f"- Classifier: `{best_tuned_benchmark_row['classifier']}`\n"
            f"- Replication mode: `{best_tuned_benchmark_row['replication_mode']}`\n"
            f"- Mean BER: `{_format_float(best_tuned_benchmark_row['mean_BER'])}`\n"
            f"- 95% CI: `{_format_float(best_tuned_benchmark_row['CI_lower_BER'])}` to `{_format_float(best_tuned_benchmark_row['CI_upper_BER'])}`\n"
            f"- Mean ROC_AUC: `{_format_float(best_tuned_benchmark_row['mean_ROC_AUC'])}`\n"
            f"- Mean PR_AUC: `{_format_float(best_tuned_benchmark_row['mean_PR_AUC'])}`\n"
            f"- Mean MCC: `{_format_float(best_tuned_benchmark_row['mean_MCC'])}`\n"
            f"- Mean F2: `{_format_float(best_tuned_benchmark_row['mean_F2'])}`"
        )
    else:
        lines.append("- Tuned benchmark summary artifact missing or empty.")
    if benchmark_tuned_ablation is not None and not benchmark_tuned_ablation.empty:
        lines.append("")
        lines.append("- Tuned missing-indicator ablation summary:")
        for row in benchmark_tuned_ablation.itertuples(index=False):
            lines.append(
                f"  - {row.selector} / {row.classifier}: delta_BER={_format_float(row.delta_BER)}"
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
                ["study", "selector", "classifier", "mode", "mean_BER", "mean_ROC_AUC", "mean_PR_AUC", "mean_MCC", "mean_F2"],
            )
        )
    lines.append("")
    lines.append("## Feature Stability and Interpretation")
    lines.append("")
    lines.append("### Original Replication")
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
    if primary_temporal_row is not None:
        lines.append(
            "- Primary temporal selector:"
            f" `{primary_temporal_row['selector']}`"
            f" with mean_BER=`{_format_float(primary_temporal_row['mean_BER'])}`"
        )
    if temporal_lockbox is not None and not temporal_lockbox.empty:
        lines.append("")
        lines.append("### Lockbox Metrics")
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
    if drift_row is not None:
        lines.append("")
        lines.append("### Drift Summary")
        lines.append("")
        lines.append(
            "- Drift status:"
            f" `{drift_row['drift_gate_status']}`"
            f" with max_PSI=`{_format_float(drift_row['max_PSI'])}`"
        )
    if mspc_lockbox_row is not None:
        lines.append("")
        lines.append("### MSPC Comparison")
        lines.append("")
        lines.extend(
            _markdown_table(
                temporal_mspc[temporal_mspc["eval_scope"] == "lockbox"],
                ["eval_scope", "best_MSPC_source", "best_MSPC_TPR_at_TNR90", "T2_AUC", "Q_AUC", "alarm_rate", "empirical_ARL0"],
                headers=["scope", "best_source", "best_TPR_at_TNR90", "T2_AUC", "Q_AUC", "alarm_rate", "empirical_ARL0"],
            )
        )
    if temporal_manager is not None and not temporal_manager.empty:
        lines.append("")
        lines.append("### Illustrative Workload Metrics")
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
        lines.append("### Illustrative Cost Curves")
        lines.append("")
        cost_columns = [col for col in temporal_cost.columns if not temporal_cost[col].isna().all()]
        lines.extend(
            _markdown_table(
                temporal_cost.sort_values("cost_ratio"),
                cost_columns,
            )
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
