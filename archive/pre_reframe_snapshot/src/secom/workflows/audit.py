from __future__ import annotations

import json
from pathlib import Path

from secom.artifacts import (
    ValidationResult,
    load_artifact_frames,
    validate_required_artifacts,
    validate_schema_and_logic,
)
from secom.config import ArtifactName, LaneAClassifier, ModelScope, ReplicationMode, SelectorName, ThresholdPolicy
from secom.qa import validate_lane_a_global_artifacts


def run_artifact_audit(output_dir: Path) -> ValidationResult:
    reports = output_dir / "reports"
    manifest = json.loads((reports / ArtifactName.MANIFEST).read_text(encoding="utf-8"))
    lane_b_feasible = bool(manifest.get("lane_b_feasible", False))

    artifact_frames = load_artifact_frames(output_dir)
    errors = []
    errors.extend(validate_required_artifacts(output_dir=output_dir, lane_b_feasible=lane_b_feasible))
    schema = validate_schema_and_logic(output_dir=output_dir, artifact_frames=artifact_frames)
    errors.extend(schema.errors)

    sweep_df = artifact_frames.get(ArtifactName.LANE_A_GLOBAL_SWEEP)
    best_df = artifact_frames.get(ArtifactName.LANE_A_GLOBAL_BEST_CONFIG)
    fold_metrics_df = artifact_frames.get(ArtifactName.LANE_A_GLOBAL_FOLD_METRICS)
    summary_df = artifact_frames.get(ArtifactName.LANE_A_GLOBAL_SUMMARY)
    ablation_df = artifact_frames.get(ArtifactName.LANE_A_GLOBAL_ABLATION)
    full_fit_df = artifact_frames.get(ArtifactName.LANE_A_GLOBAL_FULL_FIT_SUMMARY)
    if all(df is not None for df in (sweep_df, best_df, fold_metrics_df, ablation_df, summary_df, full_fit_df)):
        classifiers_run = (
            sorted(summary_df["classifier"].dropna().astype(str).unique().tolist())
            if "classifier" in summary_df.columns
            else []
        )
        selectors_run = (
            sorted(summary_df["selector"].dropna().astype(str).unique().tolist())
            if "selector" in summary_df.columns
            else []
        )
        try:
            validate_lane_a_global_artifacts(
                sweep_df=sweep_df,
                best_df=best_df,
                fold_metrics_df=fold_metrics_df,
                summary_df=summary_df,
                ablation_df=ablation_df,
                full_fit_df=full_fit_df,
                classifiers_run=classifiers_run,
                selectors_run=selectors_run,
            )
        except ValueError as exc:
            errors.append(str(exc))

        if LaneAClassifier.KRR in classifiers_run:
            f_strict = summary_df[
                (summary_df["classifier"] == LaneAClassifier.KRR)
                & (summary_df["selector"] == SelectorName.F_TEST)
                & (summary_df["replication_mode"] == ReplicationMode.STRICT)
            ]
            f_mi = summary_df[
                (summary_df["classifier"] == LaneAClassifier.KRR)
                & (summary_df["selector"] == SelectorName.F_TEST)
                & (summary_df["replication_mode"] == ReplicationMode.WITH_MISSING_INDICATORS)
            ]
            if len(f_strict) != 1:
                errors.append(
                    "benchmark claim gate requires exactly one row for "
                    "classifier=krr, selector=F-test, replication_mode=strict"
                )
            if len(f_mi) != 1:
                errors.append(
                    "benchmark claim gate requires exactly one row for "
                    "classifier=krr, selector=F-test, replication_mode=with_missing_indicators"
                )

    lock = artifact_frames.get(ArtifactName.FINAL_LOCKBOX)
    mspc = artifact_frames.get(ArtifactName.MSPC)
    drift = artifact_frames.get(ArtifactName.DRIFT_GATE)
    if lane_b_feasible and lock is not None and mspc is not None and drift is not None:
        mspc_lock = mspc[mspc["eval_scope"] == "lockbox"]
        if mspc_lock.empty:
            errors.append("mspc lockbox row missing for claim gate")
        else:
            mspc_tpr = float(mspc_lock.iloc[0]["best_MSPC_TPR_at_TNR90"])
            lock_tpr_by_role = {
                str(row.role): float(row.TPR_at_TNR90)
                for row in lock.itertuples(index=False)
                if str(row.threshold_policy) == ThresholdPolicy.SCIENTIFIC
            }
            drift_status_by_scope = {
                str(row.model_scope): str(row.drift_gate_status)
                for row in drift.itertuples(index=False)
            }
            for role, sup_tpr in lock_tpr_by_role.items():
                scope = ModelScope.PRIMARY_FROZEN if role == "primary" else ModelScope.CHALLENGER_FROZEN
                status = drift_status_by_scope.get(scope)
                if status is None:
                    errors.append(f"drift gate row missing for role={role}")
                elif status == "HIGH_SHIFT" and sup_tpr > mspc_tpr:
                    errors.append(
                        f"invalid claim condition: role={role} better than MSPC but HIGH_SHIFT"
                    )

    return ValidationResult(ok=len(errors) == 0, errors=errors)

