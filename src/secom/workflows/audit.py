from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from secom.artifacts import ValidationResult, validate_required_artifacts, validate_schema_and_logic
from secom.config import ArtifactName, LaneAClassifier, ModelScope, ReplicationMode, SelectorName, ThresholdPolicy
from secom.qa import validate_lane_a_artifacts


def run_artifact_audit(output_dir: Path) -> ValidationResult:
    reports = output_dir / "reports"
    manifest = json.loads((reports / ArtifactName.MANIFEST).read_text(encoding="utf-8"))
    lane_b_feasible = bool(manifest.get("lane_b_feasible", False))

    errors = []
    errors.extend(validate_required_artifacts(output_dir=output_dir, lane_b_feasible=lane_b_feasible))
    schema = validate_schema_and_logic(output_dir=output_dir)
    errors.extend(schema.errors)

    strict_path = reports / ArtifactName.BASELINE_STRICT
    mi_path = reports / ArtifactName.BASELINE_MI
    ablation_path = reports / ArtifactName.BASELINE_ABLATION
    summary_path = reports / ArtifactName.BASELINE_SUMMARY
    tuning_trace_path = reports / ArtifactName.BASELINE_TUNING_TRACE
    if (
        strict_path.exists()
        and mi_path.exists()
        and ablation_path.exists()
        and summary_path.exists()
        and tuning_trace_path.exists()
    ):
        strict_df = pd.read_csv(strict_path)
        mi_df = pd.read_csv(mi_path)
        ablation_df = pd.read_csv(ablation_path)
        summary_df = pd.read_csv(summary_path)
        tuning_trace_df = pd.read_csv(tuning_trace_path)
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
            validate_lane_a_artifacts(
                summary_df=summary_df,
                ablation_df=ablation_df,
                strict_df=strict_df,
                mi_df=mi_df,
                tuning_trace_df=tuning_trace_df,
                classifiers_run=classifiers_run,
                selectors_run=selectors_run,
            )
        except ValueError as exc:
            errors.append(str(exc))

        if LaneAClassifier.KRR_STRICT in classifiers_run:
            f_strict = summary_df[
                (summary_df["classifier"] == LaneAClassifier.KRR_STRICT)
                & (summary_df["selector"] == SelectorName.F_TEST)
                & (summary_df["replication_mode"] == ReplicationMode.STRICT)
            ]
            if len(f_strict) != 1:
                errors.append(
                    "benchmark claim gate requires exactly one row for "
                    "classifier=krr_strict, selector=F-test, replication_mode=strict"
                )

    if lane_b_feasible and (reports / ArtifactName.FINAL_LOCKBOX).exists():
        lock = pd.read_csv(reports / ArtifactName.FINAL_LOCKBOX)
        mspc = pd.read_csv(reports / ArtifactName.MSPC)
        drift = pd.read_csv(reports / ArtifactName.DRIFT_GATE)
        mspc_lock = mspc[mspc["eval_scope"] == "lockbox"]
        if mspc_lock.empty:
            errors.append("mspc lockbox row missing for claim gate")
        else:
            mspc_tpr = float(mspc_lock.iloc[0]["best_MSPC_TPR_at_TNR90"])
            for role in lock["role"].unique():
                row = lock[
                    (lock["role"] == role)
                    & (lock["threshold_policy"] == ThresholdPolicy.SCIENTIFIC)
                ]
                if row.empty:
                    continue
                sup_tpr = float(row.iloc[0]["TPR_at_TNR90"])
                scope = ModelScope.PRIMARY_FROZEN if role == "primary" else ModelScope.CHALLENGER_FROZEN
                drift_row = drift[drift["model_scope"] == scope]
                if drift_row.empty:
                    errors.append(f"drift gate row missing for role={role}")
                else:
                    status = str(drift_row.iloc[0]["drift_gate_status"])
                    if status == "HIGH_SHIFT" and sup_tpr > mspc_tpr:
                        errors.append(
                            f"invalid claim condition: role={role} better than MSPC but HIGH_SHIFT"
                        )

    return ValidationResult(ok=len(errors) == 0, errors=errors)

