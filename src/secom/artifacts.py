from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from secom.config import (
    ArtifactName,
    LaneAClassifier,
    MANIFEST_REQUIRED_KEYS,
    ModelScope,
    REQUIRED_ARTIFACTS_LANE_A_ONLY,
    REQUIRED_ARTIFACTS_LANE_B,
    ReplicationMode,
    ScalerName,
    SelectorName,
    ThresholdPolicy,
)


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str]


_CSV_ARTIFACT_NAMES = sorted(
    value
    for name, value in vars(ArtifactName).items()
    if not name.startswith("_") and isinstance(value, str) and value.endswith(".csv")
)


def ensure_reports_dir(output_dir: Path) -> Path:
    reports = output_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    return reports


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _normalize_float(x: float) -> float | None:
    if x is None:
        return None
    if not np.isfinite(float(x)):
        return None
    return float(f"{float(x):.6g}")


def normalize_for_manifest(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: normalize_for_manifest(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_for_manifest(v) for v in value]
    if isinstance(value, tuple):
        return [normalize_for_manifest(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (np.floating, float)):
        return _normalize_float(float(value))
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, str) or value is None:
        return value
    return str(value)


def canonical_json_bytes(data: Any) -> bytes:
    normalized = normalize_for_manifest(data)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def config_hash(config: dict[str, Any]) -> str:
    keys = ["selector", "k", "C", "scaler", "n_neighbors"]
    payload = {k: config.get(k) for k in keys}
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return digest


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    payload = normalize_for_manifest(manifest)
    missing = [k for k in MANIFEST_REQUIRED_KEYS if k not in payload]
    if missing:
        raise ValueError(f"Manifest missing required keys: {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True, indent=2, ensure_ascii=True)


def _required_artifacts(lane_b_feasible: bool) -> list[str]:
    return REQUIRED_ARTIFACTS_LANE_B if lane_b_feasible else REQUIRED_ARTIFACTS_LANE_A_ONLY


def validate_required_artifacts(output_dir: Path, lane_b_feasible: bool) -> list[str]:
    reports = output_dir / "reports"
    errors: list[str] = []
    for name in _required_artifacts(lane_b_feasible):
        if not (reports / name).exists():
            errors.append(f"missing artifact: {name}")
    return errors


def load_artifact_frames(output_dir: Path) -> dict[str, pd.DataFrame]:
    reports = output_dir / "reports"
    frames: dict[str, pd.DataFrame] = {}
    for name in _CSV_ARTIFACT_NAMES:
        df = _read_csv_if_exists(reports / name)
        if df is not None:
            frames[name] = df
    return frames


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _validate_enum_column(
    df: pd.DataFrame, column: str, allowed: set[str], errors: list[str], file_name: str
) -> None:
    if column not in df.columns:
        errors.append(f"{file_name}: missing column {column}")
        return
    bad = set(df[column].dropna().astype(str).unique()) - allowed
    if bad:
        errors.append(f"{file_name}: invalid {column} values {sorted(bad)}")


def _artifact_frame(
    *,
    name: str,
    reports: Path,
    artifact_frames: dict[str, pd.DataFrame] | None,
) -> pd.DataFrame | None:
    if artifact_frames is not None:
        return artifact_frames.get(name)
    return _read_csv_if_exists(reports / name)


def validate_schema_and_logic(
    output_dir: Path,
    artifact_frames: dict[str, pd.DataFrame] | None = None,
) -> ValidationResult:
    reports = output_dir / "reports"
    errors: list[str] = []
    lane_a_classifier_values = set(LaneAClassifier.ALL + LaneAClassifier.OPTIONAL_BENCHMARK)
    lane_a_param_cols = [
        "alpha",
        "gamma",
        "C",
        "n_neighbors",
    ]

    lane_a_sweep = _artifact_frame(
        name=ArtifactName.LANE_A_GLOBAL_SWEEP,
        reports=reports,
        artifact_frames=artifact_frames,
    )
    if lane_a_sweep is not None:
        for req in [
            "selector",
            "classifier",
            "replication_mode",
            *lane_a_param_cols,
            "threshold_oof_global",
            "mean_BER_oof",
            "std_BER_fold",
            "mean_True+_oof",
            "mean_True-_oof",
            "mean_n_selected_features",
            "min_n_selected_features",
            "max_n_selected_features",
            "n_folds",
        ]:
            if req not in lane_a_sweep.columns:
                errors.append(f"{ArtifactName.LANE_A_GLOBAL_SWEEP}: missing {req}")
        _validate_enum_column(
            lane_a_sweep,
            "classifier",
            lane_a_classifier_values,
            errors,
            ArtifactName.LANE_A_GLOBAL_SWEEP,
        )
        _validate_enum_column(
            lane_a_sweep,
            "replication_mode",
            {ReplicationMode.STRICT, ReplicationMode.WITH_MISSING_INDICATORS},
            errors,
            ArtifactName.LANE_A_GLOBAL_SWEEP,
        )

    lane_a_best = _artifact_frame(
        name=ArtifactName.LANE_A_GLOBAL_BEST_CONFIG,
        reports=reports,
        artifact_frames=artifact_frames,
    )
    if lane_a_best is not None:
        for req in [
            "selector",
            "classifier",
            "replication_mode",
            *lane_a_param_cols,
            "threshold_oof_global",
            "mean_BER_oof",
            "std_BER_fold",
            "mean_True+_oof",
            "mean_True-_oof",
            "mean_n_selected_features",
            "min_n_selected_features",
            "max_n_selected_features",
            "n_folds",
            "n_configs_evaluated",
        ]:
            if req not in lane_a_best.columns:
                errors.append(f"{ArtifactName.LANE_A_GLOBAL_BEST_CONFIG}: missing {req}")
        _validate_enum_column(
            lane_a_best,
            "classifier",
            lane_a_classifier_values,
            errors,
            ArtifactName.LANE_A_GLOBAL_BEST_CONFIG,
        )
        _validate_enum_column(
            lane_a_best,
            "replication_mode",
            {ReplicationMode.STRICT, ReplicationMode.WITH_MISSING_INDICATORS},
            errors,
            ArtifactName.LANE_A_GLOBAL_BEST_CONFIG,
        )

    lane_a_fold = _artifact_frame(
        name=ArtifactName.LANE_A_GLOBAL_FOLD_METRICS,
        reports=reports,
        artifact_frames=artifact_frames,
    )
    if lane_a_fold is not None:
        for req in [
            "selector",
            "classifier",
            "replication_mode",
            "fold",
            "BER",
            "True+",
            "True-",
            "n_train",
            "n_test",
            "n_test_fails",
            "n_selected_features",
            "threshold_oof_global",
            *lane_a_param_cols,
        ]:
            if req not in lane_a_fold.columns:
                errors.append(f"{ArtifactName.LANE_A_GLOBAL_FOLD_METRICS}: missing {req}")
        _validate_enum_column(
            lane_a_fold,
            "classifier",
            lane_a_classifier_values,
            errors,
            ArtifactName.LANE_A_GLOBAL_FOLD_METRICS,
        )
        _validate_enum_column(
            lane_a_fold,
            "replication_mode",
            {ReplicationMode.STRICT, ReplicationMode.WITH_MISSING_INDICATORS},
            errors,
            ArtifactName.LANE_A_GLOBAL_FOLD_METRICS,
        )

    lane_a_summary = _artifact_frame(
        name=ArtifactName.LANE_A_GLOBAL_SUMMARY,
        reports=reports,
        artifact_frames=artifact_frames,
    )
    if lane_a_summary is not None:
        for req in [
            "selector",
            "classifier",
            "replication_mode",
            "n_folds",
            "n_boot",
            "boot_seed",
            "mean_BER",
            "std_BER",
            "CI_lower_BER",
            "CI_upper_BER",
            "mean_True+",
            "std_True+",
            "CI_lower_True+",
            "CI_upper_True+",
            "mean_True-",
            "std_True-",
            "CI_lower_True-",
            "CI_upper_True-",
        ]:
            if req not in lane_a_summary.columns:
                errors.append(f"{ArtifactName.LANE_A_GLOBAL_SUMMARY}: missing {req}")
        _validate_enum_column(
            lane_a_summary,
            "classifier",
            lane_a_classifier_values,
            errors,
            ArtifactName.LANE_A_GLOBAL_SUMMARY,
        )
        _validate_enum_column(
            lane_a_summary,
            "replication_mode",
            {ReplicationMode.STRICT, ReplicationMode.WITH_MISSING_INDICATORS},
            errors,
            ArtifactName.LANE_A_GLOBAL_SUMMARY,
        )

    lane_a_ablation = _artifact_frame(
        name=ArtifactName.LANE_A_GLOBAL_ABLATION,
        reports=reports,
        artifact_frames=artifact_frames,
    )
    if lane_a_ablation is not None:
        for req in [
            "selector",
            "classifier",
            "BER_strict",
            "BER_MI",
            "delta_BER",
            "CI_lower",
            "CI_upper",
            "n_boot",
        ]:
            if req not in lane_a_ablation.columns:
                errors.append(f"{ArtifactName.LANE_A_GLOBAL_ABLATION}: missing {req}")
        _validate_enum_column(
            lane_a_ablation,
            "classifier",
            lane_a_classifier_values,
            errors,
            ArtifactName.LANE_A_GLOBAL_ABLATION,
        )
        if {"BER_strict", "BER_MI", "delta_BER"}.issubset(lane_a_ablation.columns):
            diff = np.abs(lane_a_ablation["delta_BER"] - (lane_a_ablation["BER_strict"] - lane_a_ablation["BER_MI"]))
            if np.any(diff > 1e-9):
                errors.append(f"{ArtifactName.LANE_A_GLOBAL_ABLATION}: delta_BER sign mismatch")

    lane_a_full = _artifact_frame(
        name=ArtifactName.LANE_A_GLOBAL_FULL_FIT_SUMMARY,
        reports=reports,
        artifact_frames=artifact_frames,
    )
    if lane_a_full is not None:
        for req in [
            "selector",
            "classifier",
            "replication_mode",
            *lane_a_param_cols,
            "threshold_oof_global",
            "threshold_full_dataset",
            "BER_full_dataset",
            "True+_full_dataset",
            "True-_full_dataset",
            "n_samples_full_dataset",
            "n_fails_full_dataset",
            "n_selected_features_full_dataset",
            "threshold_full_dataset_role",
        ]:
            if req not in lane_a_full.columns:
                errors.append(f"{ArtifactName.LANE_A_GLOBAL_FULL_FIT_SUMMARY}: missing {req}")
        _validate_enum_column(
            lane_a_full,
            "classifier",
            lane_a_classifier_values,
            errors,
            ArtifactName.LANE_A_GLOBAL_FULL_FIT_SUMMARY,
        )
        _validate_enum_column(
            lane_a_full,
            "replication_mode",
            {ReplicationMode.STRICT, ReplicationMode.WITH_MISSING_INDICATORS},
            errors,
            ArtifactName.LANE_A_GLOBAL_FULL_FIT_SUMMARY,
        )

    splitwise = _artifact_frame(
        name=ArtifactName.SPLITWISE,
        reports=reports,
        artifact_frames=artifact_frames,
    )
    if splitwise is not None:
        for req in [
            "selector",
            "outer_fold",
            "seed",
            "train_window",
            "test_window",
            "k",
            "C",
            "scaler",
            "n_neighbors",
            "threshold_policy",
            "outer_threshold",
            "n_test",
            "test_fails",
            "flagged_fraction",
            "BER",
            "True+",
            "True-",
        ]:
            if req not in splitwise.columns:
                errors.append(f"{ArtifactName.SPLITWISE}: missing {req}")
        _validate_enum_column(
            splitwise,
            "selector",
            set(SelectorName.STAGE_B),
            errors,
            ArtifactName.SPLITWISE,
        )
        _validate_enum_column(
            splitwise,
            "scaler",
            {ScalerName.STANDARD, ScalerName.ROBUST},
            errors,
            ArtifactName.SPLITWISE,
        )
        _validate_enum_column(
            splitwise,
            "threshold_policy",
            {ThresholdPolicy.OUTER_TRAIN_YOUDEN},
            errors,
            ArtifactName.SPLITWISE,
        )

    stage_b_inner = _artifact_frame(
        name=ArtifactName.STAGE_B_INNER,
        reports=reports,
        artifact_frames=artifact_frames,
    )
    if stage_b_inner is not None:
        keys = ["selector", "outer_fold", "seed"]
        for key, grp in stage_b_inner.groupby(keys):
            n_selected = int(np.sum(grp["is_selected_config"].astype(bool)))
            if n_selected != 1:
                errors.append(
                    f"{ArtifactName.STAGE_B_INNER}: {key} has {n_selected} selected configs"
                )

    freeze = _artifact_frame(
        name=ArtifactName.FREEZE,
        reports=reports,
        artifact_frames=artifact_frames,
    )
    if freeze is not None:
        if "is_frozen_config" not in freeze.columns:
            errors.append(f"{ArtifactName.FREEZE}: missing is_frozen_config")
        else:
            for role, grp in freeze.groupby("role"):
                selected_cfg = grp.loc[grp["is_frozen_config"].astype(bool), ["selector", "k", "C", "scaler", "n_neighbors"]].drop_duplicates()
                if len(selected_cfg) != 1:
                    errors.append(
                        f"{ArtifactName.FREEZE}: role={role} has {len(selected_cfg)} frozen configs"
                    )

    final_lockbox = _artifact_frame(
        name=ArtifactName.FINAL_LOCKBOX,
        reports=reports,
        artifact_frames=artifact_frames,
    )
    if final_lockbox is not None:
        _validate_enum_column(
            final_lockbox,
            "threshold_policy",
            {ThresholdPolicy.SCIENTIFIC, ThresholdPolicy.OPERATIONAL},
            errors,
            ArtifactName.FINAL_LOCKBOX,
        )
        for role, grp in final_lockbox.groupby("role"):
            if len(grp) != 2:
                continue
            cols = ["threshold_at_TNR90", "TNR_at_TNR90", "TPR_at_TNR90"]
            for col in cols:
                if grp[col].nunique(dropna=False) != 1:
                    errors.append(
                        f"{ArtifactName.FINAL_LOCKBOX}: role={role} column {col} must be identical for scientific/operational"
                    )

    mspc = _artifact_frame(
        name=ArtifactName.MSPC,
        reports=reports,
        artifact_frames=artifact_frames,
    )
    if mspc is not None:
        if "fold_index" in mspc.columns:
            vals = mspc["fold_index"].astype(str)
            if any(v == "nan" for v in vals):
                errors.append(f"{ArtifactName.MSPC}: fold_index has NaN")
        if "eval_scope" in mspc.columns:
            scopes = set(mspc["eval_scope"].astype(str).unique())
            if not {"outer_fold", "lockbox"}.issubset(scopes):
                errors.append(
                    f"{ArtifactName.MSPC}: missing required eval scopes outer_fold/lockbox"
                )

    feature_stability = _artifact_frame(
        name=ArtifactName.FEATURE_STABILITY,
        reports=reports,
        artifact_frames=artifact_frames,
    )
    if feature_stability is not None:
        _validate_enum_column(
            feature_stability,
            "feature_type",
            {"value", "missing_indicator"},
            errors,
            ArtifactName.FEATURE_STABILITY,
        )

    feature_report = _artifact_frame(
        name=ArtifactName.FEATURE_REPORT,
        reports=reports,
        artifact_frames=artifact_frames,
    )
    if feature_report is not None:
        _validate_enum_column(
            feature_report,
            "feature_type",
            {"value", "missing_indicator"},
            errors,
            ArtifactName.FEATURE_REPORT,
        )

    drift = _artifact_frame(
        name=ArtifactName.DRIFT_GATE,
        reports=reports,
        artifact_frames=artifact_frames,
    )
    if drift is not None:
        _validate_enum_column(
            drift,
            "model_scope",
            {ModelScope.PRIMARY_FROZEN, ModelScope.CHALLENGER_FROZEN},
            errors,
            ArtifactName.DRIFT_GATE,
        )

    manager_outputs = _artifact_frame(
        name=ArtifactName.MANAGER_FACING,
        reports=reports,
        artifact_frames=artifact_frames,
    )
    if manager_outputs is not None:
        for req in [
            "role",
            "selector",
            "threshold_policy",
            "dev_sample_count",
            "dev_week_count",
            "weekly_rate",
            "predicted_flag_fraction",
            "mean_weekly_flagged_wafers",
            "mean_weekly_fail_captures",
            "mean_weekly_fail_misses",
            "stage_b_mean_flagged_fraction",
            "lockbox_flagged_fraction",
        ]:
            if req not in manager_outputs.columns:
                errors.append(f"{ArtifactName.MANAGER_FACING}: missing {req}")
        _validate_enum_column(
            manager_outputs,
            "threshold_policy",
            {ThresholdPolicy.SCIENTIFIC, ThresholdPolicy.OPERATIONAL},
            errors,
            ArtifactName.MANAGER_FACING,
        )

    return ValidationResult(ok=len(errors) == 0, errors=errors)
