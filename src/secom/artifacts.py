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
    if isinstance(value, (np.floating, float)):
        return _normalize_float(float(value))
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (str, bool)) or value is None:
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


def validate_schema_and_logic(output_dir: Path) -> ValidationResult:
    reports = output_dir / "reports"
    errors: list[str] = []

    strict = _read_csv_if_exists(reports / ArtifactName.BASELINE_STRICT)
    with_mi = _read_csv_if_exists(reports / ArtifactName.BASELINE_MI)
    if strict is not None and with_mi is not None:
        for req in [
            "selector",
            "classifier",
            "fold",
            "BER",
            "True+",
            "True-",
            "n_train",
            "n_test",
            "n_test_fails",
        ]:
            if req not in strict.columns:
                errors.append(f"{ArtifactName.BASELINE_STRICT}: missing {req}")
            if req not in with_mi.columns:
                errors.append(f"{ArtifactName.BASELINE_MI}: missing {req}")
        _validate_enum_column(
            strict,
            "classifier",
            set(LaneAClassifier.ALL),
            errors,
            ArtifactName.BASELINE_STRICT,
        )
        _validate_enum_column(
            with_mi,
            "classifier",
            set(LaneAClassifier.ALL),
            errors,
            ArtifactName.BASELINE_MI,
        )

    ablation = _read_csv_if_exists(reports / ArtifactName.BASELINE_ABLATION)
    if ablation is not None:
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
            if req not in ablation.columns:
                errors.append(f"{ArtifactName.BASELINE_ABLATION}: missing {req}")
        _validate_enum_column(
            ablation,
            "classifier",
            set(LaneAClassifier.ALL),
            errors,
            ArtifactName.BASELINE_ABLATION,
        )
        if {"BER_strict", "BER_MI", "delta_BER"}.issubset(ablation.columns):
            diff = np.abs(ablation["delta_BER"] - (ablation["BER_strict"] - ablation["BER_MI"]))
            if np.any(diff > 1e-9):
                errors.append(f"{ArtifactName.BASELINE_ABLATION}: delta_BER sign mismatch")

    summary = _read_csv_if_exists(reports / ArtifactName.BASELINE_SUMMARY)
    if summary is not None:
        _validate_enum_column(
            summary,
            "classifier",
            set(LaneAClassifier.ALL),
            errors,
            ArtifactName.BASELINE_SUMMARY,
        )
        _validate_enum_column(
            summary,
            "replication_mode",
            {ReplicationMode.STRICT, ReplicationMode.WITH_MISSING_INDICATORS},
            errors,
            ArtifactName.BASELINE_SUMMARY,
        )

    tuning_trace = _read_csv_if_exists(reports / ArtifactName.BASELINE_TUNING_TRACE)
    if tuning_trace is not None:
        for req in [
            "selector",
            "classifier",
            "fold",
            "replication_mode",
            "chosen_alpha",
            "chosen_gamma",
            "chosen_C",
            "chosen_mrmr_lambda",
            "chosen_mutual_info_n_neighbors",
            "chosen_l1_selector_c",
            "threshold",
            "selector_tuning_scope",
        ]:
            if req not in tuning_trace.columns:
                errors.append(f"{ArtifactName.BASELINE_TUNING_TRACE}: missing {req}")
        _validate_enum_column(
            tuning_trace,
            "classifier",
            {LaneAClassifier.KRR_BALANCED, LaneAClassifier.LOGREG},
            errors,
            ArtifactName.BASELINE_TUNING_TRACE,
        )
        _validate_enum_column(
            tuning_trace,
            "replication_mode",
            {ReplicationMode.STRICT, ReplicationMode.WITH_MISSING_INDICATORS},
            errors,
            ArtifactName.BASELINE_TUNING_TRACE,
        )

    splitwise = _read_csv_if_exists(reports / ArtifactName.SPLITWISE)
    if splitwise is not None:
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

    stage_b_inner = _read_csv_if_exists(reports / ArtifactName.STAGE_B_INNER)
    if stage_b_inner is not None:
        keys = ["selector", "outer_fold", "seed"]
        for key, grp in stage_b_inner.groupby(keys):
            n_selected = int(np.sum(grp["is_selected_config"].astype(bool)))
            if n_selected != 1:
                errors.append(
                    f"{ArtifactName.STAGE_B_INNER}: {key} has {n_selected} selected configs"
                )

    freeze = _read_csv_if_exists(reports / ArtifactName.FREEZE)
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

    final_lockbox = _read_csv_if_exists(reports / ArtifactName.FINAL_LOCKBOX)
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

    mspc = _read_csv_if_exists(reports / ArtifactName.MSPC)
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

    feature_stability = _read_csv_if_exists(reports / ArtifactName.FEATURE_STABILITY)
    if feature_stability is not None:
        _validate_enum_column(
            feature_stability,
            "feature_type",
            {"value", "missing_indicator"},
            errors,
            ArtifactName.FEATURE_STABILITY,
        )

    feature_report = _read_csv_if_exists(reports / ArtifactName.FEATURE_REPORT)
    if feature_report is not None:
        _validate_enum_column(
            feature_report,
            "feature_type",
            {"value", "missing_indicator"},
            errors,
            ArtifactName.FEATURE_REPORT,
        )

    drift = _read_csv_if_exists(reports / ArtifactName.DRIFT_GATE)
    if drift is not None:
        _validate_enum_column(
            drift,
            "model_scope",
            {ModelScope.PRIMARY_FROZEN, ModelScope.CHALLENGER_FROZEN},
            errors,
            ArtifactName.DRIFT_GATE,
        )

    return ValidationResult(ok=len(errors) == 0, errors=errors)
