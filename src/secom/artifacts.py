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
    MANIFEST_REQUIRED_KEYS,
    REQUIRED_ARTIFACTS_PRIMARY,
    REQUIRED_ARTIFACTS_TEMPORAL,
    StudyStatus,
)


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]
    claim_restrictions: list[str]


_CSV_ARTIFACT_NAMES = sorted(
    value
    for name, value in vars(ArtifactName).items()
    if not name.startswith("_") and isinstance(value, str) and value.endswith(".csv")
)

_PRIMARY_REQUIRED_COLUMNS: dict[str, set[str]] = {
    ArtifactName.BENCHMARK_SWEEP: {
        "selector",
        "classifier",
        "replication_mode",
        "mean_BER",
        "mean_True+",
        "mean_True-",
    },
    ArtifactName.BENCHMARK_BEST_CONFIG: {
        "selector",
        "classifier",
        "replication_mode",
    },
    ArtifactName.BENCHMARK_FOLD_METRICS: {
        "selector",
        "classifier",
        "replication_mode",
        "fold",
        "BER",
        "True+",
        "True-",
    },
    ArtifactName.BENCHMARK_SUMMARY: {
        "selector",
        "classifier",
        "replication_mode",
        "mean_BER",
        "CI_lower_BER",
        "CI_upper_BER",
        "mean_True+",
        "mean_True-",
    },
    ArtifactName.BENCHMARK_ABLATION: {
        "selector",
        "classifier",
        "BER_reference",
        "BER_missing_indicator",
        "delta_BER",
    },
    ArtifactName.BENCHMARK_FULL_FIT_SUMMARY: {
        "selector",
        "classifier",
        "replication_mode",
        "BER_full_dataset",
        "True+_full_dataset",
        "True-_full_dataset",
    },
    ArtifactName.FEATURE_STABILITY: {
        "selector",
        "resample_id",
        "feature_index",
        "feature_type",
        "selected",
    },
    ArtifactName.FEATURE_REPORT: {
        "feature_index",
        "feature_type",
        "selection_frequency",
        "conditional_effect_magnitude",
        "expected_contribution",
    },
}

_TEMPORAL_REQUIRED_COLUMNS: dict[str, set[str]] = {
    ArtifactName.TEMPORAL_SPLIT_METADATA: {
        "n_total",
        "n_dev",
        "n_lockbox",
        "split_rule",
    },
    ArtifactName.TEMPORAL_SELECTOR_SCREENING: {
        "selector",
        "mean_BER",
        "std_BER",
    },
    ArtifactName.TEMPORAL_MODEL_SELECTION: {
        "selector",
        "status",
        "is_primary",
        "is_challenger",
        "mean_BER",
    },
    ArtifactName.TEMPORAL_INNER_CV: {
        "selector",
        "resample_id",
        "mean_inner_BER",
        "mean_inner_ROC_AUC",
        "is_selected_config",
    },
    ArtifactName.TEMPORAL_FREEZE: {
        "role",
        "selector",
        "is_frozen_config",
    },
    ArtifactName.TEMPORAL_LOCKBOX: {
        "role",
        "threshold_policy",
        "BER",
        "True+",
        "True-",
        "TPR_at_TNR90",
    },
    ArtifactName.TEMPORAL_DRIFT: {
        "model_scope",
        "drift_gate_status",
        "lockbox_claims_allowed",
    },
    ArtifactName.TEMPORAL_MSPC: {
        "eval_scope",
        "best_MSPC_TPR_at_TNR90",
        "best_MSPC_source",
    },
    ArtifactName.TEMPORAL_COST_CURVES: {
        "cost_ratio",
        "all_pass_baseline",
        "all_flag_baseline",
    },
    ArtifactName.TEMPORAL_MANAGER_OUTPUTS: {
        "role",
        "threshold_policy",
        "predicted_flag_fraction",
        "mean_weekly_flagged_wafers",
    },
}


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
    digest = hashlib.sha256(canonical_json_bytes(config)).hexdigest()
    return digest


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    payload = normalize_for_manifest(manifest)
    missing = [k for k in MANIFEST_REQUIRED_KEYS if k not in payload]
    if missing:
        raise ValueError(f"Manifest missing required keys: {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True, indent=2, ensure_ascii=True)


def _required_artifacts(primary_status: str, temporal_status: str) -> list[str]:
    names = [ArtifactName.MANIFEST]
    if primary_status != StudyStatus.NOT_RUN:
        names.extend(name for name in REQUIRED_ARTIFACTS_PRIMARY if name != ArtifactName.MANIFEST)
    if temporal_status != StudyStatus.NOT_RUN:
        names.extend(REQUIRED_ARTIFACTS_TEMPORAL)
    return names


def validate_required_artifacts(
    output_dir: Path,
    *,
    primary_status: str,
    temporal_status: str,
) -> list[str]:
    reports = output_dir / "reports"
    errors: list[str] = []
    for name in _required_artifacts(primary_status, temporal_status):
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


def _validate_required_columns(
    df: pd.DataFrame,
    required: set[str],
    errors: list[str],
    file_name: str,
) -> None:
    missing = required - set(df.columns)
    if missing:
        errors.append(f"{file_name}: missing columns {sorted(missing)}")


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
    manifest: dict[str, Any] | None = None,
) -> ValidationResult:
    reports = output_dir / "reports"
    errors: list[str] = []
    warnings: list[str] = []
    claim_restrictions: list[str] = []

    if manifest is None:
        manifest_path = reports / ArtifactName.MANIFEST
        if not manifest_path.exists():
            return ValidationResult(
                ok=False,
                errors=[f"missing artifact: {ArtifactName.MANIFEST}"],
                warnings=[],
                claim_restrictions=[],
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    missing_manifest_keys = [k for k in MANIFEST_REQUIRED_KEYS if k not in manifest]
    if missing_manifest_keys:
        errors.append(f"{ArtifactName.MANIFEST}: missing keys {missing_manifest_keys}")

    primary_status = str(manifest.get("primary_study_status", StudyStatus.NOT_RUN))
    temporal_status = str(manifest.get("temporal_robustness_status", StudyStatus.NOT_RUN))
    if primary_status not in StudyStatus.ALL:
        errors.append(f"{ArtifactName.MANIFEST}: invalid primary_study_status {primary_status}")
    if temporal_status not in StudyStatus.ALL:
        errors.append(f"{ArtifactName.MANIFEST}: invalid temporal_robustness_status {temporal_status}")

    restrictions = manifest.get("temporal_claim_restrictions", [])
    if not isinstance(restrictions, list):
        errors.append(f"{ArtifactName.MANIFEST}: temporal_claim_restrictions must be a list")
        restrictions = []
    else:
        claim_restrictions.extend(str(x) for x in restrictions)

    industrialization_notes = manifest.get("industrialization_notes", [])
    if not isinstance(industrialization_notes, list):
        errors.append(f"{ArtifactName.MANIFEST}: industrialization_notes must be a list")

    if primary_status == StudyStatus.FAILED:
        errors.append("primary study status indicates failure")
    elif primary_status == StudyStatus.WARNING:
        warnings.append("primary study status indicates warnings")

    if temporal_status == StudyStatus.FAILED:
        warnings.append("temporal robustness status indicates failure")
    elif temporal_status == StudyStatus.WARNING:
        warnings.append("temporal robustness status indicates warnings")

    active_primary = primary_status != StudyStatus.NOT_RUN
    active_temporal = temporal_status != StudyStatus.NOT_RUN

    for name, required in _PRIMARY_REQUIRED_COLUMNS.items():
        df = _artifact_frame(name=name, reports=reports, artifact_frames=artifact_frames)
        if df is not None:
            _validate_required_columns(df, required, errors, name)
        elif active_primary and name in REQUIRED_ARTIFACTS_PRIMARY:
            errors.append(f"missing artifact: {name}")

    for name, required in _TEMPORAL_REQUIRED_COLUMNS.items():
        df = _artifact_frame(name=name, reports=reports, artifact_frames=artifact_frames)
        if df is not None:
            _validate_required_columns(df, required, errors, name)
        elif active_temporal and name in REQUIRED_ARTIFACTS_TEMPORAL:
            errors.append(f"missing artifact: {name}")

    if not active_primary:
        for name in REQUIRED_ARTIFACTS_PRIMARY:
            if name == ArtifactName.MANIFEST:
                continue
            if _artifact_frame(name=name, reports=reports, artifact_frames=artifact_frames) is not None:
                warnings.append(f"primary artifact present while primary study status is {StudyStatus.NOT_RUN}: {name}")

    if not active_temporal:
        for name in REQUIRED_ARTIFACTS_TEMPORAL:
            if _artifact_frame(name=name, reports=reports, artifact_frames=artifact_frames) is not None:
                warnings.append(
                    f"temporal artifact present while temporal robustness status is {StudyStatus.NOT_RUN}: {name}"
                )

    deduped_warnings = list(dict.fromkeys(warnings))
    deduped_restrictions = list(dict.fromkeys(claim_restrictions))
    return ValidationResult(
        ok=len(errors) == 0,
        errors=errors,
        warnings=deduped_warnings,
        claim_restrictions=deduped_restrictions,
    )
