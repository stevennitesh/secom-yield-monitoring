"""Artifact writing, manifest normalization, and audit validation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from secom.config import ArtifactName, MANIFEST_REQUIRED_KEYS, StudyStatus


@dataclass(frozen=True)
class ValidationResult:
    """Result returned by artifact and study-audit validation."""

    ok: bool
    errors: list[str]
    warnings: list[str]
    claim_restrictions: list[str]


@dataclass(frozen=True)
class _ManifestState:
    """Validated manifest status fields used by artifact validation."""

    primary_status: str
    original_status: str
    tuned_status: str
    temporal_status: str
    claim_restrictions: list[str]


_CSV_ARTIFACT_NAMES = sorted(
    value
    for name, value in vars(ArtifactName).items()
    if not name.startswith("_") and isinstance(value, str) and value.endswith(".csv")
)

_BENCHMARK_ORIGINAL_REQUIRED_COLUMNS: dict[str, set[str]] = {
    ArtifactName.BENCHMARK_SWEEP: {
        "selector",
        "classifier",
        "replication_mode",
        "mean_BER",
        "mean_True+",
        "mean_True-",
        "mean_ROC_AUC",
        "mean_PR_AUC",
        "mean_MCC",
        "mean_F2",
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
        "ROC_AUC",
        "PR_AUC",
        "MCC",
        "F2",
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
        "mean_ROC_AUC",
        "mean_PR_AUC",
        "mean_MCC",
        "mean_F2",
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
        "ROC_AUC_full_dataset",
        "PR_AUC_full_dataset",
        "MCC_full_dataset",
        "F2_full_dataset",
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

_BENCHMARK_TUNED_REQUIRED_COLUMNS: dict[str, set[str]] = {
    ArtifactName.BENCHMARK_TUNED_SEARCH: {
        "selector",
        "classifier",
        "replication_mode",
        "fold",
        "mean_inner_ROC_AUC",
        "mean_inner_BER",
        "is_selected_config",
    },
    ArtifactName.BENCHMARK_TUNED_BEST_CONFIG: {
        "selector",
        "classifier",
        "replication_mode",
        "mean_BER",
        "mean_ROC_AUC",
        "k",
    },
    ArtifactName.BENCHMARK_TUNED_FOLD_METRICS: {
        "selector",
        "classifier",
        "replication_mode",
        "fold",
        "BER",
        "True+",
        "True-",
        "ROC_AUC",
        "PR_AUC",
        "MCC",
        "F2",
    },
    ArtifactName.BENCHMARK_TUNED_SUMMARY: {
        "selector",
        "classifier",
        "replication_mode",
        "mean_BER",
        "CI_lower_BER",
        "CI_upper_BER",
        "mean_True+",
        "mean_True-",
        "mean_ROC_AUC",
        "mean_PR_AUC",
        "mean_MCC",
        "mean_F2",
    },
    ArtifactName.BENCHMARK_TUNED_ABLATION: {
        "selector",
        "classifier",
        "BER_reference",
        "BER_missing_indicator",
        "delta_BER",
    },
    ArtifactName.BENCHMARK_TUNED_FULL_FIT_SUMMARY: {
        "selector",
        "classifier",
        "replication_mode",
        "BER_full_dataset",
        "True+_full_dataset",
        "True-_full_dataset",
        "ROC_AUC_full_dataset",
        "PR_AUC_full_dataset",
        "MCC_full_dataset",
        "F2_full_dataset",
    },
    ArtifactName.BENCHMARK_TUNED_FEATURE_STABILITY: {
        "selector",
        "classifier",
        "replication_mode",
        "resample_id",
        "feature_index",
        "feature_type",
        "selected",
    },
    ArtifactName.BENCHMARK_TUNED_FEATURE_REPORT: {
        "selector",
        "classifier",
        "replication_mode",
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

_BENCHMARK_ORIGINAL_ARTIFACTS = tuple(_BENCHMARK_ORIGINAL_REQUIRED_COLUMNS)
_BENCHMARK_TUNED_ARTIFACTS = tuple(_BENCHMARK_TUNED_REQUIRED_COLUMNS)
_TEMPORAL_ARTIFACTS = tuple(_TEMPORAL_REQUIRED_COLUMNS)


def ensure_reports_dir(output_dir: Path) -> Path:
    """Return the reports directory, creating it if needed."""
    reports = output_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    return reports


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a report artifact CSV with parent-directory creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    """Read a CSV artifact when present, otherwise return ``None``."""
    if not path.exists():
        return None
    return pd.read_csv(path)


def _normalize_float(x: float) -> float | None:
    """Normalize non-finite floats to JSON null and round finite floats."""
    if x is None:
        return None
    if not np.isfinite(float(x)):
        return None
    return float(f"{float(x):.6g}")


def normalize_for_manifest(value: Any) -> Any:
    """Recursively convert numpy/pandas-adjacent values into stable JSON values."""
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


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    """Validate and write the run manifest in deterministic JSON form."""
    payload = normalize_for_manifest(manifest)
    missing = [k for k in MANIFEST_REQUIRED_KEYS if k not in payload]
    if missing:
        raise ValueError(f"Manifest missing required keys: {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True, indent=2, ensure_ascii=True)


def read_manifest(path: Path) -> dict[str, Any]:
    """Read a run manifest JSON document from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def _required_artifacts_by_study(
    *,
    benchmark_original_status: str,
    benchmark_tuned_status: str,
    temporal_status: str,
) -> list[str]:
    """Return required artifacts using separate original/tuned/temporal statuses."""
    names = [ArtifactName.MANIFEST]
    if benchmark_original_status != StudyStatus.NOT_RUN:
        names.extend(_BENCHMARK_ORIGINAL_ARTIFACTS)
    if benchmark_tuned_status != StudyStatus.NOT_RUN:
        names.extend(_BENCHMARK_TUNED_ARTIFACTS)
    if temporal_status != StudyStatus.NOT_RUN:
        names.extend(_TEMPORAL_ARTIFACTS)
    return names


def validate_required_artifacts(
    output_dir: Path,
    *,
    benchmark_original_status: str,
    benchmark_tuned_status: str,
    temporal_status: str,
) -> list[str]:
    """Return missing artifact errors for the manifest-declared active study layers."""
    reports = output_dir / "reports"
    required = _required_artifacts_by_study(
        benchmark_original_status=benchmark_original_status,
        benchmark_tuned_status=benchmark_tuned_status,
        temporal_status=temporal_status,
    )
    return [f"missing artifact: {name}" for name in required if not (reports / name).exists()]


def load_artifact_frames(output_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all present CSV report artifacts by artifact filename."""
    reports = output_dir / "reports"
    frames: dict[str, pd.DataFrame] = {}
    for name in _CSV_ARTIFACT_NAMES:
        df = read_csv_if_exists(reports / name)
        if df is not None:
            frames[name] = df
    return frames


def _validate_required_columns(
    df: pd.DataFrame,
    required: set[str],
    errors: list[str],
    file_name: str,
) -> None:
    """Append a missing-column error for one artifact frame."""
    missing = required - set(df.columns)
    if missing:
        errors.append(f"{file_name}: missing columns {sorted(missing)}")


def _artifact_frame(
    *,
    name: str,
    reports: Path,
    artifact_frames: dict[str, pd.DataFrame] | None,
) -> pd.DataFrame | None:
    """Return a cached artifact frame or load it directly from reports."""
    if artifact_frames is not None:
        return artifact_frames.get(name)
    return read_csv_if_exists(reports / name)


def _validate_manifest_fields(
    manifest: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> _ManifestState:
    """Validate manifest fields and return the normalized status values."""
    missing_manifest_keys = [k for k in MANIFEST_REQUIRED_KEYS if k not in manifest]
    if missing_manifest_keys:
        errors.append(f"{ArtifactName.MANIFEST}: missing keys {missing_manifest_keys}")

    statuses = {
        "primary_study_status": str(manifest.get("primary_study_status", StudyStatus.NOT_RUN)),
        "benchmark_original_status": str(manifest.get("benchmark_original_status", StudyStatus.NOT_RUN)),
        "benchmark_tuned_status": str(manifest.get("benchmark_tuned_status", StudyStatus.NOT_RUN)),
        "temporal_robustness_status": str(manifest.get("temporal_robustness_status", StudyStatus.NOT_RUN)),
    }
    for key, value in statuses.items():
        if value not in StudyStatus.ALL:
            errors.append(f"{ArtifactName.MANIFEST}: invalid {key} {value}")

    restrictions = manifest.get("temporal_claim_restrictions", [])
    if not isinstance(restrictions, list):
        errors.append(f"{ArtifactName.MANIFEST}: temporal_claim_restrictions must be a list")
        claim_restrictions = []
    else:
        claim_restrictions = [str(x) for x in restrictions]

    industrialization_notes = manifest.get("industrialization_notes", [])
    if not isinstance(industrialization_notes, list):
        errors.append(f"{ArtifactName.MANIFEST}: industrialization_notes must be a list")

    if statuses["primary_study_status"] == StudyStatus.FAILED:
        errors.append("primary study status indicates failure")
    elif statuses["primary_study_status"] == StudyStatus.WARNING:
        warnings.append("primary study status indicates warnings")

    if statuses["temporal_robustness_status"] == StudyStatus.FAILED:
        warnings.append("temporal robustness status indicates failure")
    elif statuses["temporal_robustness_status"] == StudyStatus.WARNING:
        warnings.append("temporal robustness status indicates warnings")

    return _ManifestState(
        primary_status=statuses["primary_study_status"],
        original_status=statuses["benchmark_original_status"],
        tuned_status=statuses["benchmark_tuned_status"],
        temporal_status=statuses["temporal_robustness_status"],
        claim_restrictions=claim_restrictions,
    )


def _validate_artifact_family(
    *,
    reports: Path,
    artifact_frames: dict[str, pd.DataFrame] | None,
    required_columns: dict[str, set[str]],
    active: bool,
    errors: list[str],
) -> None:
    """Validate required columns for present artifacts and required presence for active layers."""
    for name, required in required_columns.items():
        df = _artifact_frame(name=name, reports=reports, artifact_frames=artifact_frames)
        if df is not None:
            _validate_required_columns(df, required, errors, name)
        elif active:
            errors.append(f"missing artifact: {name}")


def _warn_stale_artifact_family(
    *,
    reports: Path,
    artifact_frames: dict[str, pd.DataFrame] | None,
    artifact_names: tuple[str, ...],
    active: bool,
    warning_prefix: str,
    warnings: list[str],
) -> None:
    """Warn when inactive study layers still have present artifact files."""
    if active:
        return
    warnings.extend(
        f"{warning_prefix}: {name}"
        for name in artifact_names
        if _artifact_frame(name=name, reports=reports, artifact_frames=artifact_frames) is not None
    )


def validate_schema_and_logic(
    output_dir: Path,
    artifact_frames: dict[str, pd.DataFrame] | None = None,
    manifest: dict[str, Any] | None = None,
) -> ValidationResult:
    """Validate manifest status, required schemas, and artifact/status consistency."""
    reports = output_dir / "reports"
    errors: list[str] = []
    warnings: list[str] = []

    if manifest is None:
        manifest_path = reports / ArtifactName.MANIFEST
        if not manifest_path.exists():
            return ValidationResult(
                ok=False,
                errors=[f"missing artifact: {ArtifactName.MANIFEST}"],
                warnings=[],
                claim_restrictions=[],
            )
        manifest = read_manifest(manifest_path)

    state = _validate_manifest_fields(manifest=manifest, errors=errors, warnings=warnings)

    # Active layers require artifacts; inactive layers only warn if stale artifacts remain.
    active_original = state.original_status != StudyStatus.NOT_RUN
    active_tuned = state.tuned_status != StudyStatus.NOT_RUN
    active_temporal = state.temporal_status != StudyStatus.NOT_RUN

    _validate_artifact_family(
        reports=reports,
        artifact_frames=artifact_frames,
        required_columns=_BENCHMARK_ORIGINAL_REQUIRED_COLUMNS,
        active=active_original,
        errors=errors,
    )
    _validate_artifact_family(
        reports=reports,
        artifact_frames=artifact_frames,
        required_columns=_BENCHMARK_TUNED_REQUIRED_COLUMNS,
        active=active_tuned,
        errors=errors,
    )
    _validate_artifact_family(
        reports=reports,
        artifact_frames=artifact_frames,
        required_columns=_TEMPORAL_REQUIRED_COLUMNS,
        active=active_temporal,
        errors=errors,
    )

    _warn_stale_artifact_family(
        reports=reports,
        artifact_frames=artifact_frames,
        artifact_names=_BENCHMARK_ORIGINAL_ARTIFACTS,
        active=active_original,
        warning_prefix=f"original benchmark artifact present while benchmark_original_status is {StudyStatus.NOT_RUN}",
        warnings=warnings,
    )
    _warn_stale_artifact_family(
        reports=reports,
        artifact_frames=artifact_frames,
        artifact_names=_BENCHMARK_TUNED_ARTIFACTS,
        active=active_tuned,
        warning_prefix=f"tuned benchmark artifact present while benchmark_tuned_status is {StudyStatus.NOT_RUN}",
        warnings=warnings,
    )
    _warn_stale_artifact_family(
        reports=reports,
        artifact_frames=artifact_frames,
        artifact_names=_TEMPORAL_ARTIFACTS,
        active=active_temporal,
        warning_prefix=f"temporal artifact present while temporal robustness status is {StudyStatus.NOT_RUN}",
        warnings=warnings,
    )

    deduped_warnings = list(dict.fromkeys(warnings))
    deduped_restrictions = list(dict.fromkeys(state.claim_restrictions))
    return ValidationResult(
        ok=len(errors) == 0,
        errors=errors,
        warnings=deduped_warnings,
        claim_restrictions=deduped_restrictions,
    )
