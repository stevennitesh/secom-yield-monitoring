"""Artifact writing, manifest normalization, and audit validation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from secom.config import ArtifactName, MANIFEST_REQUIRED_KEYS, ModelScope, StudyStatus, ThresholdPolicy
from secom.qa import validate_benchmark_replication_artifacts, validate_tuned_benchmark_artifacts


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

_BENCHMARK_TRIPLET_COLUMNS = {"selector", "classifier", "replication_mode"}
_BENCHMARK_CONFIG_COLUMNS = {"k", "C", "alpha", "gamma", "gamma_multiplier", "n_neighbors"}
_BENCHMARK_METRIC_COLUMNS = {"BER", "True+", "True-", "ROC_AUC", "PR_AUC", "MCC", "F2"}
_BENCHMARK_MEAN_METRIC_COLUMNS = {f"mean_{metric}" for metric in _BENCHMARK_METRIC_COLUMNS}
_BENCHMARK_RANGE_COLUMNS = {f"{bound}_{metric}" for metric in _BENCHMARK_METRIC_COLUMNS for bound in ("min", "max")}
_BENCHMARK_FULL_DATASET_METRIC_COLUMNS = {f"{metric}_full_dataset" for metric in _BENCHMARK_METRIC_COLUMNS}
_BENCHMARK_FULL_DATASET_LINEAGE_COLUMNS = {"threshold_full_dataset"}
_BENCHMARK_ABLATION_COLUMNS = {"selector", "classifier", "BER_reference", "BER_missing_indicator", "delta_BER"}
_BENCHMARK_FEATURE_STABILITY_COLUMNS = {
    "resample_id",
    "feature_index",
    "feature_type",
    "feature_name_or_source_col",
    "selected",
}
_BENCHMARK_FEATURE_REPORT_COLUMNS = {
    "feature_index",
    "feature_type",
    "feature_name_or_source_col",
    "selection_frequency",
    "absolute_scaled_coefficient",
    "stability_weighted_coefficient",
}

_BENCHMARK_ORIGINAL_REQUIRED_COLUMNS: dict[str, set[str]] = {
    ArtifactName.BENCHMARK_SWEEP: {
        *_BENCHMARK_TRIPLET_COLUMNS,
        *_BENCHMARK_CONFIG_COLUMNS,
        "fold",
        "mean_inner_BER",
        "mean_inner_ROC_AUC",
        "threshold_inner_oof",
        "is_selected_config",
    },
    ArtifactName.BENCHMARK_BEST_CONFIG: {
        *_BENCHMARK_TRIPLET_COLUMNS,
        *_BENCHMARK_CONFIG_COLUMNS,
    },
    ArtifactName.BENCHMARK_FOLD_METRICS: {
        *_BENCHMARK_TRIPLET_COLUMNS,
        *_BENCHMARK_CONFIG_COLUMNS,
        "fold",
        *_BENCHMARK_METRIC_COLUMNS,
    },
    ArtifactName.BENCHMARK_SUMMARY: {
        *_BENCHMARK_TRIPLET_COLUMNS,
        *_BENCHMARK_MEAN_METRIC_COLUMNS,
        *_BENCHMARK_RANGE_COLUMNS,
    },
    ArtifactName.BENCHMARK_ABLATION: _BENCHMARK_ABLATION_COLUMNS,
    ArtifactName.BENCHMARK_FULL_FIT_SUMMARY: {
        *_BENCHMARK_TRIPLET_COLUMNS,
        *_BENCHMARK_CONFIG_COLUMNS,
        *_BENCHMARK_FULL_DATASET_LINEAGE_COLUMNS,
        *_BENCHMARK_FULL_DATASET_METRIC_COLUMNS,
    },
    ArtifactName.FEATURE_STABILITY: {
        "selector",
        "replication_mode",
        *_BENCHMARK_FEATURE_STABILITY_COLUMNS,
    },
    ArtifactName.FEATURE_REPORT: _BENCHMARK_TRIPLET_COLUMNS | _BENCHMARK_FEATURE_REPORT_COLUMNS,
}

_BENCHMARK_TUNED_REQUIRED_COLUMNS: dict[str, set[str]] = {
    ArtifactName.BENCHMARK_TUNED_SEARCH: {
        *_BENCHMARK_TRIPLET_COLUMNS,
        *_BENCHMARK_CONFIG_COLUMNS,
        "fold",
        "mean_inner_ROC_AUC",
        "mean_inner_BER",
        "is_selected_config",
    },
    ArtifactName.BENCHMARK_TUNED_BEST_CONFIG: {
        *_BENCHMARK_TRIPLET_COLUMNS,
        "mean_BER",
        "mean_ROC_AUC",
        *_BENCHMARK_CONFIG_COLUMNS,
    },
    ArtifactName.BENCHMARK_TUNED_FOLD_METRICS: {
        *_BENCHMARK_TRIPLET_COLUMNS,
        *_BENCHMARK_CONFIG_COLUMNS,
        "fold",
        *_BENCHMARK_METRIC_COLUMNS,
    },
    ArtifactName.BENCHMARK_TUNED_SUMMARY: {
        *_BENCHMARK_TRIPLET_COLUMNS,
        *_BENCHMARK_MEAN_METRIC_COLUMNS,
        *_BENCHMARK_RANGE_COLUMNS,
    },
    ArtifactName.BENCHMARK_TUNED_ABLATION: _BENCHMARK_ABLATION_COLUMNS,
    ArtifactName.BENCHMARK_TUNED_FULL_FIT_SUMMARY: {
        *_BENCHMARK_TRIPLET_COLUMNS,
        *_BENCHMARK_CONFIG_COLUMNS,
        *_BENCHMARK_FULL_DATASET_LINEAGE_COLUMNS,
        *_BENCHMARK_FULL_DATASET_METRIC_COLUMNS,
    },
    ArtifactName.BENCHMARK_TUNED_FEATURE_STABILITY: {
        *_BENCHMARK_TRIPLET_COLUMNS,
        *_BENCHMARK_FEATURE_STABILITY_COLUMNS,
    },
    ArtifactName.BENCHMARK_TUNED_FEATURE_REPORT: _BENCHMARK_TRIPLET_COLUMNS | _BENCHMARK_FEATURE_REPORT_COLUMNS,
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
        "lockbox_n",
        "lockbox_fails",
        "TP",
        "TN",
        "FP",
        "FN",
        "TPR_exact_lower",
        "TPR_exact_upper",
        "TNR_exact_lower",
        "TNR_exact_upper",
        "interval_semantics",
        "evaluation_semantics",
        "TPR_available",
        "TNR_available",
        "BER_available",
    },
    ArtifactName.TEMPORAL_DRIFT: {
        "model_scope",
        "drift_gate_status",
        "confirmatory_claims_allowed",
        "score_reference",
        "max_missingness_rate_shift",
    },
    ArtifactName.TEMPORAL_MSPC: {
        "eval_scope",
        "calibration_selected_MSPC_TPR_at_TNR90",
        "calibration_selected_MSPC_source",
        "T2_calibration_TPR_at_TNR90",
        "Q_calibration_TPR_at_TNR90",
        "frozen_threshold",
        "T2_frozen_threshold",
        "Q_frozen_threshold",
        "source_selection_region",
        "frozen_BER",
        "frozen_TPR",
        "frozen_TNR",
        "TP",
        "TN",
        "FP",
        "FN",
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
        "mean_weekly_flagged_samples",
        "mean_weekly_flag_fraction",
        "evaluation_region",
        "workload_semantics",
    },
}

_PROCEDURE_PREDICTION_COLUMNS = {
    "sample_id",
    "fold",
    "y_true",
    "score",
    "threshold",
    "prediction",
    "procedure",
    "config_id",
}
_PROCEDURE_FOLD_COLUMNS = {
    "procedure",
    "fold",
    "BER",
    "True+",
    "True-",
    "TP",
    "TN",
    "FP",
    "FN",
    "n_test",
    "n_test_fails",
}
for _prefix, _family in (
    ("benchmark", _BENCHMARK_ORIGINAL_REQUIRED_COLUMNS),
    ("benchmark_tuned", _BENCHMARK_TUNED_REQUIRED_COLUMNS),
):
    _family[f"{_prefix}_predictions.csv"] = _PROCEDURE_PREDICTION_COLUMNS
    _family[f"{_prefix}_procedure_fold_metrics.csv"] = _PROCEDURE_FOLD_COLUMNS
    _family[f"{_prefix}_procedure_summary.csv"] = {
        "procedure",
        "mean_BER",
        "std_BER",
        "min_BER",
        "max_BER",
        "pooled_TP",
        "pooled_TN",
        "pooled_FP",
        "pooled_FN",
    }
_TEMPORAL_REQUIRED_COLUMNS[ArtifactName.TEMPORAL_PREDICTIONS] = _PROCEDURE_PREDICTION_COLUMNS | {
    "timestamp",
    "fit_end_timestamp",
    "calibration_start_timestamp",
    "calibration_end_timestamp",
    "scaler",
}
_TEMPORAL_REQUIRED_COLUMNS[ArtifactName.TEMPORAL_PROCEDURE_METRICS] = _PROCEDURE_FOLD_COLUMNS

_BENCHMARK_ORIGINAL_ARTIFACTS = tuple(_BENCHMARK_ORIGINAL_REQUIRED_COLUMNS)
_BENCHMARK_TUNED_ARTIFACTS = tuple(_BENCHMARK_TUNED_REQUIRED_COLUMNS)
_TEMPORAL_ARTIFACTS = tuple(_TEMPORAL_REQUIRED_COLUMNS)
_TEMPORAL_ROLES = {ModelScope.PRIMARY, ModelScope.CHALLENGER}
_TEMPORAL_THRESHOLD_POLICIES = {ThresholdPolicy.SCIENTIFIC, ThresholdPolicy.OPERATIONAL}
_TEMPORAL_MODEL_SELECTION_STATUSES = {"primary", "challenger", "supporting"}
_TEMPORAL_DRIFT_GATE_STATUSES = {"PASS", "CAUTION", "HIGH_SHIFT"}
_TEMPORAL_MSPC_SOURCES = {"T2", "Q"}
_MANIFEST_VERSION = "3.0"
_STUDY_SPEC_PATH = "docs/spec"
_MISSING_SPEC_HASH = "MISSING"


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
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a JSON object")
    return manifest


def _required_artifacts_by_study(
    *,
    primary_status: str,
    benchmark_original_status: str,
    benchmark_tuned_status: str,
    temporal_status: str,
) -> list[str]:
    """Return required artifacts using separate original/tuned/temporal statuses."""
    names = [ArtifactName.MANIFEST]
    if benchmark_original_status != StudyStatus.NOT_RUN:
        names.extend(_BENCHMARK_ORIGINAL_ARTIFACTS)
    if benchmark_tuned_status != StudyStatus.NOT_RUN or (
        primary_status != StudyStatus.NOT_RUN and benchmark_original_status != StudyStatus.NOT_RUN
    ):
        names.extend(_BENCHMARK_TUNED_ARTIFACTS)
    if temporal_status in {StudyStatus.PASSED, StudyStatus.WARNING}:
        names.extend(_TEMPORAL_ARTIFACTS)
    return names


def validate_required_artifacts(
    output_dir: Path,
    *,
    primary_status: str = StudyStatus.NOT_RUN,
    benchmark_original_status: str,
    benchmark_tuned_status: str,
    temporal_status: str,
) -> list[str]:
    """Return missing artifact errors for the manifest-declared active study layers."""
    reports = output_dir / "reports"
    required = _required_artifacts_by_study(
        primary_status=primary_status,
        benchmark_original_status=benchmark_original_status,
        benchmark_tuned_status=benchmark_tuned_status,
        temporal_status=temporal_status,
    )
    return [f"missing artifact: {name}" for name in required if not (reports / name).exists()]


def load_artifact_frames(output_dir: Path, *, errors: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Load all present CSV report artifacts by artifact filename."""
    reports = output_dir / "reports"
    frames: dict[str, pd.DataFrame] = {}
    for name in _CSV_ARTIFACT_NAMES:
        try:
            df = read_csv_if_exists(reports / name)
        except (OSError, ValueError, UnicodeError) as exc:
            if errors is None:
                raise
            errors.append(f"{name}: cannot read CSV: {exc}")
            continue
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

    if manifest.get("manifest_version") != _MANIFEST_VERSION:
        errors.append(f"{ArtifactName.MANIFEST}: manifest_version must be {_MANIFEST_VERSION}")
    if manifest.get("study_spec_path") != _STUDY_SPEC_PATH:
        errors.append(f"{ArtifactName.MANIFEST}: study_spec_path must be {_STUDY_SPEC_PATH}")
    spec_hash = manifest.get("study_spec_sha256")
    if not isinstance(spec_hash, str) or not spec_hash.strip() or spec_hash == _MISSING_SPEC_HASH:
        errors.append(f"{ArtifactName.MANIFEST}: study_spec_sha256 must identify the active spec set")

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
    elif statuses["primary_study_status"] == StudyStatus.PASSED and (
        statuses["benchmark_original_status"] != StudyStatus.PASSED
        or statuses["benchmark_tuned_status"] != StudyStatus.PASSED
    ):
        errors.append(f"{ArtifactName.MANIFEST}: primary_study_status passed conflicts with benchmark layer statuses")

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
            if active and df.empty:
                errors.append(f"{name}: active artifact has no rows")
            if active and name.startswith("benchmark_"):
                for metric in ("BER", "True+", "True-"):
                    for column in (
                        metric,
                        f"mean_{metric}",
                        f"min_{metric}",
                        f"max_{metric}",
                        f"{metric}_full_dataset",
                    ):
                        _validate_probability_column(name, df, column, errors)
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


def _normalize_lineage_cell(value: Any) -> str:
    """Normalize scalar values before artifact-lineage comparisons."""
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.12g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _tuple_set(df: pd.DataFrame, columns: list[str]) -> set[tuple[str, ...]]:
    """Return distinct normalized row tuples for coverage checks."""
    rows = df[columns].drop_duplicates()
    return {tuple(_normalize_lineage_cell(value) for value in row) for row in rows.itertuples(index=False, name=None)}


def _append_coverage_error(
    *,
    errors: list[str],
    artifact_name: str,
    expected_label: str,
    actual: set[tuple[str, ...]],
    expected: set[tuple[str, ...]],
) -> None:
    """Append a compact coverage-mismatch error."""
    if actual != expected:
        errors.append(
            f"{artifact_name}: {expected_label} coverage mismatch "
            f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )


def _validate_binary_selected_values(name: str, df: pd.DataFrame, errors: list[str]) -> None:
    """Validate feature-stability selected flags are binary."""
    if "selected" not in df.columns:
        return
    values = pd.to_numeric(df["selected"], errors="coerce")
    valid_binary = values.notna() & values.isin([0, 1])
    if not bool(valid_binary.all()):
        errors.append(f"{name}: selected must contain only 0/1 values")


def _validate_selection_frequency_values(name: str, df: pd.DataFrame, errors: list[str]) -> None:
    """Validate feature-report selection frequencies are probabilities."""
    if "selection_frequency" not in df.columns:
        return
    values = pd.to_numeric(df["selection_frequency"], errors="coerce")
    if values.isna().any() or (values < 0.0).any() or (values > 1.0).any():
        errors.append(f"{name}: selection_frequency must be between 0 and 1")


def _validate_feature_report_coverage(
    *,
    artifact_name: str,
    feature_report: pd.DataFrame | None,
    benchmark_summary: pd.DataFrame | None,
    errors: list[str],
) -> None:
    """Require feature-report triplets to match benchmark summary triplets."""
    if feature_report is None or benchmark_summary is None:
        return
    required = _BENCHMARK_TRIPLET_COLUMNS
    if not required.issubset(feature_report.columns) or not required.issubset(benchmark_summary.columns):
        return
    _append_coverage_error(
        errors=errors,
        artifact_name=artifact_name,
        expected_label="triplet",
        actual=_tuple_set(feature_report, ["selector", "classifier", "replication_mode"]),
        expected=_tuple_set(benchmark_summary, ["selector", "classifier", "replication_mode"]),
    )
    _validate_selection_frequency_values(artifact_name, feature_report, errors)


def _validate_feature_stability_coverage(
    *,
    artifact_name: str,
    feature_stability: pd.DataFrame | None,
    benchmark_summary: pd.DataFrame | None,
    errors: list[str],
) -> None:
    """Require feature-stability lineage to match benchmark summary coverage."""
    if feature_stability is None or benchmark_summary is None:
        return
    if "classifier" in feature_stability.columns:
        if not _BENCHMARK_TRIPLET_COLUMNS.issubset(
            feature_stability.columns
        ) or not _BENCHMARK_TRIPLET_COLUMNS.issubset(benchmark_summary.columns):
            return
        _append_coverage_error(
            errors=errors,
            artifact_name=artifact_name,
            expected_label="triplet",
            actual=_tuple_set(feature_stability, ["selector", "classifier", "replication_mode"]),
            expected=_tuple_set(benchmark_summary, ["selector", "classifier", "replication_mode"]),
        )
    else:
        required = {"selector", "replication_mode"}
        if not required.issubset(feature_stability.columns) or not required.issubset(benchmark_summary.columns):
            return
        _append_coverage_error(
            errors=errors,
            artifact_name=artifact_name,
            expected_label="selector/mode",
            actual=_tuple_set(feature_stability, ["selector", "replication_mode"]),
            expected=_tuple_set(benchmark_summary, ["selector", "replication_mode"]),
        )
    _validate_binary_selected_values(artifact_name, feature_stability, errors)


def _normalized_bool_cell(value: Any) -> bool | None:
    """Normalize artifact boolean-like cells without accepting arbitrary truthy strings."""
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, (float, np.floating)) and value in {0.0, 1.0}:
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    return None


def _normalized_bool_values(df: pd.DataFrame, column: str) -> list[bool | None]:
    """Return normalized boolean values for one artifact column."""
    if column not in df.columns:
        return []
    return [_normalized_bool_cell(value) for value in df[column]]


def _validate_boolean_column(name: str, df: pd.DataFrame, column: str, errors: list[str]) -> None:
    """Validate an artifact column is encoded as boolean values."""
    values = _normalized_bool_values(df, column)
    if any(value is None for value in values):
        errors.append(f"{name}: {column} must contain only boolean values")


def _validate_allowed_values(
    name: str,
    df: pd.DataFrame,
    column: str,
    allowed: set[str],
    errors: list[str],
) -> None:
    """Validate string values in one artifact column against a controlled vocabulary."""
    if column not in df.columns:
        return
    invalid = sorted({str(value) for value in df[column].dropna().unique()} - allowed)
    if invalid:
        errors.append(f"{name}: {column} contains invalid values {invalid}")


def _numeric_values(df: pd.DataFrame, column: str) -> pd.Series | None:
    """Return numeric artifact values for a present column."""
    if column not in df.columns:
        return None
    return pd.to_numeric(df[column], errors="coerce")


def _validate_probability_column(name: str, df: pd.DataFrame, column: str, errors: list[str]) -> None:
    """Validate a numeric artifact column is finite and in [0, 1]."""
    values = _numeric_values(df, column)
    if values is None:
        return
    if values.isna().any() or (values < 0.0).any() or (values > 1.0).any():
        errors.append(f"{name}: {column} must be between 0 and 1")


def _validate_nonnegative_column(
    name: str,
    df: pd.DataFrame,
    column: str,
    errors: list[str],
    *,
    allow_missing: bool = False,
) -> None:
    """Validate a numeric artifact column is nonnegative, optionally allowing blanks."""
    values = _numeric_values(df, column)
    if values is None:
        return
    invalid_numeric = values.isna() & df[column].notna()
    if invalid_numeric.any():
        errors.append(f"{name}: {column} must be nonnegative")
        return
    checked_values = values.dropna() if allow_missing else values
    if checked_values.isna().any() or (checked_values < 0.0).any():
        errors.append(f"{name}: {column} must be nonnegative")


def _validate_positive_column(name: str, df: pd.DataFrame, column: str, errors: list[str]) -> None:
    """Validate a numeric artifact column is finite and strictly positive."""
    values = _numeric_values(df, column)
    if values is None:
        return
    if values.isna().any() or (values <= 0.0).any():
        errors.append(f"{name}: {column} must be positive")


def _validate_feature_lineage(
    *,
    artifact_frames: dict[str, pd.DataFrame] | None,
    reports: Path,
    active_original: bool,
    active_tuned: bool,
    errors: list[str],
) -> None:
    """Validate selector lineage between benchmark summaries and feature artifacts."""
    if active_original:
        original_summary = _artifact_frame(
            name=ArtifactName.BENCHMARK_SUMMARY,
            reports=reports,
            artifact_frames=artifact_frames,
        )
        _validate_feature_stability_coverage(
            artifact_name=ArtifactName.FEATURE_STABILITY,
            feature_stability=_artifact_frame(
                name=ArtifactName.FEATURE_STABILITY,
                reports=reports,
                artifact_frames=artifact_frames,
            ),
            benchmark_summary=original_summary,
            errors=errors,
        )
        _validate_feature_report_coverage(
            artifact_name=ArtifactName.FEATURE_REPORT,
            feature_report=_artifact_frame(
                name=ArtifactName.FEATURE_REPORT,
                reports=reports,
                artifact_frames=artifact_frames,
            ),
            benchmark_summary=original_summary,
            errors=errors,
        )

    if active_tuned:
        tuned_summary = _artifact_frame(
            name=ArtifactName.BENCHMARK_TUNED_SUMMARY,
            reports=reports,
            artifact_frames=artifact_frames,
        )
        _validate_feature_stability_coverage(
            artifact_name=ArtifactName.BENCHMARK_TUNED_FEATURE_STABILITY,
            feature_stability=_artifact_frame(
                name=ArtifactName.BENCHMARK_TUNED_FEATURE_STABILITY,
                reports=reports,
                artifact_frames=artifact_frames,
            ),
            benchmark_summary=tuned_summary,
            errors=errors,
        )
        _validate_feature_report_coverage(
            artifact_name=ArtifactName.BENCHMARK_TUNED_FEATURE_REPORT,
            feature_report=_artifact_frame(
                name=ArtifactName.BENCHMARK_TUNED_FEATURE_REPORT,
                reports=reports,
                artifact_frames=artifact_frames,
            ),
            benchmark_summary=tuned_summary,
            errors=errors,
        )


def _validate_config_set_equal(
    *,
    left_name: str,
    left_df: pd.DataFrame | None,
    right_name: str,
    right_df: pd.DataFrame | None,
    columns: list[str],
    errors: list[str],
) -> None:
    """Require two artifact frames to describe the same selected configs."""
    if left_df is None or right_df is None:
        return
    required = set(columns)
    if not required.issubset(left_df.columns) or not required.issubset(right_df.columns):
        return
    _append_coverage_error(
        errors=errors,
        artifact_name=f"{left_name} vs {right_name}",
        expected_label="config",
        actual=_tuple_set(left_df, columns),
        expected=_tuple_set(right_df, columns),
    )


def _validate_config_subset(
    *,
    subset_name: str,
    subset_df: pd.DataFrame | None,
    superset_name: str,
    superset_df: pd.DataFrame | None,
    columns: list[str],
    errors: list[str],
) -> None:
    """Require selected configs to exist inside the broader evaluated search space."""
    if subset_df is None or superset_df is None:
        return
    required = set(columns)
    if not required.issubset(subset_df.columns) or not required.issubset(superset_df.columns):
        return
    subset_values = _tuple_set(subset_df, columns)
    superset_values = _tuple_set(superset_df, columns)
    missing = subset_values - superset_values
    if missing:
        errors.append(f"{subset_name}: configs missing from {superset_name} {sorted(missing)}")


def _selected_tuned_search_configs(search_df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Return selected tuned search rows when the marker column is available."""
    if search_df is None or "is_selected_config" not in search_df.columns:
        return search_df
    marker = search_df["is_selected_config"]
    numeric_selected = pd.to_numeric(marker, errors="coerce").eq(1)
    string_selected = marker.astype("string").str.strip().str.lower().eq("true")
    selected_mask = (numeric_selected | string_selected).fillna(False)
    selected = search_df[selected_mask]
    return selected


def _validate_tuned_selected_config_cardinality(
    *,
    search_df: pd.DataFrame | None,
    errors: list[str],
) -> pd.DataFrame | None:
    """Require one selected tuned-search row per selector/classifier/mode/fold."""
    selected = _selected_tuned_search_configs(search_df)
    if selected is None or selected.empty:
        return selected
    group_cols = ["selector", "classifier", "replication_mode", "fold"]
    if not set(group_cols).issubset(selected.columns):
        return selected
    counts = selected.groupby(group_cols, dropna=False).size()
    duplicate_groups = counts[counts > 1]
    if not duplicate_groups.empty:
        errors.append(
            "benchmark_tuned_search.csv: each selector/classifier/mode/fold must mark exactly one selected config"
        )
    return selected


def _validate_selector_config_lineage(
    *,
    artifact_frames: dict[str, pd.DataFrame] | None,
    reports: Path,
    active_original: bool,
    active_tuned: bool,
    errors: list[str],
) -> None:
    """Validate selected selector/classifier config lineage across benchmark artifacts."""
    config_cols = [
        "selector",
        "classifier",
        "replication_mode",
        "k",
        "C",
        "alpha",
        "gamma",
        "gamma_multiplier",
        "n_neighbors",
    ]

    def candidate_view(frame):
        if frame is None or not {"gamma", "gamma_multiplier"}.issubset(frame):
            return frame
        return frame.assign(gamma=frame.gamma.where(frame.gamma_multiplier.isna()))

    fold_config_cols = [*config_cols, "fold"]

    if active_original:
        sweep_df = _artifact_frame(
            name=ArtifactName.BENCHMARK_SWEEP,
            reports=reports,
            artifact_frames=artifact_frames,
        )
        best_df = _artifact_frame(
            name=ArtifactName.BENCHMARK_BEST_CONFIG,
            reports=reports,
            artifact_frames=artifact_frames,
        )
        fold_df = _artifact_frame(
            name=ArtifactName.BENCHMARK_FOLD_METRICS,
            reports=reports,
            artifact_frames=artifact_frames,
        )
        _validate_duplicate_fold_rows(ArtifactName.BENCHMARK_FOLD_METRICS, fold_df, errors)
        full_fit_df = _artifact_frame(
            name=ArtifactName.BENCHMARK_FULL_FIT_SUMMARY,
            reports=reports,
            artifact_frames=artifact_frames,
        )
        _validate_config_subset(
            subset_name=ArtifactName.BENCHMARK_BEST_CONFIG,
            subset_df=candidate_view(best_df),
            superset_name=ArtifactName.BENCHMARK_SWEEP,
            superset_df=candidate_view(sweep_df),
            columns=config_cols,
            errors=errors,
        )
        selected = _validate_tuned_selected_config_cardinality(search_df=sweep_df, errors=errors)
        _validate_config_set_equal(
            left_name=ArtifactName.BENCHMARK_SWEEP,
            left_df=candidate_view(selected),
            right_name=ArtifactName.BENCHMARK_FOLD_METRICS,
            right_df=candidate_view(fold_df),
            columns=fold_config_cols,
            errors=errors,
        )
        _validate_config_set_equal(
            left_name=ArtifactName.BENCHMARK_BEST_CONFIG,
            left_df=candidate_view(best_df),
            right_name=ArtifactName.BENCHMARK_FULL_FIT_SUMMARY,
            right_df=candidate_view(full_fit_df),
            columns=config_cols,
            errors=errors,
        )

    if active_tuned:
        search_df = _artifact_frame(
            name=ArtifactName.BENCHMARK_TUNED_SEARCH,
            reports=reports,
            artifact_frames=artifact_frames,
        )
        best_df = _artifact_frame(
            name=ArtifactName.BENCHMARK_TUNED_BEST_CONFIG,
            reports=reports,
            artifact_frames=artifact_frames,
        )
        fold_df = _artifact_frame(
            name=ArtifactName.BENCHMARK_TUNED_FOLD_METRICS,
            reports=reports,
            artifact_frames=artifact_frames,
        )
        _validate_duplicate_fold_rows(ArtifactName.BENCHMARK_TUNED_FOLD_METRICS, fold_df, errors)
        full_fit_df = _artifact_frame(
            name=ArtifactName.BENCHMARK_TUNED_FULL_FIT_SUMMARY,
            reports=reports,
            artifact_frames=artifact_frames,
        )
        selected_search_df = _validate_tuned_selected_config_cardinality(
            search_df=search_df,
            errors=errors,
        )
        _validate_config_set_equal(
            left_name=ArtifactName.BENCHMARK_TUNED_SEARCH,
            left_df=candidate_view(selected_search_df),
            right_name=ArtifactName.BENCHMARK_TUNED_FOLD_METRICS,
            right_df=candidate_view(fold_df),
            columns=fold_config_cols,
            errors=errors,
        )
        _validate_config_set_equal(
            left_name=ArtifactName.BENCHMARK_TUNED_BEST_CONFIG,
            left_df=candidate_view(best_df),
            right_name=ArtifactName.BENCHMARK_TUNED_FULL_FIT_SUMMARY,
            right_df=candidate_view(full_fit_df),
            columns=config_cols,
            errors=errors,
        )


def _validate_duplicate_fold_rows(name: str, frame: pd.DataFrame | None, errors: list[str]) -> None:
    """Set-based config lineage must not conceal repeated fold observations."""
    keys = ["selector", "classifier", "replication_mode", "fold"]
    if frame is not None and set(keys).issubset(frame.columns) and frame.duplicated(keys).any():
        errors.append(f"{name}: duplicate selector/classifier/mode/fold rows")


def _validate_temporal_model_selection(model_selection: pd.DataFrame | None, errors: list[str]) -> None:
    """Validate temporal role-assignment semantics."""
    if model_selection is None:
        return
    name = ArtifactName.TEMPORAL_MODEL_SELECTION
    _validate_allowed_values(name, model_selection, "status", _TEMPORAL_MODEL_SELECTION_STATUSES, errors)
    _validate_boolean_column(name, model_selection, "is_primary", errors)
    _validate_boolean_column(name, model_selection, "is_challenger", errors)
    _validate_probability_column(name, model_selection, "mean_BER", errors)

    primary_flags = _normalized_bool_values(model_selection, "is_primary")
    challenger_flags = _normalized_bool_values(model_selection, "is_challenger")
    if primary_flags and primary_flags.count(True) != 1:
        errors.append(f"{name}: exactly one row must be marked primary")
    if challenger_flags and challenger_flags.count(True) > 1:
        errors.append(f"{name}: at most one row may be marked challenger")


def _validate_temporal_enums(
    *,
    freeze: pd.DataFrame | None,
    lockbox: pd.DataFrame | None,
    drift: pd.DataFrame | None,
    mspc: pd.DataFrame | None,
    manager: pd.DataFrame | None,
    errors: list[str],
) -> None:
    """Validate controlled vocabularies across temporal artifacts."""
    if freeze is not None:
        _validate_allowed_values(ArtifactName.TEMPORAL_FREEZE, freeze, "role", _TEMPORAL_ROLES, errors)
        _validate_boolean_column(ArtifactName.TEMPORAL_FREEZE, freeze, "is_frozen_config", errors)
    if lockbox is not None:
        _validate_allowed_values(ArtifactName.TEMPORAL_LOCKBOX, lockbox, "role", _TEMPORAL_ROLES, errors)
        _validate_allowed_values(
            ArtifactName.TEMPORAL_LOCKBOX,
            lockbox,
            "threshold_policy",
            _TEMPORAL_THRESHOLD_POLICIES,
            errors,
        )
    if lockbox is not None:
        from scipy.stats import binomtest

        for row in lockbox.to_dict("records"):
            try:
                positive, negative = row["TP"] + row["FN"], row["TN"] + row["FP"]
                tpr, tnr = (row["TP"] / positive if positive else 0), (row["TN"] / negative if negative else 0)
                if (
                    row["lockbox_n"] != positive + negative
                    or row["lockbox_fails"] != positive
                    or not np.isclose(row["True+"], tpr)
                    or not np.isclose(row["True-"], tnr)
                    or not np.isclose(row["BER"], 1 - 0.5 * (tpr + tnr))
                ):
                    errors.append("temporal_lockbox.csv: frozen confusion count/rate mismatch")
                if (
                    row["interval_semantics"] != "conditional_fixed_model_independent_trials_only"
                    or row["evaluation_semantics"] != "retrospective_later_block"
                ):
                    errors.append("temporal_lockbox.csv: invalid uncertainty/evaluation semantics")
                for label, successes, n in (("TPR", row["TP"], positive), ("TNR", row["TN"], negative)):
                    interval = binomtest(int(successes), int(n)).proportion_ci(method="exact") if n else None
                    for bound, value in (
                        ("lower", interval.low if interval else np.nan),
                        ("upper", interval.high if interval else np.nan),
                    ):
                        if not np.isclose(row[f"{label}_exact_{bound}"], value, equal_nan=True, rtol=0, atol=1e-9):
                            errors.append("temporal_lockbox.csv: exact binomial interval mismatch")
            except (KeyError, ValueError, TypeError):
                errors.append("temporal_lockbox.csv: incomplete frozen-count/interval lineage")
    if drift is not None:
        _validate_allowed_values(ArtifactName.TEMPORAL_DRIFT, drift, "model_scope", _TEMPORAL_ROLES, errors)
        _validate_allowed_values(
            ArtifactName.TEMPORAL_DRIFT,
            drift,
            "drift_gate_status",
            _TEMPORAL_DRIFT_GATE_STATUSES,
            errors,
        )
        _validate_boolean_column(ArtifactName.TEMPORAL_DRIFT, drift, "confirmatory_claims_allowed", errors)
        if (
            "score_reference" in drift
            and not drift.score_reference.eq("held_out_calibration_same_retained_model").all()
        ):
            errors.append(
                "temporal_drift_summary.csv: score KS reference must be held-out calibration from same retained model"
            )
        if any(value is True for value in _normalized_bool_values(drift, "confirmatory_claims_allowed")):
            errors.append("temporal_drift_summary.csv: retrospective evaluation cannot authorize confirmatory claims")
    if mspc is not None:
        _validate_allowed_values(ArtifactName.TEMPORAL_MSPC, mspc, "eval_scope", {"outer_fold", "lockbox"}, errors)
        _validate_allowed_values(
            ArtifactName.TEMPORAL_MSPC, mspc, "calibration_selected_MSPC_source", _TEMPORAL_MSPC_SOURCES, errors
        )
        for row in mspc.to_dict("records"):
            try:
                source = "T2" if row["T2_calibration_TPR_at_TNR90"] >= row["Q_calibration_TPR_at_TNR90"] else "Q"
                if (
                    row["calibration_selected_MSPC_source"] != source
                    or row["source_selection_region"] != "held_out_calibration"
                ):
                    errors.append("temporal_mspc.csv: source must be frozen using calibration metrics")
                if not np.isclose(row["frozen_threshold"], row[f"{source}_frozen_threshold"]):
                    errors.append("temporal_mspc.csv: selected frozen threshold mismatch")
                tpr = row["TP"] / (row["TP"] + row["FN"]) if row["TP"] + row["FN"] else 0
                tnr = row["TN"] / (row["TN"] + row["FP"]) if row["TN"] + row["FP"] else 0
                if not all(
                    np.isclose(row[key], expected)
                    for key, expected in (
                        ("frozen_TPR", tpr),
                        ("frozen_TNR", tnr),
                        ("frozen_BER", 1 - 0.5 * (tpr + tnr)),
                    )
                ):
                    errors.append("temporal_mspc.csv: frozen metrics disagree with confusion counts")
            except (KeyError, ValueError, TypeError, ZeroDivisionError):
                errors.append("temporal_mspc.csv: incomplete calibration source/threshold lineage")
    if manager is not None:
        _validate_allowed_values(
            ArtifactName.TEMPORAL_MANAGER_OUTPUTS, manager, "evaluation_region", {"held_out_DEV_calibration"}, errors
        )
        _validate_allowed_values(
            ArtifactName.TEMPORAL_MANAGER_OUTPUTS,
            manager,
            "workload_semantics",
            {"illustrative_mean_weekly_policy_not_per_week_cap"},
            errors,
        )
        _validate_allowed_values(ArtifactName.TEMPORAL_MANAGER_OUTPUTS, manager, "role", _TEMPORAL_ROLES, errors)
        _validate_allowed_values(
            ArtifactName.TEMPORAL_MANAGER_OUTPUTS,
            manager,
            "threshold_policy",
            _TEMPORAL_THRESHOLD_POLICIES,
            errors,
        )


def _validate_temporal_numeric_ranges(
    *,
    screening: pd.DataFrame | None,
    model_selection: pd.DataFrame | None,
    inner_cv: pd.DataFrame | None,
    lockbox: pd.DataFrame | None,
    drift: pd.DataFrame | None,
    mspc: pd.DataFrame | None,
    cost: pd.DataFrame | None,
    manager: pd.DataFrame | None,
    errors: list[str],
) -> None:
    """Validate temporal metric, workload, and cost ranges."""
    probability_columns = {
        ArtifactName.TEMPORAL_SELECTOR_SCREENING: (screening, ["mean_BER"]),
        ArtifactName.TEMPORAL_MODEL_SELECTION: (model_selection, ["mean_BER"]),
        ArtifactName.TEMPORAL_INNER_CV: (inner_cv, ["mean_inner_BER", "mean_inner_ROC_AUC"]),
        ArtifactName.TEMPORAL_LOCKBOX: (lockbox, ["BER", "True+", "True-", "TPR_at_TNR90"]),
        ArtifactName.TEMPORAL_DRIFT: (
            drift,
            ["abs_prevalence_shift", "ks_pvalue_scores", "max_missingness_rate_shift"],
        ),
        ArtifactName.TEMPORAL_MSPC: (mspc, ["calibration_selected_MSPC_TPR_at_TNR90", "T2_AUC", "Q_AUC", "alarm_rate"]),
        ArtifactName.TEMPORAL_MANAGER_OUTPUTS: (manager, ["predicted_flag_fraction", "mean_weekly_flag_fraction"]),
    }
    for artifact_name, (frame, columns) in probability_columns.items():
        if frame is None:
            continue
        for column in columns:
            _validate_probability_column(artifact_name, frame, column, errors)

    nonnegative_columns = {
        ArtifactName.TEMPORAL_SELECTOR_SCREENING: (screening, ["std_BER"]),
        ArtifactName.TEMPORAL_DRIFT: (drift, ["max_PSI"]),
        ArtifactName.TEMPORAL_MSPC: (mspc, ["observed_mean_inter_alarm_spacing"]),
        ArtifactName.TEMPORAL_COST_CURVES: (
            cost,
            [column for column in (list(cost.columns) if cost is not None else []) if column != "cost_ratio"],
        ),
        ArtifactName.TEMPORAL_MANAGER_OUTPUTS: (
            manager,
            ["mean_weekly_flagged_samples", "mean_weekly_fail_captures", "mean_weekly_fail_misses"],
        ),
    }
    for artifact_name, (frame, columns) in nonnegative_columns.items():
        if frame is None:
            continue
        for column in columns:
            _validate_nonnegative_column(
                artifact_name,
                frame,
                column,
                errors,
                allow_missing=artifact_name == ArtifactName.TEMPORAL_COST_CURVES,
            )
    if cost is not None:
        _validate_positive_column(ArtifactName.TEMPORAL_COST_CURVES, cost, "cost_ratio", errors)


def _role_policy_tuples(df: pd.DataFrame | None) -> set[tuple[str, str]]:
    """Return role/policy tuples for artifacts that expose both fields."""
    required = {"role", "threshold_policy"}
    if df is None or not required.issubset(df.columns):
        return set()
    return {
        (str(row.role), str(row.threshold_policy))
        for row in df.loc[:, ["role", "threshold_policy"]].drop_duplicates().itertuples(index=False)
    }


def _selected_temporal_roles(model_selection: pd.DataFrame | None) -> set[str]:
    """Return temporal roles marked by the model-selection artifact."""
    if model_selection is None:
        return set()
    roles: set[str] = set()
    if any(value is True for value in _normalized_bool_values(model_selection, "is_primary")):
        roles.add(ModelScope.PRIMARY)
    if any(value is True for value in _normalized_bool_values(model_selection, "is_challenger")):
        roles.add(ModelScope.CHALLENGER)
    return roles


def _artifact_roles(df: pd.DataFrame | None, column: str) -> set[str]:
    """Return distinct role/scope labels from one artifact column."""
    if df is None or column not in df.columns:
        return set()
    return {str(value) for value in df[column].dropna().unique()}


def _validate_role_coverage(
    *,
    artifact_name: str,
    actual_roles: set[str],
    expected_roles: set[str],
    errors: list[str],
) -> None:
    """Validate artifact roles match temporal model-selection roles when both are knowable."""
    if expected_roles and actual_roles != expected_roles:
        errors.append(
            f"{artifact_name}: role coverage mismatch "
            f"missing={sorted(expected_roles - actual_roles)} extra={sorted(actual_roles - expected_roles)}"
        )


def _validate_temporal_role_policy_lineage(
    *,
    model_selection: pd.DataFrame | None,
    freeze: pd.DataFrame | None,
    lockbox: pd.DataFrame | None,
    drift: pd.DataFrame | None,
    manager: pd.DataFrame | None,
    errors: list[str],
) -> None:
    """Validate temporal role and role/policy coverage across related artifacts."""
    selected_roles = _selected_temporal_roles(model_selection)
    frozen = freeze
    if frozen is not None and "is_frozen_config" in frozen.columns:
        frozen_flags = _normalized_bool_values(frozen, "is_frozen_config")
        frozen = frozen[[value is True for value in frozen_flags]]

    _validate_role_coverage(
        artifact_name=ArtifactName.TEMPORAL_FREEZE,
        actual_roles=_artifact_roles(frozen, "role"),
        expected_roles=selected_roles,
        errors=errors,
    )
    _validate_role_coverage(
        artifact_name=ArtifactName.TEMPORAL_LOCKBOX,
        actual_roles=_artifact_roles(lockbox, "role"),
        expected_roles=selected_roles,
        errors=errors,
    )
    _validate_role_coverage(
        artifact_name=ArtifactName.TEMPORAL_DRIFT,
        actual_roles=_artifact_roles(drift, "model_scope"),
        expected_roles=selected_roles,
        errors=errors,
    )
    _validate_role_coverage(
        artifact_name=ArtifactName.TEMPORAL_MANAGER_OUTPUTS,
        actual_roles=_artifact_roles(manager, "role"),
        expected_roles=selected_roles,
        errors=errors,
    )

    lockbox_tuples = _role_policy_tuples(lockbox)
    manager_tuples = _role_policy_tuples(manager)
    if lockbox_tuples and manager_tuples and manager_tuples != lockbox_tuples:
        errors.append(
            f"{ArtifactName.TEMPORAL_MANAGER_OUTPUTS}: role/policy coverage mismatch "
            f"missing={sorted(lockbox_tuples - manager_tuples)} extra={sorted(manager_tuples - lockbox_tuples)}"
        )


def _expected_temporal_claim_restrictions(
    lockbox: pd.DataFrame | None,
    drift: pd.DataFrame | None,
    mspc: pd.DataFrame | None,
) -> list[str]:
    """Recompute temporal claim restrictions from persisted temporal artifacts."""
    required_lockbox = {"role", "threshold_policy", "TPR_at_TNR90"}
    required_drift = {"model_scope", "drift_gate_status"}
    required_mspc = {"eval_scope", "calibration_selected_MSPC_TPR_at_TNR90"}
    if (
        lockbox is None
        or drift is None
        or mspc is None
        or not required_lockbox.issubset(lockbox.columns)
        or not required_drift.issubset(drift.columns)
        or not required_mspc.issubset(mspc.columns)
    ):
        return []

    mspc_lock = mspc[mspc["eval_scope"].astype(str) == "lockbox"]
    if mspc_lock.empty:
        return []
    mspc_tpr = pd.to_numeric(mspc_lock["calibration_selected_MSPC_TPR_at_TNR90"], errors="coerce").iloc[0]
    if pd.isna(mspc_tpr):
        return []

    restrictions: list[str] = [
        "retrospective_later_block_not_fresh_confirmatory_lockbox",
        "no_production_readiness_or_superiority_claim",
    ]
    scientific = lockbox[lockbox["threshold_policy"].astype(str) == ThresholdPolicy.SCIENTIFIC].copy()
    scientific["TPR_at_TNR90"] = pd.to_numeric(scientific["TPR_at_TNR90"], errors="coerce")
    for row in scientific.itertuples(index=False):
        role = str(row.role)
        if role not in _TEMPORAL_ROLES or pd.isna(row.TPR_at_TNR90):
            continue
        drift_row = drift[drift["model_scope"].astype(str) == role]
        if drift_row.empty:
            continue
        status = str(drift_row.iloc[0]["drift_gate_status"])
        if status == "HIGH_SHIFT" and float(row.TPR_at_TNR90) > float(mspc_tpr):
            restrictions.append(f"{role}_high_shift_blocks_lockbox_superiority_claim")
    return sorted(dict.fromkeys(restrictions))


def _validate_temporal_claim_restriction_lineage(
    *,
    manifest_restrictions: list[str],
    lockbox: pd.DataFrame | None,
    drift: pd.DataFrame | None,
    mspc: pd.DataFrame | None,
    errors: list[str],
) -> list[str]:
    """Validate manifest temporal restrictions against persisted artifact evidence."""
    expected = set(_expected_temporal_claim_restrictions(lockbox, drift, mspc))
    actual = set(manifest_restrictions)
    if expected != actual:
        errors.append(
            f"{ArtifactName.MANIFEST}: temporal_claim_restrictions mismatch artifact evidence "
            f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )
    return sorted(expected | actual)


def _validate_temporal_semantics(
    *,
    reports: Path,
    artifact_frames: dict[str, pd.DataFrame] | None,
    active_temporal: bool,
    manifest_restrictions: list[str],
    errors: list[str],
) -> list[str]:
    """Validate temporal artifact semantics and return derived claim restrictions."""
    if not active_temporal:
        return manifest_restrictions
    frames = {
        name: _artifact_frame(name=name, reports=reports, artifact_frames=artifact_frames)
        for name in _TEMPORAL_ARTIFACTS
    }
    _validate_temporal_model_selection(frames[ArtifactName.TEMPORAL_MODEL_SELECTION], errors)
    _validate_temporal_enums(
        freeze=frames[ArtifactName.TEMPORAL_FREEZE],
        lockbox=frames[ArtifactName.TEMPORAL_LOCKBOX],
        drift=frames[ArtifactName.TEMPORAL_DRIFT],
        mspc=frames[ArtifactName.TEMPORAL_MSPC],
        manager=frames[ArtifactName.TEMPORAL_MANAGER_OUTPUTS],
        errors=errors,
    )
    _validate_temporal_numeric_ranges(
        screening=frames[ArtifactName.TEMPORAL_SELECTOR_SCREENING],
        model_selection=frames[ArtifactName.TEMPORAL_MODEL_SELECTION],
        inner_cv=frames[ArtifactName.TEMPORAL_INNER_CV],
        lockbox=frames[ArtifactName.TEMPORAL_LOCKBOX],
        drift=frames[ArtifactName.TEMPORAL_DRIFT],
        mspc=frames[ArtifactName.TEMPORAL_MSPC],
        cost=frames[ArtifactName.TEMPORAL_COST_CURVES],
        manager=frames[ArtifactName.TEMPORAL_MANAGER_OUTPUTS],
        errors=errors,
    )
    _validate_temporal_role_policy_lineage(
        model_selection=frames[ArtifactName.TEMPORAL_MODEL_SELECTION],
        freeze=frames[ArtifactName.TEMPORAL_FREEZE],
        lockbox=frames[ArtifactName.TEMPORAL_LOCKBOX],
        drift=frames[ArtifactName.TEMPORAL_DRIFT],
        manager=frames[ArtifactName.TEMPORAL_MANAGER_OUTPUTS],
        errors=errors,
    )
    return _validate_temporal_claim_restriction_lineage(
        manifest_restrictions=manifest_restrictions,
        lockbox=frames[ArtifactName.TEMPORAL_LOCKBOX],
        drift=frames[ArtifactName.TEMPORAL_DRIFT],
        mspc=frames[ArtifactName.TEMPORAL_MSPC],
        errors=errors,
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
        try:
            manifest = read_manifest(manifest_path)
        except (OSError, ValueError, UnicodeError) as exc:
            return ValidationResult(
                ok=False,
                errors=[f"{ArtifactName.MANIFEST}: cannot read manifest: {exc}"],
                warnings=[],
                claim_restrictions=[],
            )

    if artifact_frames is None:
        artifact_frames = load_artifact_frames(output_dir, errors=errors)

    state = _validate_manifest_fields(manifest=manifest, errors=errors, warnings=warnings)

    if state.temporal_status in {StudyStatus.PASSED, StudyStatus.WARNING}:
        for name, field in (
            (ArtifactName.TEMPORAL_LOCKBOX, "lockbox_fails"),
            (ArtifactName.TEMPORAL_PROCEDURE_METRICS, "n_test_fails"),
        ):
            frame = artifact_frames.get(name)
            if frame is not None and field in frame and (pd.to_numeric(frame[field], errors="coerce") < 20).any():
                warnings.append(
                    f"{name}: sparse evaluation failures; rates and exact conditional intervals have limited precision"
                )
    # Active layers require artifacts; inactive layers only warn if stale artifacts remain.
    active_original = state.original_status != StudyStatus.NOT_RUN
    active_tuned = state.tuned_status != StudyStatus.NOT_RUN or (
        state.primary_status != StudyStatus.NOT_RUN and state.original_status != StudyStatus.NOT_RUN
    )
    active_temporal = state.temporal_status in {StudyStatus.PASSED, StudyStatus.WARNING}

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
    _validate_feature_lineage(
        artifact_frames=artifact_frames,
        reports=reports,
        active_original=active_original,
        active_tuned=active_tuned,
        errors=errors,
    )
    _validate_selector_config_lineage(
        artifact_frames=artifact_frames,
        reports=reports,
        active_original=active_original,
        active_tuned=active_tuned,
        errors=errors,
    )
    for status, validator, families in (
        (
            state.original_status,
            validate_benchmark_replication_artifacts,
            {
                "sweep_df": ArtifactName.BENCHMARK_SWEEP,
                "best_df": ArtifactName.BENCHMARK_BEST_CONFIG,
                "fold_metrics_df": ArtifactName.BENCHMARK_FOLD_METRICS,
                "summary_df": ArtifactName.BENCHMARK_SUMMARY,
                "ablation_df": ArtifactName.BENCHMARK_ABLATION,
                "full_fit_df": ArtifactName.BENCHMARK_FULL_FIT_SUMMARY,
            },
        ),
        (
            state.tuned_status,
            validate_tuned_benchmark_artifacts,
            {
                "search_df": ArtifactName.BENCHMARK_TUNED_SEARCH,
                "best_df": ArtifactName.BENCHMARK_TUNED_BEST_CONFIG,
                "fold_metrics_df": ArtifactName.BENCHMARK_TUNED_FOLD_METRICS,
                "summary_df": ArtifactName.BENCHMARK_TUNED_SUMMARY,
                "ablation_df": ArtifactName.BENCHMARK_TUNED_ABLATION,
                "full_fit_df": ArtifactName.BENCHMARK_TUNED_FULL_FIT_SUMMARY,
            },
        ),
    ):
        if status == StudyStatus.PASSED and all(name in artifact_frames for name in families.values()):
            try:
                validator(**{key: artifact_frames[name] for key, name in families.items()})
            except (ValueError, TypeError, KeyError) as exc:
                errors.append(str(exc))
    from secom.workflows.benchmark_procedures import validate_prediction_lineage, procedure_summary, selected_candidate

    paired_predictions = []
    for prefix, active in (("benchmark", active_original), ("benchmark_tuned", active_tuned)):
        if not active:
            continue
        try:
            predictions = artifact_frames[f"{prefix}_predictions.csv"]
            folds = artifact_frames[f"{prefix}_procedure_fold_metrics.csv"]
            summary = artifact_frames[f"{prefix}_procedure_summary.csv"]
            validate_prediction_lineage(predictions, folds)
            profile = manifest.get("dataset", {})
            if "n_samples" in profile:
                joint = predictions[predictions.procedure == "joint"]
                if set(joint.sample_id) != set(range(int(profile["n_samples"]))) or int(joint.y_true.sum()) != int(
                    profile["n_fails"]
                ):
                    raise ValueError("held-out predictions do not match input sample IDs/failure counts")
            expected = procedure_summary(folds, predictions).set_index("procedure")
            if summary.procedure.duplicated().any() or set(summary.procedure) != set(expected.index):
                raise ValueError("procedure summary coverage mismatch")
            for row in summary.to_dict("records"):
                for column in expected.columns:
                    if column not in row or not np.isclose(
                        float(row[column]), float(expected.loc[row["procedure"], column]), atol=1e-9, rtol=0
                    ):
                        raise ValueError("procedure summary mismatch with held-out predictions/folds")
            search = artifact_frames[
                ArtifactName.BENCHMARK_SWEEP if prefix == "benchmark" else ArtifactName.BENCHMARK_TUNED_SEARCH
            ]
            for name, width_column in (
                (f"{prefix}_fold_metrics.csv", "n_selected_features"),
                (f"{prefix}_full_fit_summary.csv", "n_selected_features_full_dataset"),
            ):
                configured = artifact_frames[name]
                relative = configured[configured["gamma_multiplier"].notna()]
                if not relative.empty and (
                    width_column not in relative
                    or not np.allclose(
                        relative.gamma, relative.gamma_multiplier / relative[width_column], rtol=0, atol=1e-14
                    )
                ):
                    raise ValueError("effective gamma must match actual selected matrix width")
            relative_search = search[search["gamma_multiplier"].notna()]
            for candidate in relative_search.to_dict("records"):
                widths = [int(width) for width in str(candidate["inner_selected_widths"]).split(",")]
                if min(widths) < 1 or not np.isclose(
                    candidate["gamma"], candidate["gamma_multiplier"] / widths[0], rtol=0, atol=1e-14
                ):
                    raise ValueError("inner effective gamma must match selected width receipt")
            from secom.workflows.benchmark_tuned import _select_best_tuned_config

            for key, group in search.groupby(["selector", "classifier", "replication_mode", "fold"], dropna=False):
                selected = group[group.is_selected_config.map(lambda value: _normalized_bool_cell(value) is True)]
                if len(selected) != 1:
                    raise ValueError("each family/fold must select exactly one inner config")
                best = _select_best_tuned_config(group.to_dict("records"))
                for column in ("k", "C", "alpha", "gamma", "n_neighbors", "threshold_inner_oof", "mean_inner_BER"):
                    if _normalize_lineage_cell(selected.iloc[0][column]) != _normalize_lineage_cell(best[column]):
                        raise ValueError("selected family config must minimize inner BER with declared tie order")
            for fold, frame in search.groupby("fold"):
                candidate_rows = frame[
                    frame.is_selected_config.map(lambda value: _normalized_bool_cell(value) is True)
                ].to_dict("records")
                candidates = [(row, {}, np.array([])) for row in candidate_rows]
                for procedure, mode in (
                    ("joint", None),
                    ("values_only", "strict"),
                    ("values_and_indicators", "with_missing_indicators"),
                ):
                    best, _, _ = selected_candidate(candidates, mode)
                    emitted = predictions[(predictions.fold == fold) & (predictions.procedure == procedure)].iloc[0]
                    columns = [
                        "selector",
                        "classifier",
                        "replication_mode",
                        "k",
                        "C",
                        "alpha",
                        "gamma_multiplier",
                        "n_neighbors",
                    ]
                    if pd.isna(best.get("gamma_multiplier")):
                        columns.append("gamma")
                    if any(
                        _normalize_lineage_cell(best[c]) != _normalize_lineage_cell(emitted[c]) for c in columns
                    ) or not np.isclose(best["threshold_inner_oof"], emitted.threshold):
                        raise ValueError("joint procedure must match inner-selected config and frozen threshold")
            paired_predictions.append(predictions[predictions.procedure == "joint"])
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            errors.append(f"{prefix} procedures: {exc}")
    if len(paired_predictions) == 2:
        identity = ["sample_id", "fold", "y_true"]
        if _tuple_set(paired_predictions[0], identity) != _tuple_set(paired_predictions[1], identity):
            errors.append("original/tuned joint procedures: held-out sample/fold/label pairing mismatch")
    if active_temporal:
        try:
            from secom.workflows.temporal_comparator_audit import validate_temporal_comparator

            validate_temporal_comparator(artifact_frames, warnings)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            errors.append(f"temporal comparator/calibration: {exc}")
        try:
            validate_prediction_lineage(
                artifact_frames[ArtifactName.TEMPORAL_PREDICTIONS],
                artifact_frames[ArtifactName.TEMPORAL_PROCEDURE_METRICS],
                temporal=True,
            )
            from secom.selection.tuning import select_ber_config
            from secom.workflows.temporal_robustness import _selector_config_simplicity_key

            inner = artifact_frames[ArtifactName.TEMPORAL_INNER_CV]
            for fold, frame in artifact_frames[ArtifactName.TEMPORAL_PREDICTIONS].groupby("fold"):
                candidates = inner[
                    (inner.resample_id == f"outer_{fold}_seed_42")
                    & inner.is_selected_config.map(lambda value: _normalized_bool_cell(value) is True)
                ].to_dict("records")
                best = select_ber_config(
                    candidates, simplicity_key=lambda row: (*_selector_config_simplicity_key(row), str(row["selector"]))
                )
                emitted = frame.iloc[0]
                if any(
                    _normalize_lineage_cell(best[c]) != _normalize_lineage_cell(emitted[c])
                    for c in ("selector", "k", "C", "scaler", "n_neighbors")
                ):
                    raise ValueError("temporal joint config must be selected using chronological inner predictions")
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"temporal procedures: {exc}")
    temporal_claim_restrictions = _validate_temporal_semantics(
        reports=reports,
        artifact_frames=artifact_frames,
        active_temporal=active_temporal,
        manifest_restrictions=state.claim_restrictions,
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
        warning_prefix="temporal artifact present without completed temporal robustness status",
        warnings=warnings,
    )

    deduped_warnings = list(dict.fromkeys(warnings))
    deduped_restrictions = list(dict.fromkeys(temporal_claim_restrictions))
    return ValidationResult(
        ok=len(errors) == 0,
        errors=errors,
        warnings=deduped_warnings,
        claim_restrictions=deduped_restrictions,
    )
