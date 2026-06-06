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

_BENCHMARK_TRIPLET_COLUMNS = {"selector", "classifier", "replication_mode"}
_BENCHMARK_CONFIG_COLUMNS = {"k", "C", "alpha", "gamma", "n_neighbors"}
_BENCHMARK_METRIC_COLUMNS = {"BER", "True+", "True-", "ROC_AUC", "PR_AUC", "MCC", "F2"}
_BENCHMARK_MEAN_METRIC_COLUMNS = {f"mean_{metric}" for metric in _BENCHMARK_METRIC_COLUMNS}
_BENCHMARK_BER_CI_COLUMNS = {"CI_lower_BER", "CI_upper_BER"}
_BENCHMARK_FULL_DATASET_METRIC_COLUMNS = {f"{metric}_full_dataset" for metric in _BENCHMARK_METRIC_COLUMNS}
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
    "conditional_effect_magnitude",
    "expected_contribution",
}

_BENCHMARK_ORIGINAL_REQUIRED_COLUMNS: dict[str, set[str]] = {
    ArtifactName.BENCHMARK_SWEEP: {
        *_BENCHMARK_TRIPLET_COLUMNS,
        *_BENCHMARK_CONFIG_COLUMNS,
        *_BENCHMARK_MEAN_METRIC_COLUMNS,
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
        *_BENCHMARK_BER_CI_COLUMNS,
    },
    ArtifactName.BENCHMARK_ABLATION: _BENCHMARK_ABLATION_COLUMNS,
    ArtifactName.BENCHMARK_FULL_FIT_SUMMARY: {
        *_BENCHMARK_TRIPLET_COLUMNS,
        *_BENCHMARK_CONFIG_COLUMNS,
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
        *_BENCHMARK_BER_CI_COLUMNS,
    },
    ArtifactName.BENCHMARK_TUNED_ABLATION: _BENCHMARK_ABLATION_COLUMNS,
    ArtifactName.BENCHMARK_TUNED_FULL_FIT_SUMMARY: {
        *_BENCHMARK_TRIPLET_COLUMNS,
        *_BENCHMARK_CONFIG_COLUMNS,
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
    if temporal_status != StudyStatus.NOT_RUN:
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
    selected = search_df[search_df["is_selected_config"].astype(bool)]
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
    config_cols = ["selector", "classifier", "replication_mode", "k", "C", "alpha", "gamma", "n_neighbors"]
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
        full_fit_df = _artifact_frame(
            name=ArtifactName.BENCHMARK_FULL_FIT_SUMMARY,
            reports=reports,
            artifact_frames=artifact_frames,
        )
        _validate_config_subset(
            subset_name=ArtifactName.BENCHMARK_BEST_CONFIG,
            subset_df=best_df,
            superset_name=ArtifactName.BENCHMARK_SWEEP,
            superset_df=sweep_df,
            columns=config_cols,
            errors=errors,
        )
        _validate_config_set_equal(
            left_name=ArtifactName.BENCHMARK_BEST_CONFIG,
            left_df=best_df,
            right_name=ArtifactName.BENCHMARK_FOLD_METRICS,
            right_df=fold_df,
            columns=config_cols,
            errors=errors,
        )
        _validate_config_set_equal(
            left_name=ArtifactName.BENCHMARK_BEST_CONFIG,
            left_df=best_df,
            right_name=ArtifactName.BENCHMARK_FULL_FIT_SUMMARY,
            right_df=full_fit_df,
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
        full_fit_df = _artifact_frame(
            name=ArtifactName.BENCHMARK_TUNED_FULL_FIT_SUMMARY,
            reports=reports,
            artifact_frames=artifact_frames,
        )
        _validate_config_set_equal(
            left_name=ArtifactName.BENCHMARK_TUNED_SEARCH,
            left_df=_selected_tuned_search_configs(search_df),
            right_name=ArtifactName.BENCHMARK_TUNED_FOLD_METRICS,
            right_df=fold_df,
            columns=fold_config_cols,
            errors=errors,
        )
        _validate_config_set_equal(
            left_name=ArtifactName.BENCHMARK_TUNED_BEST_CONFIG,
            left_df=best_df,
            right_name=ArtifactName.BENCHMARK_TUNED_FULL_FIT_SUMMARY,
            right_df=full_fit_df,
            columns=config_cols,
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
        manifest = read_manifest(manifest_path)

    state = _validate_manifest_fields(manifest=manifest, errors=errors, warnings=warnings)

    # Active layers require artifacts; inactive layers only warn if stale artifacts remain.
    active_original = state.original_status != StudyStatus.NOT_RUN
    active_tuned = state.tuned_status != StudyStatus.NOT_RUN or (
        state.primary_status != StudyStatus.NOT_RUN and state.original_status != StudyStatus.NOT_RUN
    )
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
