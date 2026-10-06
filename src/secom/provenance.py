"""Dataset and execution provenance for the complete study command."""

from __future__ import annotations

import hashlib
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from secom.artifacts import read_manifest, write_manifest
from secom.common.meta import installed_package_versions, source_tree_identity
from secom.config import (
    ArtifactName,
    BENCHMARK_INNER_SPLITS,
    BENCHMARK_KRR_ALPHA_GRID,
    BENCHMARK_KRR_GAMMA_GRID,
    TUNED_KRR_ALPHA_GRID,
    TUNED_KRR_GAMMA_MULTIPLIERS,
    TEMPORAL_KRR_CALIBRATION_FRACTIONS,
    BENCHMARK_LOGREG_C_GRID,
    BenchmarkClassifier,
    LOCKBOX_FRAC,
    SEED_BENCHMARK,
    SelectorName,
)
from secom.io import load_raw_secom, parse_sort_and_label
from secom.feature_select.relief import relief_backend_name, relief_worker_count, RELIEF_CACHE_SIZE
from secom.workflows.manifest import initial_study_manifest

UCI_DATASET_URL = "https://archive.ics.uci.edu/dataset/179/secom"
UCI_ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"
SECOM_FILE_SHA256 = {
    "secom.data": "20f0e7ee434f7dcbae0eea9ffff009a2b57f42d6b0dc9a5bd4f00782c0a3374c",
    "secom_labels.data": "126884cf453705c9e61a903fe906f0665a3b45ce3639e621edc5c93c89627e03",
}


def sha256_file(path: Path) -> str:
    """Hash exact file bytes, without normalizing external dataset or artifact content."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset_profile(input_dir: Path) -> dict[str, Any]:
    """Describe observed inputs; missingness diagnostics are full-sample descriptive evidence."""
    loaded = load_raw_secom(input_dir)
    data = parse_sort_and_label(loaded.frame)
    x = data[loaded.feature_columns]
    files = {
        name: {"sha256": sha256_file(input_dir / name), "bytes": (input_dir / name).stat().st_size}
        for name in SECOM_FILE_SHA256
    }
    month = data["timestamp"].dt.strftime("%Y-%m")
    patterns: dict[bytes, list[str]] = {}
    for idx, name in enumerate(loaded.feature_columns):
        missing = x[name].isna().to_numpy()
        if missing.any() and not missing.all():
            patterns.setdefault(np.packbits(missing).tobytes(), []).append(f"M{idx}")
    groups = []
    for features in patterns.values():
        if len(features) < 2:
            continue
        missing = x.iloc[:, int(features[0][1:])].isna()
        monthly = {str(key): float(missing[month == key].mean()) for key in sorted(month.unique())}
        groups.append(
            {
                "features": features,
                "missing_rate": float(missing.mean()),
                "pass_missing_rate": float(missing[data["y_bin"] == 0].mean()),
                "fail_missing_rate": float(missing[data["y_bin"] == 1].mean()),
                "monthly_missing_rates": monthly,
                "monthly_range": max(monthly.values()) - min(monthly.values()),
            }
        )
    groups.sort(key=lambda group: (-group["monthly_range"], group["features"][0]))
    return {
        "source_url": UCI_DATASET_URL,
        "doi": "10.24432/C54305",
        "license": "CC BY 4.0",
        "files": files,
        "matches_reference_files": all(files[name]["sha256"] == sha for name, sha in SECOM_FILE_SHA256.items()),
        "n_samples": len(data),
        "n_features": len(loaded.feature_columns),
        "n_passes": int((data["y_bin"] == 0).sum()),
        "n_fails": int(data["y_bin"].sum()),
        "missing_cells": int(x.isna().sum().sum()),
        "missing_fraction": float(x.isna().to_numpy().mean()),
        "timestamp_min": data["timestamp"].min().isoformat(),
        "timestamp_max": data["timestamp"].max().isoformat(),
        "invalid_timestamps": 0,
        "row_grain": "production entity; physical unit unspecified",
        "monthly_sample_counts": {str(key): int((month == key).sum()) for key in sorted(month.unique())},
        "shared_missingness_patterns": groups,
    }


def begin_full_study(input_dir: Path, output_dir: Path, project_root: Path, classifiers_run: list[str] | None) -> None:
    """Start a fresh run so completed artifacts cannot acquire mixed input/source provenance."""
    reports = output_dir / "reports"
    if reports.exists() and any(reports.iterdir()):
        raise ValueError("Full study requires a fresh output directory; preserve prior runs and choose a new path")
    from secom.workflows.benchmark_common import reset_model_score_cache

    reset_model_score_cache()
    manifest = initial_study_manifest(project_root)
    manifest["source_tree"] = source_tree_identity(project_root)
    manifest["resolved_packages"] = installed_package_versions()
    manifest["dataset"] = dataset_profile(input_dir)
    manifest["execution"] = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "SECOM_RELIEF_N_JOBS",
                "SECOM_RELIEF_BACKEND",
            )
        },
        "settings": {
            "classifiers": classifiers_run or list(BenchmarkClassifier.TUNED_DEFAULT),
            "original_selectors": list(SelectorName.ORIGINAL_BENCHMARK),
            "tuned_selectors": list(SelectorName.ORIGINAL_BENCHMARK),
            "benchmark_seed": SEED_BENCHMARK,
            "benchmark_outer_folds": 10,
            "benchmark_inner_folds": BENCHMARK_INNER_SPLITS,
            "original_feature_budget": 40,
            "tuned_feature_budgets": [10, 20, 40],
            "relief_neighbors_original": 10,
            "relief_neighbors_tuned": [5, 10, 20],
            "relief_backend": relief_backend_name(),
            "relief_reference_version": "0.8.4",
            "relief_n_jobs": relief_worker_count(),
            "relief_cache_max_rankings": RELIEF_CACHE_SIZE,
            "original_krr_alpha_grid": BENCHMARK_KRR_ALPHA_GRID,
            "original_krr_gamma_grid": BENCHMARK_KRR_GAMMA_GRID,
            "tuned_krr_alpha_grid": TUNED_KRR_ALPHA_GRID,
            "tuned_krr_gamma_multipliers": TUNED_KRR_GAMMA_MULTIPLIERS,
            "tuned_gamma_semantics": "multiplier / actual_selected_feature_count_per_fit",
            "tuned_krr_budget": "16 versus 12 original classifier configurations per selector budget",
            "temporal_krr_calibration_fractions": TEMPORAL_KRR_CALIBRATION_FRACTIONS,
            "temporal_krr_scope": "DEV_only_joint_values_only_values_and_indicators_StandardScaler",
            "logreg_C_grid": BENCHMARK_LOGREG_C_GRID,
            "temporal_classifier": "balanced logistic regression",
            "lockbox_fraction": LOCKBOX_FRAC,
            "benchmark_inner_objective": "BER",
            "benchmark_threshold_source": "pooled_inner_out_of_fold",
            "benchmark_headline": "joint_inner_selected_procedure",
            "temporal_inner_plan": "three_chronological_expanding_splits_min_two_feasible",
            "temporal_calibration_fraction": 0.20,
            "temporal_eval_semantics": "retrospective_later_block",
            "fold_uncertainty_semantics": "descriptive_mean_std_range",
        },
    }
    write_manifest(manifest, reports / ArtifactName.MANIFEST)


def finish_full_study(output_dir: Path, durations: dict[str, float]) -> None:
    """Bind completed CSV artifacts to the run metadata before auditing or reporting."""
    reports = output_dir / "reports"
    path = reports / ArtifactName.MANIFEST
    if not path.exists():
        return
    manifest = read_manifest(path)
    execution = manifest.setdefault("execution", {})
    execution["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    execution["duration_seconds"] = durations
    from secom.workflows.benchmark_common import model_score_cache_info

    execution["model_score_cache"] = model_score_cache_info()
    manifest["artifact_sha256"] = {p.name: sha256_file(p) for p in sorted(reports.glob("*.csv"))}
    write_manifest(manifest, path)
