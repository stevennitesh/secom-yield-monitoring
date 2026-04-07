from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

SEED_BENCHMARK: Final[int] = 42
SEEDS_STAGE_B: Final[list[int]] = [42, 11, 23, 37, 59]
SEEDS_PHASE2: Final[list[int]] = [42, 11, 23, 37, 59]
SEED_PHASE3: Final[int] = 42

LOCKBOX_FRAC: Final[float] = 0.15
MIN_TEST_FAILS: Final[int] = 20
INNER_MIN_CLASS: Final[int] = 5
EPS_SELECTOR: Final[float] = 1e-12
EPS_PSI: Final[float] = 1e-6
PSI_MAX_FEATURES: Final[int] = 10

COST_RATIOS: Final[list[int]] = [1, 2, 5, 10, 20]
BENCHMARK_KRR_ALPHA_GRID: Final[list[float]] = [0.1, 1.0, 10.0]
BENCHMARK_KRR_GAMMA_GRID: Final[list[float | None]] = [None, 0.01, 0.1, 1.0]
BENCHMARK_LOGREG_C_GRID: Final[list[float]] = [0.01, 0.1, 1.0, 10.0]
BENCHMARK_INNER_SPLITS: Final[int] = 3


class SelectorName:
    S2N = "S2N"
    WELCH_T = "Welch-t"
    F_TEST = "F-test"
    PEARSON = "Pearson"
    RELIEFF = "ReliefF"
    GRAM_SCHMIDT = "Gram-Schmidt"
    CORE = [S2N, WELCH_T, F_TEST, RELIEFF, GRAM_SCHMIDT]
    EXPERIMENTAL: list[str] = [PEARSON]
    ALL = CORE + EXPERIMENTAL
    ACTIVE = CORE


class ScalerName:
    STANDARD = "StandardScaler"
    ROBUST = "RobustScaler"
    ALL = [STANDARD, ROBUST]


class ThresholdPolicy:
    SCIENTIFIC = "scientific"
    OPERATIONAL = "operational"


class EvalScope:
    BENCHMARK = "benchmark"
    TEMPORAL = "temporal"
    LOCKBOX = "lockbox"


class ModelScope:
    PRIMARY = "primary"
    CHALLENGER = "challenger"


class ReplicationMode:
    STRICT = "strict"
    WITH_MISSING_INDICATORS = "with_missing_indicators"


class BenchmarkClassifier:
    KRR = "krr"
    LOGREG = "logreg"
    KRR_STRICT = "krr_strict"
    ALL = [KRR, LOGREG]
    OPTIONAL_BENCHMARK = [KRR_STRICT]


class FoldPlanName:
    PRIMARY_3FOLD = "primary_3fold"
    FALLBACK_3FOLD = "fallback_3fold"
    FALLBACK_2FOLD = "fallback_2fold"


class StudyStatus:
    NOT_RUN = "not_run"
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    ALL = [NOT_RUN, PASSED, FAILED, WARNING]


class ArtifactName:
    BENCHMARK_SWEEP = "benchmark_sweep.csv"
    BENCHMARK_BEST_CONFIG = "benchmark_best_config.csv"
    BENCHMARK_FOLD_METRICS = "benchmark_fold_metrics.csv"
    BENCHMARK_SUMMARY = "benchmark_summary.csv"
    BENCHMARK_ABLATION = "benchmark_ablation.csv"
    BENCHMARK_FULL_FIT_SUMMARY = "benchmark_full_fit_summary.csv"
    FEATURE_STABILITY = "feature_stability.csv"
    FEATURE_REPORT = "feature_report.csv"
    BENCHMARK_TUNED_SEARCH = "benchmark_tuned_search.csv"
    BENCHMARK_TUNED_BEST_CONFIG = "benchmark_tuned_best_config.csv"
    BENCHMARK_TUNED_FOLD_METRICS = "benchmark_tuned_fold_metrics.csv"
    BENCHMARK_TUNED_SUMMARY = "benchmark_tuned_summary.csv"
    BENCHMARK_TUNED_ABLATION = "benchmark_tuned_ablation.csv"
    BENCHMARK_TUNED_FULL_FIT_SUMMARY = "benchmark_tuned_full_fit_summary.csv"
    BENCHMARK_TUNED_FEATURE_STABILITY = "benchmark_tuned_feature_stability.csv"
    BENCHMARK_TUNED_FEATURE_REPORT = "benchmark_tuned_feature_report.csv"
    TEMPORAL_SPLIT_METADATA = "temporal_split_metadata.csv"
    TEMPORAL_SELECTOR_SCREENING = "temporal_selector_screening.csv"
    TEMPORAL_MODEL_SELECTION = "temporal_model_selection.csv"
    TEMPORAL_INNER_CV = "temporal_inner_cv.csv"
    TEMPORAL_FREEZE = "temporal_freeze.csv"
    TEMPORAL_LOCKBOX = "temporal_lockbox.csv"
    TEMPORAL_DRIFT = "temporal_drift_summary.csv"
    TEMPORAL_MSPC = "temporal_mspc.csv"
    TEMPORAL_COST_CURVES = "temporal_cost_curves.csv"
    TEMPORAL_MANAGER_OUTPUTS = "temporal_manager_outputs.csv"
    MANIFEST = "run_manifest.json"
    FINAL_REPORT = "final_report.md"
    REPORT_SKELETON = "final_report_skeleton.md"


REQUIRED_ARTIFACTS_PRIMARY: Final[list[str]] = [
    ArtifactName.BENCHMARK_SWEEP,
    ArtifactName.BENCHMARK_BEST_CONFIG,
    ArtifactName.BENCHMARK_FOLD_METRICS,
    ArtifactName.BENCHMARK_SUMMARY,
    ArtifactName.BENCHMARK_ABLATION,
    ArtifactName.BENCHMARK_FULL_FIT_SUMMARY,
    ArtifactName.FEATURE_STABILITY,
    ArtifactName.FEATURE_REPORT,
    ArtifactName.BENCHMARK_TUNED_SEARCH,
    ArtifactName.BENCHMARK_TUNED_BEST_CONFIG,
    ArtifactName.BENCHMARK_TUNED_FOLD_METRICS,
    ArtifactName.BENCHMARK_TUNED_SUMMARY,
    ArtifactName.BENCHMARK_TUNED_ABLATION,
    ArtifactName.BENCHMARK_TUNED_FULL_FIT_SUMMARY,
    ArtifactName.BENCHMARK_TUNED_FEATURE_STABILITY,
    ArtifactName.BENCHMARK_TUNED_FEATURE_REPORT,
    ArtifactName.MANIFEST,
]

REQUIRED_ARTIFACTS_TEMPORAL: Final[list[str]] = [
    ArtifactName.TEMPORAL_SPLIT_METADATA,
    ArtifactName.TEMPORAL_SELECTOR_SCREENING,
    ArtifactName.TEMPORAL_MODEL_SELECTION,
    ArtifactName.TEMPORAL_INNER_CV,
    ArtifactName.TEMPORAL_FREEZE,
    ArtifactName.TEMPORAL_LOCKBOX,
    ArtifactName.TEMPORAL_DRIFT,
    ArtifactName.TEMPORAL_MSPC,
    ArtifactName.TEMPORAL_COST_CURVES,
    ArtifactName.TEMPORAL_MANAGER_OUTPUTS,
]


@dataclass(frozen=True)
class Paths:
    project_root: Path
    input_dir: Path
    output_dir: Path

    @property
    def reports_dir(self) -> Path:
        return self.output_dir / "reports"


MANIFEST_REQUIRED_KEYS: Final[list[str]] = [
    "manifest_version",
    "study_spec_path",
    "study_spec_sha256",
    "git_commit",
    "git_dirty",
    "python_executable",
    "library_versions",
    "primary_study_status",
    "benchmark_original_status",
    "benchmark_tuned_status",
    "temporal_robustness_status",
    "temporal_claim_restrictions",
    "industrialization_notes",
]
