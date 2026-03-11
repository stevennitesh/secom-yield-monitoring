from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

SEED_LANE_A: Final[int] = 42
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
LANE_A_KRR_ALPHA_GRID: Final[list[float]] = [0.1, 1.0, 10.0]
LANE_A_KRR_GAMMA_GRID: Final[list[float | None]] = [None, 0.01, 0.1, 1.0]
LANE_A_LOGREG_C_GRID: Final[list[float]] = [0.01, 0.1, 1.0, 10.0]
LANE_A_KRR_INNER_SPLITS: Final[int] = 3


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
    STAGE_B = [S2N, WELCH_T, F_TEST, RELIEFF, GRAM_SCHMIDT]


class ScalerName:
    STANDARD = "StandardScaler"
    ROBUST = "RobustScaler"
    ALL = [STANDARD, ROBUST]


class ThresholdPolicy:
    OUTER_TRAIN_YOUDEN = "outer_train_youden_ber_optimal"
    SCIENTIFIC = "scientific"
    OPERATIONAL = "operational"


class EvalScope:
    OUTER_FOLD = "outer_fold"
    LOCKBOX = "lockbox"


class ModelScope:
    PRIMARY_FROZEN = "primary_frozen"
    CHALLENGER_FROZEN = "challenger_frozen"


class ReplicationMode:
    STRICT = "strict"
    WITH_MISSING_INDICATORS = "with_missing_indicators"


class LaneAClassifier:
    KRR = "krr"
    LOGREG = "logreg"
    KRR_STRICT = "krr_strict"
    ALL = [KRR, LOGREG]
    OPTIONAL_BENCHMARK = [KRR_STRICT]


class FoldPlanName:
    PRIMARY_3FOLD = "primary_3fold"
    FALLBACK_3FOLD = "fallback_3fold"
    FALLBACK_2FOLD = "fallback_2fold"


class ArtifactName:
    LANE_A_GLOBAL_SWEEP = "lane_a_global_sweep.csv"
    LANE_A_GLOBAL_BEST_CONFIG = "lane_a_global_best_config.csv"
    LANE_A_GLOBAL_FOLD_METRICS = "lane_a_global_fold_metrics.csv"
    LANE_A_GLOBAL_SUMMARY = "lane_a_global_summary.csv"
    LANE_A_GLOBAL_ABLATION = "lane_a_global_ablation.csv"
    LANE_A_GLOBAL_FULL_FIT_SUMMARY = "lane_a_global_full_fit_summary.csv"
    STAGE_A = "timeaware_selector_screening.csv"
    SPLITWISE = "splitwise_timeaware_results.csv"
    STAGE_B_INNER = "stage_b_inner_cv_results.csv"
    MODEL_SELECTION = "timeaware_model_selection.csv"
    SEED_STABILITY = "seed_stability_summary.csv"
    FEATURE_STABILITY = "feature_stability_by_seed.csv"
    FREEZE = "hyperparameter_freeze_results.csv"
    FINAL_LOCKBOX = "final_lockbox_result.csv"
    MSPC = "mspc_baseline.csv"
    COST_CURVES = "operational_cost_curves.csv"
    MANAGER_FACING = "manager_facing_outputs.csv"
    FEATURE_REPORT = "feature_report.csv"
    DRIFT_GATE = "drift_gate_summary.csv"
    MANIFEST = "run_manifest.json"


REQUIRED_ARTIFACTS_LANE_B: Final[list[str]] = [
    ArtifactName.LANE_A_GLOBAL_SWEEP,
    ArtifactName.LANE_A_GLOBAL_BEST_CONFIG,
    ArtifactName.LANE_A_GLOBAL_FOLD_METRICS,
    ArtifactName.LANE_A_GLOBAL_SUMMARY,
    ArtifactName.LANE_A_GLOBAL_ABLATION,
    ArtifactName.LANE_A_GLOBAL_FULL_FIT_SUMMARY,
    ArtifactName.STAGE_A,
    ArtifactName.SPLITWISE,
    ArtifactName.STAGE_B_INNER,
    ArtifactName.MODEL_SELECTION,
    ArtifactName.SEED_STABILITY,
    ArtifactName.FEATURE_STABILITY,
    ArtifactName.FREEZE,
    ArtifactName.FINAL_LOCKBOX,
    ArtifactName.MSPC,
    ArtifactName.COST_CURVES,
    ArtifactName.MANAGER_FACING,
    ArtifactName.FEATURE_REPORT,
    ArtifactName.DRIFT_GATE,
    ArtifactName.MANIFEST,
]

REQUIRED_ARTIFACTS_LANE_A_ONLY: Final[list[str]] = [
    ArtifactName.LANE_A_GLOBAL_SWEEP,
    ArtifactName.LANE_A_GLOBAL_BEST_CONFIG,
    ArtifactName.LANE_A_GLOBAL_FOLD_METRICS,
    ArtifactName.LANE_A_GLOBAL_SUMMARY,
    ArtifactName.LANE_A_GLOBAL_ABLATION,
    ArtifactName.LANE_A_GLOBAL_FULL_FIT_SUMMARY,
    ArtifactName.MANIFEST,
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
    "strategy_doc_path",
    "strategy_doc_sha256",
    "git_commit",
    "git_dirty",
    "python_executable",
    "library_versions",
    "seed_policy",
    "dev_lockbox_split",
    "outer_fold_plan_used",
    "outer_fold_week_ranges",
    "lane_b_feasible",
    "lane_b_infeasible_reason",
    "challenger_available",
    "challenger_unavailable_reason",
    "frozen_primary",
    "frozen_challenger",
    "frozen_thresholds",
    "drift_gate_results",
    "empirical_ARL0_nan_reason",
]
