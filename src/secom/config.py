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
LANE_A_KRR_BALANCED_ALPHA_GRID: Final[list[float]] = [0.1, 1.0, 10.0]
LANE_A_KRR_BALANCED_GAMMA_GRID: Final[list[float | None]] = [None, 0.01, 0.1, 1.0]
LANE_A_LOGREG_C_GRID: Final[list[float]] = [0.01, 0.1, 1.0, 10.0]
LANE_A_MRMR_LAMBDA_GRID: Final[list[float]] = [0.5, 1.0, 2.0]
LANE_A_MUTUAL_INFO_N_NEIGHBORS_GRID: Final[list[int]] = [3, 5, 10]
LANE_A_L1_SELECTOR_C_GRID: Final[list[float]] = [0.01, 0.1, 1.0, 10.0]
LANE_A_KRR_BALANCED_INNER_SPLITS: Final[int] = 3


class SelectorName:
    S2N = "S2N"
    WELCH_T = "Welch-t"
    F_TEST = "F-test"
    PEARSON = "Pearson"
    RELIEFF = "ReliefF"
    GRAM_SCHMIDT = "Gram-Schmidt"
    MUTUAL_INFO = "MutualInfo"
    MRMR = "mRMR"
    L1_LOGREG = "L1-LogReg"

    CORE = [S2N, WELCH_T, F_TEST, PEARSON, RELIEFF, GRAM_SCHMIDT]
    EXPERIMENTAL = [MUTUAL_INFO, MRMR, L1_LOGREG]
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
    KRR_STRICT = "krr_strict"
    KRR_BALANCED = "krr_balanced"
    LOGREG = "logreg"
    ALL = [KRR_STRICT, KRR_BALANCED, LOGREG]


class FoldPlanName:
    PRIMARY_3FOLD = "primary_3fold"
    FALLBACK_3FOLD = "fallback_3fold"
    FALLBACK_2FOLD = "fallback_2fold"


class ArtifactName:
    BASELINE_STRICT = "baseline_replication_strict.csv"
    BASELINE_MI = "baseline_replication_with_missing_indicators.csv"
    BASELINE_ABLATION = "baseline_missing_indicator_ablation.csv"
    BASELINE_SUMMARY = "baseline_replication_summary.csv"
    BASELINE_TUNING_TRACE = "baseline_lane_a_tuning_trace.csv"
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
    FEATURE_REPORT = "feature_report.csv"
    DRIFT_GATE = "drift_gate_summary.csv"
    MANIFEST = "run_manifest.json"


REQUIRED_ARTIFACTS_LANE_B: Final[list[str]] = [
    ArtifactName.BASELINE_STRICT,
    ArtifactName.BASELINE_MI,
    ArtifactName.BASELINE_ABLATION,
    ArtifactName.BASELINE_SUMMARY,
    ArtifactName.BASELINE_TUNING_TRACE,
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
    ArtifactName.FEATURE_REPORT,
    ArtifactName.DRIFT_GATE,
    ArtifactName.MANIFEST,
]

REQUIRED_ARTIFACTS_LANE_A_ONLY: Final[list[str]] = [
    ArtifactName.BASELINE_STRICT,
    ArtifactName.BASELINE_MI,
    ArtifactName.BASELINE_ABLATION,
    ArtifactName.BASELINE_SUMMARY,
    ArtifactName.BASELINE_TUNING_TRACE,
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
