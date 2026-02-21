from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from secom.artifacts import (
    ValidationResult,
    config_hash,
    ensure_reports_dir,
    validate_required_artifacts,
    validate_schema_and_logic,
    write_csv,
    write_manifest,
)
from secom.config import (
    ArtifactName,
    COST_RATIOS,
    EPS_PSI,
    INNER_MIN_CLASS,
    LaneAClassifier,
    LANE_A_KRR_BALANCED_ALPHA_GRID,
    LANE_A_KRR_BALANCED_GAMMA_GRID,
    LANE_A_KRR_BALANCED_INNER_SPLITS,
    LANE_A_LOGREG_C_GRID,
    LOCKBOX_FRAC,
    MIN_TEST_FAILS,
    ModelScope,
    PSI_MAX_FEATURES,
    ReplicationMode,
    ScalerName,
    SEED_LANE_A,
    SEED_PHASE3,
    SEEDS_PHASE2,
    SEEDS_STAGE_B,
    SelectorName,
    ThresholdPolicy,
)
from secom.cv import (
    OuterFoldPlanResult,
    add_dev_week_bins,
    choose_outer_fold_plan,
    fold_plan_manifest_ranges,
    lane_b_feasibility_gate,
    split_dev_lockbox,
    to_time_window_string,
)
from secom.feature_select.gram_schmidt import gram_schmidt_rank_features
from secom.feature_select.relief import relief_rank_features
from secom.feature_select.univariate import rank_features
from secom.io import LoadedSecom, load_raw_secom, parse_sort_and_label
from secom.metrics import (
    binary_metrics_at_threshold,
    bootstrap_ci_for_mean,
    candidate_thresholds,
    confusion_counts,
    expected_cost_per_wafer,
    extract_tpr_at_tnr,
    find_ber_optimal_threshold,
    safe_std,
    true_pos_rate,
)
from secom.models import (
    fit_lane_a_balanced_classifier,
    make_lane_a_classifier,
    make_lane_a_logreg_tuned_classifier,
    fit_lane_b_classifier,
)
from secom.preprocess import (
    build_feature_universe,
    local_to_global_feature_indices,
    make_imputer,
    make_scaler,
    transformed_feature_metadata_from_imputer,
)
from secom.qa import validate_lane_a_artifacts


@dataclass(frozen=True)
class DataBundle:
    all_data: pd.DataFrame
    dev: pd.DataFrame
    lockbox: pd.DataFrame
    feature_columns: list[str]
    dev_with_weeks: pd.DataFrame
    fold_plan: OuterFoldPlanResult | None
    lane_b_feasible: bool
    lane_b_infeasible_reason: str | None


@dataclass(frozen=True)
class RoleConfig:
    role: str
    selector: str
    k: int
    c_value: float
    scaler: str
    n_neighbors: int | None

    def to_hash_payload(self) -> dict[str, Any]:
        return {
            "selector": self.selector,
            "k": int(self.k),
            "C": float(self.c_value),
            "scaler": self.scaler,
            "n_neighbors": None if self.n_neighbors is None else int(self.n_neighbors),
        }


@dataclass
class FittedRoleModel:
    config: RoleConfig
    imputer: Any
    scaler: Any
    selected_local_idx: np.ndarray
    selected_global_idx: list[int]
    clf: Any
    dev_scores: np.ndarray
    scientific_threshold: float
    operational_threshold: float
    threshold_at_tnr90_dev: float
    tnr_at_tnr90_dev: float
    tpr_at_tnr90_dev: float
    feature_meta: list[Any]


def _git_commit_and_dirty(project_root: Path) -> tuple[str, bool]:
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True)
            .strip()
        )
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=project_root, text=True).strip()
        )
        return commit, dirty
    except Exception:
        return "UNKNOWN", True


def _strategy_sha256(project_root: Path) -> str:
    strategy = project_root / "docs" / "final_end_to_end_report_strategy_merged.md"
    if not strategy.exists():
        return "MISSING"
    h = hashlib.sha256()
    h.update(strategy.read_bytes())
    return h.hexdigest()


def _library_versions() -> dict[str, str]:
    import numpy
    import pandas
    import scipy
    import sklearn

    try:
        import skrebate

        skrebate_v = getattr(skrebate, "__version__", "UNKNOWN")
    except Exception:
        skrebate_v = "UNAVAILABLE"

    return {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "sklearn": sklearn.__version__,
        "scipy": scipy.__version__,
        "skrebate": skrebate_v,
    }


def _selector_pick(
    method: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
    n_neighbors: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if method in {SelectorName.S2N, SelectorName.WELCH_T, SelectorName.F_TEST, SelectorName.PEARSON}:
        order, scores = rank_features(method, x_train, y_train)
        selected = order[: min(k, order.shape[0])]
        return selected, scores
    if method == SelectorName.RELIEFF:
        if n_neighbors is None:
            raise ValueError("ReliefF requires n_neighbors")
        order, scores = relief_rank_features(x_train, y_train, n_neighbors=n_neighbors)
        selected = order[: min(k, order.shape[0])]
        return selected, scores
    if method == SelectorName.GRAM_SCHMIDT:
        order, scores = gram_schmidt_rank_features(x_train, y_train, k=k)
        if order.shape[0] > k:
            order = order[:k]
        return order, scores
    raise ValueError(f"Unknown selector {method}")


def _fit_selector_pipeline(
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
    x_eval_raw: np.ndarray,
    method: str,
    k: int,
    scaler_name: str,
    add_indicator: bool,
    n_neighbors: int | None,
) -> tuple[np.ndarray, np.ndarray, list[Any], np.ndarray, Any, Any]:
    imputer = make_imputer(add_indicator=add_indicator)
    x_train_imp = imputer.fit_transform(x_train_raw)
    x_eval_imp = imputer.transform(x_eval_raw)
    scaler = make_scaler(scaler_name)
    x_train_scaled = scaler.fit_transform(x_train_imp)
    x_eval_scaled = scaler.transform(x_eval_imp)

    selected_local, _scores = _selector_pick(
        method=method,
        x_train=x_train_scaled,
        y_train=y_train,
        k=int(k),
        n_neighbors=n_neighbors,
    )
    feature_meta = transformed_feature_metadata_from_imputer(
        imputer=imputer, raw_feature_count=x_train_raw.shape[1]
    )
    x_train_sel = x_train_scaled[:, selected_local]
    x_eval_sel = x_eval_scaled[:, selected_local]
    return x_train_sel, x_eval_sel, feature_meta, selected_local, imputer, scaler


def _lane_b_fit_eval_with_labels(
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
    x_eval_raw: np.ndarray,
    y_eval: np.ndarray,
    method: str,
    k: int,
    c_value: float,
    scaler_name: str,
    n_neighbors: int | None,
) -> tuple[dict[str, float], float, list[Any], np.ndarray, Any, Any, Any]:
    x_train_sel, x_eval_sel, feature_meta, selected_local, imputer, scaler = _fit_selector_pipeline(
        x_train_raw=x_train_raw,
        y_train=y_train,
        x_eval_raw=x_eval_raw,
        method=method,
        k=k,
        scaler_name=scaler_name,
        add_indicator=True,
        n_neighbors=n_neighbors,
    )
    clf = fit_lane_b_classifier(x_train_sel, y_train, c_value=c_value)
    train_scores = clf.predict_proba(x_train_sel)[:, 1]
    eval_scores = clf.predict_proba(x_eval_sel)[:, 1]
    threshold, _ = find_ber_optimal_threshold(y_train, train_scores)
    metrics = binary_metrics_at_threshold(y_eval, eval_scores, threshold)
    return metrics, threshold, feature_meta, selected_local, clf, imputer, scaler


def _weekly_flag_fraction(scores: np.ndarray, threshold: float, week_labels: np.ndarray) -> float:
    preds = (scores >= threshold).astype(int)
    weeks = np.asarray(week_labels, dtype=int)
    fractions: list[float] = []
    for w in sorted(np.unique(weeks).tolist()):
        idx = np.where(weeks == w)[0]
        if idx.size == 0:
            continue
        fractions.append(float(np.mean(preds[idx])))
    if not fractions:
        return 0.0
    return float(np.mean(fractions))


def _operational_threshold(scores: np.ndarray, y_true: np.ndarray, week_labels: np.ndarray) -> float:
    best_threshold = None
    best_tpr = -np.inf
    for t in candidate_thresholds(scores):
        frac = _weekly_flag_fraction(scores=scores, threshold=float(t), week_labels=week_labels)
        if frac <= 0.10:
            counts = confusion_counts(y_true, (scores >= t).astype(int))
            tpr = true_pos_rate(counts)
            if tpr > best_tpr:
                best_tpr = tpr
                best_threshold = float(t)
            elif np.isclose(tpr, best_tpr):
                if best_threshold is None or float(t) < best_threshold:
                    best_threshold = float(t)
    if best_threshold is None:
        return float(np.inf)
    return best_threshold


def run_01_data_contract_and_split(input_dir: Path, output_dir: Path, project_root: Path) -> DataBundle:
    reports = ensure_reports_dir(output_dir)
    loaded: LoadedSecom = load_raw_secom(input_dir)
    all_sorted = parse_sort_and_label(loaded.frame)

    split = split_dev_lockbox(all_sorted, lockbox_frac=LOCKBOX_FRAC)
    dev_weeks = add_dev_week_bins(split.dev)
    plan = choose_outer_fold_plan(dev_weeks, min_test_fails=MIN_TEST_FAILS)
    feasible, reason = lane_b_feasibility_gate(
        dev=dev_weeks, plan=plan, min_class_count=INNER_MIN_CLASS
    )

    split_meta = pd.DataFrame(
        [
            {
                "N_total_after_NaT_drop": split.n_total_after_nat_drop,
                "N_dev": split.n_dev,
                "N_lockbox": split.n_lockbox,
                "lockbox_rule": "last floor(0.15*N) rows after stable sort by (timestamp, raw_row_id)",
                "outer_fold_plan_used": None if plan is None else plan.plan_name,
                "lane_b_feasible": feasible,
                "lane_b_infeasible_reason": reason,
            }
        ]
    )
    write_csv(split_meta, reports / "split_metadata.csv")

    commit, dirty = _git_commit_and_dirty(project_root)
    manifest = {
        "manifest_version": "1.0",
        "strategy_doc_path": "docs/final_end_to_end_report_strategy_merged.md",
        "strategy_doc_sha256": _strategy_sha256(project_root),
        "git_commit": commit,
        "git_dirty": dirty,
        "python_executable": sys.executable,
        "library_versions": _library_versions(),
        "seed_policy": {
            "lane_a": [SEED_LANE_A],
            "stage_b": SEEDS_STAGE_B,
            "phase_2": SEEDS_PHASE2,
            "phase_3": [SEED_PHASE3],
        },
        "dev_lockbox_split": {
            "N_total_after_NaT_drop": split.n_total_after_nat_drop,
            "N_dev": split.n_dev,
            "N_lockbox": split.n_lockbox,
            "lockbox_rule": "last floor(0.15*N) rows after stable sort by (timestamp, raw_row_id)",
        },
        "outer_fold_plan_used": None if plan is None else plan.plan_name,
        "outer_fold_week_ranges": [] if plan is None else fold_plan_manifest_ranges(plan),
        "lane_b_feasible": feasible,
        "lane_b_infeasible_reason": reason,
        "challenger_available": None,
        "challenger_unavailable_reason": None,
        "frozen_primary": None,
        "frozen_challenger": None,
        "frozen_thresholds": None,
        "drift_gate_results": None,
        "empirical_ARL0_nan_reason": None,
    }
    write_manifest(manifest, reports / ArtifactName.MANIFEST)
    write_csv(dev_weeks, reports / "dev_sorted_with_weeks.csv")
    write_csv(split.lockbox, reports / "lockbox_sorted.csv")

    return DataBundle(
        all_data=all_sorted,
        dev=split.dev,
        lockbox=split.lockbox,
        feature_columns=loaded.feature_columns,
        dev_with_weeks=dev_weeks,
        fold_plan=plan,
        lane_b_feasible=feasible,
        lane_b_infeasible_reason=reason,
    )


def _gamma_sort_key(gamma: float | None) -> float:
    return -1.0 if gamma is None else float(gamma)


def _select_krr_balanced_config_with_inner_cv(
    x_train_sel: np.ndarray,
    y_train: np.ndarray,
) -> tuple[float, float | None, Any, float]:
    y_train = np.asarray(y_train, dtype=int)
    n_fail = int(np.sum(y_train == 1))
    n_pass = int(np.sum(y_train == 0))
    min_class = min(n_fail, n_pass)
    n_splits = min(int(LANE_A_KRR_BALANCED_INNER_SPLITS), min_class)
    sorted_alphas = sorted(float(a) for a in LANE_A_KRR_BALANCED_ALPHA_GRID)
    sorted_gammas = sorted((None if g is None else float(g) for g in LANE_A_KRR_BALANCED_GAMMA_GRID), key=_gamma_sort_key)

    if n_splits < 2:
        fallback_alpha = float(sorted_alphas[0])
        fallback_gamma = sorted_gammas[0]
        fallback_clf = fit_lane_a_balanced_classifier(
            x_train_sel,
            y_train,
            alpha=fallback_alpha,
            gamma=fallback_gamma,
        )
        fallback_train_scores = np.asarray(fallback_clf.predict(x_train_sel), dtype=float)
        fallback_threshold, _ = find_ber_optimal_threshold(y_train, fallback_train_scores)
        return fallback_alpha, fallback_gamma, fallback_clf, float(fallback_threshold)

    inner_cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED_LANE_A,
    )
    best_alpha: float | None = None
    best_gamma: float | None = None
    best_inner_ber = np.inf

    # NOTE: selector fixed from outer-train; inner CV tunes classifier only (explicit approximation).
    for alpha, gamma in product(sorted_alphas, sorted_gammas):
        fold_bers: list[float] = []
        for inner_train_idx, inner_val_idx in inner_cv.split(x_train_sel, y_train):
            x_inner_train = x_train_sel[inner_train_idx]
            y_inner_train = y_train[inner_train_idx]
            x_inner_val = x_train_sel[inner_val_idx]
            y_inner_val = y_train[inner_val_idx]

            clf_inner = fit_lane_a_balanced_classifier(
                x_inner_train,
                y_inner_train,
                alpha=alpha,
                gamma=gamma,
            )
            inner_train_scores = np.asarray(clf_inner.predict(x_inner_train), dtype=float)
            inner_threshold, _ = find_ber_optimal_threshold(y_inner_train, inner_train_scores)
            inner_val_scores = np.asarray(clf_inner.predict(x_inner_val), dtype=float)
            inner_metrics = binary_metrics_at_threshold(
                y_inner_val,
                inner_val_scores,
                threshold=float(inner_threshold),
            )
            fold_bers.append(float(inner_metrics["BER"]))

        mean_inner_ber = float(np.mean(fold_bers))
        if mean_inner_ber < best_inner_ber - 1e-12:
            best_inner_ber = mean_inner_ber
            best_alpha = alpha
            best_gamma = gamma
        elif np.isclose(mean_inner_ber, best_inner_ber):
            if best_alpha is None or alpha < best_alpha:
                best_alpha = alpha
                best_gamma = gamma
            elif best_alpha is not None and np.isclose(alpha, best_alpha):
                if _gamma_sort_key(gamma) < _gamma_sort_key(best_gamma):
                    best_gamma = gamma

    if best_alpha is None:
        raise RuntimeError("krr_balanced: failed to choose (alpha, gamma) from inner CV")

    final_clf = fit_lane_a_balanced_classifier(
        x_train_sel,
        y_train,
        alpha=float(best_alpha),
        gamma=best_gamma,
    )
    final_train_scores = np.asarray(final_clf.predict(x_train_sel), dtype=float)
    final_threshold, _ = find_ber_optimal_threshold(y_train, final_train_scores)
    return float(best_alpha), best_gamma, final_clf, float(final_threshold)


def _select_logreg_config_with_inner_cv(
    x_train_sel: np.ndarray,
    y_train: np.ndarray,
) -> tuple[float, Any, float]:
    y_train = np.asarray(y_train, dtype=int)
    n_fail = int(np.sum(y_train == 1))
    n_pass = int(np.sum(y_train == 0))
    min_class = min(n_fail, n_pass)
    n_splits = min(int(LANE_A_KRR_BALANCED_INNER_SPLITS), min_class)
    sorted_c_values = sorted(float(c) for c in LANE_A_LOGREG_C_GRID)

    if n_splits < 2:
        fallback_c = float(sorted_c_values[0])
        fallback_clf = make_lane_a_logreg_tuned_classifier(c_value=fallback_c)
        fallback_clf.fit(x_train_sel, y_train)
        fallback_train_scores = np.asarray(fallback_clf.predict_proba(x_train_sel)[:, 1], dtype=float)
        fallback_threshold, _ = find_ber_optimal_threshold(y_train, fallback_train_scores)
        return fallback_c, fallback_clf, float(fallback_threshold)

    inner_cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED_LANE_A,
    )
    best_c: float | None = None
    best_inner_ber = np.inf

    # NOTE: selector fixed from outer-train; inner CV tunes classifier only (explicit approximation).
    for c_value in sorted_c_values:
        fold_bers: list[float] = []
        for inner_train_idx, inner_val_idx in inner_cv.split(x_train_sel, y_train):
            x_inner_train = x_train_sel[inner_train_idx]
            y_inner_train = y_train[inner_train_idx]
            x_inner_val = x_train_sel[inner_val_idx]
            y_inner_val = y_train[inner_val_idx]

            clf_inner = make_lane_a_logreg_tuned_classifier(c_value=c_value)
            clf_inner.fit(x_inner_train, y_inner_train)
            inner_train_scores = np.asarray(clf_inner.predict_proba(x_inner_train)[:, 1], dtype=float)
            inner_threshold, _ = find_ber_optimal_threshold(y_inner_train, inner_train_scores)
            inner_val_scores = np.asarray(clf_inner.predict_proba(x_inner_val)[:, 1], dtype=float)
            inner_metrics = binary_metrics_at_threshold(
                y_inner_val,
                inner_val_scores,
                threshold=float(inner_threshold),
            )
            fold_bers.append(float(inner_metrics["BER"]))

        mean_inner_ber = float(np.mean(fold_bers))
        if mean_inner_ber < best_inner_ber - 1e-12:
            best_inner_ber = mean_inner_ber
            best_c = c_value
        elif np.isclose(mean_inner_ber, best_inner_ber):
            if best_c is None or c_value < best_c:
                best_c = c_value

    if best_c is None:
        raise RuntimeError("logreg: failed to choose C from inner CV")

    final_clf = make_lane_a_logreg_tuned_classifier(c_value=float(best_c))
    final_clf.fit(x_train_sel, y_train)
    final_train_scores = np.asarray(final_clf.predict_proba(x_train_sel)[:, 1], dtype=float)
    final_threshold, _ = find_ber_optimal_threshold(y_train, final_train_scores)
    return float(best_c), final_clf, float(final_threshold)


def _lane_a_run_mode(
    df: pd.DataFrame,
    feature_cols: list[str],
    selector: str,
    add_indicator: bool,
    classifier: str,
    replication_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = df[feature_cols].to_numpy(dtype=float)
    y = df["y_bin"].to_numpy(dtype=int)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED_LANE_A)
    rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(x, y), start=1):
        x_train_raw = x[train_idx]
        y_train = y[train_idx]
        x_test_raw = x[test_idx]
        y_test = y[test_idx]

        imputer = make_imputer(add_indicator=add_indicator)
        x_train_imp = imputer.fit_transform(x_train_raw)
        x_test_imp = imputer.transform(x_test_raw)
        scaler = make_scaler(ScalerName.STANDARD)
        x_train = scaler.fit_transform(x_train_imp)
        x_test = scaler.transform(x_test_imp)

        selected_local, _ = _selector_pick(
            method=selector,
            x_train=x_train,
            y_train=y_train,
            k=40,
            n_neighbors=10 if selector == SelectorName.RELIEFF else None,
        )

        x_train_sel = x_train[:, selected_local]
        x_test_sel = x_test[:, selected_local]

        chosen_alpha: float | None = None
        chosen_gamma: float | None = None
        chosen_c: float | None = None
        threshold: float
        scores: np.ndarray

        if classifier == LaneAClassifier.KRR_STRICT:
            strict_clf = make_lane_a_classifier(alpha=1.0, gamma=None)
            y_train_krr = 2 * np.asarray(y_train, dtype=int) - 1
            strict_clf.fit(x_train_sel, y_train_krr)
            strict_train_scores = np.asarray(strict_clf.predict(x_train_sel), dtype=float)
            threshold, _ = find_ber_optimal_threshold(y_train, strict_train_scores)
            scores = np.asarray(strict_clf.predict(x_test_sel), dtype=float)
        elif classifier == LaneAClassifier.KRR_BALANCED:
            chosen_alpha, chosen_gamma, best_clf, threshold = _select_krr_balanced_config_with_inner_cv(
                x_train_sel=x_train_sel,
                y_train=y_train,
            )
            scores = np.asarray(best_clf.predict(x_test_sel), dtype=float)
            tuning_rows.append(
                {
                    "selector": selector,
                    "classifier": classifier,
                    "fold": fold_i,
                    "replication_mode": replication_mode,
                    "chosen_alpha": chosen_alpha,
                    "chosen_gamma": np.nan if chosen_gamma is None else float(chosen_gamma),
                    "chosen_C": np.nan,
                    "threshold": float(threshold),
                    "selector_tuning_scope": "outer_train_fixed",
                }
            )
        elif classifier == LaneAClassifier.LOGREG:
            chosen_c, best_clf, threshold = _select_logreg_config_with_inner_cv(
                x_train_sel=x_train_sel,
                y_train=y_train,
            )
            scores = np.asarray(best_clf.predict_proba(x_test_sel)[:, 1], dtype=float)
            tuning_rows.append(
                {
                    "selector": selector,
                    "classifier": classifier,
                    "fold": fold_i,
                    "replication_mode": replication_mode,
                    "chosen_alpha": np.nan,
                    "chosen_gamma": np.nan,
                    "chosen_C": chosen_c,
                    "threshold": float(threshold),
                    "selector_tuning_scope": "outer_train_fixed",
                }
            )
        else:
            raise ValueError(f"Unknown Lane A classifier mode: {classifier}")

        metrics = binary_metrics_at_threshold(y_test, scores, threshold=threshold)
        rows.append(
            {
                "selector": selector,
                "classifier": classifier,
                "fold": fold_i,
                "BER": metrics["BER"],
                "True+": metrics["True+"],
                "True-": metrics["True-"],
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_test_fails": int(np.sum(y_test == 1)),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(tuning_rows)


def run_02_lane_a_replication(
    bundle: DataBundle,
    output_dir: Path,
    lane_a_classifier: str | None = None,
) -> None:
    reports = ensure_reports_dir(output_dir)
    classifiers_run = (
        list(LaneAClassifier.ALL) if lane_a_classifier is None else [str(lane_a_classifier)]
    )
    bad = set(classifiers_run) - set(LaneAClassifier.ALL)
    if bad:
        raise ValueError(f"Unknown Lane A classifier(s): {sorted(bad)}")

    strict_rows: list[pd.DataFrame] = []
    mi_rows: list[pd.DataFrame] = []
    tuning_rows: list[pd.DataFrame] = []
    for classifier in classifiers_run:
        for selector in SelectorName.ALL:
            strict_frame, strict_tuning = _lane_a_run_mode(
                df=bundle.all_data,
                feature_cols=bundle.feature_columns,
                selector=selector,
                add_indicator=False,
                classifier=classifier,
                replication_mode=ReplicationMode.STRICT,
            )
            mi_frame, mi_tuning = _lane_a_run_mode(
                df=bundle.all_data,
                feature_cols=bundle.feature_columns,
                selector=selector,
                add_indicator=True,
                classifier=classifier,
                replication_mode=ReplicationMode.WITH_MISSING_INDICATORS,
            )
            strict_rows.append(strict_frame)
            mi_rows.append(mi_frame)
            tuning_rows.extend([strict_tuning, mi_tuning])

    strict_df = pd.concat(strict_rows, ignore_index=True)
    mi_df = pd.concat(mi_rows, ignore_index=True)
    write_csv(strict_df, reports / ArtifactName.BASELINE_STRICT)
    write_csv(mi_df, reports / ArtifactName.BASELINE_MI)

    tuning_trace_df = (
        pd.concat([df for df in tuning_rows if not df.empty], ignore_index=True)
        if any(not df.empty for df in tuning_rows)
        else pd.DataFrame(
            columns=[
                "selector",
                "classifier",
                "fold",
                "replication_mode",
                "chosen_alpha",
                "chosen_gamma",
                "chosen_C",
                "threshold",
                "selector_tuning_scope",
            ]
        )
    )
    write_csv(tuning_trace_df, reports / ArtifactName.BASELINE_TUNING_TRACE)

    ablation_rows = []
    summary_rows = []
    for classifier in classifiers_run:
        for selector in SelectorName.ALL:
            s = strict_df.loc[
                (strict_df["selector"] == selector) & (strict_df["classifier"] == classifier)
            ].sort_values("fold")
            m = mi_df.loc[
                (mi_df["selector"] == selector) & (mi_df["classifier"] == classifier)
            ].sort_values("fold")
            delta = s["BER"].to_numpy() - m["BER"].to_numpy()
            lo, hi = bootstrap_ci_for_mean(delta, n_boot=1000, seed=42, alpha=0.95)
            ablation_rows.append(
                {
                    "selector": selector,
                    "classifier": classifier,
                    "BER_strict": float(np.mean(s["BER"])),
                    "BER_MI": float(np.mean(m["BER"])),
                    "delta_BER": float(np.mean(delta)),
                    "CI_lower": lo,
                    "CI_upper": hi,
                    "n_boot": 1000,
                }
            )

            for mode_name, frame in [
                (ReplicationMode.STRICT, s),
                (ReplicationMode.WITH_MISSING_INDICATORS, m),
            ]:
                ber_lo, ber_hi = bootstrap_ci_for_mean(
                    frame["BER"].to_numpy(), n_boot=1000, seed=42
                )
                tp_lo, tp_hi = bootstrap_ci_for_mean(
                    frame["True+"].to_numpy(), n_boot=1000, seed=42
                )
                tn_lo, tn_hi = bootstrap_ci_for_mean(
                    frame["True-"].to_numpy(), n_boot=1000, seed=42
                )
                summary_rows.append(
                    {
                        "selector": selector,
                        "classifier": classifier,
                        "replication_mode": mode_name,
                        "n_folds": int(len(frame)),
                        "n_boot": 1000,
                        "boot_seed": 42,
                        "mean_BER": float(np.mean(frame["BER"])),
                        "std_BER": safe_std(frame["BER"].to_numpy()),
                        "CI_lower_BER": ber_lo,
                        "CI_upper_BER": ber_hi,
                        "mean_True+": float(np.mean(frame["True+"])),
                        "std_True+": safe_std(frame["True+"].to_numpy()),
                        "CI_lower_True+": tp_lo,
                        "CI_upper_True+": tp_hi,
                        "mean_True-": float(np.mean(frame["True-"])),
                        "std_True-": safe_std(frame["True-"].to_numpy()),
                        "CI_lower_True-": tn_lo,
                        "CI_upper_True-": tn_hi,
                    }
                )

    ablation_df = pd.DataFrame(ablation_rows)
    summary_df = pd.DataFrame(summary_rows)
    write_csv(ablation_df, reports / ArtifactName.BASELINE_ABLATION)
    write_csv(summary_df, reports / ArtifactName.BASELINE_SUMMARY)
    validate_lane_a_artifacts(
        summary_df=summary_df,
        ablation_df=ablation_df,
        strict_df=strict_df,
        mi_df=mi_df,
        tuning_trace_df=tuning_trace_df,
        classifiers_run=classifiers_run,
    )


def _stage_a_configs() -> list[dict[str, Any]]:
    return [
        {
            "selector": s,
            "k": 40,
            "C": 1.0,
            "scaler": ScalerName.ROBUST,
            "n_neighbors": 10 if s == SelectorName.RELIEFF else None,
        }
        for s in SelectorName.ALL
    ]


def _stage_b_configs(selector: str) -> list[dict[str, Any]]:
    ks = [10, 20, 40]
    cs = [0.01, 0.1, 1.0, 10.0]
    scalers = [ScalerName.STANDARD, ScalerName.ROBUST]
    configs: list[dict[str, Any]] = []
    if selector == SelectorName.RELIEFF:
        for nn in [5, 10, 20]:
            for k in ks:
                for c in cs:
                    for scaler in scalers:
                        configs.append(
                            {
                                "selector": selector,
                                "k": k,
                                "C": c,
                                "scaler": scaler,
                                "n_neighbors": nn,
                            }
                        )
    else:
        for k in ks:
            for c in cs:
                for scaler in scalers:
                    configs.append(
                        {
                            "selector": selector,
                            "k": k,
                            "C": c,
                            "scaler": scaler,
                            "n_neighbors": None,
                        }
                    )
    return configs


def _inner_cv_scores(
    x_outer_train_raw: np.ndarray,
    y_outer_train: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> tuple[float, float]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs: list[float] = []
    bers: list[float] = []

    for inner_train_idx, inner_val_idx in skf.split(x_outer_train_raw, y_outer_train):
        x_inner_train = x_outer_train_raw[inner_train_idx]
        y_inner_train = y_outer_train[inner_train_idx]
        x_inner_val = x_outer_train_raw[inner_val_idx]
        y_inner_val = y_outer_train[inner_val_idx]

        x_train_sel, x_val_sel, _meta, _sel, _imp, _scaler = _fit_selector_pipeline(
            x_train_raw=x_inner_train,
            y_train=y_inner_train,
            x_eval_raw=x_inner_val,
            method=config["selector"],
            k=config["k"],
            scaler_name=config["scaler"],
            add_indicator=True,
            n_neighbors=config.get("n_neighbors"),
        )
        clf = fit_lane_b_classifier(x_train_sel, y_inner_train, c_value=config["C"])
        train_scores = clf.predict_proba(x_train_sel)[:, 1]
        val_scores = clf.predict_proba(x_val_sel)[:, 1]
        threshold, _ = find_ber_optimal_threshold(y_inner_train, train_scores)
        m = binary_metrics_at_threshold(y_inner_val, val_scores, threshold)
        aucs.append(float(m["ROC_AUC"]) if np.isfinite(m["ROC_AUC"]) else 0.5)
        bers.append(float(m["BER"]))

    return float(np.mean(aucs)), float(np.mean(bers))


def _select_best_inner_config(config_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not config_rows:
        raise ValueError("No configs to select")
    best_auc = max(r["mean_inner_ROC_AUC"] for r in config_rows)
    near = [
        r
        for r in config_rows
        if r["mean_inner_ROC_AUC"] >= best_auc - 0.01 - 1e-12
    ]
    min_ber = min(r["mean_inner_BER"] for r in near)
    tied = [r for r in near if np.isclose(r["mean_inner_BER"], min_ber)]

    def key(row):
        nn = row.get("n_neighbors")
        nn_key = math.inf if nn is None else nn
        scaler_pref = 0 if row["scaler"] == ScalerName.STANDARD else 1
        return (row["k"], row["C"], scaler_pref, nn_key)

    return sorted(tied, key=key)[0]


def run_03_lane_b_stage_ab(bundle: DataBundle, output_dir: Path) -> dict[str, Any]:
    reports = ensure_reports_dir(output_dir)
    if not bundle.lane_b_feasible or bundle.fold_plan is None:
        return {"lane_b_feasible": False, "reason": bundle.lane_b_infeasible_reason}

    x_dev = bundle.dev_with_weeks[bundle.feature_columns].to_numpy(dtype=float)
    y_dev = bundle.dev_with_weeks["y_bin"].to_numpy(dtype=int)

    stage_a_rows: list[dict[str, Any]] = []
    for cfg in _stage_a_configs():
        selector = cfg["selector"]
        fold_metrics = []
        for fold in bundle.fold_plan.folds:
            metrics, threshold, _meta, _sel, _clf, _imp, _scl = _lane_b_fit_eval_with_labels(
                x_train_raw=x_dev[fold.train_index],
                y_train=y_dev[fold.train_index],
                x_eval_raw=x_dev[fold.test_index],
                y_eval=y_dev[fold.test_index],
                method=selector,
                k=cfg["k"],
                c_value=cfg["C"],
                scaler_name=cfg["scaler"],
                n_neighbors=cfg.get("n_neighbors"),
            )
            fold_metrics.append(
                {
                    "BER": metrics["BER"],
                    "True+": metrics["True+"],
                    "True-": metrics["True-"],
                    "test_fails": int(np.sum(y_dev[fold.test_index] == 1)),
                    "threshold": threshold,
                }
            )

        stage_a_rows.append(
            {
                "selector": selector,
                "n_splits": len(bundle.fold_plan.folds),
                "mean_BER": float(np.mean([m["BER"] for m in fold_metrics])),
                "std_BER": safe_std([m["BER"] for m in fold_metrics]),
                "mean_True+": float(np.mean([m["True+"] for m in fold_metrics])),
                "mean_True-": float(np.mean([m["True-"] for m in fold_metrics])),
                "min_test_fails": int(np.min([m["test_fails"] for m in fold_metrics])),
            }
        )
    write_csv(pd.DataFrame(stage_a_rows), reports / ArtifactName.STAGE_A)

    splitwise_rows: list[dict[str, Any]] = []
    stage_b_inner_rows: list[dict[str, Any]] = []
    feature_stability_rows: list[dict[str, Any]] = []
    feature_universe = build_feature_universe(raw_feature_count=len(bundle.feature_columns))

    for selector in SelectorName.STAGE_B:
        configs = _stage_b_configs(selector)
        for fold in bundle.fold_plan.folds:
            x_outer_train = x_dev[fold.train_index]
            y_outer_train = y_dev[fold.train_index]
            x_outer_test = x_dev[fold.test_index]
            y_outer_test = y_dev[fold.test_index]

            for seed in SEEDS_STAGE_B:
                config_scores = []
                for cfg in configs:
                    mean_auc, mean_ber = _inner_cv_scores(
                        x_outer_train_raw=x_outer_train,
                        y_outer_train=y_outer_train,
                        config=cfg,
                        seed=seed,
                    )
                    row = dict(cfg)
                    row["selector"] = selector
                    row["outer_fold"] = fold.outer_fold
                    row["seed"] = seed
                    row["mean_inner_ROC_AUC"] = mean_auc
                    row["mean_inner_BER"] = mean_ber
                    config_scores.append(row)

                best = _select_best_inner_config(config_scores)
                for row in config_scores:
                    row["is_selected_config"] = (
                        row["k"] == best["k"]
                        and np.isclose(row["C"], best["C"])
                        and row["scaler"] == best["scaler"]
                        and row.get("n_neighbors") == best.get("n_neighbors")
                    )
                    stage_b_inner_rows.append(row)

                metrics, threshold, meta, selected_local, _clf, _imp, _scl = _lane_b_fit_eval_with_labels(
                    x_train_raw=x_outer_train,
                    y_train=y_outer_train,
                    x_eval_raw=x_outer_test,
                    y_eval=y_outer_test,
                    method=selector,
                    k=int(best["k"]),
                    c_value=float(best["C"]),
                    scaler_name=best["scaler"],
                    n_neighbors=best.get("n_neighbors"),
                )
                selected_global = set(local_to_global_feature_indices(selected_local, meta))

                splitwise_rows.append(
                    {
                        "selector": selector,
                        "outer_fold": fold.outer_fold,
                        "seed": seed,
                        "train_window": to_time_window_string(fold.train_start_ts, fold.train_end_ts),
                        "test_window": to_time_window_string(fold.test_start_ts, fold.test_end_ts),
                        "k": int(best["k"]),
                        "C": float(best["C"]),
                        "scaler": best["scaler"],
                        "n_neighbors": best.get("n_neighbors"),
                        "threshold_policy": ThresholdPolicy.OUTER_TRAIN_YOUDEN,
                        "outer_threshold": threshold,
                        "test_fails": int(np.sum(y_outer_test == 1)),
                        "BER": metrics["BER"],
                        "True+": metrics["True+"],
                        "True-": metrics["True-"],
                    }
                )

                for feat in feature_universe:
                    feature_stability_rows.append(
                        {
                            "selector": selector,
                            "seed": seed,
                            "outer_fold": fold.outer_fold,
                            "feature_index": feat.feature_index,
                            "feature_type": feat.feature_type,
                            "selected": 1 if feat.feature_index in selected_global else 0,
                        }
                    )

    splitwise_df = pd.DataFrame(splitwise_rows)
    stage_b_inner_df = pd.DataFrame(stage_b_inner_rows)
    feature_stability_df = pd.DataFrame(feature_stability_rows)
    write_csv(splitwise_df, reports / ArtifactName.SPLITWISE)
    write_csv(stage_b_inner_df, reports / ArtifactName.STAGE_B_INNER)
    write_csv(feature_stability_df, reports / ArtifactName.FEATURE_STABILITY)

    selector_stats = []
    deciding_outer_fold = max(f.outer_fold for f in bundle.fold_plan.folds)
    for selector, grp in splitwise_df.groupby("selector"):
        mu_f = grp.groupby("outer_fold")["BER"].mean()
        deciding_vote = grp[
            (grp["seed"] == 42) & (grp["outer_fold"] == deciding_outer_fold)
        ]
        vote_ber = float(deciding_vote["BER"].iloc[0]) if not deciding_vote.empty else np.inf
        vote_true_pos = float(deciding_vote["True+"].iloc[0]) if not deciding_vote.empty else -np.inf
        selector_stats.append(
            {
                "selector": selector,
                "n_folds": int(grp["outer_fold"].nunique()),
                "n_seeds": int(grp["seed"].nunique()),
                "mean_BER": float(grp["BER"].mean()),
                "std_BER": safe_std(grp["BER"].to_numpy()),
                "std_per_fold_BER_means": safe_std(mu_f.to_numpy()),
                "mean_True+": float(grp["True+"].mean()),
                "std_True+": safe_std(grp["True+"].to_numpy()),
                "mean_True-": float(grp["True-"].mean()),
                "std_True-": safe_std(grp["True-"].to_numpy()),
                "modal_k": int(grp["k"].mode().min()),
                "modal_C": float(grp["C"].mode().min()),
                "modal_scaler": (
                    ScalerName.STANDARD
                    if (grp["scaler"] == ScalerName.STANDARD).sum()
                    >= (grp["scaler"] == ScalerName.ROBUST).sum()
                    else ScalerName.ROBUST
                ),
                "modal_n_neighbors": (
                    float(grp["n_neighbors"].dropna().mode().min())
                    if selector == SelectorName.RELIEFF and grp["n_neighbors"].notna().any()
                    else np.nan
                ),
                "vote_outer_fold": deciding_outer_fold,
                "vote_seed": 42,
                "vote_outer_BER": vote_ber,
                "vote_outer_True+": vote_true_pos,
            }
        )

    model_sel = pd.DataFrame(selector_stats)

    def _selector_rank_key(row):
        scaler_pref = 0 if row["modal_scaler"] == ScalerName.STANDARD else 1
        nn = row["modal_n_neighbors"]
        nn_key = math.inf if pd.isna(nn) else float(nn)
        return (
            row["mean_BER"],
            row["std_per_fold_BER_means"],
            -row["mean_True+"],
            row["modal_k"],
            row["modal_C"],
            scaler_pref,
            nn_key,
            row["vote_outer_BER"],
            -row["vote_outer_True+"],
            row["selector"],
        )

    ranked = sorted(model_sel.to_dict("records"), key=_selector_rank_key)
    primary = ranked[0]["selector"]
    eligible = [r for r in ranked[1:] if r["mean_BER"] <= 0.40]
    challenger_available = len(eligible) > 0
    challenger = None
    if challenger_available:
        eligible_sorted = sorted(
            eligible,
            key=lambda r: (
                -r["mean_True-"],
                r["mean_BER"],
                r["std_per_fold_BER_means"],
                r["selector"],
            ),
        )
        challenger = eligible_sorted[0]["selector"]

    model_sel["is_primary"] = model_sel["selector"] == primary
    model_sel["is_challenger"] = (
        (model_sel["selector"] == challenger) if challenger is not None else False
    )
    model_sel_out = model_sel[
        [
            "selector",
            "n_folds",
            "n_seeds",
            "mean_BER",
            "std_BER",
            "std_per_fold_BER_means",
            "mean_True+",
            "std_True+",
            "mean_True-",
            "std_True-",
            "is_primary",
            "is_challenger",
        ]
    ].copy()
    write_csv(model_sel_out, reports / ArtifactName.MODEL_SELECTION)

    seed_rows = []
    for (selector, seed), grp in splitwise_df.groupby(["selector", "seed"]):
        seed_rows.append(
            {
                "selector": selector,
                "seed": seed,
                "mean_outer_BER": float(grp["BER"].mean()),
                "std_outer_BER": safe_std(grp["BER"].to_numpy()),
                "mean_outer_True+": float(grp["True+"].mean()),
                "mean_outer_True-": float(grp["True-"].mean()),
                "modal_k": int(grp["k"].mode().min()),
                "modal_C": float(grp["C"].mode().min()),
                "modal_scaler": (
                    ScalerName.STANDARD
                    if (grp["scaler"] == ScalerName.STANDARD).sum()
                    >= (grp["scaler"] == ScalerName.ROBUST).sum()
                    else ScalerName.ROBUST
                ),
                "modal_n_neighbors": (
                    float(grp["n_neighbors"].dropna().mode().min())
                    if selector == SelectorName.RELIEFF and grp["n_neighbors"].notna().any()
                    else np.nan
                ),
                "n_outer_folds": int(grp["outer_fold"].nunique()),
            }
        )
    write_csv(pd.DataFrame(seed_rows), reports / ArtifactName.SEED_STABILITY)

    return {
        "lane_b_feasible": True,
        "primary_selector": primary,
        "challenger_selector": challenger,
        "challenger_available": challenger_available,
        "splitwise": splitwise_df,
        "model_selection": model_sel_out,
    }


def _phase2_freeze_for_role(
    role: str,
    selector: str,
    x_dev: np.ndarray,
    y_dev: np.ndarray,
) -> tuple[pd.DataFrame, RoleConfig]:
    configs = _stage_b_configs(selector)
    per_config: dict[tuple, list[dict[str, Any]]] = {}

    for cfg in configs:
        key = (cfg["k"], float(cfg["C"]), cfg["scaler"], cfg.get("n_neighbors"))
        per_config[key] = []
        for seed in SEEDS_PHASE2:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for inner_fold_i, (tr, va) in enumerate(skf.split(x_dev, y_dev), start=1):
                x_tr = x_dev[tr]
                y_tr = y_dev[tr]
                x_va = x_dev[va]
                y_va = y_dev[va]

                x_train_sel, x_val_sel, _meta, _sel, _imp, _scaler = _fit_selector_pipeline(
                    x_train_raw=x_tr,
                    y_train=y_tr,
                    x_eval_raw=x_va,
                    method=selector,
                    k=int(cfg["k"]),
                    scaler_name=cfg["scaler"],
                    add_indicator=True,
                    n_neighbors=cfg.get("n_neighbors"),
                )
                clf = fit_lane_b_classifier(x_train_sel, y_tr, c_value=float(cfg["C"]))
                tr_scores = clf.predict_proba(x_train_sel)[:, 1]
                va_scores = clf.predict_proba(x_val_sel)[:, 1]
                threshold, _ = find_ber_optimal_threshold(y_tr, tr_scores)
                m = binary_metrics_at_threshold(y_va, va_scores, threshold)
                per_config[key].append(
                    {
                        "role": role,
                        "selector": selector,
                        "k": int(cfg["k"]),
                        "C": float(cfg["C"]),
                        "scaler": cfg["scaler"],
                        "n_neighbors": cfg.get("n_neighbors"),
                        "seed": seed,
                        "inner_fold": inner_fold_i,
                        "inner_ROC_AUC": float(m["ROC_AUC"]) if np.isfinite(m["ROC_AUC"]) else 0.5,
                        "inner_BER": float(m["BER"]),
                    }
                )

    config_rows = []
    for (k, c, scaler, nn), items in per_config.items():
        config_rows.append(
            {
                "k": k,
                "C": c,
                "scaler": scaler,
                "n_neighbors": nn,
                "mean_inner_ROC_AUC": float(np.mean([r["inner_ROC_AUC"] for r in items])),
                "mean_inner_BER": float(np.mean([r["inner_BER"] for r in items])),
            }
        )
    best = _select_best_inner_config(config_rows)
    best_key = (best["k"], float(best["C"]), best["scaler"], best.get("n_neighbors"))

    all_rows = []
    for key, items in per_config.items():
        mean_auc = float(np.mean([r["inner_ROC_AUC"] for r in items]))
        mean_ber = float(np.mean([r["inner_BER"] for r in items]))
        is_best = key == best_key
        for r in items:
            out = dict(r)
            out["mean_inner_ROC_AUC_by_config"] = mean_auc
            out["mean_inner_BER_by_config"] = mean_ber
            out["is_frozen_config"] = is_best
            all_rows.append(out)

    freeze_df = pd.DataFrame(all_rows)
    role_cfg = RoleConfig(
        role=role,
        selector=selector,
        k=int(best["k"]),
        c_value=float(best["C"]),
        scaler=str(best["scaler"]),
        n_neighbors=None if best.get("n_neighbors") is None else int(best["n_neighbors"]),
    )
    return freeze_df, role_cfg


def _fit_phase3_role_model(
    role_cfg: RoleConfig,
    x_dev_raw: np.ndarray,
    y_dev: np.ndarray,
    week_labels: np.ndarray,
    raw_feature_count: int,
) -> FittedRoleModel:
    imputer = make_imputer(add_indicator=True)
    x_dev_imp = imputer.fit_transform(x_dev_raw)
    scaler = make_scaler(role_cfg.scaler)
    x_dev_scaled = scaler.fit_transform(x_dev_imp)
    selected_local, _scores = _selector_pick(
        method=role_cfg.selector,
        x_train=x_dev_scaled,
        y_train=y_dev,
        k=role_cfg.k,
        n_neighbors=role_cfg.n_neighbors,
    )
    meta = transformed_feature_metadata_from_imputer(imputer, raw_feature_count=raw_feature_count)
    selected_global = local_to_global_feature_indices(selected_local, meta)
    x_dev_sel = x_dev_scaled[:, selected_local]

    clf = fit_lane_b_classifier(x_dev_sel, y_dev, c_value=role_cfg.c_value)
    dev_scores = clf.predict_proba(x_dev_sel)[:, 1]
    sci_threshold, _ = find_ber_optimal_threshold(y_dev, dev_scores)
    op_threshold = _operational_threshold(dev_scores, y_dev, week_labels=week_labels)
    t90, tnr90, tpr90 = extract_tpr_at_tnr(y_dev, dev_scores, target_tnr=0.90)

    return FittedRoleModel(
        config=role_cfg,
        imputer=imputer,
        scaler=scaler,
        selected_local_idx=selected_local,
        selected_global_idx=selected_global,
        clf=clf,
        dev_scores=dev_scores,
        scientific_threshold=float(sci_threshold),
        operational_threshold=float(op_threshold),
        threshold_at_tnr90_dev=float(t90),
        tnr_at_tnr90_dev=float(tnr90),
        tpr_at_tnr90_dev=float(tpr90),
        feature_meta=meta,
    )


def _score_lockbox_for_role(model: FittedRoleModel, x_lock_raw: np.ndarray, y_lock: np.ndarray) -> pd.DataFrame:
    x_lock_imp = model.imputer.transform(x_lock_raw)
    x_lock_scaled = model.scaler.transform(x_lock_imp)
    x_lock_sel = x_lock_scaled[:, model.selected_local_idx]
    lock_scores = model.clf.predict_proba(x_lock_sel)[:, 1]
    t90, tnr90, tpr90 = extract_tpr_at_tnr(y_lock, lock_scores, target_tnr=0.90)

    rows = []
    for policy, th in [
        (ThresholdPolicy.SCIENTIFIC, model.scientific_threshold),
        (ThresholdPolicy.OPERATIONAL, model.operational_threshold),
    ]:
        m = binary_metrics_at_threshold(y_lock, lock_scores, th)
        rows.append(
            {
                "role": model.config.role,
                "selector": model.config.selector,
                "threshold_policy": policy,
                "threshold_value": float(th),
                "BER": m["BER"],
                "True+": m["True+"],
                "True-": m["True-"],
                "ROC_AUC": m["ROC_AUC"],
                "PR_AUC": m["PR_AUC"],
                "MCC": m["MCC"],
                "F2": m["F2"],
                "lockbox_n": int(m["lockbox_n"]),
                "lockbox_fails": int(m["lockbox_fails"]),
                "threshold_at_TNR90": float(t90),
                "TNR_at_TNR90": float(tnr90),
                "TPR_at_TNR90": float(tpr90),
                "FP": int(m["FP"]),
                "FN": int(m["FN"]),
            }
        )
    return pd.DataFrame(rows)


def _psi_for_feature(dev_vals: np.ndarray, lock_vals: np.ndarray) -> float:
    dev = np.asarray(dev_vals, dtype=float)
    lock = np.asarray(lock_vals, dtype=float)
    dev_nm = dev[~np.isnan(dev)]
    if dev_nm.size > 0:
        q = np.quantile(dev_nm, np.arange(0.1, 1.0, 0.1))
        edges = np.unique(np.asarray(q, dtype=float))
    else:
        edges = np.array([], dtype=float)

    def bin_index(val: float) -> int:
        if np.isnan(val):
            return len(edges) + 1
        if edges.size == 0:
            return 0
        for i, e in enumerate(edges):
            if val <= e:
                return i
        return len(edges)

    n_bins = (len(edges) + 1) + 1
    dev_counts = np.zeros(n_bins, dtype=float)
    lock_counts = np.zeros(n_bins, dtype=float)
    for v in dev:
        dev_counts[bin_index(float(v))] += 1
    for v in lock:
        lock_counts[bin_index(float(v))] += 1

    p = dev_counts / max(dev.shape[0], 1)
    q = lock_counts / max(lock.shape[0], 1)
    psi = np.sum((p - q) * np.log((p + EPS_PSI) / (q + EPS_PSI)))
    return float(psi)


def _drift_gate_for_role(
    model: FittedRoleModel,
    x_dev_raw: np.ndarray,
    y_dev: np.ndarray,
    x_lock_raw: np.ndarray,
    y_lock: np.ndarray,
) -> dict[str, Any]:
    x_lock_imp = model.imputer.transform(x_lock_raw)
    x_lock_scaled = model.scaler.transform(x_lock_imp)
    x_lock_sel = x_lock_scaled[:, model.selected_local_idx]
    lock_scores = model.clf.predict_proba(x_lock_sel)[:, 1]

    dev_fail_rate = float(np.mean(y_dev == 1))
    lock_fail_rate = float(np.mean(y_lock == 1))
    abs_prev = abs(lock_fail_rate - dev_fail_rate)

    ks_p = float(ks_2samp(model.dev_scores, lock_scores, alternative="two-sided", mode="auto").pvalue)

    coef_abs = np.abs(model.clf.coef_[0])
    sel_meta = [model.feature_meta[int(i)] for i in model.selected_local_idx.tolist()]
    candidates = [
        (coef_abs[i], meta.raw_index)
        for i, meta in enumerate(sel_meta)
        if meta.feature_type == "value"
    ]
    candidates = sorted(candidates, key=lambda x: (-x[0], x[1]))
    top_raw_idx = [raw_idx for _, raw_idx in candidates[:PSI_MAX_FEATURES]]

    psi_vals = [_psi_for_feature(x_dev_raw[:, idx], x_lock_raw[:, idx]) for idx in top_raw_idx]
    max_psi = 0.0 if not psi_vals else float(np.max(psi_vals))
    med_psi = 0.0 if not psi_vals else float(np.median(psi_vals))

    violated = 0
    if abs_prev >= 0.02:
        violated += 1
    if ks_p < 0.01:
        violated += 1
    if max_psi >= 0.30:
        violated += 1
    status = "PASS" if violated == 0 else ("CAUTION" if violated == 1 else "HIGH_SHIFT")

    return {
        "model_scope": ModelScope.PRIMARY_FROZEN
        if model.config.role == "primary"
        else ModelScope.CHALLENGER_FROZEN,
        "dev_fail_rate": dev_fail_rate,
        "lockbox_fail_rate": lock_fail_rate,
        "abs_prevalence_shift": abs_prev,
        "ks_pvalue_scores": ks_p,
        "max_PSI": max_psi,
        "median_PSI": med_psi,
        "psi_feature_count": int(len(top_raw_idx)),
        "drift_gate_status": status,
        "lockbox_claims_allowed": status in {"PASS", "CAUTION"},
    }


def _mspc_fit_and_score(
    x_train_pass: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
) -> dict[str, Any]:
    imputer = SimpleImputer(strategy="median", keep_empty_features=True, add_indicator=False)
    scaler = StandardScaler()
    x_train_imp = imputer.fit_transform(x_train_pass)
    x_train_s = scaler.fit_transform(x_train_imp)
    n_comp = max(1, min(10, x_train_s.shape[1], x_train_s.shape[0] - 1))
    pca = PCA(n_components=n_comp, random_state=42)
    t_train = pca.fit_transform(x_train_s)
    xhat_train = pca.inverse_transform(t_train)
    q_train = np.sum((x_train_s - xhat_train) ** 2, axis=1)
    ev = pca.explained_variance_
    t2_train = np.sum((t_train**2) / (ev + 1e-12), axis=1)
    ucl_t2 = float(np.quantile(t2_train, 0.99))
    ucl_q = float(np.quantile(q_train, 0.99))

    x_eval_s = scaler.transform(imputer.transform(x_eval))
    t_eval = pca.transform(x_eval_s)
    xhat_eval = pca.inverse_transform(t_eval)
    q_eval = np.sum((x_eval_s - xhat_eval) ** 2, axis=1)
    t2_eval = np.sum((t_eval**2) / (ev + 1e-12), axis=1)

    t2_thr, _, t2_tpr90 = extract_tpr_at_tnr(y_eval, t2_eval, target_tnr=0.90)
    q_thr, _, q_tpr90 = extract_tpr_at_tnr(y_eval, q_eval, target_tnr=0.90)

    alarm = ((t2_eval > ucl_t2) | (q_eval > ucl_q)).astype(int)
    alarm_rate = float(np.mean(alarm))
    alarm_positions = np.where(alarm == 1)[0]
    if alarm_positions.size < 2:
        arl0 = np.nan
    else:
        arl0 = float(np.mean(np.diff(alarm_positions)))

    best_tpr = max(float(t2_tpr90), float(q_tpr90))
    best_src = "T2" if np.isclose(best_tpr, float(t2_tpr90)) else "Q"

    from sklearn.metrics import roc_auc_score

    if np.unique(y_eval).size == 2:
        t2_auc = float(roc_auc_score(y_eval, t2_eval))
        q_auc = float(roc_auc_score(y_eval, q_eval))
    else:
        t2_auc = np.nan
        q_auc = np.nan

    _ = t2_thr, q_thr
    return {
        "T2_AUC": t2_auc,
        "Q_AUC": q_auc,
        "alarm_rate": alarm_rate,
        "empirical_ARL0": arl0,
        "T2_TPR_at_TNR90": float(t2_tpr90),
        "Q_TPR_at_TNR90": float(q_tpr90),
        "best_MSPC_TPR_at_TNR90": float(best_tpr),
        "best_MSPC_source": best_src,
    }


def run_04_phase2_phase3_freeze_lockbox(
    bundle: DataBundle,
    stage3: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    reports = ensure_reports_dir(output_dir)
    if not bundle.lane_b_feasible or bundle.fold_plan is None:
        return {"lane_b_feasible": False}

    primary_selector = stage3["primary_selector"]
    challenger_selector = stage3.get("challenger_selector")
    challenger_available = bool(stage3.get("challenger_available", False))

    x_dev = bundle.dev_with_weeks[bundle.feature_columns].to_numpy(dtype=float)
    y_dev = bundle.dev_with_weeks["y_bin"].to_numpy(dtype=int)
    week_dev = bundle.dev_with_weeks["week_label"].to_numpy(dtype=int)
    x_lock = bundle.lockbox[bundle.feature_columns].to_numpy(dtype=float)
    y_lock = bundle.lockbox["y_bin"].to_numpy(dtype=int)

    freeze_frames = []
    frozen_roles: list[RoleConfig] = []
    f_primary_df, cfg_primary = _phase2_freeze_for_role("primary", primary_selector, x_dev, y_dev)
    freeze_frames.append(f_primary_df)
    frozen_roles.append(cfg_primary)

    if challenger_available and challenger_selector is not None:
        f_ch_df, cfg_ch = _phase2_freeze_for_role(
            "challenger", challenger_selector, x_dev, y_dev
        )
        freeze_frames.append(f_ch_df)
        frozen_roles.append(cfg_ch)

    freeze_df = pd.concat(freeze_frames, ignore_index=True)
    write_csv(freeze_df, reports / ArtifactName.FREEZE)

    fitted_models = [
        _fit_phase3_role_model(
            role_cfg=cfg,
            x_dev_raw=x_dev,
            y_dev=y_dev,
            week_labels=week_dev,
            raw_feature_count=len(bundle.feature_columns),
        )
        for cfg in frozen_roles
    ]

    lock_rows = []
    drift_rows = []
    for fitted in fitted_models:
        lock_rows.append(_score_lockbox_for_role(fitted, x_lock_raw=x_lock, y_lock=y_lock))
        drift_rows.append(
            _drift_gate_for_role(
                model=fitted,
                x_dev_raw=x_dev,
                y_dev=y_dev,
                x_lock_raw=x_lock,
                y_lock=y_lock,
            )
        )

    final_lock_df = pd.concat(lock_rows, ignore_index=True)
    write_csv(final_lock_df, reports / ArtifactName.FINAL_LOCKBOX)
    drift_df = pd.DataFrame(drift_rows)
    write_csv(drift_df, reports / ArtifactName.DRIFT_GATE)

    mspc_rows = []
    for fold in bundle.fold_plan.folds:
        tr = fold.train_index
        te = fold.test_index
        x_train_pass = x_dev[tr][y_dev[tr] == 0]
        m = _mspc_fit_and_score(x_train_pass=x_train_pass, x_eval=x_dev[te], y_eval=y_dev[te])
        mspc_rows.append({"fold_index": str(fold.outer_fold), "eval_scope": "outer_fold", **m})
    x_dev_pass = x_dev[y_dev == 0]
    m = _mspc_fit_and_score(x_train_pass=x_dev_pass, x_eval=x_lock, y_eval=y_lock)
    mspc_rows.append({"fold_index": "LOCKBOX", "eval_scope": "lockbox", **m})
    mspc_df = pd.DataFrame(mspc_rows)
    write_csv(mspc_df, reports / ArtifactName.MSPC)

    cost_rows = []
    for ratio in COST_RATIOS:
        row: dict[str, Any] = {"cost_ratio": ratio}
        for role in ["primary", "challenger"]:
            for policy in [ThresholdPolicy.SCIENTIFIC, ThresholdPolicy.OPERATIONAL]:
                key = f"{role}_{policy}"
                sub = final_lock_df[
                    (final_lock_df["role"] == role)
                    & (final_lock_df["threshold_policy"] == policy)
                ]
                if sub.empty:
                    row[key] = np.nan
                else:
                    rr = sub.iloc[0]
                    row[key] = expected_cost_per_wafer(
                        fp=float(rr["FP"]),
                        fn=float(rr["FN"]),
                        n=float(rr["lockbox_n"]),
                        cost_ratio=ratio,
                    )
        n = len(y_lock)
        fails = int(np.sum(y_lock == 1))
        row["all_pass_baseline"] = float((ratio * fails) / max(n, 1))
        row["all_flag_baseline"] = float((n - fails) / max(n, 1))
        cost_rows.append(row)
    write_csv(pd.DataFrame(cost_rows), reports / ArtifactName.COST_CURVES)

    primary_model = [m for m in fitted_models if m.config.role == "primary"][0]
    coefs = np.abs(primary_model.clf.coef_[0])
    feature_stability = pd.read_csv(reports / ArtifactName.FEATURE_STABILITY)
    fs_primary = feature_stability[
        feature_stability["selector"] == primary_model.config.selector
    ].copy()
    selected_map = fs_primary.groupby("feature_index")["selected"].mean().to_dict()

    tuples = fs_primary[["outer_fold", "seed"]].drop_duplicates()
    m_tuples = len(tuples)
    pair_count = max((m_tuples * (m_tuples - 1)) // 2, 1)
    stability_map: dict[int, float] = {}
    for feat_idx, grp in fs_primary.groupby("feature_index"):
        vals = grp.sort_values(["outer_fold", "seed"])["selected"].to_numpy(dtype=int)
        c = 0
        for a, b in combinations(range(vals.size), 2):
            if vals[a] == 1 and vals[b] == 1:
                c += 1
        stability_map[int(feat_idx)] = c / pair_count

    x_dev_imp = primary_model.imputer.transform(x_dev)
    value_x = x_dev_imp[:, : len(bundle.feature_columns)]
    corr = np.corrcoef(value_x, rowvar=False)
    p = corr.shape[0]
    adj = {i: set() for i in range(p)}
    for i in range(p):
        for j in range(i + 1, p):
            cij = corr[i, j]
            if np.isfinite(cij) and abs(cij) >= 0.95:
                adj[i].add(j)
                adj[j].add(i)
    cluster_id: dict[int, int] = {}
    cid = 0
    seen = set()
    for i in range(p):
        if i in seen:
            continue
        stack = [i]
        seen.add(i)
        while stack:
            v = stack.pop()
            cluster_id[v] = cid
            for nb in adj[v]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        cid += 1

    feature_rows = []
    sel_meta = [primary_model.feature_meta[int(i)] for i in primary_model.selected_local_idx.tolist()]
    for i, meta in enumerate(sel_meta):
        idx = int(meta.feature_index)
        sf = float(selected_map.get(idx, 0.0))
        cmag = float(coefs[i])
        cid_val = cluster_id.get(meta.raw_index, np.nan) if meta.feature_type == "value" else np.nan
        feature_rows.append(
            {
                "feature_index": idx,
                "feature_type": meta.feature_type,
                "feature_name_or_source_col": meta.feature_name_or_source_col,
                "selection_frequency": sf,
                "conditional_effect_magnitude": cmag,
                "expected_contribution": sf * cmag,
                "fold_jaccard_stability": float(stability_map.get(idx, 0.0)),
                "cluster_id": cid_val,
            }
        )
    write_csv(pd.DataFrame(feature_rows), reports / ArtifactName.FEATURE_REPORT)

    manifest_path = reports / ArtifactName.MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["challenger_available"] = challenger_available
    manifest["challenger_unavailable_reason"] = (
        None if challenger_available else "no_eligible_method_under_BER_0.40"
    )
    manifest["frozen_primary"] = {
        **cfg_primary.to_hash_payload(),
        "config_hash": config_hash(cfg_primary.to_hash_payload()),
    }
    if challenger_available and challenger_selector is not None:
        cfg_ch = [c for c in frozen_roles if c.role == "challenger"][0]
        manifest["frozen_challenger"] = {
            **cfg_ch.to_hash_payload(),
            "config_hash": config_hash(cfg_ch.to_hash_payload()),
        }
    else:
        manifest["frozen_challenger"] = None
    manifest["frozen_thresholds"] = {
        m.config.role: {
            "scientific": m.scientific_threshold,
            "operational": m.operational_threshold,
        }
        for m in fitted_models
    }
    manifest["drift_gate_results"] = {
        r["model_scope"]: {
            "drift_gate_status": r["drift_gate_status"],
            "lockbox_claims_allowed": r["lockbox_claims_allowed"],
        }
        for r in drift_rows
    }
    if mspc_df["empirical_ARL0"].isna().any():
        manifest["empirical_ARL0_nan_reason"] = "fewer_than_two_alarms_in_evaluated_sequence"
    write_manifest(manifest, manifest_path)
    return {"lane_b_feasible": True, "challenger_available": challenger_available}


def run_05_artifact_and_claim_audit(output_dir: Path) -> ValidationResult:
    reports = output_dir / "reports"
    manifest = json.loads((reports / ArtifactName.MANIFEST).read_text(encoding="utf-8"))
    lane_b_feasible = bool(manifest.get("lane_b_feasible", False))

    errors = []
    errors.extend(validate_required_artifacts(output_dir=output_dir, lane_b_feasible=lane_b_feasible))
    schema = validate_schema_and_logic(output_dir=output_dir)
    errors.extend(schema.errors)

    strict_path = reports / ArtifactName.BASELINE_STRICT
    mi_path = reports / ArtifactName.BASELINE_MI
    ablation_path = reports / ArtifactName.BASELINE_ABLATION
    summary_path = reports / ArtifactName.BASELINE_SUMMARY
    tuning_trace_path = reports / ArtifactName.BASELINE_TUNING_TRACE
    if (
        strict_path.exists()
        and mi_path.exists()
        and ablation_path.exists()
        and summary_path.exists()
        and tuning_trace_path.exists()
    ):
        strict_df = pd.read_csv(strict_path)
        mi_df = pd.read_csv(mi_path)
        ablation_df = pd.read_csv(ablation_path)
        summary_df = pd.read_csv(summary_path)
        tuning_trace_df = pd.read_csv(tuning_trace_path)
        classifiers_run = (
            sorted(summary_df["classifier"].dropna().astype(str).unique().tolist())
            if "classifier" in summary_df.columns
            else []
        )
        try:
            validate_lane_a_artifacts(
                summary_df=summary_df,
                ablation_df=ablation_df,
                strict_df=strict_df,
                mi_df=mi_df,
                tuning_trace_df=tuning_trace_df,
                classifiers_run=classifiers_run,
            )
        except ValueError as exc:
            errors.append(str(exc))

        if LaneAClassifier.KRR_STRICT in classifiers_run:
            f_strict = summary_df[
                (summary_df["classifier"] == LaneAClassifier.KRR_STRICT)
                & (summary_df["selector"] == SelectorName.F_TEST)
                & (summary_df["replication_mode"] == ReplicationMode.STRICT)
            ]
            if len(f_strict) != 1:
                errors.append(
                    "benchmark claim gate requires exactly one row for "
                    "classifier=krr_strict, selector=F-test, replication_mode=strict"
                )

    if lane_b_feasible and (reports / ArtifactName.FINAL_LOCKBOX).exists():
        lock = pd.read_csv(reports / ArtifactName.FINAL_LOCKBOX)
        mspc = pd.read_csv(reports / ArtifactName.MSPC)
        drift = pd.read_csv(reports / ArtifactName.DRIFT_GATE)
        mspc_lock = mspc[mspc["eval_scope"] == "lockbox"]
        if mspc_lock.empty:
            errors.append("mspc lockbox row missing for claim gate")
        else:
            mspc_tpr = float(mspc_lock.iloc[0]["best_MSPC_TPR_at_TNR90"])
            for role in lock["role"].unique():
                row = lock[
                    (lock["role"] == role)
                    & (lock["threshold_policy"] == ThresholdPolicy.SCIENTIFIC)
                ]
                if row.empty:
                    continue
                sup_tpr = float(row.iloc[0]["TPR_at_TNR90"])
                scope = ModelScope.PRIMARY_FROZEN if role == "primary" else ModelScope.CHALLENGER_FROZEN
                drift_row = drift[drift["model_scope"] == scope]
                if drift_row.empty:
                    errors.append(f"drift gate row missing for role={role}")
                else:
                    status = str(drift_row.iloc[0]["drift_gate_status"])
                    if status == "HIGH_SHIFT" and sup_tpr > mspc_tpr:
                        errors.append(
                            f"invalid claim condition: role={role} better than MSPC but HIGH_SHIFT"
                        )

    return ValidationResult(ok=len(errors) == 0, errors=errors)
