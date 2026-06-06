"""Temporal robustness workflow with screening, freeze, lockbox, drift, and MSPC artifacts."""

from __future__ import annotations

from contextlib import suppress
import math
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from secom.artifacts import ensure_reports_dir, write_csv
from secom.common.drift import psi_for_feature
from secom.common.paths import project_root_from_repo_structure
from secom.common.thresholds import operational_threshold
from secom.config import (
    ArtifactName,
    COST_RATIOS,
    ModelScope,
    PSI_MAX_FEATURES,
    SEEDS_PHASE2,
    SEEDS_STAGE_B,
    ScalerName,
    SelectorName,
    StudyStatus,
    ThresholdPolicy,
    validate_selector_name,
)
from secom.cv import (
    add_dev_week_bins,
    choose_outer_fold_plan,
    split_dev_lockbox,
    temporal_feasibility_gate,
    to_time_window_string,
)
from secom.io import load_raw_secom, parse_sort_and_label
from secom.metrics import (
    binary_metrics_at_threshold,
    core_binary_metrics_at_threshold,
    expected_cost_per_wafer,
    extract_tpr_at_tnr,
    find_ber_optimal_threshold,
    predict_from_threshold,
    roc_auc_or_default,
    safe_std,
)
from secom.models import fit_temporal_logreg_model
from secom.preprocess import make_imputer, make_scaler, transformed_feature_metadata_from_imputer
from secom.selection.engine import fit_selector_pipeline, select_features
from secom.selection.tuning import select_best_inner_config
from secom.types import DataBundle, FittedRoleModel, RoleConfig
from secom.workflows.manifest import write_temporal_failure, write_temporal_status

STAGE_A_FEATURE_BUDGET = 40
STAGE_B_K_VALUES = [10, 20, 40]
STAGE_B_C_VALUES = [0.01, 0.1, 1.0, 10.0]
STAGE_B_SCALERS = [ScalerName.STANDARD, ScalerName.ROBUST]
RELIEFF_NEIGHBOR_VALUES = [5, 10, 20]
INNER_CV_SPLITS = 5
PREVALENCE_SHIFT_CAUTION = 0.02
SCORE_KS_PVALUE_CAUTION = 0.01
MAX_PSI_CAUTION = 0.30
MSPC_QUANTILE = 0.99
THRESHOLD_POLICIES = (ThresholdPolicy.SCIENTIFIC, ThresholdPolicy.OPERATIONAL)
TEMPORAL_CHALLENGER_MAX_BER = 0.40


def _build_bundle(input_dir: Path) -> DataBundle:
    """Load SECOM data, create the chronological split, and assess temporal feasibility."""
    loaded = load_raw_secom(input_dir)
    all_sorted = parse_sort_and_label(loaded.frame)
    split = split_dev_lockbox(all_sorted)
    dev_weeks = add_dev_week_bins(split.dev)
    plan = choose_outer_fold_plan(dev_weeks)
    feasible, reason = temporal_feasibility_gate(dev=dev_weeks, plan=plan)
    return DataBundle(
        all_data=all_sorted,
        dev=split.dev,
        lockbox=split.lockbox,
        feature_columns=loaded.feature_columns,
        dev_with_weeks=dev_weeks,
        fold_plan=plan,
        temporal_feasible=feasible,
        temporal_infeasible_reason=reason,
    )


def _fit_eval_with_labels(
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
    x_eval_raw: np.ndarray,
    y_eval: np.ndarray,
    method: str,
    k: int,
    c_value: float,
    scaler_name: str,
    n_neighbors: int | None,
) -> tuple[dict[str, float], float]:
    """Fit a temporal selector/logreg view and return eval metrics plus threshold."""
    prepared = _prepare_selector_eval_view(
        x_train_raw=x_train_raw,
        y_train=y_train,
        x_eval_raw=x_eval_raw,
        y_eval=y_eval,
        method=method,
        k=k,
        scaler_name=scaler_name,
        add_indicator=True,
        n_neighbors=n_neighbors,
    )
    metrics, threshold, _clf, _train_scores, _eval_scores = _score_temporal_logreg_view(
        prepared_view=prepared,
        c_value=c_value,
    )
    return metrics, threshold


def _prepare_selector_eval_view(
    *,
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
    x_eval_raw: np.ndarray,
    y_eval: np.ndarray,
    method: str,
    k: int,
    scaler_name: str,
    add_indicator: bool,
    n_neighbors: int | None,
) -> dict[str, Any]:
    """Prepare selected train/eval matrices while retaining transform metadata."""
    try:
        x_train_sel, x_eval_sel, feature_meta, selected_local, imputer, scaler = fit_selector_pipeline(
            x_train_raw=x_train_raw,
            y_train=y_train,
            x_eval_raw=x_eval_raw,
            method=method,
            k=k,
            scaler_name=scaler_name,
            add_indicator=add_indicator,
            n_neighbors=n_neighbors,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "temporal selector failure "
            f"selector={method} k={int(k)} scaler={scaler_name} "
            f"add_indicator={add_indicator} n_neighbors={n_neighbors}"
        ) from exc
    return {
        "x_train_sel": x_train_sel,
        "y_train": np.asarray(y_train, dtype=int),
        "x_eval_sel": x_eval_sel,
        "y_eval": np.asarray(y_eval, dtype=int),
        "feature_meta": feature_meta,
        "selected_local": selected_local,
        "imputer": imputer,
        "scaler": scaler,
    }


def _score_temporal_logreg_view(
    *,
    prepared_view: dict[str, Any],
    c_value: float,
) -> tuple[dict[str, float], float, Any, np.ndarray, np.ndarray]:
    """Fit temporal logistic regression and score eval data at a train-frozen threshold."""
    x_train_sel = prepared_view["x_train_sel"]
    y_train = prepared_view["y_train"]
    x_eval_sel = prepared_view["x_eval_sel"]
    y_eval = prepared_view["y_eval"]
    clf = fit_temporal_logreg_model(x_train_sel, y_train, c_value=c_value)
    train_scores = clf.predict_proba(x_train_sel)[:, 1]
    eval_scores = clf.predict_proba(x_eval_sel)[:, 1]
    threshold, _ = find_ber_optimal_threshold(y_train, train_scores)
    metrics = binary_metrics_at_threshold(y_eval, eval_scores, threshold)
    return metrics, float(threshold), clf, train_scores, eval_scores


def _prepare_resampled_selector_views(
    *,
    x_raw: np.ndarray,
    y: np.ndarray,
    splits_with_meta: list[tuple[dict[str, Any], np.ndarray, np.ndarray]],
    selector: str,
    k: int,
    scaler_name: str,
    add_indicator: bool,
    n_neighbors: int | None,
) -> list[dict[str, Any]]:
    """Prepare selector views for repeated CV splits with caller-provided metadata."""
    prepared_views: list[dict[str, Any]] = []
    for meta, train_idx, eval_idx in splits_with_meta:
        prepared = _prepare_selector_eval_view(
            x_train_raw=x_raw[train_idx],
            y_train=y[train_idx],
            x_eval_raw=x_raw[eval_idx],
            y_eval=y[eval_idx],
            method=selector,
            k=k,
            scaler_name=scaler_name,
            add_indicator=add_indicator,
            n_neighbors=n_neighbors,
        )
        prepared_views.append({**meta, **prepared})
    return prepared_views


def _stage_a_configs(selectors_run: list[str]) -> list[dict[str, Any]]:
    """Build the fixed selector-screening configs used before temporal model selection."""
    selectors = [validate_selector_name(s) for s in selectors_run]
    return [
        {
            "selector": s,
            "k": STAGE_A_FEATURE_BUDGET,
            "C": 1.0,
            "scaler": ScalerName.ROBUST,
            "n_neighbors": 10 if s == SelectorName.RELIEFF else None,
        }
        for s in selectors
    ]


def build_stage_b_config_grid(selector: str) -> list[dict[str, Any]]:
    """Return the temporal Stage-B grid for one selector."""
    selector = validate_selector_name(selector)
    if selector == SelectorName.RELIEFF:
        return [
            {
                "selector": selector,
                "k": k,
                "C": c,
                "scaler": scaler,
                "n_neighbors": nn,
            }
            for nn, k, c, scaler in product(
                RELIEFF_NEIGHBOR_VALUES,
                STAGE_B_K_VALUES,
                STAGE_B_C_VALUES,
                STAGE_B_SCALERS,
            )
        ]
    return [
        {
            "selector": selector,
            "k": k,
            "C": c,
            "scaler": scaler,
            "n_neighbors": None,
        }
        for k, c, scaler in product(STAGE_B_K_VALUES, STAGE_B_C_VALUES, STAGE_B_SCALERS)
    ]


def _group_stage_b_configs_by_preparation(
    configs: list[dict[str, Any]],
) -> dict[tuple[int, str, int | None], list[dict[str, Any]]]:
    """Group Stage-B configs that can reuse the same selector preparation."""
    grouped: dict[tuple[int, str, int | None], list[dict[str, Any]]] = {}
    for cfg in configs:
        key = (int(cfg["k"]), str(cfg["scaler"]), cfg.get("n_neighbors"))
        grouped.setdefault(key, []).append(cfg)
    return grouped


def _prepare_inner_cv_views(
    x_outer_train_raw: np.ndarray,
    y_outer_train: np.ndarray,
    selector: str,
    k: int,
    scaler_name: str,
    n_neighbors: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Prepare selected inner-CV folds for one outer temporal split and seed."""
    skf = StratifiedKFold(n_splits=INNER_CV_SPLITS, shuffle=True, random_state=seed)
    splits_with_meta = [
        ({}, inner_train_idx, inner_val_idx)
        for inner_train_idx, inner_val_idx in skf.split(x_outer_train_raw, y_outer_train)
    ]
    return _prepare_resampled_selector_views(
        x_raw=x_outer_train_raw,
        y=y_outer_train,
        splits_with_meta=splits_with_meta,
        selector=selector,
        k=k,
        scaler_name=scaler_name,
        add_indicator=True,
        n_neighbors=n_neighbors,
    )


def _score_prepared_inner_cv(
    prepared_views: list[dict[str, Any]],
    c_value: float,
) -> tuple[float, float]:
    """Return mean inner AUC and BER for one C value over prepared folds."""
    aucs: list[float] = []
    bers: list[float] = []
    for prepared in prepared_views:
        m, _threshold, _clf, _train_scores, _eval_scores = _score_temporal_logreg_view(
            prepared_view=prepared,
            c_value=c_value,
        )
        aucs.append(float(m["ROC_AUC"]) if np.isfinite(m["ROC_AUC"]) else 0.5)
        bers.append(float(m["BER"]))
    return float(np.mean(aucs)), float(np.mean(bers))


def _phase2_fold_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[float, float]:
    """Compute freeze-phase BER and AUC from frozen-threshold scores."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    metrics = core_binary_metrics_at_threshold(y_true=y_true, scores=scores, threshold=threshold)
    auc = roc_auc_or_default(y_true, scores)
    return float(metrics["BER"]), auc


def _prepare_phase2_inner_views(
    *,
    selector: str,
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    k: int,
    scaler_name: str,
    n_neighbors: int | None,
) -> list[dict[str, Any]]:
    """Prepare repeated Phase-2 inner views used to freeze role configs."""
    splits_with_meta: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []
    for seed in SEEDS_PHASE2:
        skf = StratifiedKFold(n_splits=INNER_CV_SPLITS, shuffle=True, random_state=seed)
        for inner_fold_i, (tr, va) in enumerate(skf.split(x_dev, y_dev), start=1):
            splits_with_meta.append(
                (
                    {"seed": seed, "inner_fold": inner_fold_i},
                    tr,
                    va,
                )
            )
    return _prepare_resampled_selector_views(
        x_raw=x_dev,
        y=y_dev,
        splits_with_meta=splits_with_meta,
        selector=selector,
        k=k,
        scaler_name=scaler_name,
        add_indicator=True,
        n_neighbors=n_neighbors,
    )


def _phase2_freeze_for_role(
    role: str,
    selector: str,
    x_dev: np.ndarray,
    y_dev: np.ndarray,
) -> tuple[pd.DataFrame, RoleConfig]:
    """Freeze one role's selector, feature budget, C, scaler, and ReliefF neighbors."""
    configs = build_stage_b_config_grid(selector)
    per_config: dict[tuple[int, float, str, int | None], list[dict[str, Any]]] = {}

    for (k, scaler_name, n_neighbors), prep_configs in _group_stage_b_configs_by_preparation(configs).items():
        prepared_views = _prepare_phase2_inner_views(
            selector=selector,
            x_dev=x_dev,
            y_dev=y_dev,
            k=k,
            scaler_name=scaler_name,
            n_neighbors=n_neighbors,
        )
        for cfg in prep_configs:
            key = (k, float(cfg["C"]), scaler_name, n_neighbors)
            items: list[dict[str, Any]] = []
            for prepared in prepared_views:
                _metrics, threshold, _clf, _tr_scores, va_scores = _score_temporal_logreg_view(
                    prepared_view=prepared,
                    c_value=float(cfg["C"]),
                )
                ber, auc = _phase2_fold_metrics(prepared["y_eval"], va_scores, threshold)
                items.append(
                    {
                        "role": role,
                        "selector": selector,
                        "k": k,
                        "C": float(cfg["C"]),
                        "scaler": scaler_name,
                        "n_neighbors": n_neighbors,
                        "resample_id": f"seed_{prepared['seed']}_fold_{prepared['inner_fold']}",
                        "mean_inner_ROC_AUC": auc,
                        "mean_inner_BER": ber,
                    }
                )
            per_config[key] = items

    config_rows: list[dict[str, Any]] = []
    for (k, c, scaler, nn), items in per_config.items():
        config_rows.append(
            {
                "k": k,
                "C": c,
                "scaler": scaler,
                "n_neighbors": nn,
                "mean_inner_ROC_AUC": float(np.mean([r["mean_inner_ROC_AUC"] for r in items])),
                "mean_inner_BER": float(np.mean([r["mean_inner_BER"] for r in items])),
            }
        )
    best = select_best_inner_config(config_rows)
    best_key = (best["k"], float(best["C"]), best["scaler"], best.get("n_neighbors"))

    rows: list[dict[str, Any]] = []
    for key, items in per_config.items():
        is_best = key == best_key
        for r in items:
            row = dict(r)
            row["is_frozen_config"] = is_best
            rows.append(row)

    freeze_df = pd.DataFrame(rows)
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
    """Fit the frozen role model on all DEV data and freeze scientific/operational thresholds."""
    imputer = make_imputer(add_indicator=True)
    x_dev_imp = imputer.fit_transform(x_dev_raw)
    scaler = make_scaler(role_cfg.scaler)
    x_dev_scaled = scaler.fit_transform(x_dev_imp)
    selected_local, _scores = select_features(
        method=role_cfg.selector,
        x_train=x_dev_scaled,
        y_train=y_dev,
        k=role_cfg.k,
        n_neighbors=role_cfg.n_neighbors,
    )
    meta = transformed_feature_metadata_from_imputer(imputer, raw_feature_count=raw_feature_count)
    if selected_local.size <= 0:
        raise RuntimeError(
            "temporal role selector failure "
            f"role={role_cfg.role} selector={role_cfg.selector} k={role_cfg.k} "
            f"scaler={role_cfg.scaler} n_neighbors={role_cfg.n_neighbors}"
        )
    selected_global = [meta[int(i)].feature_index for i in selected_local.tolist()]
    x_dev_sel = x_dev_scaled[:, selected_local]

    clf = fit_temporal_logreg_model(x_dev_sel, y_dev, c_value=role_cfg.c_value)
    dev_scores = clf.predict_proba(x_dev_sel)[:, 1]
    sci_threshold, _ = find_ber_optimal_threshold(y_dev, dev_scores)
    op_threshold = operational_threshold(dev_scores, y_dev, week_labels=week_labels)
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


def _prepare_lockbox_eval_context(
    *,
    model: FittedRoleModel,
    x_lock_raw: np.ndarray,
    y_lock: np.ndarray,
) -> dict[str, Any]:
    """Transform lockbox data through the fitted role model and compute TNR90 context."""
    x_lock_imp = model.imputer.transform(x_lock_raw)
    x_lock_scaled = model.scaler.transform(x_lock_imp)
    x_lock_sel = x_lock_scaled[:, model.selected_local_idx]
    lock_scores = model.clf.predict_proba(x_lock_sel)[:, 1]
    t90, tnr90, tpr90 = extract_tpr_at_tnr(y_lock, lock_scores, target_tnr=0.90)
    return {
        "lock_scores": lock_scores,
        "threshold_at_tnr90": float(t90),
        "tnr_at_tnr90": float(tnr90),
        "tpr_at_tnr90": float(tpr90),
    }


def _threshold_values_for_model(model: FittedRoleModel) -> tuple[tuple[str, float], ...]:
    """Return role thresholds in artifact column order."""
    return (
        (ThresholdPolicy.SCIENTIFIC, model.scientific_threshold),
        (ThresholdPolicy.OPERATIONAL, model.operational_threshold),
    )


def _score_lockbox_for_role(
    model: FittedRoleModel,
    y_lock: np.ndarray,
    lock_ctx: dict[str, Any],
) -> pd.DataFrame:
    """Score lockbox metrics for scientific and operational threshold policies."""
    lock_scores = np.asarray(lock_ctx["lock_scores"], dtype=float)
    rows = []
    for policy, th in _threshold_values_for_model(model):
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
                "threshold_at_TNR90": float(lock_ctx["threshold_at_tnr90"]),
                "TNR_at_TNR90": float(lock_ctx["tnr_at_tnr90"]),
                "TPR_at_TNR90": float(lock_ctx["tpr_at_tnr90"]),
                "FP": int(m["FP"]),
                "FN": int(m["FN"]),
            }
        )
    return pd.DataFrame(rows)


def _drift_gate_for_role(
    model: FittedRoleModel,
    x_dev_raw: np.ndarray,
    y_dev: np.ndarray,
    x_lock_raw: np.ndarray,
    y_lock: np.ndarray,
    lock_ctx: dict[str, Any],
) -> dict[str, Any]:
    """Apply prevalence, score-shift, and selected-feature PSI drift gates for one role."""
    lock_scores = np.asarray(lock_ctx["lock_scores"], dtype=float)
    dev_fail_rate = float(np.mean(y_dev == 1))
    lock_fail_rate = float(np.mean(y_lock == 1))
    abs_prev = abs(lock_fail_rate - dev_fail_rate)
    ks_p = float(ks_2samp(model.dev_scores, lock_scores, alternative="two-sided", mode="auto").pvalue)

    coef_abs = np.abs(model.clf.coef_[0])
    sel_meta = [model.feature_meta[int(i)] for i in model.selected_local_idx.tolist()]
    candidates = [(coef_abs[i], meta.raw_index) for i, meta in enumerate(sel_meta) if meta.feature_type == "value"]
    candidates = sorted(candidates, key=lambda x: (-x[0], x[1]))
    top_raw_idx = [raw_idx for _, raw_idx in candidates[:PSI_MAX_FEATURES]]
    psi_vals = [psi_for_feature(x_dev_raw[:, idx], x_lock_raw[:, idx]) for idx in top_raw_idx]
    max_psi = 0.0 if not psi_vals else float(np.max(psi_vals))
    med_psi = 0.0 if not psi_vals else float(np.median(psi_vals))

    violated = 0
    if abs_prev >= PREVALENCE_SHIFT_CAUTION:
        violated += 1
    if ks_p < SCORE_KS_PVALUE_CAUTION:
        violated += 1
    if max_psi >= MAX_PSI_CAUTION:
        violated += 1
    status = "PASS" if violated == 0 else ("CAUTION" if violated == 1 else "HIGH_SHIFT")
    return {
        "model_scope": ModelScope.PRIMARY if model.config.role == "primary" else ModelScope.CHALLENGER,
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
    """Fit a PCA MSPC baseline on pass wafers and score an eval window."""
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
    ucl_t2 = float(np.quantile(t2_train, MSPC_QUANTILE))
    ucl_q = float(np.quantile(q_train, MSPC_QUANTILE))

    x_eval_s = scaler.transform(imputer.transform(x_eval))
    t_eval = pca.transform(x_eval_s)
    xhat_eval = pca.inverse_transform(t_eval)
    q_eval = np.sum((x_eval_s - xhat_eval) ** 2, axis=1)
    t2_eval = np.sum((t_eval**2) / (ev + 1e-12), axis=1)
    _t2_thr, _, t2_tpr90 = extract_tpr_at_tnr(y_eval, t2_eval, target_tnr=0.90)
    _q_thr, _, q_tpr90 = extract_tpr_at_tnr(y_eval, q_eval, target_tnr=0.90)
    alarm = ((t2_eval > ucl_t2) | (q_eval > ucl_q)).astype(int)
    alarm_rate = float(np.mean(alarm))
    alarm_positions = np.where(alarm == 1)[0]
    arl0 = np.nan if alarm_positions.size < 2 else float(np.mean(np.diff(alarm_positions)))
    best_tpr = max(float(t2_tpr90), float(q_tpr90))
    best_src = "T2" if np.isclose(best_tpr, float(t2_tpr90)) else "Q"
    t2_auc = roc_auc_or_default(y_eval, t2_eval, default=np.nan)
    q_auc = roc_auc_or_default(y_eval, q_eval, default=np.nan)
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


def _manager_weekly_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    week_labels: np.ndarray,
) -> dict[str, float]:
    """Summarize weekly operations-facing flag and fail-capture rates."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    weeks = np.asarray(week_labels, dtype=int)
    preds = predict_from_threshold(scores, threshold)
    _, week_codes = np.unique(weeks, return_inverse=True)
    week_count = int(week_codes.max()) + 1 if week_codes.size else 0
    preds_float = preds.astype(float, copy=False)
    fail_mask = (y_true == 1).astype(float)
    flagged_counts = (
        np.bincount(week_codes, weights=preds_float, minlength=week_count) if week_count else np.array([], dtype=float)
    )
    tp_counts = (
        np.bincount(week_codes, weights=(preds_float * fail_mask), minlength=week_count)
        if week_count
        else np.array([], dtype=float)
    )
    fn_counts = (
        np.bincount(week_codes, weights=((1.0 - preds_float) * fail_mask), minlength=week_count)
        if week_count
        else np.array([], dtype=float)
    )
    sample_count = len(y_true)
    return {
        "predicted_flag_fraction": float(np.mean(preds)) if sample_count else 0.0,
        "mean_weekly_flagged_wafers": float(np.mean(flagged_counts)) if flagged_counts.size else 0.0,
        "mean_weekly_fail_captures": float(np.mean(tp_counts)) if tp_counts.size else 0.0,
        "mean_weekly_fail_misses": float(np.mean(fn_counts)) if fn_counts.size else 0.0,
    }


def _build_temporal_cost_curves(lockbox_df: pd.DataFrame, y_lock: np.ndarray) -> pd.DataFrame:
    """Build illustrative cost-curve rows from lockbox confusion counts."""
    rows = []
    for ratio in COST_RATIOS:
        row: dict[str, Any] = {"cost_ratio": ratio}
        for role in ["primary", "challenger"]:
            for policy in THRESHOLD_POLICIES:
                key = f"{role}_{policy}"
                sub = lockbox_df[(lockbox_df["role"] == role) & (lockbox_df["threshold_policy"] == policy)]
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
        rows.append(row)
    return pd.DataFrame(rows)


def _build_manager_outputs(
    fitted_models: list[FittedRoleModel], y_dev: np.ndarray, week_dev: np.ndarray
) -> pd.DataFrame:
    """Build manager-facing workload rows for each frozen role threshold."""
    rows = []
    for fitted in fitted_models:
        for policy, threshold in _threshold_values_for_model(fitted):
            weekly = _manager_weekly_metrics(
                y_true=y_dev,
                scores=fitted.dev_scores,
                threshold=float(threshold),
                week_labels=week_dev,
            )
            rows.append(
                {
                    "role": fitted.config.role,
                    "selector": fitted.config.selector,
                    "threshold_policy": policy,
                    "predicted_flag_fraction": float(weekly["predicted_flag_fraction"]),
                    "mean_weekly_flagged_wafers": float(weekly["mean_weekly_flagged_wafers"]),
                    "mean_weekly_fail_captures": float(weekly["mean_weekly_fail_captures"]),
                    "mean_weekly_fail_misses": float(weekly["mean_weekly_fail_misses"]),
                }
            )
    return pd.DataFrame(rows)


def _temporal_claim_restrictions(lockbox_df: pd.DataFrame, drift_df: pd.DataFrame, mspc_df: pd.DataFrame) -> list[str]:
    """Return temporal claim restrictions implied by lockbox, drift, and MSPC artifacts."""
    restrictions: list[str] = []
    mspc_lock = mspc_df[mspc_df["eval_scope"] == "lockbox"]
    if mspc_lock.empty:
        return restrictions

    mspc_tpr = float(mspc_lock.iloc[0]["best_MSPC_TPR_at_TNR90"])
    lockbox_scientific = lockbox_df[lockbox_df["threshold_policy"] == ThresholdPolicy.SCIENTIFIC]
    for row in lockbox_scientific.itertuples(index=False):
        scope = ModelScope.PRIMARY if row.role == "primary" else ModelScope.CHALLENGER
        drift_row = drift_df[drift_df["model_scope"] == scope]
        if not drift_row.empty:
            status = str(drift_row.iloc[0]["drift_gate_status"])
            if status == "HIGH_SHIFT" and float(row.TPR_at_TNR90) > mspc_tpr:
                restrictions.append(f"{scope}_high_shift_blocks_lockbox_superiority_claim")
    return restrictions


def _is_selected_stage_b_config(row: dict[str, Any], best: dict[str, Any]) -> bool:
    """Return whether a Stage-B config row matches the selected inner-CV config."""
    return (
        row["k"] == best["k"]
        and np.isclose(row["C"], best["C"])
        and row["scaler"] == best["scaler"]
        and row.get("n_neighbors") == best.get("n_neighbors")
    )


def _stage_b_inner_artifact_row(
    *,
    selector: str,
    resample_id: str,
    row: dict[str, Any],
    best: dict[str, Any],
) -> dict[str, Any]:
    """Normalize one Stage-B inner-CV score row for the artifact."""
    return {
        "selector": selector,
        "resample_id": resample_id,
        "k": int(row["k"]),
        "C": float(row["C"]),
        "scaler": row["scaler"],
        "n_neighbors": row.get("n_neighbors"),
        "mean_inner_ROC_AUC": float(row["mean_inner_ROC_AUC"]),
        "mean_inner_BER": float(row["mean_inner_BER"]),
        "is_selected_config": _is_selected_stage_b_config(row, best),
    }


def _flagged_fraction(metrics: dict[str, float]) -> float:
    """Return the fraction of wafers flagged by a binary metric payload."""
    flagged = float(metrics["FP"]) + (float(metrics["lockbox_fails"]) - float(metrics["FN"]))
    return float(flagged / max(float(metrics["lockbox_n"]), 1.0))


def _outer_eval_artifact_row(
    *,
    selector: str,
    fold: Any,
    seed: int,
    resample_id: str,
    best: dict[str, Any],
    threshold: float,
    metrics: dict[str, float],
) -> dict[str, Any]:
    """Normalize one temporal outer-evaluation row for the artifact."""
    return {
        "selector": selector,
        "outer_fold": int(fold.outer_fold),
        "seed": int(seed),
        "resample_id": resample_id,
        "train_window": to_time_window_string(fold.train_start_ts, fold.train_end_ts),
        "test_window": to_time_window_string(fold.test_start_ts, fold.test_end_ts),
        "k": int(best["k"]),
        "C": float(best["C"]),
        "scaler": best["scaler"],
        "n_neighbors": best.get("n_neighbors"),
        "outer_threshold": float(threshold),
        "BER": float(metrics["BER"]),
        "True+": float(metrics["True+"]),
        "True-": float(metrics["True-"]),
        "flagged_fraction": _flagged_fraction(metrics),
    }


def _neighbor_sort_key(value: Any) -> float:
    """Return a deterministic sort key for optional ReliefF neighbors."""
    return math.inf if value is None or pd.isna(value) else float(value)


def _selector_config_simplicity_key(row: dict[str, Any]) -> tuple[int, float, int, float]:
    """Rank selected temporal configs by deterministic simplicity."""
    scaler_pref = 0 if row["scaler"] == ScalerName.STANDARD else 1
    return (int(row["k"]), float(row["C"]), scaler_pref, _neighbor_sort_key(row.get("n_neighbors")))


def _modal_selected_config(group: pd.DataFrame) -> dict[str, Any]:
    """Return the modal selected config tuple for one selector's outer evaluations."""
    config_counts = (
        group.groupby(["k", "C", "scaler", "n_neighbors"], dropna=False, sort=False)
        .size()
        .reset_index(name="selection_count")
    )
    best = min(
        config_counts.to_dict("records"),
        key=lambda row: (-int(row["selection_count"]), *_selector_config_simplicity_key(row)),
    )
    nn = best["n_neighbors"]
    return {
        "modal_k": int(best["k"]),
        "modal_C": float(best["C"]),
        "modal_scaler": str(best["scaler"]),
        "modal_n_neighbors": np.nan if nn is None or pd.isna(nn) else float(nn),
    }


def _summarize_temporal_selector_results(
    *,
    outer_eval_df: pd.DataFrame,
    deciding_outer_fold: int,
) -> list[dict[str, Any]]:
    """Summarize outer-evaluation selector results for temporal role assignment."""
    selector_stats: list[dict[str, Any]] = []
    for selector, grp in outer_eval_df.groupby("selector", sort=False):
        deciding_vote = grp[(grp["seed"] == SEEDS_STAGE_B[0]) & (grp["outer_fold"] == deciding_outer_fold)]
        vote_ber = float(deciding_vote["BER"].iloc[0]) if not deciding_vote.empty else np.inf
        vote_true_pos = float(deciding_vote["True+"].iloc[0]) if not deciding_vote.empty else -np.inf
        selector_stats.append(
            {
                "selector": selector,
                "mean_BER": float(grp["BER"].mean()),
                "std_BER": safe_std(grp["BER"].to_numpy(dtype=float)),
                "mean_True+": float(grp["True+"].mean()),
                "mean_True-": float(grp["True-"].mean()),
                **_modal_selected_config(grp),
                "vote_outer_BER": vote_ber,
                "vote_outer_True+": vote_true_pos,
            }
        )
    return selector_stats


def _selector_rank_key(row: dict[str, Any]) -> tuple[float, float, float, float, int, float, int, float, str]:
    """Rank temporal selectors by study priority and deterministic simplicity."""
    modal_config = {
        "k": row["modal_k"],
        "C": row["modal_C"],
        "scaler": row["modal_scaler"],
        "n_neighbors": row["modal_n_neighbors"],
    }
    modal_k, modal_c, scaler_pref, nn_key = _selector_config_simplicity_key(modal_config)
    return (
        float(row["mean_BER"]),
        -float(row["mean_True+"]),
        float(row["vote_outer_BER"]),
        -float(row["vote_outer_True+"]),
        modal_k,
        modal_c,
        scaler_pref,
        nn_key,
        str(row["selector"]),
    )


def _choose_temporal_roles(selector_stats: list[dict[str, Any]]) -> tuple[str, str | None]:
    """Choose primary and optional challenger selectors from temporal summaries."""
    if not selector_stats:
        raise ValueError("No temporal selector statistics available for role assignment")
    ranked = sorted(selector_stats, key=_selector_rank_key)
    primary = ranked[0]["selector"]
    eligible = [row for row in ranked[1:] if float(row["mean_BER"]) <= TEMPORAL_CHALLENGER_MAX_BER]
    challenger = eligible[0]["selector"] if eligible else None
    return str(primary), None if challenger is None else str(challenger)


def _model_selection_artifact_row(
    *,
    row: dict[str, Any],
    primary: str,
    challenger: str | None,
) -> dict[str, Any]:
    """Normalize one selector summary into the temporal model-selection artifact."""
    is_primary = row["selector"] == primary
    is_challenger = challenger is not None and row["selector"] == challenger
    status = "primary" if is_primary else ("challenger" if is_challenger else "supporting")
    return {
        "selector": row["selector"],
        "status": status,
        "is_primary": is_primary,
        "is_challenger": is_challenger,
        "mean_BER": float(row["mean_BER"]),
        "std_BER": float(row["std_BER"]),
        "mean_True+": float(row["mean_True+"]),
        "mean_True-": float(row["mean_True-"]),
        "modal_k": int(row["modal_k"]),
        "modal_C": float(row["modal_C"]),
        "modal_scaler": row["modal_scaler"],
        "modal_n_neighbors": row["modal_n_neighbors"],
    }


def _run_stage_b_model_selection(
    *,
    bundle: DataBundle,
    selectors_run: list[str],
    x_dev: np.ndarray,
    y_dev: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run temporal Stage-B inner search and outer evaluation for all selectors."""
    if bundle.fold_plan is None:
        raise ValueError("Temporal fold plan is required for Stage-B model selection")

    inner_rows: list[dict[str, Any]] = []
    outer_eval_rows: list[dict[str, Any]] = []
    for selector in selectors_run:
        configs = build_stage_b_config_grid(selector)
        config_groups = _group_stage_b_configs_by_preparation(configs)

        for fold in bundle.fold_plan.folds:
            x_outer_train = x_dev[fold.train_index]
            y_outer_train = y_dev[fold.train_index]
            x_outer_test = x_dev[fold.test_index]
            y_outer_test = y_dev[fold.test_index]

            for seed in SEEDS_STAGE_B:
                config_scores = []
                resample_id = f"outer_{fold.outer_fold}_seed_{seed}"
                for (k_value, scaler_name, n_neighbors), cfg_group in config_groups.items():
                    prepared_views = _prepare_inner_cv_views(
                        x_outer_train_raw=x_outer_train,
                        y_outer_train=y_outer_train,
                        selector=selector,
                        k=k_value,
                        scaler_name=scaler_name,
                        n_neighbors=n_neighbors,
                        seed=seed,
                    )
                    for cfg in cfg_group:
                        mean_auc, mean_ber = _score_prepared_inner_cv(
                            prepared_views=prepared_views,
                            c_value=float(cfg["C"]),
                        )
                        row = dict(cfg)
                        row["selector"] = selector
                        row["resample_id"] = resample_id
                        row["mean_inner_ROC_AUC"] = mean_auc
                        row["mean_inner_BER"] = mean_ber
                        config_scores.append(row)

                best = select_best_inner_config(config_scores)
                inner_rows.extend(
                    _stage_b_inner_artifact_row(
                        selector=selector,
                        resample_id=resample_id,
                        row=row,
                        best=best,
                    )
                    for row in config_scores
                )

                metrics, threshold = _fit_eval_with_labels(
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
                outer_eval_rows.append(
                    _outer_eval_artifact_row(
                        selector=selector,
                        fold=fold,
                        seed=seed,
                        resample_id=resample_id,
                        best=best,
                        threshold=threshold,
                        metrics=metrics,
                    )
                )

    return pd.DataFrame(inner_rows), pd.DataFrame(outer_eval_rows)


def run_temporal_robustness(
    input_dir: Path,
    output_dir: Path,
    *,
    selectors_run: list[str] | None = None,
) -> dict[str, Any]:
    """Run temporal robustness and persist failed status before re-raising errors."""
    try:
        return _run_temporal_robustness(
            input_dir=input_dir,
            output_dir=output_dir,
            selectors_run=selectors_run,
        )
    except Exception as exc:
        with suppress(Exception):
            ensure_reports_dir(output_dir)
            write_temporal_failure(
                manifest_path=output_dir / "reports" / ArtifactName.MANIFEST,
                project_root=project_root_from_repo_structure(),
                reason=str(exc),
            )
        raise


def _run_temporal_robustness(
    input_dir: Path,
    output_dir: Path,
    *,
    selectors_run: list[str] | None = None,
) -> dict[str, Any]:
    """Run the temporal robustness study and write all temporal artifacts."""
    reports = ensure_reports_dir(output_dir)
    selectors_run = list(SelectorName.ACTIVE) if selectors_run is None else [str(s) for s in selectors_run]
    bundle = _build_bundle(input_dir)

    split_meta = pd.DataFrame(
        [
            {
                "n_total": len(bundle.all_data),
                "n_dev": len(bundle.dev),
                "n_lockbox": len(bundle.lockbox),
                "split_rule": "last floor(0.15*N) rows after stable sort by (timestamp, raw_row_id)",
            }
        ]
    )
    write_csv(split_meta, reports / ArtifactName.TEMPORAL_SPLIT_METADATA)

    manifest_path = reports / ArtifactName.MANIFEST
    project_root = project_root_from_repo_structure()
    if not bundle.temporal_feasible or bundle.fold_plan is None:
        # Still write split metadata so the audit explains why temporal artifacts are absent.
        infeasible_reason = bundle.temporal_infeasible_reason or "no_feasible_plan"
        write_temporal_status(
            manifest_path=manifest_path,
            project_root=project_root,
            temporal_status=StudyStatus.NOT_RUN,
            industrialization_note=f"temporal robustness not run: {infeasible_reason}",
        )
        return {
            "temporal_robustness_status": StudyStatus.NOT_RUN,
            "reason": bundle.temporal_infeasible_reason,
        }

    x_dev = bundle.dev_with_weeks[bundle.feature_columns].to_numpy(dtype=float)
    y_dev = bundle.dev_with_weeks["y_bin"].to_numpy(dtype=int)
    week_dev = bundle.dev_with_weeks["week_label"].to_numpy(dtype=int)
    x_lock = bundle.lockbox[bundle.feature_columns].to_numpy(dtype=float)
    y_lock = bundle.lockbox["y_bin"].to_numpy(dtype=int)

    stage_a_rows: list[dict[str, Any]] = []
    for cfg in _stage_a_configs(selectors_run):
        selector = cfg["selector"]
        fold_ber_values: list[float] = []
        for fold in bundle.fold_plan.folds:
            metrics, _threshold = _fit_eval_with_labels(
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
            fold_ber_values.append(float(metrics["BER"]))
        stage_a_rows.append(
            {
                "selector": selector,
                "mean_BER": float(np.mean(fold_ber_values)),
                "std_BER": safe_std(fold_ber_values),
            }
        )
    write_csv(pd.DataFrame(stage_a_rows), reports / ArtifactName.TEMPORAL_SELECTOR_SCREENING)

    inner_df, outer_eval_df = _run_stage_b_model_selection(
        bundle=bundle,
        selectors_run=selectors_run,
        x_dev=x_dev,
        y_dev=y_dev,
    )
    write_csv(inner_df, reports / ArtifactName.TEMPORAL_INNER_CV)

    deciding_outer_fold = max(f.outer_fold for f in bundle.fold_plan.folds)
    selector_stats = _summarize_temporal_selector_results(
        outer_eval_df=outer_eval_df,
        deciding_outer_fold=deciding_outer_fold,
    )
    primary, challenger = _choose_temporal_roles(selector_stats)

    model_selection_rows = [
        _model_selection_artifact_row(row=row, primary=primary, challenger=challenger) for row in selector_stats
    ]
    write_csv(pd.DataFrame(model_selection_rows), reports / ArtifactName.TEMPORAL_MODEL_SELECTION)

    freeze_frames = []
    frozen_roles: list[RoleConfig] = []
    freeze_primary_df, cfg_primary = _phase2_freeze_for_role("primary", primary, x_dev, y_dev)
    freeze_frames.append(freeze_primary_df)
    frozen_roles.append(cfg_primary)
    if challenger is not None:
        freeze_ch_df, cfg_ch = _phase2_freeze_for_role("challenger", challenger, x_dev, y_dev)
        freeze_frames.append(freeze_ch_df)
        frozen_roles.append(cfg_ch)
    freeze_df = pd.concat(freeze_frames, ignore_index=True)
    write_csv(freeze_df, reports / ArtifactName.TEMPORAL_FREEZE)

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
        lock_ctx = _prepare_lockbox_eval_context(model=fitted, x_lock_raw=x_lock, y_lock=y_lock)
        lock_rows.append(_score_lockbox_for_role(fitted, y_lock=y_lock, lock_ctx=lock_ctx))
        drift_rows.append(
            _drift_gate_for_role(
                model=fitted,
                x_dev_raw=x_dev,
                y_dev=y_dev,
                x_lock_raw=x_lock,
                y_lock=y_lock,
                lock_ctx=lock_ctx,
            )
        )
    lockbox_df = pd.concat(lock_rows, ignore_index=True)
    drift_df = pd.DataFrame(drift_rows)
    write_csv(lockbox_df, reports / ArtifactName.TEMPORAL_LOCKBOX)
    write_csv(drift_df, reports / ArtifactName.TEMPORAL_DRIFT)

    mspc_rows = []
    for fold in bundle.fold_plan.folds:
        tr = fold.train_index
        te = fold.test_index
        x_train_pass = x_dev[tr][y_dev[tr] == 0]
        m = _mspc_fit_and_score(x_train_pass=x_train_pass, x_eval=x_dev[te], y_eval=y_dev[te])
        mspc_rows.append({"eval_scope": "outer_fold", "fold_index": str(fold.outer_fold), **m})
    x_dev_pass = x_dev[y_dev == 0]
    m = _mspc_fit_and_score(x_train_pass=x_dev_pass, x_eval=x_lock, y_eval=y_lock)
    mspc_rows.append({"eval_scope": "lockbox", "fold_index": "LOCKBOX", **m})
    mspc_df = pd.DataFrame(mspc_rows)
    write_csv(mspc_df, reports / ArtifactName.TEMPORAL_MSPC)

    write_csv(
        _build_temporal_cost_curves(lockbox_df=lockbox_df, y_lock=y_lock), reports / ArtifactName.TEMPORAL_COST_CURVES
    )
    write_csv(
        _build_manager_outputs(fitted_models=fitted_models, y_dev=y_dev, week_dev=week_dev),
        reports / ArtifactName.TEMPORAL_MANAGER_OUTPUTS,
    )

    restrictions = _temporal_claim_restrictions(lockbox_df=lockbox_df, drift_df=drift_df, mspc_df=mspc_df)

    manifest = write_temporal_status(
        manifest_path=manifest_path,
        project_root=project_root,
        temporal_status=StudyStatus.WARNING if restrictions else StudyStatus.PASSED,
        claim_restrictions=restrictions,
    )
    return {
        "temporal_robustness_status": manifest["temporal_robustness_status"],
        "primary_selector": primary,
        "challenger_selector": challenger,
        "claim_restrictions": restrictions,
    }
