from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from secom.artifacts import config_hash, ensure_reports_dir, write_csv, write_manifest
from secom.common.drift import psi_for_feature
from secom.common.meta import git_commit_and_dirty, library_versions, strategy_sha256
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
    expected_cost_per_wafer,
    extract_tpr_at_tnr,
    find_ber_optimal_threshold,
    safe_std,
)
from secom.models import fit_temporal_logreg_model
from secom.preprocess import make_imputer, make_scaler, transformed_feature_metadata_from_imputer
from secom.selection.engine import fit_selector_pipeline, select_features
from secom.selection.tuning import select_best_inner_config
from secom.types import DataBundle, FittedRoleModel, RoleConfig


def _build_bundle(input_dir: Path) -> DataBundle:
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
) -> tuple[dict[str, float], float, list[Any], np.ndarray, Any, Any, Any]:
    x_train_sel, x_eval_sel, feature_meta, selected_local, imputer, scaler = fit_selector_pipeline(
        x_train_raw=x_train_raw,
        y_train=y_train,
        x_eval_raw=x_eval_raw,
        method=method,
        k=k,
        scaler_name=scaler_name,
        add_indicator=True,
        n_neighbors=n_neighbors,
    )
    clf = fit_temporal_logreg_model(x_train_sel, y_train, c_value=c_value)
    train_scores = clf.predict_proba(x_train_sel)[:, 1]
    eval_scores = clf.predict_proba(x_eval_sel)[:, 1]
    threshold, _ = find_ber_optimal_threshold(y_train, train_scores)
    metrics = binary_metrics_at_threshold(y_eval, eval_scores, threshold)
    return metrics, threshold, feature_meta, selected_local, clf, imputer, scaler


def _stage_a_configs(selectors_run: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "selector": s,
            "k": 40,
            "C": 1.0,
            "scaler": ScalerName.ROBUST,
            "n_neighbors": 10 if s == SelectorName.RELIEFF else None,
        }
        for s in selectors_run
    ]


def build_stage_b_config_grid(selector: str) -> list[dict[str, Any]]:
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


def _prepare_inner_cv_views(
    x_outer_train_raw: np.ndarray,
    y_outer_train: np.ndarray,
    selector: str,
    k: int,
    scaler_name: str,
    n_neighbors: int | None,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    prepared_folds: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for inner_train_idx, inner_val_idx in skf.split(x_outer_train_raw, y_outer_train):
        x_inner_train = x_outer_train_raw[inner_train_idx]
        y_inner_train = y_outer_train[inner_train_idx]
        x_inner_val = x_outer_train_raw[inner_val_idx]
        y_inner_val = y_outer_train[inner_val_idx]
        x_train_sel, x_val_sel, _meta, _sel, _imp, _scaler = fit_selector_pipeline(
            x_train_raw=x_inner_train,
            y_train=y_inner_train,
            x_eval_raw=x_inner_val,
            method=selector,
            k=k,
            scaler_name=scaler_name,
            add_indicator=True,
            n_neighbors=n_neighbors,
        )
        prepared_folds.append((x_train_sel, y_inner_train, x_val_sel, y_inner_val))
    return prepared_folds


def _score_prepared_inner_cv(
    prepared_folds: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    c_value: float,
) -> tuple[float, float]:
    aucs: list[float] = []
    bers: list[float] = []
    for x_train_sel, y_inner_train, x_val_sel, y_inner_val in prepared_folds:
        clf = fit_temporal_logreg_model(x_train_sel, y_inner_train, c_value=c_value)
        train_scores = clf.predict_proba(x_train_sel)[:, 1]
        val_scores = clf.predict_proba(x_val_sel)[:, 1]
        threshold, _ = find_ber_optimal_threshold(y_inner_train, train_scores)
        m = binary_metrics_at_threshold(y_inner_val, val_scores, threshold)
        aucs.append(float(m["ROC_AUC"]) if np.isfinite(m["ROC_AUC"]) else 0.5)
        bers.append(float(m["BER"]))
    return float(np.mean(aucs)), float(np.mean(bers))


def _phase2_fold_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    preds = (scores >= threshold).astype(int)
    pos_mask = y_true == 1
    neg_mask = ~pos_mask
    tp = int(np.sum(pos_mask & (preds == 1)))
    fn = int(np.sum(pos_mask & (preds == 0)))
    tn = int(np.sum(neg_mask & (preds == 0)))
    fp = int(np.sum(neg_mask & (preds == 1)))
    tpr = float(tp / (tp + fn)) if (tp + fn) else 0.0
    tnr = float(tn / (tn + fp)) if (tn + fp) else 0.0
    ber = float(1.0 - 0.5 * (tpr + tnr))
    auc = float(roc_auc_score(y_true, scores)) if np.unique(y_true).size == 2 else 0.5
    return ber, auc


def _prepare_phase2_inner_views(
    *,
    selector: str,
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    k: int,
    scaler_name: str,
    n_neighbors: int | None,
) -> list[dict[str, Any]]:
    prepared_views: list[dict[str, Any]] = []
    for seed in SEEDS_PHASE2:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for inner_fold_i, (tr, va) in enumerate(skf.split(x_dev, y_dev), start=1):
            x_tr = x_dev[tr]
            y_tr = y_dev[tr]
            x_va = x_dev[va]
            y_va = y_dev[va]
            x_train_sel, x_val_sel, _meta, _sel, _imp, _scaler = fit_selector_pipeline(
                x_train_raw=x_tr,
                y_train=y_tr,
                x_eval_raw=x_va,
                method=selector,
                k=int(k),
                scaler_name=scaler_name,
                add_indicator=True,
                n_neighbors=n_neighbors,
            )
            prepared_views.append(
                {
                    "seed": seed,
                    "inner_fold": inner_fold_i,
                    "y_train": y_tr,
                    "y_val": y_va,
                    "x_train_sel": x_train_sel,
                    "x_val_sel": x_val_sel,
                }
            )
    return prepared_views


def _phase2_freeze_for_role(
    role: str,
    selector: str,
    x_dev: np.ndarray,
    y_dev: np.ndarray,
) -> tuple[pd.DataFrame, RoleConfig]:
    configs = build_stage_b_config_grid(selector)
    per_config: dict[tuple, list[dict[str, Any]]] = {}
    grouped_configs: dict[tuple[int, str, int | None], list[dict[str, Any]]] = {}
    for cfg in configs:
        prep_key = (int(cfg["k"]), str(cfg["scaler"]), cfg.get("n_neighbors"))
        grouped_configs.setdefault(prep_key, []).append(cfg)

    for (k, scaler_name, n_neighbors), prep_configs in grouped_configs.items():
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
                y_tr = prepared["y_train"]
                y_va = prepared["y_val"]
                x_train_sel = prepared["x_train_sel"]
                x_val_sel = prepared["x_val_sel"]
                clf = fit_temporal_logreg_model(x_train_sel, y_tr, c_value=float(cfg["C"]))
                tr_scores = clf.predict_proba(x_train_sel)[:, 1]
                va_scores = clf.predict_proba(x_val_sel)[:, 1]
                threshold, _ = find_ber_optimal_threshold(y_tr, tr_scores)
                ber, auc = _phase2_fold_metrics(y_va, va_scores, threshold)
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

    config_rows = []
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

    rows = []
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


def _score_lockbox_for_role(
    model: FittedRoleModel,
    y_lock: np.ndarray,
    lock_ctx: dict[str, Any],
) -> pd.DataFrame:
    lock_scores = np.asarray(lock_ctx["lock_scores"], dtype=float)
    rows = []
    for policy, th in (
        (ThresholdPolicy.SCIENTIFIC, model.scientific_threshold),
        (ThresholdPolicy.OPERATIONAL, model.operational_threshold),
    ):
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
    lock_scores = np.asarray(lock_ctx["lock_scores"], dtype=float)
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
    psi_vals = [psi_for_feature(x_dev_raw[:, idx], x_lock_raw[:, idx]) for idx in top_raw_idx]
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
    _t2_thr, _, t2_tpr90 = extract_tpr_at_tnr(y_eval, t2_eval, target_tnr=0.90)
    _q_thr, _, q_tpr90 = extract_tpr_at_tnr(y_eval, q_eval, target_tnr=0.90)
    alarm = ((t2_eval > ucl_t2) | (q_eval > ucl_q)).astype(int)
    alarm_rate = float(np.mean(alarm))
    alarm_positions = np.where(alarm == 1)[0]
    arl0 = np.nan if alarm_positions.size < 2 else float(np.mean(np.diff(alarm_positions)))
    best_tpr = max(float(t2_tpr90), float(q_tpr90))
    best_src = "T2" if np.isclose(best_tpr, float(t2_tpr90)) else "Q"
    if np.unique(y_eval).size == 2:
        t2_auc = float(roc_auc_score(y_eval, t2_eval))
        q_auc = float(roc_auc_score(y_eval, q_eval))
    else:
        t2_auc = np.nan
        q_auc = np.nan
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
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    weeks = np.asarray(week_labels, dtype=int)
    preds = (scores >= threshold).astype(int)
    _, week_codes = np.unique(weeks, return_inverse=True)
    week_count = int(week_codes.max()) + 1 if week_codes.size else 0
    preds_float = preds.astype(float, copy=False)
    fail_mask = (y_true == 1).astype(float)
    flagged_counts = np.bincount(week_codes, weights=preds_float, minlength=week_count) if week_count else np.array([], dtype=float)
    tp_counts = np.bincount(week_codes, weights=(preds_float * fail_mask), minlength=week_count) if week_count else np.array([], dtype=float)
    fn_counts = np.bincount(week_codes, weights=((1.0 - preds_float) * fail_mask), minlength=week_count) if week_count else np.array([], dtype=float)
    sample_count = len(y_true)
    return {
        "predicted_flag_fraction": float(np.mean(preds)) if sample_count else 0.0,
        "mean_weekly_flagged_wafers": float(np.mean(flagged_counts)) if flagged_counts.size else 0.0,
        "mean_weekly_fail_captures": float(np.mean(tp_counts)) if tp_counts.size else 0.0,
        "mean_weekly_fail_misses": float(np.mean(fn_counts)) if fn_counts.size else 0.0,
    }


def _init_manifest(output_dir: Path) -> dict[str, Any]:
    reports = ensure_reports_dir(output_dir)
    path = reports / ArtifactName.MANIFEST
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    project_root = Path(__file__).resolve().parents[3]
    commit, dirty = git_commit_and_dirty(project_root)
    return {
        "manifest_version": "2.0",
        "study_spec_path": "docs/spec/README.md",
        "study_spec_sha256": strategy_sha256(project_root),
        "git_commit": commit,
        "git_dirty": dirty,
        "python_executable": sys.executable,
        "library_versions": library_versions(),
        "primary_study_status": StudyStatus.NOT_RUN,
        "benchmark_original_status": StudyStatus.NOT_RUN,
        "benchmark_tuned_status": StudyStatus.NOT_RUN,
        "temporal_robustness_status": StudyStatus.NOT_RUN,
        "temporal_claim_restrictions": [],
        "industrialization_notes": [],
    }


def run_temporal_robustness(
    input_dir: Path,
    output_dir: Path,
    *,
    selectors_run: list[str] | None = None,
) -> dict[str, Any]:
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

    manifest = _init_manifest(output_dir)
    if not bundle.temporal_feasible or bundle.fold_plan is None:
        manifest["temporal_robustness_status"] = StudyStatus.NOT_RUN
        notes = list(manifest.get("industrialization_notes", []))
        notes.append(
            f"temporal robustness not run: {bundle.temporal_infeasible_reason or 'no_feasible_plan'}"
        )
        manifest["industrialization_notes"] = notes
        write_manifest(manifest, reports / ArtifactName.MANIFEST)
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
        fold_metrics = []
        for fold in bundle.fold_plan.folds:
            metrics, _threshold, _meta, _sel, _clf, _imp, _scl = _fit_eval_with_labels(
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
            fold_metrics.append({"BER": metrics["BER"]})
        stage_a_rows.append(
            {
                "selector": selector,
                "mean_BER": float(np.mean([m["BER"] for m in fold_metrics])),
                "std_BER": safe_std([m["BER"] for m in fold_metrics]),
            }
        )
    write_csv(pd.DataFrame(stage_a_rows), reports / ArtifactName.TEMPORAL_SELECTOR_SCREENING)

    outer_eval_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    for selector in selectors_run:
        configs = build_stage_b_config_grid(selector)
        config_groups: dict[tuple[int, str, int | None], list[dict[str, Any]]] = {}
        for cfg in configs:
            key = (int(cfg["k"]), str(cfg["scaler"]), cfg.get("n_neighbors"))
            config_groups.setdefault(key, []).append(cfg)

        for fold in bundle.fold_plan.folds:
            x_outer_train = x_dev[fold.train_index]
            y_outer_train = y_dev[fold.train_index]
            x_outer_test = x_dev[fold.test_index]
            y_outer_test = y_dev[fold.test_index]

            for seed in SEEDS_STAGE_B:
                config_scores = []
                resample_id = f"outer_{fold.outer_fold}_seed_{seed}"
                for (k_value, scaler_name, n_neighbors), cfg_group in config_groups.items():
                    prepared_folds = _prepare_inner_cv_views(
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
                            prepared_folds=prepared_folds,
                            c_value=float(cfg["C"]),
                        )
                        row = dict(cfg)
                        row["selector"] = selector
                        row["resample_id"] = resample_id
                        row["mean_inner_ROC_AUC"] = mean_auc
                        row["mean_inner_BER"] = mean_ber
                        config_scores.append(row)

                best = select_best_inner_config(config_scores)
                for row in config_scores:
                    inner_rows.append(
                        {
                            "selector": selector,
                            "resample_id": resample_id,
                            "k": int(row["k"]),
                            "C": float(row["C"]),
                            "scaler": row["scaler"],
                            "n_neighbors": row.get("n_neighbors"),
                            "mean_inner_ROC_AUC": float(row["mean_inner_ROC_AUC"]),
                            "mean_inner_BER": float(row["mean_inner_BER"]),
                            "is_selected_config": (
                                row["k"] == best["k"]
                                and np.isclose(row["C"], best["C"])
                                and row["scaler"] == best["scaler"]
                                and row.get("n_neighbors") == best.get("n_neighbors")
                            ),
                        }
                    )

                metrics, threshold, _meta, _selected, _clf, _imp, _scl = _fit_eval_with_labels(
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
                    {
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
                        "flagged_fraction": float(
                            (float(metrics["FP"]) + (float(metrics["lockbox_fails"]) - float(metrics["FN"])))
                            / max(float(metrics["lockbox_n"]), 1.0)
                        ),
                    }
                )

    inner_df = pd.DataFrame(inner_rows)
    write_csv(inner_df, reports / ArtifactName.TEMPORAL_INNER_CV)

    outer_eval_df = pd.DataFrame(outer_eval_rows)
    selector_stats = []
    deciding_outer_fold = max(f.outer_fold for f in bundle.fold_plan.folds)
    for selector, grp in outer_eval_df.groupby("selector"):
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
                "vote_outer_BER": vote_ber,
                "vote_outer_True+": vote_true_pos,
            }
        )

    def _selector_rank_key(row: dict[str, Any]) -> tuple[float, float, int, float, int, float, float, str]:
        scaler_pref = 0 if row["modal_scaler"] == ScalerName.STANDARD else 1
        nn = row["modal_n_neighbors"]
        nn_key = math.inf if pd.isna(nn) else float(nn)
        return (
            row["mean_BER"],
            -row["mean_True+"],
            row["modal_k"],
            row["modal_C"],
            scaler_pref,
            nn_key,
            row["vote_outer_BER"],
            row["selector"],
        )

    ranked = sorted(selector_stats, key=_selector_rank_key)
    primary = ranked[0]["selector"]
    eligible = [r for r in ranked[1:] if r["mean_BER"] <= 0.40]
    challenger = None
    if eligible:
        challenger = sorted(
            eligible,
            key=lambda r: (-r["mean_True-"], r["mean_BER"], r["selector"]),
        )[0]["selector"]

    model_selection_rows = []
    for row in selector_stats:
        is_primary = row["selector"] == primary
        is_challenger = challenger is not None and row["selector"] == challenger
        status = "primary" if is_primary else ("challenger" if is_challenger else "supporting")
        model_selection_rows.append(
            {
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
        )
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

    cost_rows = []
    for ratio in COST_RATIOS:
        row: dict[str, Any] = {"cost_ratio": ratio}
        for role in ["primary", "challenger"]:
            for policy in [ThresholdPolicy.SCIENTIFIC, ThresholdPolicy.OPERATIONAL]:
                key = f"{role}_{policy}"
                sub = lockbox_df[
                    (lockbox_df["role"] == role) & (lockbox_df["threshold_policy"] == policy)
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
    write_csv(pd.DataFrame(cost_rows), reports / ArtifactName.TEMPORAL_COST_CURVES)

    manager_rows = []
    for fitted in fitted_models:
        for policy, threshold in (
            (ThresholdPolicy.SCIENTIFIC, fitted.scientific_threshold),
            (ThresholdPolicy.OPERATIONAL, fitted.operational_threshold),
        ):
            weekly = _manager_weekly_metrics(
                y_true=y_dev,
                scores=fitted.dev_scores,
                threshold=float(threshold),
                week_labels=week_dev,
            )
            manager_rows.append(
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
    write_csv(pd.DataFrame(manager_rows), reports / ArtifactName.TEMPORAL_MANAGER_OUTPUTS)

    restrictions: list[str] = []
    mspc_lock = mspc_df[mspc_df["eval_scope"] == "lockbox"]
    if not mspc_lock.empty:
        mspc_tpr = float(mspc_lock.iloc[0]["best_MSPC_TPR_at_TNR90"])
        for row in lockbox_df[lockbox_df["threshold_policy"] == ThresholdPolicy.SCIENTIFIC].itertuples(index=False):
            scope = ModelScope.PRIMARY if row.role == "primary" else ModelScope.CHALLENGER
            drift_row = drift_df[drift_df["model_scope"] == scope]
            if not drift_row.empty:
                status = str(drift_row.iloc[0]["drift_gate_status"])
                if status == "HIGH_SHIFT" and float(row.TPR_at_TNR90) > mspc_tpr:
                    restrictions.append(f"{scope}_high_shift_blocks_lockbox_superiority_claim")

    manifest["temporal_robustness_status"] = StudyStatus.WARNING if restrictions else StudyStatus.PASSED
    manifest["temporal_claim_restrictions"] = restrictions
    write_manifest(manifest, reports / ArtifactName.MANIFEST)
    return {
        "temporal_robustness_status": manifest["temporal_robustness_status"],
        "primary_selector": primary,
        "challenger_selector": challenger,
        "claim_restrictions": restrictions,
    }
