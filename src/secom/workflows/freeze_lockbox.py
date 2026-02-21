from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from secom.artifacts import config_hash, ensure_reports_dir, write_csv, write_manifest
from secom.common.drift import psi_for_feature
from secom.common.thresholds import operational_threshold
from secom.config import (
    ArtifactName,
    COST_RATIOS,
    ModelScope,
    PSI_MAX_FEATURES,
    SEEDS_PHASE2,
    ThresholdPolicy,
)
from secom.metrics import (
    binary_metrics_at_threshold,
    expected_cost_per_wafer,
    extract_tpr_at_tnr,
    find_ber_optimal_threshold,
)
from secom.models import fit_lane_b_classifier
from secom.preprocess import (
    local_to_global_feature_indices,
    make_imputer,
    make_scaler,
    transformed_feature_metadata_from_imputer,
)
from secom.selection.engine import fit_selector_pipeline, select_features
from secom.selection.tuning import select_best_inner_config
from secom.types import DataBundle, FittedRoleModel, RoleConfig
from secom.workflows.lane_b import build_stage_b_config_grid


def _phase2_freeze_for_role(
    role: str,
    selector: str,
    x_dev: np.ndarray,
    y_dev: np.ndarray,
) -> tuple[pd.DataFrame, RoleConfig]:
    configs = build_stage_b_config_grid(selector)
    per_config: dict[tuple, list[dict[str, Any]]] = {}

    for cfg in configs:
        key = (cfg["k"], float(cfg["C"]), cfg["scaler"], cfg.get("n_neighbors"))
        per_config[key] = []
        for seed in SEEDS_PHASE2:
            from sklearn.model_selection import StratifiedKFold

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
    best = select_best_inner_config(config_rows)
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
    selected_local, _scores = select_features(
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


def run_freeze_lockbox(
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

