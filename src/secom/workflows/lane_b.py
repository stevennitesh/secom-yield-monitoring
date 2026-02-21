from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from secom.artifacts import ensure_reports_dir, write_csv
from secom.config import ArtifactName, SEEDS_STAGE_B, ScalerName, SelectorName, ThresholdPolicy
from secom.cv import to_time_window_string
from secom.metrics import binary_metrics_at_threshold, find_ber_optimal_threshold, safe_std
from secom.models import fit_lane_b_classifier
from secom.preprocess import build_feature_universe, local_to_global_feature_indices
from secom.selection.engine import fit_selector_pipeline
from secom.selection.tuning import select_best_inner_config
from secom.types import DataBundle


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
    clf = fit_lane_b_classifier(x_train_sel, y_train, c_value=c_value)
    train_scores = clf.predict_proba(x_train_sel)[:, 1]
    eval_scores = clf.predict_proba(x_eval_sel)[:, 1]
    threshold, _ = find_ber_optimal_threshold(y_train, train_scores)
    metrics = binary_metrics_at_threshold(y_eval, eval_scores, threshold)
    return metrics, threshold, feature_meta, selected_local, clf, imputer, scaler


def _stage_a_configs() -> list[dict[str, Any]]:
    return [
        {
            "selector": s,
            "k": 40,
            "C": 1.0,
            "scaler": ScalerName.ROBUST,
            "n_neighbors": 10 if s == SelectorName.RELIEFF else None,
        }
        for s in SelectorName.ACTIVE
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

        x_train_sel, x_val_sel, _meta, _sel, _imp, _scaler = fit_selector_pipeline(
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


def run_lane_b_stage_ab(bundle: DataBundle, output_dir: Path) -> dict[str, Any]:
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
            metrics, threshold, _meta, _sel, _clf, _imp, _scl = _fit_eval_with_labels(
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
        configs = build_stage_b_config_grid(selector)
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

                best = select_best_inner_config(config_scores)
                for row in config_scores:
                    row["is_selected_config"] = (
                        row["k"] == best["k"]
                        and np.isclose(row["C"], best["C"])
                        and row["scaler"] == best["scaler"]
                        and row.get("n_neighbors") == best.get("n_neighbors")
                    )
                    stage_b_inner_rows.append(row)

                metrics, threshold, meta, selected_local, _clf, _imp, _scl = _fit_eval_with_labels(
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

    def _selector_rank_key(row: dict[str, Any]) -> tuple[float, float, float, int, float, int, float, float, float, str]:
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

    ranked = sorted(model_sel.to_dict("records"), key=_selector_rank_key) # type: ignore
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

