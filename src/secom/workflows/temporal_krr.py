"""Bounded DEV-only KRR contrast; separate from the LR roles and later block."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from secom.config import ReplicationMode, ScalerName, TEMPORAL_KRR_CALIBRATION_FRACTIONS
from secom.cv import chronological_inner_splits, fit_calibration_indices
from secom.metrics import binary_metrics_at_threshold, find_ber_optimal_threshold, roc_auc_or_default
from secom.selection.engine import fit_selector_pipeline
from secom.selection.tuning import select_ber_config
from secom.workflows.benchmark_common import (
    effective_classifier_config,
    fit_classifier_scores,
    tuned_classifier_param_grid,
)
from secom.workflows.benchmark_procedures import (
    CONFIG_COLUMNS,
    prediction_frame,
    prediction_metrics,
    procedure_config_key,
)
from secom.workflows.benchmark_tuned import _tuned_selector_param_grid
from secom.workflows.calibration import calibration_receipt

KRR_PROCEDURES = ("joint", "values_only", "values_and_indicators")
SEARCH_COLUMNS = [
    "fold",
    "calibration_fraction",
    *CONFIG_COLUMNS,
    "candidate_id",
    "inner_fold",
    "mean_inner_BER",
    "inner_BER",
    "inner_ROC_AUC",
    "threshold",
    "TP",
    "TN",
    "FP",
    "FN",
    "fit_sample_ids",
    "calibration_sample_ids",
    "validation_sample_ids",
    "inner_selected_width",
    "fit_end_timestamp",
    "calibration_start_timestamp",
    "calibration_end_timestamp",
    "validation_start_timestamp",
    "inner_timestamp_tie",
    "is_selected_joint",
    "is_selected_values_only",
    "is_selected_values_and_indicators",
]
PREDICTION_COLUMNS = [
    "sample_id",
    "fold",
    "y_true",
    "score",
    "threshold",
    "prediction",
    "procedure",
    "study",
    *CONFIG_COLUMNS,
    "config_id",
    "calibration_fraction",
    "timestamp",
    "fit_end_timestamp",
    "calibration_start_timestamp",
    "calibration_end_timestamp",
    "fit_n",
    "calibration_n",
]


def _prepared(x, y, train, evaluation, selector, mode, neighbors, fraction):
    fit, calibration = fit_calibration_indices(len(train), fraction)
    fit, calibration = train[fit], train[calibration]
    if any(len(np.unique(y[ids])) < 2 for ids in (fit, calibration)):
        raise ValueError("insufficient_classes_in_fit_or_calibration")
    selected, evaluated, *_ = fit_selector_pipeline(
        x_train_raw=x[fit],
        y_train=y[fit],
        x_eval_raw=x[np.concatenate((calibration, evaluation))],
        method=selector,
        k=40,
        scaler_name=ScalerName.STANDARD,
        add_indicator=mode == ReplicationMode.WITH_MISSING_INDICATORS,
        n_neighbors=neighbors,
    )
    return selected, evaluated, fit, calibration, evaluation


def _score(view, y, k, config):
    selected, evaluated, fit, calibration, evaluation = view
    selected, evaluated = selected[:, :k], evaluated[:, :k]
    effective = effective_classifier_config(
        {key: config[key] for key in ("alpha", "gamma_multiplier")}, selected.shape[1]
    )
    _, scores = fit_classifier_scores("krr", selected, y[fit], evaluated, effective, include_train_scores=False)
    cal_scores, eval_scores = scores[: len(calibration)], scores[len(calibration) :]
    threshold, _ = find_ber_optimal_threshold(y[calibration], cal_scores)
    metrics = binary_metrics_at_threshold(y[evaluation], eval_scores, threshold)
    return metrics, float(threshold), cal_scores, eval_scores, effective, selected.shape[1]


def run_krr_comparator(*, data, x_raw, folds, selectors, progress=None):
    """Select independently inside each fraction's FIT; reuse rankings and selected fits."""
    x, y = np.asarray(x_raw, dtype=float), data.y_bin.to_numpy(dtype=int)
    searches, predictions, metrics_rows, calibration_frames, diagnostics = [], [], [], [], []
    for fraction in TEMPORAL_KRR_CALIBRATION_FRACTIONS:
        for fold in folds:
            train, test = fold.train_index, fold.test_index
            prefix = f"krr_cal{round(100 * fraction)}"
            if progress:
                progress(f"temporal KRR comparator calibration={fraction:.0%} fold={fold.outer_fold}")
            try:
                fit, calibration = fit_calibration_indices(len(train), fraction)
                fit, calibration = train[fit], train[calibration]
                if any(len(np.unique(y[ids])) < 2 for ids in (fit, calibration)):
                    raise ValueError("insufficient_classes_in_fit_or_calibration")
                splits = chronological_inner_splits(y[fit], fraction)
            except ValueError as exc:
                metrics_rows.extend(
                    {
                        "fold": fold.outer_fold,
                        "calibration_fraction": fraction,
                        "procedure": f"{prefix}_{procedure}",
                        "available": False,
                        "unavailable_reason": str(exc),
                    }
                    for procedure in KRR_PROCEDURES
                )
                continue
            candidates, candidate_rows = [], {}
            for selector in selectors:
                for mode in (ReplicationMode.STRICT, ReplicationMode.WITH_MISSING_INDICATORS):
                    prepared = {}
                    for selector_config in _tuned_selector_param_grid(selector):
                        neighbors = selector_config["n_neighbors"]
                        if neighbors not in prepared:
                            prepared[neighbors] = [
                                _prepared(x, y, fit[tr], fit[val], selector, mode, neighbors, fraction)
                                for tr, val in splits
                            ]
                        for config in tuned_classifier_param_grid("krr"):
                            cfg = {
                                "selector": selector,
                                "classifier": "krr",
                                "replication_mode": mode,
                                "scaler": ScalerName.STANDARD,
                                **selector_config,
                                **config,
                                "C": np.nan,
                            }
                            records = []
                            for inner_fold, view in enumerate(prepared[neighbors], 1):
                                m, th, _, _, effective, width = _score(view, y, cfg["k"], config)
                                _, _, inner_fit, inner_cal, inner_val = view
                                records.append(
                                    {
                                        **cfg,
                                        "gamma": effective["gamma"],
                                        "inner_fold": inner_fold,
                                        "inner_selected_width": width,
                                        "inner_BER": m["BER"],
                                        "inner_ROC_AUC": m["ROC_AUC"],
                                        "threshold": th,
                                        "TP": int(m["lockbox_fails"] - m["FN"]),
                                        "TN": int(m["lockbox_n"] - m["lockbox_fails"] - m["FP"]),
                                        "FP": int(m["FP"]),
                                        "FN": int(m["FN"]),
                                        "fit_end_timestamp": data.loc[inner_fit].timestamp.iloc[-1],
                                        "calibration_start_timestamp": data.loc[inner_cal].timestamp.iloc[0],
                                        "calibration_end_timestamp": data.loc[inner_cal].timestamp.iloc[-1],
                                        "validation_start_timestamp": data.loc[inner_val].timestamp.iloc[0],
                                        "inner_timestamp_tie": bool(
                                            data.loc[inner_fit].timestamp.iloc[-1]
                                            == data.loc[inner_cal].timestamp.iloc[0]
                                            or data.loc[inner_cal].timestamp.iloc[-1]
                                            == data.loc[inner_val].timestamp.iloc[0]
                                        ),
                                        **{
                                            name: ",".join(str(v) for v in data.loc[ids].raw_row_id)
                                            for name, ids in (
                                                ("fit_sample_ids", inner_fit),
                                                ("calibration_sample_ids", inner_cal),
                                                ("validation_sample_ids", inner_val),
                                            )
                                        },
                                    }
                                )
                            cfg["gamma"] = records[0]["gamma"]
                            cfg["mean_inner_BER"] = float(np.mean([row["inner_BER"] for row in records]))
                            cfg["candidate_id"] = json.dumps(
                                {key: cfg.get(key) for key in CONFIG_COLUMNS if key != "gamma"}, sort_keys=True
                            )
                            candidates.append(cfg)
                            candidate_rows[cfg["candidate_id"]] = records
            winners = {}
            for procedure, mode in (
                ("joint", None),
                ("values_only", ReplicationMode.STRICT),
                ("values_and_indicators", ReplicationMode.WITH_MISSING_INDICATORS),
            ):
                winners[procedure] = select_ber_config(
                    [cfg for cfg in candidates if mode is None or cfg["replication_mode"] == mode],
                    simplicity_key=procedure_config_key,
                )
            for cfg in candidates:
                for row in candidate_rows[cfg["candidate_id"]]:
                    searches.append(
                        {
                            **row,
                            "fold": fold.outer_fold,
                            "calibration_fraction": fraction,
                            "mean_inner_BER": cfg["mean_inner_BER"],
                            "candidate_id": cfg["candidate_id"],
                            **{f"is_selected_{name}": winner is cfg for name, winner in winners.items()},
                        }
                    )
            outer_views, scored = {}, {}
            for procedure, cfg in winners.items():
                key = cfg["selector"], cfg["replication_mode"], cfg["n_neighbors"]
                if key not in outer_views:
                    outer_views[key] = _prepared(x, y, train, test, *key, fraction)
                if cfg["candidate_id"] not in scored:
                    scored[cfg["candidate_id"]] = _score(outer_views[key], y, cfg["k"], cfg)
                _, th, cal_scores, eval_scores, effective, width = scored[cfg["candidate_id"]]
                config = {**cfg, "gamma": effective["gamma"]}
                name = f"{prefix}_{procedure}"
                frame = prediction_frame(
                    sample_ids=data.loc[test].raw_row_id.to_numpy(),
                    fold=fold.outer_fold,
                    y_true=y[test],
                    scores=eval_scores,
                    threshold=th,
                    config=config,
                    procedure=name,
                    study="DEV_chronological_KRR_secondary",
                )
                cal_frame, diagnostic = calibration_receipt(
                    data=data,
                    fit_ids=fit,
                    calibration_ids=calibration,
                    scores=cal_scores,
                    threshold=th,
                    config=config,
                    procedure=name,
                    fold=fold.outer_fold,
                    calibration_fraction=fraction,
                    classifier="krr",
                )
                for column in (
                    "fit_end_timestamp",
                    "calibration_start_timestamp",
                    "calibration_end_timestamp",
                    "fit_n",
                ):
                    frame[column] = diagnostic[column]
                frame["timestamp"] = data.loc[test].timestamp.to_numpy()
                frame["calibration_n"] = len(calibration)
                frame["calibration_fraction"] = fraction
                predictions.append(frame)
                metrics_rows.append(
                    {
                        "fold": fold.outer_fold,
                        "procedure": name,
                        "calibration_fraction": fraction,
                        "available": True,
                        "unavailable_reason": "",
                        "candidate_id": cfg["candidate_id"],
                        "mean_inner_BER": cfg["mean_inner_BER"],
                        "n_selected_features": width,
                        "n_inner_timestamp_ties": sum(
                            row["inner_timestamp_tie"] for row in candidate_rows[cfg["candidate_id"]]
                        ),
                        "ROC_AUC": roc_auc_or_default(y[test], eval_scores),
                        **prediction_metrics(frame),
                    }
                )
                calibration_frames.append(cal_frame)
                diagnostics.append(diagnostic)
    return (
        pd.DataFrame(searches, columns=SEARCH_COLUMNS),
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(columns=PREDICTION_COLUMNS),
        pd.DataFrame(metrics_rows),
        calibration_frames,
        diagnostics,
    )
