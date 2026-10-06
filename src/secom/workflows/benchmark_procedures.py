"""Joint inner-selected procedures, simple baselines, and held-out lineage."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from secom.config import ReplicationMode
from secom.metrics import binary_metrics_at_threshold, find_ber_optimal_threshold, safe_std
from secom.models import make_benchmark_logreg_model
from secom.selection.tuning import select_ber_config
from secom.workflows.benchmark_common import BENCHMARK_METRICS, benchmark_metric_fields, config_tie_break_key

PROCEDURES = ("joint", "values_only", "values_and_indicators", "missingness_only", "all_pass")
CONFIG_COLUMNS = (
    "selector",
    "classifier",
    "replication_mode",
    "k",
    "C",
    "alpha",
    "gamma",
    "gamma_multiplier",
    "n_neighbors",
    "scaler",
)


def procedure_config_key(row: dict[str, Any]) -> tuple:
    """Shared family/joint simplicity key, then deterministic family/input order."""
    return (
        *config_tie_break_key(str(row["selector"]), str(row["classifier"]), row, row),
        str(row["selector"]),
        str(row["classifier"]),
        str(row["replication_mode"]),
    )


def selected_candidate(candidates: list[tuple[dict, dict, np.ndarray]], mode: str | None = None) -> tuple:
    """Choose from inner scores only; the outer metric payload is never consulted."""
    eligible = [item for item in candidates if mode is None or item[0]["replication_mode"] == mode]
    winner = select_ber_config([item[0] for item in eligible], simplicity_key=procedure_config_key)
    return next(item for item in eligible if item[0] is winner)


def missingness_scores(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """Balanced C=1 logistic regression on original-column masks, fitted on train only."""
    train_mask = np.isnan(x_train).astype(float)
    eval_mask = np.isnan(x_eval).astype(float)
    varying = np.ptp(train_mask, axis=0) > 0
    if not np.any(varying):
        return np.full(len(x_eval), 0.5)
    scaler = StandardScaler().fit(train_mask[:, varying])
    model = make_benchmark_logreg_model(c_value=1.0)
    model.fit(scaler.transform(train_mask[:, varying]), y_train)
    return model.predict_proba(scaler.transform(eval_mask[:, varying]))[:, 1]


def missingness_baseline(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> tuple[np.ndarray, float]:
    """Calibrate a frozen threshold using masks transformed within each inner split."""
    scores_oof = np.empty(len(y_train))
    n_splits = min(3, np.bincount(y_train, minlength=2).min())
    cv = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=42)
    for train, valid in cv.split(x_train, y_train):
        scores_oof[valid] = missingness_scores(x_train[train], y_train[train], x_train[valid])
    threshold, _ = find_ber_optimal_threshold(y_train, scores_oof)
    return missingness_scores(x_train, y_train, x_eval), float(threshold)


def prediction_frame(
    *, sample_ids, fold: int, y_true, scores, threshold: float, config: dict, procedure: str, study: str
) -> pd.DataFrame:
    """Compact raw-ID held-out prediction receipt; no matrices or fitted models."""
    normalized = {
        name: None if config.get(name) is None or pd.isna(config.get(name)) else config[name] for name in CONFIG_COLUMNS
    }
    return pd.DataFrame(
        {
            "sample_id": sample_ids,
            "fold": fold,
            "y_true": y_true,
            "score": scores,
            "threshold": threshold,
            "prediction": (scores >= threshold).astype(int),
            "procedure": procedure,
            "study": study,
            **normalized,
            "config_id": json.dumps(normalized, sort_keys=True, separators=(",", ":")),
        }
    )


def prediction_metrics(frame: pd.DataFrame) -> dict:
    """Confusion counts from per-example frozen predictions, never a pooled threshold."""
    metrics = binary_metrics_at_threshold(frame.y_true.to_numpy(), frame.prediction.to_numpy(), 0.5)
    return {
        "BER": metrics["BER"],
        "True+": metrics["True+"],
        "True-": metrics["True-"],
        "TP": int(metrics["lockbox_fails"] - metrics["FN"]),
        "TN": int(metrics["lockbox_n"] - metrics["lockbox_fails"] - metrics["FP"]),
        "FP": int(metrics["FP"]),
        "FN": int(metrics["FN"]),
        "n_test": len(frame),
        "n_test_fails": int(frame.y_true.sum()),
        "TPR_available": bool(frame.y_true.sum() > 0),
        "TNR_available": bool(frame.y_true.sum() < len(frame)),
        "BER_available": bool(0 < frame.y_true.sum() < len(frame)),
    }


def procedure_summary(folds: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Descriptive fold spread and pooled held-out confusion counts."""
    rows = []
    for procedure, frame in folds.groupby("procedure", sort=False):
        row = {"procedure": procedure, "n_folds": len(frame)}
        for metric in ("BER", "True+", "True-"):
            values = frame[metric].to_numpy(dtype=float)
            row.update(
                {
                    f"mean_{metric}": float(values.mean()),
                    f"std_{metric}": safe_std(values),
                    f"min_{metric}": float(values.min()),
                    f"max_{metric}": float(values.max()),
                }
            )
        pooled = prediction_metrics(predictions[predictions.procedure == procedure])
        row.update({f"pooled_{key}": value for key, value in pooled.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def build_procedure_artifacts(*, x, y, folds, sample_ids, candidates_by_fold, study):
    """Reuse family outer fits for joint procedures; fit only the independent simple baseline."""
    predictions, metrics_rows = [], []
    for fold, (train, test) in enumerate(folds, start=1):
        candidates = candidates_by_fold[fold]
        for procedure, mode in (
            ("joint", None),
            ("values_only", ReplicationMode.STRICT),
            ("values_and_indicators", ReplicationMode.WITH_MISSING_INDICATORS),
        ):
            best, metrics, scores = selected_candidate(candidates, mode)
            frame = prediction_frame(
                sample_ids=sample_ids[test],
                fold=fold,
                y_true=y[test],
                scores=scores,
                threshold=best["threshold_inner_oof"],
                config=best,
                procedure=procedure,
                study=study,
            )
            predictions.append(frame)
            metrics_rows.append(
                {
                    "procedure": procedure,
                    "fold": fold,
                    **{key: best.get(key) for key in CONFIG_COLUMNS},
                    "mean_inner_BER": best["mean_inner_BER"],
                    "threshold_inner_oof": best["threshold_inner_oof"],
                    **benchmark_metric_fields(metrics),
                    **prediction_metrics(frame),
                }
            )
        scores, threshold = missingness_baseline(x[train], y[train], x[test])
        for procedure, score, th in (("missingness_only", scores, threshold), ("all_pass", np.zeros(len(test)), 1.0)):
            config = {
                "selector": "none",
                "classifier": "logreg" if procedure == "missingness_only" else "constant",
                "replication_mode": procedure,
                "k": 0,
                "C": 1.0 if procedure == "missingness_only" else None,
                "scaler": "StandardScaler" if procedure == "missingness_only" else None,
            }
            frame = prediction_frame(
                sample_ids=sample_ids[test],
                fold=fold,
                y_true=y[test],
                scores=score,
                threshold=th,
                config=config,
                procedure=procedure,
                study=study,
            )
            predictions.append(frame)
            metrics_rows.append(
                {
                    "procedure": procedure,
                    "fold": fold,
                    **config,
                    "threshold_inner_oof": th,
                    **{key: float("nan") for key in BENCHMARK_METRICS},
                    **prediction_metrics(frame),
                }
            )
    prediction_df, folds_df = pd.concat(predictions, ignore_index=True), pd.DataFrame(metrics_rows)
    return prediction_df, folds_df, procedure_summary(folds_df, prediction_df)


def validate_prediction_lineage(predictions: pd.DataFrame, folds: pd.DataFrame, *, temporal: bool = False) -> None:
    """Recompute held-out counts and reject duplicate IDs, thresholds, or changed predictions."""
    required = {"sample_id", "fold", "y_true", "score", "threshold", "prediction", "procedure", "config_id"}
    if not required.issubset(predictions.columns):
        raise ValueError("held-out predictions: missing required lineage columns")
    if predictions.empty or folds.empty:
        raise ValueError("held-out predictions: empty active procedure")
    ids = pd.to_numeric(predictions.sample_id, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(ids).all() or (ids < 0).any() or not np.equal(ids, np.floor(ids)).all():
        raise ValueError("held-out predictions: sample IDs must be nonnegative integers")
    if predictions.duplicated(["procedure", "sample_id"]).any():
        raise ValueError("held-out predictions: duplicate evaluation sample IDs")
    if temporal:
        columns = ("timestamp", "fit_end_timestamp", "calibration_start_timestamp", "calibration_end_timestamp")
        dates = {column: pd.to_datetime(predictions[column], errors="raise") for column in columns}
        if not (
            (dates["fit_end_timestamp"] <= dates["calibration_start_timestamp"])
            & (dates["calibration_start_timestamp"] <= dates["calibration_end_timestamp"])
            & (dates["calibration_end_timestamp"] < dates["timestamp"])
        ).all():
            raise ValueError("temporal predictions: FIT/calibration/evaluation chronology violated")
        periods = predictions.groupby("fold")["timestamp"].agg(["min", "max"]).sort_index()
        if len(periods) > 1 and not (periods["max"].iloc[:-1].to_numpy() < periods["min"].iloc[1:].to_numpy()).all():
            raise ValueError("temporal predictions: overlapping calendar evaluation periods")
    if not predictions.y_true.isin([0, 1]).all() or not predictions.prediction.isin([0, 1]).all():
        raise ValueError("held-out predictions: labels and predictions must be binary")
    if not np.isfinite(predictions.score.to_numpy(dtype=float)).all() or predictions.threshold.isna().any():
        raise ValueError("held-out predictions: invalid score or threshold")
    if not np.array_equal(predictions.prediction, (predictions.score >= predictions.threshold).astype(int)):
        raise ValueError("held-out predictions: frozen-threshold prediction mismatch")
    expected_procedures = {"temporal_joint"} if temporal else set(PROCEDURES)
    if set(predictions.procedure) != expected_procedures or set(folds.procedure) != expected_procedures:
        raise ValueError("held-out predictions: incomplete procedure coverage")
    keys = ["procedure", "fold"]
    if folds.duplicated(keys).any() or set(map(tuple, folds[keys].to_numpy())) != set(
        map(tuple, predictions[keys].drop_duplicates().to_numpy())
    ):
        raise ValueError("held-out predictions: inconsistent fold coverage")
    identities = {}
    fold_identities = {}
    for (procedure, fold), frame in predictions.groupby(keys):
        if frame.threshold.nunique(dropna=False) != 1 or frame.config_id.nunique() != 1:
            raise ValueError("held-out predictions: mixed config/threshold within a fold")
        identities.setdefault(procedure, set()).update(frame.sample_id)
        fold_identities[(procedure, fold)] = set(frame.sample_id)
        row = folds[(folds.procedure == procedure) & (folds.fold == fold)].iloc[0]
        for key, value in prediction_metrics(frame).items():
            if key not in row or not np.isclose(float(row[key]), value, rtol=0, atol=1e-9):
                raise ValueError(f"held-out predictions: {procedure} fold {fold} {key} mismatch")
        if not temporal:
            for name in CONFIG_COLUMNS:
                if (
                    name in row
                    and name in frame
                    and not (pd.isna(row[name]) and frame[name].isna().all())
                    and not (frame[name] == row[name]).all()
                ):
                    raise ValueError("held-out predictions: fold config mismatch")
            if "threshold_inner_oof" in row and not np.isclose(
                float(row.threshold_inner_oof), float(frame.threshold.iloc[0])
            ):
                raise ValueError("held-out predictions: fold frozen threshold mismatch")
            for config_row in frame[list(CONFIG_COLUMNS) + ["config_id"]].drop_duplicates().to_dict("records"):
                expected = {name: None if pd.isna(config_row[name]) else config_row[name] for name in CONFIG_COLUMNS}
                actual = json.loads(config_row["config_id"])
                if set(actual) != set(expected) or any(
                    not np.isclose(actual[name], expected[name], rtol=0, atol=1e-14)
                    if isinstance(expected[name], (int, float)) and isinstance(actual.get(name), (int, float))
                    else actual.get(name) != expected[name]
                    for name in expected
                ):
                    raise ValueError("held-out predictions: config identity mismatch")
    if not temporal:
        if any(ids != identities["joint"] for ids in identities.values()):
            raise ValueError("held-out predictions: baselines and procedures have different sample IDs")
        if any(ids != fold_identities[("joint", fold)] for (_procedure, fold), ids in fold_identities.items()):
            raise ValueError("held-out predictions: baseline/procedure fold sample pairing mismatch")
        if set(predictions.fold) != set(range(1, 11)):
            raise ValueError("held-out predictions: expected 10 benchmark folds")
