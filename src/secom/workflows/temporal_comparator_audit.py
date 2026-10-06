"""Recompute comparator selection, disjoint lineage and calibration diagnostics."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from secom.config import ArtifactName, ReplicationMode, TEMPORAL_KRR_CALIBRATION_FRACTIONS
from secom.metrics import find_ber_optimal_threshold, roc_auc_or_default
from secom.selection.tuning import select_ber_config
from secom.workflows.benchmark_procedures import CONFIG_COLUMNS, prediction_metrics, procedure_config_key
from secom.workflows.calibration import calibration_diagnostics
from secom.workflows.temporal_krr import KRR_PROCEDURES, SEARCH_COLUMNS, PREDICTION_COLUMNS
from secom.workflows.benchmark_common import tuned_classifier_param_grid
from secom.workflows.benchmark_tuned import _tuned_selector_param_grid


def _candidate_identity(config):
    return json.dumps({key: config.get(key) for key in CONFIG_COLUMNS if key != "gamma"}, sort_keys=True)


def _ids(value) -> set[int]:
    return {int(v) for v in str(value).split(",") if v}


def _same(actual, expected):
    if isinstance(expected, str):
        return actual == expected
    return bool(np.isclose(float(actual), float(expected), rtol=0, atol=1e-9, equal_nan=True))


def validate_temporal_comparator(frames: dict, warnings: list[str]) -> None:
    """Schema/count/hash failures are hard errors; unavailable paths and sparse calibration warn."""
    search = frames[ArtifactName.TEMPORAL_KRR_SEARCH]
    predictions = frames[ArtifactName.TEMPORAL_KRR_PREDICTIONS]
    metrics = frames[ArtifactName.TEMPORAL_KRR_METRICS]
    scores = frames[ArtifactName.TEMPORAL_CALIBRATION_SCORES]
    diagnostics = frames[ArtifactName.TEMPORAL_CALIBRATION_DIAGNOSTICS]
    for name, frame, required in (
        ("KRR search", search, SEARCH_COLUMNS),
        ("KRR predictions", predictions, PREDICTION_COLUMNS),
        ("KRR metrics", metrics, ["fold", "procedure", "available", "calibration_fraction", "unavailable_reason"]),
        (
            "calibration scores",
            scores,
            [
                "sample_id",
                "fold",
                "procedure",
                "y_true",
                "score",
                "threshold",
                "config_id",
                "fit_sample_ids",
                "timestamp",
                "fit_end_timestamp",
                "calibration_start_timestamp",
                "calibration_end_timestamp",
                "calibration_fraction",
                "fit_n",
                "fit_fails",
                "fit_calibration_timestamp_tie",
            ],
        ),
        (
            "calibration diagnostics",
            diagnostics,
            [
                "procedure",
                "fold",
                "calibration_n",
                "calibration_fails",
                "calibration_passes",
                "threshold",
                "lofo_available",
                "fragile_calibration",
                "config_id",
                "diagnostic_semantics",
            ],
        ),
    ):
        if not set(required).issubset(frame.columns):
            raise ValueError(f"{name}: missing required schema columns")
    lr_predictions = frames[ArtifactName.TEMPORAL_PREDICTIONS]
    expected_keys = {
        (int(fold), f"krr_cal{round(fraction * 100)}_{procedure}")
        for fold in lr_predictions.fold.unique()
        for fraction in TEMPORAL_KRR_CALIBRATION_FRACTIONS
        for procedure in KRR_PROCEDURES
    }
    if metrics.duplicated(["fold", "procedure"]).any() or set(zip(metrics.fold, metrics.procedure)) != expected_keys:
        raise ValueError("KRR metrics: incomplete fixed period/procedure coverage")
    if predictions.duplicated(["procedure", "sample_id"]).any():
        raise ValueError("KRR predictions: duplicate evaluation IDs")
    if (
        not predictions.y_true.isin([0, 1]).all()
        or not np.isfinite(pd.to_numeric(predictions.score, errors="coerce").to_numpy(dtype=float)).all()
    ):
        raise ValueError("KRR predictions: invalid labels/scores")
    if not np.array_equal(predictions.prediction, (predictions.score >= predictions.threshold).astype(int)):
        raise ValueError("KRR predictions: frozen threshold mismatch")
    expected_calibration = {(int(fold), "temporal_joint") for fold in lr_predictions.fold.unique()}
    final_roles = frames[ArtifactName.TEMPORAL_FREEZE].role.unique()
    expected_calibration.update((0, f"logreg_final_{role}") for role in final_roles)
    objective_cache = {}
    path_lineage = {}
    for row in metrics.to_dict("records"):
        fold, procedure = int(row["fold"]), row["procedure"]
        frame = predictions[(predictions.fold == fold) & (predictions.procedure == procedure)]
        if not row["available"]:
            if not frame.empty or not isinstance(row["unavailable_reason"], str) or not row["unavailable_reason"]:
                raise ValueError("KRR unavailable path: missing reason or unexpected predictions")
            warnings.append(f"KRR comparator {procedure} period {fold} unavailable: {row['unavailable_reason']}")
            continue
        expected_calibration.add((fold, procedure))
        reference = lr_predictions[lr_predictions.fold == fold]
        if set(zip(frame.sample_id, frame.y_true)) != set(zip(reference.sample_id, reference.y_true)):
            raise ValueError("KRR comparator: evaluation IDs/labels differ from fixed LR periods")
        if frame.config_id.nunique() != 1 or frame.threshold.nunique(dropna=False) != 1:
            raise ValueError("KRR comparator: mixed config/threshold")
        for key, value in {
            **prediction_metrics(frame),
            "ROC_AUC": roc_auc_or_default(frame.y_true.to_numpy(), frame.score.to_numpy()),
        }.items():
            if key not in row or not _same(row[key], value):
                raise ValueError(f"KRR comparator: recomputed {key} mismatch")
        fraction = float(row["calibration_fraction"])
        if fraction not in TEMPORAL_KRR_CALIBRATION_FRACTIONS or not procedure.startswith(
            f"krr_cal{round(100 * fraction)}_"
        ):
            raise ValueError("KRR comparator: procedure/fraction identity mismatch")
        candidates = search[(search.fold == fold) & (search.calibration_fraction == fraction)]
        expected_candidates = set()
        for selector in frames[ArtifactName.TEMPORAL_SELECTOR_SCREENING].selector.unique():
            for mode in (ReplicationMode.STRICT, ReplicationMode.WITH_MISSING_INDICATORS):
                for selector_config in _tuned_selector_param_grid(selector):
                    for classifier_config in tuned_classifier_param_grid("krr"):
                        expected_candidates.add(
                            _candidate_identity(
                                {
                                    "selector": selector,
                                    "classifier": "krr",
                                    "replication_mode": mode,
                                    "scaler": "StandardScaler",
                                    "C": np.nan,
                                    **selector_config,
                                    **classifier_config,
                                }
                            )
                        )
        if set(candidates.candidate_id) != expected_candidates:
            raise ValueError("KRR search: incomplete declared candidate grid")
        objective_rows = []
        outer_cal = scores[(scores.fold == fold) & (scores.procedure == procedure)]
        forbidden = set(outer_cal.sample_id) | set(reference.sample_id)
        allowed = _ids(outer_cal.fit_sample_ids.iloc[0])
        path_key = (fold, fraction)
        lineage = (frozenset(allowed), frozenset(forbidden))
        if path_key in path_lineage and path_lineage[path_key] != lineage:
            raise ValueError("KRR calibration: procedures in a fraction use different train/calibration IDs")
        path_lineage[path_key] = lineage
        if path_key not in objective_cache:
            for candidate_id, group in candidates.groupby("candidate_id"):
                if len(group) < 2 or group.inner_fold.duplicated().any():
                    raise ValueError("KRR search: requires two or more distinct chronological inner periods")
                for inner in group.to_dict("records"):
                    fit_ids, cal_ids, val_ids = (
                        _ids(inner[key])
                        for key in ("fit_sample_ids", "calibration_sample_ids", "validation_sample_ids")
                    )
                    if fit_ids & cal_ids or (fit_ids | cal_ids) & val_ids or (fit_ids | cal_ids | val_ids) & forbidden:
                        raise ValueError("KRR search: calibration/evaluation labels entered selected FIT search")
                    if not (fit_ids | cal_ids | val_ids).issubset(allowed):
                        raise ValueError("KRR search: inner labels lie outside outer FIT")
                    if len(cal_ids) != len(fit_ids | cal_ids) - int(np.floor((1 - fraction) * len(fit_ids | cal_ids))):
                        raise ValueError("KRR search: inner calibration fraction mismatch")
                    if not (
                        pd.Timestamp(inner["fit_end_timestamp"])
                        <= pd.Timestamp(inner["calibration_start_timestamp"])
                        <= pd.Timestamp(inner["calibration_end_timestamp"])
                        <= pd.Timestamp(inner["validation_start_timestamp"])
                    ):
                        raise ValueError("KRR search: inner chronology violated")
                    tp, tn, fp, fn = (float(inner[c]) for c in ("TP", "TN", "FP", "FN"))
                    if min(tp, tn, fp, fn) < 0 or tp + fn <= 0 or tn + fp <= 0:
                        raise ValueError("KRR search: invalid inner confusion counts")
                    if tp + tn + fp + fn != len(val_ids) or any(v != int(v) for v in (tp, tn, fp, fn)):
                        raise ValueError("KRR search: validation count mismatch")
                    tied = pd.Timestamp(inner["fit_end_timestamp"]) == pd.Timestamp(
                        inner["calibration_start_timestamp"]
                    ) or pd.Timestamp(inner["calibration_end_timestamp"]) == pd.Timestamp(
                        inner["validation_start_timestamp"]
                    )
                    if bool(inner["inner_timestamp_tie"]) != tied:
                        raise ValueError("KRR search: timestamp tie flag mismatch")
                    ber = 0.5 * (fn / (tp + fn) + fp / (tn + fp))
                    if not _same(inner["inner_BER"], ber) or not _same(
                        inner["gamma"], inner["gamma_multiplier"] / inner["inner_selected_width"]
                    ):
                        raise ValueError("KRR search: inner metric/effective gamma mismatch")
                mean = float(group.inner_BER.mean())
                if not np.allclose(group.mean_inner_BER, mean, rtol=0, atol=1e-9):
                    raise ValueError("KRR search: mean objective does not recompute")
                objective_rows.append({**group.iloc[0].to_dict(), "candidate_id": candidate_id})
            objective_cache[path_key] = objective_rows
        objective_rows = objective_cache[path_key]
        mode = (
            None
            if procedure.endswith("_joint")
            else (
                ReplicationMode.STRICT
                if procedure.endswith("_values_only")
                else ReplicationMode.WITH_MISSING_INDICATORS
            )
        )
        best = select_ber_config(
            [cfg for cfg in objective_rows if mode is None or cfg["replication_mode"] == mode],
            simplicity_key=procedure_config_key,
        )
        if best["candidate_id"] != row["candidate_id"] or not _same(best["mean_inner_BER"], row["mean_inner_BER"]):
            raise ValueError("KRR comparator: selected config must minimize chronological inner BER")
        selected_search = candidates[candidates.candidate_id == best["candidate_id"]]
        if int(row["n_inner_timestamp_ties"]) != int(selected_search.inner_timestamp_tie.sum()):
            raise ValueError("KRR comparator: selected inner timestamp tie count mismatch")
        selected_name = "is_selected_" + (
            "joint" if mode is None else "values_only" if mode == ReplicationMode.STRICT else "values_and_indicators"
        )
        if set(candidates.loc[candidates[selected_name].astype(bool), "candidate_id"]) != {best["candidate_id"]}:
            raise ValueError("KRR comparator: selected flags disagree with inner BER selection")
        config = json.loads(frame.config_id.iloc[0])
        for key in CONFIG_COLUMNS:
            expected = None if pd.isna(frame[key].iloc[0]) else frame[key].iloc[0]
            if not (_same(config[key], expected) if isinstance(expected, (float, int)) else config[key] == expected):
                raise ValueError("KRR predictions: config identity mismatch")
            if key != "gamma" and (None if pd.isna(best[key]) else best[key]) != config[key]:
                raise ValueError("KRR predictions: selected search config mismatch")
        if not _same(config["gamma"], config["gamma_multiplier"] / row["n_selected_features"]):
            raise ValueError("KRR predictions: outer effective gamma mismatch")
        if not (pd.to_datetime(frame.calibration_end_timestamp) < pd.to_datetime(frame.timestamp)).all():
            raise ValueError("KRR predictions: calibration overlaps evaluation")
        for key in (
            "config_id",
            "threshold",
            "fit_n",
            "calibration_n",
            "calibration_fraction",
            "fit_end_timestamp",
            "calibration_start_timestamp",
            "calibration_end_timestamp",
        ):
            value = outer_cal[key].iloc[0] if key != "calibration_n" else len(outer_cal)
            if not (frame[key] == value).all():
                raise ValueError("KRR predictions: calibration receipt mismatch")
    if (
        scores.duplicated(["procedure", "fold", "sample_id"]).any()
        or diagnostics.duplicated(["procedure", "fold"]).any()
    ):
        raise ValueError("calibration receipts: duplicate identities")
    if (
        set(zip(diagnostics.fold, diagnostics.procedure)) != expected_calibration
        or set(zip(scores.fold, scores.procedure)) != expected_calibration
    ):
        raise ValueError("calibration receipts: incomplete selected LR/KRR coverage")
    for (fold, procedure), group in scores.groupby(["fold", "procedure"]):
        row = diagnostics[(diagnostics.fold == fold) & (diagnostics.procedure == procedure)].iloc[0]
        if (
            group.threshold.nunique(dropna=False) != 1
            or group.config_id.nunique() != 1
            or row.config_id != group.config_id.iloc[0]
        ):
            raise ValueError("calibration receipts: mixed or inconsistent config/threshold")
        if not group.y_true.isin([0, 1]).all() or not np.isfinite(group.score).all():
            raise ValueError("calibration receipts: invalid labels/scores")
        config = json.loads(row.config_id)
        if procedure == "temporal_joint":
            emitted = lr_predictions[lr_predictions.fold == fold].iloc[0]
            if emitted.config_id != row.config_id or not _same(emitted.threshold, row.threshold):
                raise ValueError("LR outer calibration: selected model/threshold mismatch")
        elif procedure.startswith("logreg_final_"):
            role = procedure.removeprefix("logreg_final_")
            frozen = frames[ArtifactName.TEMPORAL_FREEZE]
            selected = frozen[(frozen.role == role) & frozen.is_frozen_config.astype(bool)].iloc[0]
            for key in ("selector", "k", "C", "scaler", "n_neighbors"):
                value = None if pd.isna(selected[key]) else selected[key]
                if config[key] != value:
                    raise ValueError("LR final calibration: frozen role config mismatch")
            lock = frames[ArtifactName.TEMPORAL_LOCKBOX]
            lock = lock[(lock.role == role) & (lock.threshold_policy == "scientific")]
            if not np.allclose(lock["threshold_value"], row.threshold, rtol=0, atol=1e-9):
                raise ValueError("LR final calibration: frozen role threshold mismatch")
        fit_ids = _ids(group.fit_sample_ids.iloc[0])
        if len(fit_ids) != int(row.fit_n) or fit_ids & set(group.sample_id):
            raise ValueError("calibration receipts: fit/calibration IDs overlap or count mismatch")
        if procedure == "temporal_joint":
            reference = lr_predictions[lr_predictions.fold == fold]
            if (fit_ids | set(group.sample_id)) & set(reference.sample_id):
                raise ValueError("LR outer calibration: FIT/calibration IDs overlap evaluation")
            if not (
                pd.Timestamp(row["fit_end_timestamp"])
                <= pd.Timestamp(row["calibration_start_timestamp"])
                <= pd.Timestamp(row["calibration_end_timestamp"])
                < pd.to_datetime(reference["timestamp"]).min()
            ):
                raise ValueError("LR outer calibration: FIT/calibration chronology overlaps evaluation")
            for key in ("fit_end_timestamp", "calibration_start_timestamp", "calibration_end_timestamp"):
                if not (pd.to_datetime(reference[key]) == pd.Timestamp(row[key])).all():
                    raise ValueError("LR outer calibration: prediction boundary receipt mismatch")
        if not (pd.to_datetime(group.fit_end_timestamp) <= pd.to_datetime(group.timestamp)).all():
            raise ValueError("calibration receipts: chronology mismatch")
        for key in (
            "fit_n",
            "fit_fails",
            "fit_calibration_timestamp_tie",
            "fit_sample_ids",
            "fit_start_timestamp",
            "fit_end_timestamp",
            "calibration_fraction",
            "calibration_start_timestamp",
            "calibration_end_timestamp",
        ):
            if group[key].nunique(dropna=False) != 1 or not _same(group[key].iloc[0], row[key]):
                raise ValueError("calibration receipts: inconsistent boundary/count metadata")
        if pd.to_datetime(group.timestamp).min() != pd.Timestamp(row.calibration_start_timestamp) or pd.to_datetime(
            group.timestamp
        ).max() != pd.Timestamp(row.calibration_end_timestamp):
            raise ValueError("calibration receipts: timestamp bounds mismatch")
        if len(group) != len(group) + int(row.fit_n) - int(
            np.floor((1 - float(row.calibration_fraction)) * (len(group) + int(row.fit_n)))
        ):
            raise ValueError("calibration receipts: declared fraction/count mismatch")
        if bool(row.fit_calibration_timestamp_tie) != (
            pd.Timestamp(row.fit_end_timestamp) == pd.Timestamp(row.calibration_start_timestamp)
        ):
            raise ValueError("calibration receipts: timestamp tie flag mismatch")
        th, _ = find_ber_optimal_threshold(group.y_true.to_numpy(), group.score.to_numpy())
        if not _same(group.threshold.iloc[0], th):
            raise ValueError("calibration receipts: threshold must be selected only on held-out scores")
        for key, value in calibration_diagnostics(group.y_true, group.score, th).items():
            if key not in row or not _same(row[key], value):
                raise ValueError(f"calibration diagnostics: recomputed {key} mismatch")
        if row.fragile_calibration:
            warnings.append(f"{procedure} period {fold}: fewer than ten calibration failures; threshold is fragile")
