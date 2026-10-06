"""Descriptive fixed-model calibration fragility, with recomputable score receipts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from secom.metrics import find_ber_optimal_threshold


def calibration_diagnostics(labels, scores, threshold: float) -> dict:
    """LOFO recalibrates scores only; ranges are neither CIs nor future uncertainty."""
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    failures, passes = int(labels.sum()), int(len(labels) - labels.sum())
    thresholds, fractions = [], []
    if failures > 1 and passes > 0:
        for index in np.flatnonzero(labels == 1):
            keep = np.arange(len(labels)) != index
            candidate, _ = find_ber_optimal_threshold(labels[keep], scores[keep])
            thresholds.append(float(candidate))
            # Evaluate every recalibrated threshold on the SAME full calibration score set.
            fractions.append(float(np.mean(scores >= candidate)))
    return {
        "calibration_n": len(labels),
        "calibration_fails": failures,
        "calibration_passes": passes,
        "BER_step_failure": 0.5 / failures if failures else np.nan,
        "BER_step_pass": 0.5 / passes if passes else np.nan,
        "threshold": float(threshold),
        "calibration_flagged_fraction": float(np.mean(scores >= threshold)),
        "fragile_calibration": failures < 10,
        "lofo_available": bool(thresholds),
        "lofo_threshold_min": min(thresholds) if thresholds else np.nan,
        "lofo_threshold_max": max(thresholds) if thresholds else np.nan,
        "lofo_flagged_fraction_min": min(fractions) if fractions else np.nan,
        "lofo_flagged_fraction_max": max(fractions) if fractions else np.nan,
        "diagnostic_semantics": "fixed_model_calibration_instability_only_not_CI",
    }


def calibration_receipt(
    *,
    data,
    fit_ids,
    calibration_ids,
    scores,
    threshold,
    config,
    procedure,
    fold,
    calibration_fraction=0.20,
    classifier="logreg",
) -> tuple[pd.DataFrame, dict]:
    """Record exact chronological sample lineage and the retained-model score vector."""
    from secom.workflows.benchmark_procedures import prediction_frame

    fit_data, cal_data = data.loc[fit_ids], data.loc[calibration_ids]
    frame = prediction_frame(
        sample_ids=cal_data.raw_row_id.to_numpy(),
        fold=fold,
        y_true=cal_data.y_bin.to_numpy(),
        scores=scores,
        threshold=threshold,
        config=config,
        procedure=procedure,
        study="held_out_DEV_calibration",
    )
    lineage = {
        "calibration_fraction": calibration_fraction,
        "classifier": classifier,
        "fit_calibration_timestamp_tie": bool(fit_data.timestamp.iloc[-1] == cal_data.timestamp.iloc[0]),
        "fit_n": len(fit_ids),
        "fit_fails": int(fit_data.y_bin.sum()),
        "fit_start_timestamp": fit_data.timestamp.iloc[0],
        "fit_end_timestamp": fit_data.timestamp.iloc[-1],
        "calibration_start_timestamp": cal_data.timestamp.iloc[0],
        "calibration_end_timestamp": cal_data.timestamp.iloc[-1],
        "fit_sample_ids": ",".join(str(v) for v in fit_data.raw_row_id),
    }
    for key, value in lineage.items():
        frame[key] = value
    frame["timestamp"] = cal_data.timestamp.to_numpy()
    row = {
        "procedure": procedure,
        "fold": fold,
        "config_id": frame.config_id.iloc[0],
        **lineage,
        **calibration_diagnostics(cal_data.y_bin, scores, threshold),
    }
    return frame, row
