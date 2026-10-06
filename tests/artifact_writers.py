"""Test helpers for writing artifact CSV rows."""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache

import pandas as pd
import numpy as np

from secom.artifacts import write_csv
from secom.config import ArtifactName


def write_artifact_rows(reports_dir: Path, artifact_name: str, rows: list[dict[str, object]]) -> None:
    """Write multiple artifact rows through the production CSV writer."""
    write_csv(pd.DataFrame(rows), reports_dir / artifact_name)


def write_artifact_row(reports_dir: Path, artifact_name: str, row: dict[str, object]) -> None:
    """Write one artifact row through the production CSV writer."""
    write_artifact_rows(reports_dir, artifact_name, [row])


def complete_benchmark_fixture(reports: Path, *, tuned: bool = False) -> None:
    """Materialize paired modes and ten folds for compact completed-study fixture builders."""
    prefix = "benchmark_tuned_" if tuned else "benchmark_"
    names = ["search" if tuned else "sweep", "best_config", "fold_metrics", "summary", "full_fit_summary"]
    feature_names = (
        [ArtifactName.BENCHMARK_TUNED_FEATURE_STABILITY, ArtifactName.BENCHMARK_TUNED_FEATURE_REPORT]
        if tuned
        else [ArtifactName.FEATURE_STABILITY, ArtifactName.FEATURE_REPORT]
    )
    for name in [*(prefix + suffix + ".csv" for suffix in names), *feature_names]:
        path = reports / name
        frame = pd.read_csv(path)
        if "gamma" in frame:
            frame["gamma_multiplier"] = np.nan
        if "fold" in frame.columns:
            frame = pd.concat([frame.assign(fold=fold) for fold in range(1, 11)], ignore_index=True)
        if "n_folds" in frame.columns:
            frame["n_folds"] = 10
        if "selection_count" in frame.columns:
            frame["selection_count"] = 10
        frame = pd.concat([frame, frame.assign(replication_mode="with_missing_indicators")], ignore_index=True)
        write_csv(frame, path)
    summary = pd.read_csv(reports / (prefix + "summary.csv"))
    ablation = pd.read_csv(reports / (prefix + "ablation.csv"))
    for index, row in ablation.iterrows():
        mean_ber = summary.loc[
            (summary["selector"] == row["selector"]) & (summary["classifier"] == row["classifier"]), "mean_BER"
        ].iloc[0]
        ablation.loc[index, ["BER_reference", "BER_missing_indicator", "delta_BER"]] = [mean_ber, mean_ber, 0]
    write_csv(ablation, reports / (prefix + "ablation.csv"))
    from secom.workflows.benchmark_common import build_benchmark_summary_df

    folds = pd.read_csv(reports / (prefix + "fold_metrics.csv"))
    if {"True+", "True-"}.issubset(folds):
        folds["BER"] = 1 - 0.5 * (folds["True+"] + folds["True-"])
        write_csv(folds, reports / (prefix + "fold_metrics.csv"))
    summary = build_benchmark_summary_df(folds)
    write_csv(summary, reports / (prefix + "summary.csv"))
    for index, row in ablation.iterrows():
        value = summary.loc[
            (summary.selector == row.selector) & (summary.classifier == row.classifier), "mean_BER"
        ].iloc[0]
        ablation.loc[index, ["BER_reference", "BER_missing_indicator", "delta_BER"]] = [value, value, 0]
    write_csv(ablation, reports / (prefix + "ablation.csv"))
    for suffix in ("best_config", "full_fit_summary"):
        path = reports / (prefix + suffix + ".csv")
        best = pd.read_csv(path)
        best["selection_count"] = 10
        best["mean_inner_ROC_AUC"] = 0.7
        best["mean_inner_BER"] = 0.25
        for metric in ("BER", "True+", "True-", "ROC_AUC", "PR_AUC", "MCC", "F2"):
            if f"mean_{metric}" not in best:
                best[f"mean_{metric}"] = best.apply(
                    lambda row: summary.loc[
                        (summary.selector == row.selector)
                        & (summary.classifier == row.classifier)
                        & (summary.replication_mode == row.replication_mode),
                        f"mean_{metric}",
                    ].iloc[0],
                    axis=1,
                )
        write_csv(best, path)
    # Materialize genuine held-out prediction/count lineage, not just schema-shaped rows.
    from secom.workflows.benchmark_procedures import prediction_frame, prediction_metrics, procedure_summary

    search_path = reports / (prefix + ("search.csv" if tuned else "sweep.csv"))
    search = pd.read_csv(search_path)
    if "fold" not in search:
        search = pd.concat([search.assign(fold=fold) for fold in range(1, 11)], ignore_index=True)
    search["mean_inner_BER"] = search.get("mean_inner_BER", 0.25)
    search["mean_inner_ROC_AUC"] = search.get("mean_inner_ROC_AUC", 0.7)
    search["threshold_inner_oof"] = 0.5
    search["scaler"] = "StandardScaler"
    search["gamma_multiplier"] = np.nan
    search["is_selected_config"] = True
    write_csv(search, search_path)
    predictions, rows = [], []
    for fold in range(1, 11):
        config = search[(search.fold == fold) & (search.replication_mode == "strict")].iloc[0].to_dict()
        family = summary[summary.replication_mode == "strict"].iloc[0]
        labels = np.repeat([1, 0], 50)
        tp, tn = round(float(family["mean_True+"]) * 50), round(float(family["mean_True-"]) * 50)
        scores = np.r_[np.ones(tp), np.zeros(50 - tp), np.zeros(tn), np.ones(50 - tn)]
        for procedure in ("joint", "values_only", "values_and_indicators", "missingness_only", "all_pass"):
            selected = dict(config)
            if procedure == "values_and_indicators":
                selected["replication_mode"] = "with_missing_indicators"
            if procedure in ("missingness_only", "all_pass"):
                selected.update(
                    selector="none",
                    classifier="logreg" if procedure == "missingness_only" else "constant",
                    replication_mode=procedure,
                    k=0,
                    C=1.0 if procedure == "missingness_only" else None,
                )
            evaluated = np.zeros(100) if procedure == "all_pass" else scores
            frame = prediction_frame(
                sample_ids=np.arange((fold - 1) * 100, fold * 100),
                fold=fold,
                y_true=labels,
                scores=evaluated,
                threshold=0.5,
                config=selected,
                procedure=procedure,
                study="tuned" if tuned else "original",
            )
            predictions.append(frame)
            rows.append({"procedure": procedure, "fold": fold, **prediction_metrics(frame)})
    prediction_df, fold_df = pd.concat(predictions, ignore_index=True), pd.DataFrame(rows)
    write_csv(prediction_df, reports / (prefix + "predictions.csv"))
    write_csv(fold_df, reports / (prefix + "procedure_fold_metrics.csv"))
    write_csv(procedure_summary(fold_df, prediction_df), reports / (prefix + "procedure_summary.csv"))


def complete_temporal_fixture(reports: Path) -> None:
    """Bind the compact temporal fixture to disjoint evaluation receipts and current restrictions."""
    from secom.artifacts import read_manifest, write_manifest, _expected_temporal_claim_restrictions
    from secom.workflows.benchmark_procedures import prediction_frame, prediction_metrics

    predictions = []
    for fold in range(1, 4):
        predictions.append(
            prediction_frame(
                sample_ids=np.arange(fold * 20, (fold + 1) * 20),
                fold=fold,
                y_true=np.tile([0, 1], 10),
                scores=np.tile([0.1, 0.8], 10),
                threshold=0.8,
                config={
                    "selector": "F-test",
                    "classifier": "logreg",
                    "replication_mode": "with_missing_indicators",
                    "k": 10,
                    "C": 1,
                    "scaler": "StandardScaler",
                },
                procedure="temporal_joint",
                study="DEV_chronological",
            )
        )
    frame = pd.concat(predictions, ignore_index=True)
    frame["timestamp"] = pd.Timestamp("2008-01-01") + pd.to_timedelta(frame.sample_id, unit="h")
    for fold in range(1, 4):
        fit_n = int(np.floor(0.8 * fold * 20))
        frame.loc[frame.fold == fold, "fit_end_timestamp"] = pd.Timestamp("2008-01-01") + pd.Timedelta(hours=fit_n - 1)
        frame.loc[frame.fold == fold, "calibration_start_timestamp"] = pd.Timestamp("2008-01-01") + pd.Timedelta(
            hours=fit_n
        )
        frame.loc[frame.fold == fold, "calibration_end_timestamp"] = pd.Timestamp("2008-01-01") + pd.Timedelta(
            hours=fold * 20 - 1
        )
    write_csv(frame, reports / ArtifactName.TEMPORAL_PREDICTIONS)
    write_csv(
        pd.DataFrame(
            [
                {"procedure": "temporal_joint", "fold": fold, **prediction_metrics(group)}
                for fold, group in frame.groupby("fold")
            ]
        ),
        reports / ArtifactName.TEMPORAL_PROCEDURE_METRICS,
    )
    from scipy.stats import binomtest

    lock_path = reports / ArtifactName.TEMPORAL_LOCKBOX
    lock = pd.read_csv(lock_path)
    for index, row in lock.iterrows():
        tp, tn = round(float(row["True+"]) * 100), round(float(row["True-"]) * 100)
        lock.loc[index, ["TP", "TN", "FP", "FN", "lockbox_n", "lockbox_fails"]] = [tp, tn, 100 - tn, 100 - tp, 200, 100]
        lock.loc[index, ["BER", "True+", "True-"]] = [1 - 0.5 * (tp / 100 + tn / 100), tp / 100, tn / 100]
        for label, successes in (("TPR", tp), ("TNR", tn)):
            interval = binomtest(successes, 100).proportion_ci(method="exact")
            lock.loc[index, [f"{label}_exact_lower", f"{label}_exact_upper"]] = [interval.low, interval.high]
    lock["threshold_value"] = 0.8
    lock["interval_semantics"] = "conditional_fixed_model_independent_trials_only"
    lock["evaluation_semantics"] = "retrospective_later_block"
    lock["TPR_available"] = lock["TNR_available"] = lock["BER_available"] = True
    write_csv(lock, lock_path)
    drift_path = reports / ArtifactName.TEMPORAL_DRIFT
    drift = pd.read_csv(drift_path)
    drift["confirmatory_claims_allowed"] = False
    drift["score_reference"] = "held_out_calibration_same_retained_model"
    drift["max_missingness_rate_shift"] = 0.1
    manager_path = reports / ArtifactName.TEMPORAL_MANAGER_OUTPUTS
    manager = pd.read_csv(manager_path)
    manager["evaluation_region"] = "held_out_DEV_calibration"
    manager["workload_semantics"] = "illustrative_mean_weekly_policy_not_per_week_cap"
    manager["mean_weekly_flag_fraction"] = 0.1
    write_csv(manager, manager_path)
    inner_path = reports / ArtifactName.TEMPORAL_INNER_CV
    inner = pd.read_csv(inner_path)
    inner["selector"], inner["k"], inner["C"], inner["scaler"], inner["n_neighbors"] = (
        "F-test",
        10,
        1,
        "StandardScaler",
        np.nan,
    )
    inner = pd.concat([inner.assign(resample_id=f"outer_{fold}_seed_42") for fold in range(1, 4)], ignore_index=True)
    inner["is_selected_config"] = True
    write_csv(inner, inner_path)
    freeze_path = reports / ArtifactName.TEMPORAL_FREEZE
    freeze = pd.read_csv(freeze_path)
    freeze["k"], freeze["C"], freeze["scaler"], freeze["n_neighbors"] = 10, 1.0, "StandardScaler", np.nan
    write_csv(freeze, freeze_path)
    write_csv(drift, drift_path)
    mspc_path = reports / ArtifactName.TEMPORAL_MSPC
    mspc = pd.read_csv(mspc_path)
    mspc["source_selection_region"] = "held_out_calibration"
    mspc["T2_calibration_TPR_at_TNR90"] = (mspc.calibration_selected_MSPC_source == "T2").astype(float)
    mspc["Q_calibration_TPR_at_TNR90"] = (mspc.calibration_selected_MSPC_source == "Q").astype(float)
    mspc["frozen_threshold"] = mspc["T2_frozen_threshold"] = mspc["Q_frozen_threshold"] = 0.5
    mspc["TP"], mspc["TN"], mspc["FP"], mspc["FN"] = 5, 8, 2, 5
    mspc["frozen_TPR"], mspc["frozen_TNR"], mspc["frozen_BER"] = 0.5, 0.8, 0.35
    write_csv(mspc, mspc_path)
    manifest = read_manifest(reports / ArtifactName.MANIFEST)
    manifest["temporal_claim_restrictions"] = _expected_temporal_claim_restrictions(
        pd.read_csv(reports / ArtifactName.TEMPORAL_LOCKBOX), drift, pd.read_csv(reports / ArtifactName.TEMPORAL_MSPC)
    )
    write_manifest(manifest, reports / ArtifactName.MANIFEST)
    write_comparator_fixture(reports)


def write_comparator_fixture(reports: Path) -> None:
    """Use a cached small real comparator run rather than fabricated winner receipts."""
    from secom.workflows.calibration import calibration_receipt

    selectors = tuple(pd.read_csv(reports / ArtifactName.TEMPORAL_SELECTOR_SCREENING).selector.unique())
    search, predictions, metrics, cal, diag, data = _comparator_fixture(selectors)
    calibration_frames, diagnostic_rows = list(cal), list(diag)
    for fold in range(1, 4):
        train = np.arange(fold * 20)
        cut = int(np.floor(0.8 * len(train)))
        ids = train[cut:]
        scores = np.where(data.loc[ids].y_bin == 1, 0.8, 0.1)
        from secom.metrics import find_ber_optimal_threshold

        threshold, _ = find_ber_optimal_threshold(data.loc[ids].y_bin.to_numpy(), scores)
        frame, row = calibration_receipt(
            data=data,
            fit_ids=train[:cut],
            calibration_ids=ids,
            scores=scores,
            threshold=threshold,
            config={
                "selector": "F-test",
                "classifier": "logreg",
                "replication_mode": "with_missing_indicators",
                "k": 10,
                "C": 1,
                "scaler": "StandardScaler",
            },
            procedure="temporal_joint",
            fold=fold,
        )
        calibration_frames.append(frame)
        diagnostic_rows.append(row)
    for role in pd.read_csv(reports / ArtifactName.TEMPORAL_FREEZE).role.unique():
        frozen = pd.read_csv(reports / ArtifactName.TEMPORAL_FREEZE)
        selected = frozen[(frozen.role == role) & frozen.is_frozen_config].iloc[0]
        frame, row = calibration_receipt(
            data=data,
            fit_ids=np.arange(64),
            calibration_ids=np.arange(64, 80),
            scores=np.where(data.loc[64:79].y_bin == 1, 0.8, 0.1),
            threshold=threshold,
            config={
                "selector": selected.selector,
                "classifier": "logreg",
                "replication_mode": "with_missing_indicators",
                "k": 10,
                "C": 1,
                "scaler": "StandardScaler",
            },
            procedure=f"logreg_final_{role}",
            fold=0,
        )
        calibration_frames.append(frame)
        diagnostic_rows.append(row)
    for frame, name in (
        (search, ArtifactName.TEMPORAL_KRR_SEARCH),
        (predictions, ArtifactName.TEMPORAL_KRR_PREDICTIONS),
        (metrics, ArtifactName.TEMPORAL_KRR_METRICS),
        (pd.concat(calibration_frames, ignore_index=True), ArtifactName.TEMPORAL_CALIBRATION_SCORES),
        (pd.DataFrame(diagnostic_rows), ArtifactName.TEMPORAL_CALIBRATION_DIAGNOSTICS),
    ):
        write_csv(frame, reports / name)


@lru_cache(maxsize=4)
def _comparator_fixture(selectors):
    from types import SimpleNamespace
    from secom.workflows.temporal_krr import run_krr_comparator

    data = pd.DataFrame(
        {
            "raw_row_id": np.arange(80),
            "y_bin": np.tile([0, 1], 40),
            "timestamp": pd.Timestamp("2008-01-01") + pd.to_timedelta(np.arange(80), unit="h"),
        }
    )
    data.attrs["raw_features"] = np.random.default_rng(88).normal(size=(80, 12))
    folds = [
        SimpleNamespace(outer_fold=i, train_index=np.arange(i * 20), test_index=np.arange(i * 20, (i + 1) * 20))
        for i in range(1, 4)
    ]
    return (
        *run_krr_comparator(data=data, x_raw=data.attrs["raw_features"], folds=folds, selectors=list(selectors)),
        data,
    )
