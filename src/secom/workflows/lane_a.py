from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from secom.artifacts import ensure_reports_dir, write_csv
from secom.config import (
    ArtifactName,
    LANE_A_KRR_ALPHA_GRID,
    LANE_A_KRR_GAMMA_GRID,
    LANE_A_LOGREG_C_GRID,
    LaneAClassifier,
    ReplicationMode,
    ScalerName,
    SEED_LANE_A,
    SelectorName,
)
from secom.metrics import (
    binary_metrics_at_threshold,
    bootstrap_ci_for_mean,
    find_ber_optimal_threshold,
    safe_std,
)
from secom.models import (
    fit_lane_a_krr_classifier,
    make_lane_a_classifier,
    make_lane_a_logreg_tuned_classifier,
)
from secom.preprocess import make_imputer, make_scaler
from secom.qa import validate_lane_a_global_artifacts
from secom.selection.engine import select_features
from secom.selection.tuning import gamma_sort_key
from secom.types import DataBundle


class LaneAThresholdMode:
    GLOBAL_OOF = "global_oof"
    PER_FOLD_TRAIN = "per_fold_train"
    ALL = [GLOBAL_OOF, PER_FOLD_TRAIN]


def _lane_a_selector_param_grid(selector: str) -> list[dict[str, Any]]:
    if selector == SelectorName.RELIEFF:
        return [{"n_neighbors": 10}]
    return [{}]


def _lane_a_classifier_param_grid(classifier: str) -> list[dict[str, Any]]:
    if classifier == LaneAClassifier.KRR:
        alphas = sorted(float(v) for v in LANE_A_KRR_ALPHA_GRID)
        gammas = sorted(
            (None if g is None else float(g) for g in LANE_A_KRR_GAMMA_GRID),
            key=gamma_sort_key,
        )
        return [{"alpha": float(alpha), "gamma": gamma} for alpha, gamma in product(alphas, gammas)]
    if classifier == LaneAClassifier.LOGREG:
        return [{"C": float(v)} for v in sorted(float(c) for c in LANE_A_LOGREG_C_GRID)]
    if classifier == LaneAClassifier.KRR_STRICT:
        return [{"alpha": 1.0, "gamma": None}]
    raise ValueError(f"Unknown Lane A classifier mode: {classifier}")


def _lane_a_selector_kwargs(selector: str, selector_config: dict[str, Any]) -> dict[str, Any]:
    if selector == SelectorName.RELIEFF:
        return {"n_neighbors": int(selector_config.get("n_neighbors", 10))}
    return {}


def _lane_a_config_fields(
    selector_config: dict[str, Any],
    classifier_config: dict[str, Any],
) -> dict[str, Any]:
    gamma = classifier_config.get("gamma")
    return {
        "alpha": np.nan if classifier_config.get("alpha") is None else float(classifier_config["alpha"]),
        "gamma": np.nan if gamma is None else float(gamma),
        "C": np.nan if classifier_config.get("C") is None else float(classifier_config["C"]),
        "n_neighbors": np.nan
        if selector_config.get("n_neighbors") is None
        else int(selector_config["n_neighbors"]),
    }


def _lane_a_config_tie_break_key(
    selector: str,
    classifier: str,
    selector_config: dict[str, Any],
    classifier_config: dict[str, Any],
) -> tuple[Any, ...]:
    selector_key: tuple[Any, ...]
    if selector == SelectorName.RELIEFF:
        selector_key = (int(selector_config.get("n_neighbors", 10)),)
    else:
        selector_key = ()

    classifier_key: tuple[Any, ...]
    if classifier == LaneAClassifier.KRR:
        classifier_key = (
            float(classifier_config["alpha"]),
            float(gamma_sort_key(classifier_config.get("gamma"))),
        )
    elif classifier == LaneAClassifier.LOGREG:
        classifier_key = (float(classifier_config["C"]),)
    else:
        classifier_key = ()
    return selector_key + classifier_key


def _prepare_lane_a_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    x = df[feature_cols].to_numpy(dtype=float)
    y = df["y_bin"].to_numpy(dtype=int)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED_LANE_A)
    folds = [(train_idx, test_idx) for train_idx, test_idx in skf.split(x, y)]
    return x, y, folds


def _evaluate_lane_a_config_over_folds(
    x: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    selector: str,
    classifier: str,
    add_indicator: bool,
    replication_mode: str,
    selector_config: dict[str, Any],
    classifier_config: dict[str, Any],
    threshold_mode: str,
    k: int = 40,
) -> dict[str, Any]:
    fold_scores: list[np.ndarray] = []
    fold_labels: list[np.ndarray] = []
    fold_rows: list[dict[str, Any]] = []
    fold_train_thresholds: list[float] = []
    n_selected_per_fold: list[int] = []
    selector_kwargs = _lane_a_selector_kwargs(selector=selector, selector_config=selector_config)

    for fold_i, (train_idx, test_idx) in enumerate(folds, start=1):
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

        selected_local, _ = select_features(
            method=selector,
            x_train=x_train,
            y_train=y_train,
            k=int(k),
            **selector_kwargs,
        )
        if selected_local.size <= 0:
            raise RuntimeError("Lane A config produced zero selected features")
        n_selected_per_fold.append(int(selected_local.size))
        x_train_sel = x_train[:, selected_local]
        x_test_sel = x_test[:, selected_local] # type: ignore

        if classifier == LaneAClassifier.KRR:
            clf = fit_lane_a_krr_classifier(
                x_train_sel,
                y_train,
                alpha=float(classifier_config["alpha"]),
                gamma=classifier_config.get("gamma"),
            )
            train_scores = np.asarray(clf.predict(x_train_sel), dtype=float)
            scores = np.asarray(clf.predict(x_test_sel), dtype=float)
        elif classifier == LaneAClassifier.LOGREG:
            clf = make_lane_a_logreg_tuned_classifier(c_value=float(classifier_config["C"]))
            clf.fit(x_train_sel, y_train)
            train_scores = np.asarray(clf.predict_proba(x_train_sel)[:, 1], dtype=float)
            scores = np.asarray(clf.predict_proba(x_test_sel)[:, 1], dtype=float)
        elif classifier == LaneAClassifier.KRR_STRICT:
            clf = make_lane_a_classifier(alpha=1.0, gamma=None)
            y_train_krr = 2 * np.asarray(y_train, dtype=int) - 1
            clf.fit(x_train_sel, y_train_krr)
            train_scores = np.asarray(clf.predict(x_train_sel), dtype=float)
            scores = np.asarray(clf.predict(x_test_sel), dtype=float)
        else:
            raise ValueError(f"Unknown Lane A classifier mode: {classifier}")
        train_threshold, _ = find_ber_optimal_threshold(y_train, train_scores)
        fold_train_thresholds.append(float(train_threshold))

        fold_scores.append(scores)
        fold_labels.append(np.asarray(y_test, dtype=int))
        fold_rows.append(
            {
                "selector": selector,
                "classifier": classifier,
                "replication_mode": replication_mode,
                "fold": int(fold_i),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_test_fails": int(np.sum(y_test == 1)),
                "n_selected_features": int(selected_local.size),
            }
        )

    oof_scores = np.concatenate(fold_scores)
    oof_labels = np.concatenate(fold_labels)
    threshold_oof, _ = find_ber_optimal_threshold(oof_labels, oof_scores)
    oof_metrics = binary_metrics_at_threshold(y_true=oof_labels, scores=oof_scores, threshold=float(threshold_oof))

    fold_ber_values: list[float] = []
    fold_tp_values: list[float] = []
    fold_tn_values: list[float] = []
    for row, y_fold, scores_fold, train_threshold in zip(
        fold_rows,
        fold_labels,
        fold_scores,
        fold_train_thresholds,
    ):
        if threshold_mode == LaneAThresholdMode.PER_FOLD_TRAIN:
            applied_threshold = float(train_threshold)
        elif threshold_mode == LaneAThresholdMode.GLOBAL_OOF:
            applied_threshold = float(threshold_oof)
        else:
            raise ValueError(f"Unknown Lane A threshold mode: {threshold_mode}")
        fold_metrics = binary_metrics_at_threshold(
            y_true=y_fold,
            scores=scores_fold,
            threshold=applied_threshold,
        )
        row["BER"] = float(fold_metrics["BER"])
        row["True+"] = float(fold_metrics["True+"])
        row["True-"] = float(fold_metrics["True-"])
        row["threshold_mode"] = threshold_mode
        row["threshold_applied"] = applied_threshold
        row["threshold_train_fold"] = float(train_threshold)
        row["threshold_oof_global"] = float(threshold_oof)
        fold_ber_values.append(float(fold_metrics["BER"]))
        fold_tp_values.append(float(fold_metrics["True+"]))
        fold_tn_values.append(float(fold_metrics["True-"]))

    if threshold_mode == LaneAThresholdMode.PER_FOLD_TRAIN:
        mean_ber = float(np.mean(np.asarray(fold_ber_values, dtype=float)))
        mean_tp = float(np.mean(np.asarray(fold_tp_values, dtype=float)))
        mean_tn = float(np.mean(np.asarray(fold_tn_values, dtype=float)))
    else:
        mean_ber = float(oof_metrics["BER"])
        mean_tp = float(oof_metrics["True+"])
        mean_tn = float(oof_metrics["True-"])

    return {
        "threshold_mode": threshold_mode,
        "threshold_oof_global": float(threshold_oof),
        "mean_BER_oof": mean_ber,
        "mean_True+_oof": mean_tp,
        "mean_True-_oof": mean_tn,
        "std_BER_fold": safe_std(np.asarray(fold_ber_values, dtype=float)),
        "mean_n_selected_features": float(np.mean(np.asarray(n_selected_per_fold, dtype=float))),
        "min_n_selected_features": int(np.min(np.asarray(n_selected_per_fold, dtype=int))),
        "max_n_selected_features": int(np.max(np.asarray(n_selected_per_fold, dtype=int))),
        "n_folds": int(len(folds)),
        "fold_rows": fold_rows,
    }


def _fit_lane_a_full_dataset(
    x: np.ndarray,
    y: np.ndarray,
    selector: str,
    classifier: str,
    add_indicator: bool,
    selector_config: dict[str, Any],
    classifier_config: dict[str, Any],
    k: int = 40,
) -> dict[str, Any]:
    imputer = make_imputer(add_indicator=add_indicator)
    x_imp = imputer.fit_transform(x)
    scaler = make_scaler(ScalerName.STANDARD)
    x_scaled = scaler.fit_transform(x_imp)
    selector_kwargs = _lane_a_selector_kwargs(selector=selector, selector_config=selector_config)
    selected_local, _ = select_features(
        method=selector,
        x_train=x_scaled,
        y_train=y,
        k=int(k),
        **selector_kwargs,
    )
    if selected_local.size <= 0:
        raise RuntimeError("Lane A full-data fit produced zero selected features")
    x_sel = x_scaled[:, selected_local]

    if classifier == LaneAClassifier.KRR:
        clf = fit_lane_a_krr_classifier(
            x_sel,
            y,
            alpha=float(classifier_config["alpha"]),
            gamma=classifier_config.get("gamma"),
        )
        scores = np.asarray(clf.predict(x_sel), dtype=float)
    elif classifier == LaneAClassifier.LOGREG:
        clf = make_lane_a_logreg_tuned_classifier(c_value=float(classifier_config["C"]))
        clf.fit(x_sel, y)
        scores = np.asarray(clf.predict_proba(x_sel)[:, 1], dtype=float)
    elif classifier == LaneAClassifier.KRR_STRICT:
        clf = make_lane_a_classifier(alpha=1.0, gamma=None)
        y_krr = 2 * np.asarray(y, dtype=int) - 1
        clf.fit(x_sel, y_krr)
        scores = np.asarray(clf.predict(x_sel), dtype=float)
    else:
        raise ValueError(f"Unknown Lane A classifier mode: {classifier}")

    threshold_full, _ = find_ber_optimal_threshold(y, scores)
    metrics_full = binary_metrics_at_threshold(y_true=y, scores=scores, threshold=float(threshold_full))
    return {
        "threshold_full_dataset": float(threshold_full),
        "BER_full_dataset": float(metrics_full["BER"]),
        "True+_full_dataset": float(metrics_full["True+"]),
        "True-_full_dataset": float(metrics_full["True-"]),
        "n_samples_full_dataset": int(y.size),
        "n_fails_full_dataset": int(np.sum(y == 1)),
        "n_selected_features_full_dataset": int(selected_local.size),
        "threshold_full_dataset_role": "diagnostic_only",
    }


def run_lane_a_replication(
    bundle: DataBundle,
    output_dir: Path,
    lane_a_classifier: str | None = None,
    selectors_run: list[str] | None = None,
    threshold_mode: str = LaneAThresholdMode.GLOBAL_OOF,
) -> None:
    reports = ensure_reports_dir(output_dir)
    allowed_classifiers = set(LaneAClassifier.ALL + LaneAClassifier.OPTIONAL_BENCHMARK)
    classifiers_run = list(LaneAClassifier.ALL) if lane_a_classifier is None else [str(lane_a_classifier)]
    bad = set(classifiers_run) - allowed_classifiers
    if bad:
        raise ValueError(f"Unknown Lane A classifier(s): {sorted(bad)}")
    selectors_run = list(SelectorName.ACTIVE) if selectors_run is None else [str(s) for s in selectors_run]
    bad_selectors = set(selectors_run) - set(SelectorName.ALL)
    if bad_selectors:
        raise ValueError(f"Unknown Lane A selector(s): {sorted(bad_selectors)}")
    if threshold_mode not in LaneAThresholdMode.ALL:
        raise ValueError(
            f"Unknown Lane A threshold mode {threshold_mode!r}. "
            f"Expected one of: {LaneAThresholdMode.ALL}"
        )

    x, y, folds = _prepare_lane_a_cv(
        df=bundle.all_data,
        feature_cols=bundle.feature_columns,
    )
    sweep_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    full_fit_rows: list[dict[str, Any]] = []

    for classifier in classifiers_run:
        classifier_grid = _lane_a_classifier_param_grid(classifier=classifier)
        for selector in selectors_run:
            selector_grid = _lane_a_selector_param_grid(selector=selector)
            for replication_mode, add_indicator in (
                (ReplicationMode.STRICT, False),
                (ReplicationMode.WITH_MISSING_INDICATORS, True),
            ):
                best_payload: dict[str, Any] | None = None
                best_selector_config: dict[str, Any] | None = None
                best_classifier_config: dict[str, Any] | None = None
                best_obj = np.inf
                best_tie_key: tuple[Any, ...] | None = None

                for selector_config in selector_grid:
                    for classifier_config in classifier_grid:
                        payload = _evaluate_lane_a_config_over_folds(
                            x=x,
                            y=y,
                            folds=folds,
                            selector=selector,
                            classifier=classifier,
                            add_indicator=add_indicator,
                            replication_mode=replication_mode,
                            selector_config=selector_config,
                            classifier_config=classifier_config,
                            threshold_mode=threshold_mode,
                            k=40,
                        )
                        config_fields = _lane_a_config_fields(
                            selector_config=selector_config,
                            classifier_config=classifier_config,
                        )
                        sweep_rows.append(
                            {
                                "selector": selector,
                                "classifier": classifier,
                                "replication_mode": replication_mode,
                                **config_fields,
                                "threshold_oof_global": float(payload["threshold_oof_global"]),
                                "mean_BER_oof": float(payload["mean_BER_oof"]),
                                "std_BER_fold": float(payload["std_BER_fold"]),
                                "mean_True+_oof": float(payload["mean_True+_oof"]),
                                "mean_True-_oof": float(payload["mean_True-_oof"]),
                                "mean_n_selected_features": float(payload["mean_n_selected_features"]),
                                "min_n_selected_features": int(payload["min_n_selected_features"]),
                                "max_n_selected_features": int(payload["max_n_selected_features"]),
                                "n_folds": int(payload["n_folds"]),
                            }
                        )

                        objective = float(payload["mean_BER_oof"])
                        tie_key = _lane_a_config_tie_break_key(
                            selector=selector,
                            classifier=classifier,
                            selector_config=selector_config,
                            classifier_config=classifier_config,
                        )
                        is_better = False
                        if objective < best_obj - 1e-12:
                            is_better = True
                        elif np.isclose(objective, best_obj):
                            if best_tie_key is None or tie_key < best_tie_key:
                                is_better = True
                        if is_better:
                            best_obj = objective
                            best_tie_key = tie_key
                            best_payload = payload
                            best_selector_config = dict(selector_config)
                            best_classifier_config = dict(classifier_config)

                if best_payload is None or best_selector_config is None or best_classifier_config is None:
                    raise RuntimeError("Lane A global OOF search failed to select a best config")

                best_fields = _lane_a_config_fields(
                    selector_config=best_selector_config,
                    classifier_config=best_classifier_config,
                )
                best_rows.append(
                    {
                        "selector": selector,
                        "classifier": classifier,
                        "replication_mode": replication_mode,
                        **best_fields,
                        "threshold_oof_global": float(best_payload["threshold_oof_global"]),
                        "mean_BER_oof": float(best_payload["mean_BER_oof"]),
                        "std_BER_fold": float(best_payload["std_BER_fold"]),
                        "mean_True+_oof": float(best_payload["mean_True+_oof"]),
                        "mean_True-_oof": float(best_payload["mean_True-_oof"]),
                        "mean_n_selected_features": float(best_payload["mean_n_selected_features"]),
                        "min_n_selected_features": int(best_payload["min_n_selected_features"]),
                        "max_n_selected_features": int(best_payload["max_n_selected_features"]),
                        "n_folds": int(best_payload["n_folds"]),
                        "n_configs_evaluated": int(len(selector_grid) * len(classifier_grid)),
                    }
                )

                for fold_row in best_payload["fold_rows"]:
                    fold_metric_rows.append({**fold_row, **best_fields})

                full_fit_payload = _fit_lane_a_full_dataset(
                    x=x,
                    y=y,
                    selector=selector,
                    classifier=classifier,
                    add_indicator=add_indicator,
                    selector_config=best_selector_config,
                    classifier_config=best_classifier_config,
                    k=40,
                )
                full_fit_rows.append(
                    {
                        "selector": selector,
                        "classifier": classifier,
                        "replication_mode": replication_mode,
                        **best_fields,
                        "threshold_oof_global": float(best_payload["threshold_oof_global"]),
                        **full_fit_payload,
                    }
                )

    sweep_df = pd.DataFrame(sweep_rows)
    best_df = pd.DataFrame(best_rows)
    fold_metrics_df = pd.DataFrame(fold_metric_rows)
    full_fit_df = pd.DataFrame(full_fit_rows)

    summary_rows: list[dict[str, Any]] = []
    for (selector, classifier, mode), frame in fold_metrics_df.groupby(
        ["selector", "classifier", "replication_mode"], sort=False
    ):
        ber_values = frame["BER"].to_numpy(dtype=float)
        tp_values = frame["True+"].to_numpy(dtype=float)
        tn_values = frame["True-"].to_numpy(dtype=float)
        ber_lo, ber_hi = bootstrap_ci_for_mean(ber_values, n_boot=1000, seed=42)
        tp_lo, tp_hi = bootstrap_ci_for_mean(tp_values, n_boot=1000, seed=42)
        tn_lo, tn_hi = bootstrap_ci_for_mean(tn_values, n_boot=1000, seed=42)
        summary_rows.append(
            {
                "selector": selector,
                "classifier": classifier,
                "replication_mode": mode,
                "n_folds": int(len(frame)),
                "n_boot": 1000,
                "boot_seed": 42,
                "mean_BER": float(np.mean(ber_values)),
                "std_BER": safe_std(ber_values),
                "CI_lower_BER": ber_lo,
                "CI_upper_BER": ber_hi,
                "mean_True+": float(np.mean(tp_values)),
                "std_True+": safe_std(tp_values),
                "CI_lower_True+": tp_lo,
                "CI_upper_True+": tp_hi,
                "mean_True-": float(np.mean(tn_values)),
                "std_True-": safe_std(tn_values),
                "CI_lower_True-": tn_lo,
                "CI_upper_True-": tn_hi,
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    ablation_rows: list[dict[str, Any]] = []
    for (selector, classifier), frame in fold_metrics_df.groupby(["selector", "classifier"], sort=False):
        strict_frame = frame[frame["replication_mode"] == ReplicationMode.STRICT].sort_values("fold")
        mi_frame = frame[frame["replication_mode"] == ReplicationMode.WITH_MISSING_INDICATORS].sort_values("fold")
        delta = strict_frame["BER"].to_numpy(dtype=float) - mi_frame["BER"].to_numpy(dtype=float)
        lo, hi = bootstrap_ci_for_mean(delta, n_boot=1000, seed=42, alpha=0.95)
        ablation_rows.append(
            {
                "selector": selector,
                "classifier": classifier,
                "BER_strict": float(np.mean(strict_frame["BER"].to_numpy(dtype=float))),
                "BER_MI": float(np.mean(mi_frame["BER"].to_numpy(dtype=float))),
                "delta_BER": float(np.mean(delta)),
                "CI_lower": lo,
                "CI_upper": hi,
                "n_boot": 1000,
            }
        )
    ablation_df = pd.DataFrame(ablation_rows)

    write_csv(sweep_df, reports / ArtifactName.LANE_A_GLOBAL_SWEEP)
    write_csv(best_df, reports / ArtifactName.LANE_A_GLOBAL_BEST_CONFIG)
    write_csv(fold_metrics_df, reports / ArtifactName.LANE_A_GLOBAL_FOLD_METRICS)
    write_csv(summary_df, reports / ArtifactName.LANE_A_GLOBAL_SUMMARY)
    write_csv(ablation_df, reports / ArtifactName.LANE_A_GLOBAL_ABLATION)
    write_csv(full_fit_df, reports / ArtifactName.LANE_A_GLOBAL_FULL_FIT_SUMMARY)

    validate_lane_a_global_artifacts(
        sweep_df=sweep_df,
        best_df=best_df,
        fold_metrics_df=fold_metrics_df,
        summary_df=summary_df,
        ablation_df=ablation_df,
        full_fit_df=full_fit_df,
        classifiers_run=classifiers_run,
        selectors_run=selectors_run,
    )
