from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from secom.artifacts import ensure_reports_dir, write_csv
from secom.config import ArtifactName, LaneAClassifier, ReplicationMode, ScalerName, SEED_LANE_A, SelectorName
from secom.metrics import binary_metrics_at_threshold, bootstrap_ci_for_mean, find_ber_optimal_threshold, safe_std
from secom.models import make_lane_a_classifier
from secom.preprocess import make_imputer, make_scaler
from secom.qa import validate_lane_a_artifacts
from secom.selection.engine import select_features
from secom.selection.tuning import (
    lane_a_param_grid_for_selector,
    select_krr_balanced_config_with_inner_cv,
    select_logreg_config_with_inner_cv,
    select_selector_param_for_fold,
)
from secom.types import DataBundle


def _lane_a_run_mode(
    df: pd.DataFrame,
    feature_cols: list[str],
    selector: str,
    add_indicator: bool,
    classifier: str,
    replication_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = df[feature_cols].to_numpy(dtype=float)
    y = df["y_bin"].to_numpy(dtype=int)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED_LANE_A)
    rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(x, y), start=1):
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

        chosen_mrmr_lambda: float | None = None
        chosen_mi_n_neighbors: int | None = None
        chosen_l1_selector_c: float | None = None
        cached_model: Any | None = None
        cached_threshold: float | None = None
        tuned_payload: dict[str, Any] | None = None
        if selector == SelectorName.MRMR:
            chosen_mrmr_lambda, selected_local, tuned_payload = select_selector_param_for_fold(
                x_train=x_train,
                y_train=y_train,
                classifier=classifier,
                selector_method=SelectorName.MRMR,
                param_values=lane_a_param_grid_for_selector(SelectorName.MRMR, "mrmr_lambda"),
                param_name="mrmr_lambda",
                k=40,
            )
            cached_model = tuned_payload.get("model")
            cached_threshold = tuned_payload.get("threshold")
        elif selector == SelectorName.MUTUAL_INFO:
            chosen_mi_n_neighbors, selected_local, tuned_payload = select_selector_param_for_fold(
                x_train=x_train,
                y_train=y_train,
                classifier=classifier,
                selector_method=SelectorName.MUTUAL_INFO,
                param_values=lane_a_param_grid_for_selector(
                    SelectorName.MUTUAL_INFO, "mutual_info_n_neighbors"
                ),
                param_name="mutual_info_n_neighbors",
                k=40,
            )
            cached_model = tuned_payload.get("model")
            cached_threshold = tuned_payload.get("threshold")
        elif selector == SelectorName.L1_LOGREG:
            chosen_l1_selector_c, selected_local, tuned_payload = select_selector_param_for_fold(
                x_train=x_train,
                y_train=y_train,
                classifier=classifier,
                selector_method=SelectorName.L1_LOGREG,
                param_values=lane_a_param_grid_for_selector(SelectorName.L1_LOGREG, "l1_selector_c"),
                param_name="l1_selector_c",
                k=40,
            )
            cached_model = tuned_payload.get("model")
            cached_threshold = tuned_payload.get("threshold")
        else:
            selected_local, _ = select_features(
                method=selector,
                x_train=x_train,
                y_train=y_train,
                k=40,
                n_neighbors=10 if selector == SelectorName.RELIEFF else None,
            )

        x_train_sel = x_train[:, selected_local]
        x_test_sel = x_test[:, selected_local]

        chosen_alpha: float | None = None
        chosen_gamma: float | None = None
        chosen_c: float | None = None
        threshold: float
        scores: np.ndarray

        if classifier == LaneAClassifier.KRR_STRICT:
            strict_clf = make_lane_a_classifier(alpha=1.0, gamma=None)
            y_train_krr = 2 * np.asarray(y_train, dtype=int) - 1
            strict_clf.fit(x_train_sel, y_train_krr)
            strict_train_scores = np.asarray(strict_clf.predict(x_train_sel), dtype=float)
            threshold, _ = find_ber_optimal_threshold(y_train, strict_train_scores)
            scores = np.asarray(strict_clf.predict(x_test_sel), dtype=float)
        elif classifier == LaneAClassifier.KRR_BALANCED:
            if cached_model is not None and cached_threshold is not None:
                best_clf = cached_model
                threshold = float(cached_threshold)
                if tuned_payload is None:
                    raise RuntimeError("Missing tuned payload for cached model")
                chosen_alpha = (
                    float(tuned_payload["chosen_alpha"])
                    if tuned_payload.get("chosen_alpha") is not None
                    else None
                )
                chosen_gamma = tuned_payload.get("chosen_gamma")
            else:
                chosen_alpha, chosen_gamma, best_clf, threshold, _ = select_krr_balanced_config_with_inner_cv(
                    x_train_sel=x_train_sel,
                    y_train=y_train,
                )
            scores = np.asarray(best_clf.predict(x_test_sel), dtype=float)
            tuning_rows.append(
                {
                    "selector": selector,
                    "classifier": classifier,
                    "fold": fold_i,
                    "replication_mode": replication_mode,
                    "chosen_alpha": chosen_alpha,
                    "chosen_gamma": np.nan if chosen_gamma is None else float(chosen_gamma),
                    "chosen_C": np.nan,
                    "chosen_mrmr_lambda": np.nan if chosen_mrmr_lambda is None else float(chosen_mrmr_lambda),
                    "chosen_mutual_info_n_neighbors": np.nan
                    if chosen_mi_n_neighbors is None
                    else int(chosen_mi_n_neighbors),
                    "chosen_l1_selector_c": np.nan
                    if chosen_l1_selector_c is None
                    else float(chosen_l1_selector_c),
                    "threshold": float(threshold),
                    "selector_tuning_scope": "outer_train_fixed",
                }
            )
        elif classifier == LaneAClassifier.LOGREG:
            if cached_model is not None and cached_threshold is not None:
                best_clf = cached_model
                threshold = float(cached_threshold)
                if tuned_payload is None:
                    raise RuntimeError("Missing tuned payload for cached model")
                chosen_c = (
                    float(tuned_payload["chosen_C"]) if tuned_payload.get("chosen_C") is not None else None
                )
            else:
                chosen_c, best_clf, threshold, _ = select_logreg_config_with_inner_cv(
                    x_train_sel=x_train_sel,
                    y_train=y_train,
                )
            scores = np.asarray(best_clf.predict_proba(x_test_sel)[:, 1], dtype=float)
            tuning_rows.append(
                {
                    "selector": selector,
                    "classifier": classifier,
                    "fold": fold_i,
                    "replication_mode": replication_mode,
                    "chosen_alpha": np.nan,
                    "chosen_gamma": np.nan,
                    "chosen_C": chosen_c,
                    "chosen_mrmr_lambda": np.nan if chosen_mrmr_lambda is None else float(chosen_mrmr_lambda),
                    "chosen_mutual_info_n_neighbors": np.nan
                    if chosen_mi_n_neighbors is None
                    else int(chosen_mi_n_neighbors),
                    "chosen_l1_selector_c": np.nan
                    if chosen_l1_selector_c is None
                    else float(chosen_l1_selector_c),
                    "threshold": float(threshold),
                    "selector_tuning_scope": "outer_train_fixed",
                }
            )
        else:
            raise ValueError(f"Unknown Lane A classifier mode: {classifier}")

        metrics = binary_metrics_at_threshold(y_test, scores, threshold=threshold)
        rows.append(
            {
                "selector": selector,
                "classifier": classifier,
                "fold": fold_i,
                "BER": metrics["BER"],
                "True+": metrics["True+"],
                "True-": metrics["True-"],
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_test_fails": int(np.sum(y_test == 1)),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(tuning_rows)


def run_lane_a_replication(
    bundle: DataBundle,
    output_dir: Path,
    lane_a_classifier: str | None = None,
    selectors_run: list[str] | None = None,
) -> None:
    reports = ensure_reports_dir(output_dir)
    classifiers_run = (
        list(LaneAClassifier.ALL) if lane_a_classifier is None else [str(lane_a_classifier)]
    )
    bad = set(classifiers_run) - set(LaneAClassifier.ALL)
    if bad:
        raise ValueError(f"Unknown Lane A classifier(s): {sorted(bad)}")
    selectors_run = list(SelectorName.ACTIVE) if selectors_run is None else [str(s) for s in selectors_run]
    bad_selectors = set(selectors_run) - set(SelectorName.ALL)
    if bad_selectors:
        raise ValueError(f"Unknown Lane A selector(s): {sorted(bad_selectors)}")

    strict_rows: list[pd.DataFrame] = []
    mi_rows: list[pd.DataFrame] = []
    tuning_rows: list[pd.DataFrame] = []
    for classifier in classifiers_run:
        for selector in selectors_run:
            strict_frame, strict_tuning = _lane_a_run_mode(
                df=bundle.all_data,
                feature_cols=bundle.feature_columns,
                selector=selector,
                add_indicator=False,
                classifier=classifier,
                replication_mode=ReplicationMode.STRICT,
            )
            mi_frame, mi_tuning = _lane_a_run_mode(
                df=bundle.all_data,
                feature_cols=bundle.feature_columns,
                selector=selector,
                add_indicator=True,
                classifier=classifier,
                replication_mode=ReplicationMode.WITH_MISSING_INDICATORS,
            )
            strict_rows.append(strict_frame)
            mi_rows.append(mi_frame)
            tuning_rows.extend([strict_tuning, mi_tuning])

    strict_df = pd.concat(strict_rows, ignore_index=True)
    mi_df = pd.concat(mi_rows, ignore_index=True)
    write_csv(strict_df, reports / ArtifactName.BASELINE_STRICT)
    write_csv(mi_df, reports / ArtifactName.BASELINE_MI)

    tuning_trace_df = (
        pd.concat([df for df in tuning_rows if not df.empty], ignore_index=True)
        if any(not df.empty for df in tuning_rows)
        else pd.DataFrame(
            columns=[
                "selector",
                "classifier",
                "fold",
                "replication_mode",
                "chosen_alpha",
                "chosen_gamma",
                "chosen_C",
                "chosen_mrmr_lambda",
                "chosen_mutual_info_n_neighbors",
                "chosen_l1_selector_c",
                "threshold",
                "selector_tuning_scope",
            ]
        )
    )
    write_csv(tuning_trace_df, reports / ArtifactName.BASELINE_TUNING_TRACE)

    ablation_rows = []
    summary_rows = []
    for classifier in classifiers_run:
        for selector in selectors_run:
            s = strict_df.loc[
                (strict_df["selector"] == selector) & (strict_df["classifier"] == classifier)
            ].sort_values("fold")
            m = mi_df.loc[
                (mi_df["selector"] == selector) & (mi_df["classifier"] == classifier)
            ].sort_values("fold")
            delta = s["BER"].to_numpy() - m["BER"].to_numpy()
            lo, hi = bootstrap_ci_for_mean(delta, n_boot=1000, seed=42, alpha=0.95)
            ablation_rows.append(
                {
                    "selector": selector,
                    "classifier": classifier,
                    "BER_strict": float(np.mean(s["BER"])),
                    "BER_MI": float(np.mean(m["BER"])),
                    "delta_BER": float(np.mean(delta)),
                    "CI_lower": lo,
                    "CI_upper": hi,
                    "n_boot": 1000,
                }
            )

            for mode_name, frame in [
                (ReplicationMode.STRICT, s),
                (ReplicationMode.WITH_MISSING_INDICATORS, m),
            ]:
                ber_lo, ber_hi = bootstrap_ci_for_mean(
                    frame["BER"].to_numpy(), n_boot=1000, seed=42
                )
                tp_lo, tp_hi = bootstrap_ci_for_mean(
                    frame["True+"].to_numpy(), n_boot=1000, seed=42
                )
                tn_lo, tn_hi = bootstrap_ci_for_mean(
                    frame["True-"].to_numpy(), n_boot=1000, seed=42
                )
                summary_rows.append(
                    {
                        "selector": selector,
                        "classifier": classifier,
                        "replication_mode": mode_name,
                        "n_folds": int(len(frame)),
                        "n_boot": 1000,
                        "boot_seed": 42,
                        "mean_BER": float(np.mean(frame["BER"])),
                        "std_BER": safe_std(frame["BER"].to_numpy()),
                        "CI_lower_BER": ber_lo,
                        "CI_upper_BER": ber_hi,
                        "mean_True+": float(np.mean(frame["True+"])),
                        "std_True+": safe_std(frame["True+"].to_numpy()),
                        "CI_lower_True+": tp_lo,
                        "CI_upper_True+": tp_hi,
                        "mean_True-": float(np.mean(frame["True-"])),
                        "std_True-": safe_std(frame["True-"].to_numpy()),
                        "CI_lower_True-": tn_lo,
                        "CI_upper_True-": tn_hi,
                    }
                )

    ablation_df = pd.DataFrame(ablation_rows)
    summary_df = pd.DataFrame(summary_rows)
    write_csv(ablation_df, reports / ArtifactName.BASELINE_ABLATION)
    write_csv(summary_df, reports / ArtifactName.BASELINE_SUMMARY)
    validate_lane_a_artifacts(
        summary_df=summary_df,
        ablation_df=ablation_df,
        strict_df=strict_df,
        mi_df=mi_df,
        tuning_trace_df=tuning_trace_df,
        classifiers_run=classifiers_run,
        selectors_run=selectors_run,
    )

