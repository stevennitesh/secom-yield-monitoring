from __future__ import annotations

from typing import Any

import numpy as np

from secom.config import SelectorName
from secom.feature_select.gram_schmidt import gram_schmidt_rank_features
from secom.feature_select.relief import relief_rank_features
from secom.feature_select.univariate import rank_features
from secom.preprocess import (
    make_imputer,
    make_scaler,
    transformed_feature_metadata_from_imputer,
)


def select_features(
    method: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
    n_neighbors: int | None = None,
    mrmr_lambda: float = 1.0,
    mutual_info_n_neighbors: int = 3,
    l1_selector_c: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    if method in {
        SelectorName.S2N,
        SelectorName.WELCH_T,
        SelectorName.F_TEST,
        SelectorName.PEARSON,
        SelectorName.MUTUAL_INFO,
        SelectorName.MRMR,
        SelectorName.L1_LOGREG,
    }:
        order, scores = rank_features(
            method,
            x_train,
            y_train,
            mrmr_lambda=float(mrmr_lambda),
            mutual_info_n_neighbors=int(mutual_info_n_neighbors),
            l1_selector_c=float(l1_selector_c),
        )
        selected = order[: min(k, order.shape[0])]
        return selected, scores
    if method == SelectorName.RELIEFF:
        if n_neighbors is None:
            raise ValueError("ReliefF requires n_neighbors")
        order, scores = relief_rank_features(x_train, y_train, n_neighbors=n_neighbors)
        selected = order[: min(k, order.shape[0])]
        return selected, scores
    if method == SelectorName.GRAM_SCHMIDT:
        order, scores = gram_schmidt_rank_features(x_train, y_train, k=k)
        if order.shape[0] > k:
            order = order[:k]
        return order, scores
    raise ValueError(f"Unknown selector {method}")


def fit_selector_pipeline(
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
    x_eval_raw: np.ndarray,
    method: str,
    k: int,
    scaler_name: str,
    add_indicator: bool,
    n_neighbors: int | None,
    mrmr_lambda: float = 1.0,
    mutual_info_n_neighbors: int = 3,
    l1_selector_c: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, list[Any], np.ndarray, Any, Any]:
    imputer = make_imputer(add_indicator=add_indicator)
    x_train_imp = imputer.fit_transform(x_train_raw)
    x_eval_imp = imputer.transform(x_eval_raw)
    scaler = make_scaler(scaler_name)
    x_train_scaled = scaler.fit_transform(x_train_imp)
    x_eval_scaled = scaler.transform(x_eval_imp)

    selected_local, _scores = select_features(
        method=method,
        x_train=x_train_scaled,
        y_train=y_train,
        k=int(k),
        n_neighbors=n_neighbors,
        mrmr_lambda=float(mrmr_lambda),
        mutual_info_n_neighbors=int(mutual_info_n_neighbors),
        l1_selector_c=float(l1_selector_c),
    )
    feature_meta = transformed_feature_metadata_from_imputer(
        imputer=imputer, raw_feature_count=x_train_raw.shape[1]
    )
    x_train_sel = x_train_scaled[:, selected_local]
    x_eval_sel = x_eval_scaled[:, selected_local] # type: ignore
    return x_train_sel, x_eval_sel, feature_meta, selected_local, imputer, scaler
