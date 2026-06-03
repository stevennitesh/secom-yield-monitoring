"""Feature selection dispatch and preprocessing pipeline assembly."""

from __future__ import annotations

import numpy as np

from secom.config import SelectorName
from secom.feature_select.gram_schmidt import gram_schmidt_rank_features
from secom.feature_select.relief import relief_rank_features
from secom.feature_select.univariate import rank_features
from secom.preprocess import (
    TransformedFeature,
    make_imputer,
    make_scaler,
    transformed_feature_metadata_from_imputer,
)

_UNIVARIATE_SELECTORS = {
    SelectorName.S2N,
    SelectorName.WELCH_T,
    SelectorName.F_TEST,
    SelectorName.PEARSON,
}


def _top_k(order: np.ndarray, k: int) -> np.ndarray:
    """Return the bounded top-k slice while preserving selector order."""
    return order[: min(int(k), order.shape[0])]


def _selector_order_and_scores(
    method: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_neighbors: int | None,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to the selector implementation and return full order plus scores."""
    if method in _UNIVARIATE_SELECTORS:
        return rank_features(method, x_train, y_train)

    if method == SelectorName.RELIEFF:
        if n_neighbors is None:
            raise ValueError("ReliefF requires n_neighbors")
        return relief_rank_features(x_train, y_train, n_neighbors=n_neighbors)

    if method == SelectorName.GRAM_SCHMIDT:
        return gram_schmidt_rank_features(x_train, y_train, k=k)

    raise ValueError(f"Unknown selector {method}")


def select_features(
    method: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
    n_neighbors: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return selected local feature indices and full selector scores."""
    order, scores = _selector_order_and_scores(
        method=method,
        x_train=x_train,
        y_train=y_train,
        n_neighbors=n_neighbors,
        k=int(k),
    )
    return _top_k(order, k), scores


def fit_selector_pipeline(
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
    x_eval_raw: np.ndarray,
    method: str,
    k: int,
    scaler_name: str,
    add_indicator: bool,
    n_neighbors: int | None,
) -> tuple[np.ndarray, np.ndarray, list[TransformedFeature], np.ndarray, object, object]:
    """Fit imputer/scaler/selector on train data and transform train plus eval matrices."""
    imputer = make_imputer(add_indicator=add_indicator)
    x_train_imp = imputer.fit_transform(x_train_raw)
    x_eval_imp = imputer.transform(x_eval_raw)

    scaler = make_scaler(scaler_name)
    x_train_scaled = scaler.fit_transform(x_train_imp)
    x_eval_scaled = scaler.transform(x_eval_imp)

    # selected_local indexes transformed columns, not raw SECOM feature numbers.
    selected_local, _scores = select_features(
        method=method,
        x_train=x_train_scaled,
        y_train=y_train,
        k=int(k),
        n_neighbors=n_neighbors,
    )
    feature_meta = transformed_feature_metadata_from_imputer(imputer=imputer, raw_feature_count=x_train_raw.shape[1])

    x_train_sel = x_train_scaled[:, selected_local]
    x_eval_sel = x_eval_scaled[:, selected_local]
    return x_train_sel, x_eval_sel, feature_meta, selected_local, imputer, scaler
