from __future__ import annotations

import numpy as np

from secom.feature_select import univariate


def test_score_f_test_skips_constant_columns(monkeypatch) -> None:
    x = np.array(
        [
            [1.0, 0.0, 5.0, 2.0],
            [1.0, 1.0, 5.0, 3.0],
            [1.0, 0.0, 5.0, 4.0],
            [1.0, 1.0, 5.0, 5.0],
        ],
        dtype=float,
    )
    y = np.array([0, 1, 0, 1], dtype=int)

    called = {}

    def fake_f_classif(x_sub: np.ndarray, y_sub: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        called["shape"] = x_sub.shape
        assert np.array_equal(y_sub, y)
        return np.array([7.0, 3.0], dtype=float), np.array([0.1, 0.2], dtype=float)

    monkeypatch.setattr(univariate, "f_classif", fake_f_classif)

    scores = univariate.score_f_test(x, y)

    assert called["shape"] == (4, 2)
    assert np.isneginf(scores[[0, 2]]).all()
    assert np.allclose(scores[[1, 3]], [7.0, 3.0])
