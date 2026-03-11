from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from secom.models import fit_lane_b_classifier


def test_fit_lane_b_classifier_suppresses_convergence_warning(monkeypatch) -> None:
    class FakeClassifier:
        def fit(self, x_train, y_train):
            warnings.warn(
                "lbfgs failed to converge after 3000 iteration(s)",
                ConvergenceWarning,
            )
            return self

    import secom.models as models

    monkeypatch.setattr(models, "make_lane_b_classifier", lambda c_value: FakeClassifier(), raising=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_lane_b_classifier(
            x_train=np.array([[0.0], [1.0]], dtype=float),
            y_train_bin=np.array([0, 1], dtype=int),
            c_value=1.0,
        )

    assert not any(issubclass(w.category, ConvergenceWarning) for w in caught)
