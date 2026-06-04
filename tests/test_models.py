"""Tests for model construction and warning handling."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from secom.models import fit_temporal_logreg_model


def test_fit_temporal_logreg_model_suppresses_convergence_warning(monkeypatch) -> None:
    """Temporal logistic fitting should hide expected convergence noise."""

    class FakeClassifier:
        """Classifier double that emits the warning wrapper code suppresses."""

        def fit(self, x_train, y_train):
            """Emit a convergence warning while preserving sklearn-like chaining."""
            warnings.warn(
                "lbfgs failed to converge after 3000 iteration(s)",
                ConvergenceWarning,
            )
            return self

    import secom.models as models

    monkeypatch.setattr(models, "make_temporal_logreg_model", lambda c_value: FakeClassifier(), raising=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_temporal_logreg_model(
            x_train=np.array([[0.0], [1.0]], dtype=float),
            y_train_bin=np.array([0, 1], dtype=int),
            c_value=1.0,
        )

    assert not any(issubclass(w.category, ConvergenceWarning) for w in caught)
