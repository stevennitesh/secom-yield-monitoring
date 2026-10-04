"""Model reuse must bind complete inputs and preserve independent score arrays."""

import numpy as np

from secom.workflows import benchmark_common as common


def test_model_score_cache_binds_inputs_and_returns_copies(monkeypatch):
    common.reset_model_score_cache()
    original = common.fit_benchmark_krr_model
    calls = []

    def spy(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(common, "fit_benchmark_krr_model", spy)
    x = np.random.default_rng(42).normal(size=(12, 3))
    y = np.tile([0, 1], 6)
    eval_x = x[:4].copy()
    cfg = {"alpha": 1.0, "gamma": None}
    train, evaluated = common.fit_classifier_scores("krr", x, y, eval_x, cfg)
    expected = (train.copy(), evaluated.copy())
    train[:] = 99
    evaluated[:] = 99
    cached = common.fit_classifier_scores("krr", x.copy(), y.copy(), eval_x.copy(), cfg.copy())
    assert len(calls) == 1
    for actual, wanted in zip(cached, expected):
        np.testing.assert_array_equal(actual, wanted)
    common.fit_classifier_scores("krr", x + 0.1, y, eval_x, cfg)
    common.fit_classifier_scores("krr", x, 1 - y, eval_x, cfg)
    common.fit_classifier_scores("krr", x, y, eval_x + 0.1, cfg)
    common.fit_classifier_scores("krr", x, y, eval_x, {"alpha": 0.1, "gamma": None})
    common.fit_classifier_scores("krr", x, y, eval_x, cfg, include_train_scores=False)
    assert len(calls) == 6
    assert common.model_score_cache_info()["hits"] == 1


def test_model_score_cache_distinguishes_memory_layout():
    """Equal values with different strides may follow different floating-point reductions."""
    common.reset_model_score_cache()
    x = np.random.default_rng(42).normal(size=(12, 3))
    y = np.tile([0, 1], 6)
    for values in (x, np.asfortranarray(x)):
        common.fit_classifier_scores("krr", values, y, values[:4], {"alpha": 1.0, "gamma": None})
    assert common.model_score_cache_info()["misses"] == 2


def test_model_score_cache_evicts_and_distinguishes_factory_changes(monkeypatch):
    common.reset_model_score_cache()
    monkeypatch.setattr(common, "MODEL_SCORE_CACHE_SIZE", 2)
    original = common.fit_benchmark_krr_model
    calls = []

    def spy(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(common, "fit_benchmark_krr_model", spy)
    x = np.random.default_rng(42).normal(size=(12, 3))
    y = np.tile([0, 1], 6)
    for alpha in (1.0, 0.1, 10.0, 1.0):
        common.fit_classifier_scores("krr", x, y, x[:4], {"alpha": alpha, "gamma": None})
    assert len(calls) == 4
    assert common.model_score_cache_info()["entries"] == 2
    monkeypatch.setattr(common, "fit_benchmark_krr_model", original)
    common.fit_classifier_scores("krr", x, y, x[:4], {"alpha": 1.0, "gamma": None})
    assert common.model_score_cache_info()["misses"] == 5
