"""Distinguishing regressions for input, fitting and publication integrity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from secom.evidence import _publish_snapshot
from secom.config import ArtifactName
from secom.io import load_raw_secom
from secom.models import validated_model_scores
from secom.selection.engine import fit_selector_pipeline
from secom.workflows import benchmark_common, benchmark_tuned, temporal_robustness
from secom.workflows.audit import run_study_audit


@pytest.mark.parametrize("token", ["garbage", "NULL", "inf", "-Infinity"])
def test_invalid_numeric_input_rejected_at_loader(workspace_tmp_dir: Path, token: str) -> None:
    (workspace_tmp_dir / "secom.data").write_text(f"1 {token}\n2 3\n", encoding="utf-8")
    (workspace_tmp_dir / "secom_labels.data").write_text(
        '-1 "19/07/2008 11:55:00"\n1 "19/07/2008 12:55:00"\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="feature cells"):
        load_raw_secom(workspace_tmp_dir)


def test_nan_features_remain_missing_values(workspace_tmp_dir: Path) -> None:
    (workspace_tmp_dir / "secom.data").write_text("1 NaN\n2 3\n", encoding="utf-8")
    (workspace_tmp_dir / "secom_labels.data").write_text(
        '-1 "19/07/2008 11:55:00"\n1 "19/07/2008 12:55:00"\n', encoding="utf-8"
    )
    assert np.isnan(load_raw_secom(workspace_tmp_dir).frame.loc[0, "x1"])


@pytest.mark.parametrize("labels", ['-1 "19/07/2008 11:55:00"\n\n', '-1 "19/07/2008 11:55:00" extra\n'])
def test_blank_or_extra_label_fields_cannot_silently_realign_rows(workspace_tmp_dir: Path, labels: str) -> None:
    (workspace_tmp_dir / "secom.data").write_text("1 2\n2 3\n", encoding="utf-8")
    (workspace_tmp_dir / "secom_labels.data").write_text(labels, encoding="utf-8")
    with pytest.raises(ValueError):
        load_raw_secom(workspace_tmp_dir)


@pytest.mark.parametrize("n_pos", [0, 1, 9])
def test_benchmark_cv_rejects_insufficient_class_count(n_pos: int) -> None:
    frame = pd.DataFrame({"x0": np.arange(30), "y_bin": [1] * n_pos + [0] * (30 - n_pos)})
    with pytest.raises(ValueError, match="at least ten"):
        benchmark_common.prepare_cv(frame, ["x0"])


@pytest.mark.parametrize("labels", [[], [0, 0], [0, 0, 1]])
def test_tuned_inner_cv_never_substitutes_in_sample_evaluation(labels: list[int]) -> None:
    with pytest.raises(ValueError, match="at least two"):
        benchmark_tuned._prepare_inner_selector_views(
            x_outer_train_raw=np.ones((len(labels), 2)),
            y_outer_train=np.array(labels),
            selector="F-test",
            add_indicator=False,
            selector_config={"k": 2, "n_neighbors": None},
        )


def test_tuned_inner_selector_fits_only_inner_training_rows(monkeypatch) -> None:
    original = benchmark_tuned.fit_selector_pipeline
    calls = []

    def spy(**kwargs):
        train_ids = set(kwargs["x_train_raw"][:, 0])
        eval_ids = set(kwargs["x_eval_raw"][:, 0])
        assert not train_ids & eval_ids
        calls.append((train_ids, eval_ids))
        return original(**kwargs)

    monkeypatch.setattr(benchmark_tuned, "fit_selector_pipeline", spy)
    benchmark_tuned._prepare_inner_selector_views(
        x_outer_train_raw=np.column_stack((np.arange(20), np.random.default_rng(42).normal(size=20))),
        y_outer_train=np.tile([0, 1], 10),
        selector="F-test",
        add_indicator=False,
        selector_config={"k": 2, "n_neighbors": None},
    )
    assert len(calls) >= 2
    assert set.union(*(evaluated for _trained, evaluated in calls)) == set(range(20))


def test_preprocessing_and_selection_ignore_evaluation_extremes() -> None:
    train = np.array([[0, np.nan, 0], [1, 3, 0], [2, 5, 1], [3, 7, 1]], dtype=float)
    labels = np.array([0, 0, 1, 1])
    outputs = [
        fit_selector_pipeline(train, labels, evaluated, "F-test", 2, "StandardScaler", True, None)
        for evaluated in (np.array([[4, np.nan, 0]], dtype=float), np.array([[1e9, -1e9, np.nan]]))
    ]
    for position in (0, 3):
        np.testing.assert_array_equal(outputs[0][position], outputs[1][position])
    np.testing.assert_array_equal(outputs[0][4].statistics_, outputs[1][4].statistics_)
    np.testing.assert_array_equal(outputs[0][5].mean_, outputs[1][5].mean_)
    assert outputs[0][2] == outputs[1][2]


@pytest.mark.parametrize("bad_score", [np.nan, np.inf, -np.inf])
def test_invalid_fitted_scores_never_enter_benchmark_cache(monkeypatch, bad_score: float) -> None:
    benchmark_common.reset_model_score_cache()

    class BadModel:
        def predict(self, x):
            return np.full(len(x), bad_score)

    monkeypatch.setattr(benchmark_common, "fit_benchmark_krr_model", lambda *args, **kwargs: BadModel())
    with pytest.raises(ValueError, match="finite array"):
        benchmark_common.fit_classifier_scores(
            "krr", np.ones((4, 2)), np.array([0, 1, 0, 1]), np.ones((2, 2)), {"alpha": 1}
        )
    assert benchmark_common.model_score_cache_info()["entries"] == 0


def test_temporal_predictions_are_validated_before_threshold_search(monkeypatch) -> None:
    class BadModel:
        def predict_proba(self, x):
            return np.full((len(x), 2), np.nan)

    monkeypatch.setattr(temporal_robustness, "fit_temporal_logreg_model", lambda *args, **kwargs: BadModel())
    with pytest.raises(ValueError, match="finite array"):
        temporal_robustness._score_temporal_logreg_view(
            prepared_view={
                "x_train_sel": np.ones((4, 2)),
                "y_train": np.array([0, 1, 0, 1]),
                "x_eval_sel": np.ones((2, 2)),
                "x_calibration_sel": np.ones((2, 2)),
                "y_calibration": np.array([0, 1]),
                "y_eval": np.array([0, 1]),
            },
            c_value=1,
        )


def test_model_score_validation_keeps_valid_prediction_bytes() -> None:
    scores = np.array([-3, 0, 0.25, 5.0])
    assert validated_model_scores(scores) is scores


def _files(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes() for path in directory.rglob("*") if path.is_file()
    }


def _publication_inputs(root: Path) -> tuple[Path, Path]:
    staged, public = root / "staged", root / "public"
    for directory, version in ((staged, "new"), (public, "old")):
        (directory / "evidence").mkdir(parents=True)
        (directory / "final_report.md").write_text(version)
        (directory / "evidence/audit_receipt.json").write_text(version)
    (public / "notes.txt").write_text("manual")
    (public / "evidence/legacy.csv").write_text("old generated")
    return staged, public


@pytest.mark.parametrize("failure", ["copy", "rename"])
def test_publication_failure_preserves_entire_previous_snapshot(
    workspace_tmp_dir: Path, monkeypatch, failure: str
) -> None:
    import secom.evidence as evidence

    staged, public = _publication_inputs(workspace_tmp_dir)
    before = _files(public)
    if failure == "copy":
        original = evidence.shutil.copy2
        original_tree = evidence.shutil.copytree
        copied = []

        def fail_copy(source, target, **kwargs):
            if str(source).startswith(str(staged)):
                copied.append(source)
                if len(copied) == 2:
                    raise OSError("injected staging copy interruption")
            return original(source, target, **kwargs)

        def copy_tree(source, target, *args, **kwargs):
            if "copy_function" not in kwargs and not args:
                kwargs["copy_function"] = fail_copy
            return original_tree(source, target, *args, **kwargs)

        monkeypatch.setattr(evidence.shutil, "copytree", copy_tree)
    else:
        original = Path.rename

        def fail_rename(source, target):
            if source.name == "public" and source != public:
                raise OSError("injected directory rename interruption")
            return original(source, target)

        monkeypatch.setattr(Path, "rename", fail_rename)
    with pytest.raises(OSError, match="injected"):
        _publish_snapshot(staged, public, {"legacy.csv"})
    assert _files(public) == before


def test_publication_recovers_between_renames_and_reruns_consistently(workspace_tmp_dir: Path) -> None:
    staged, public = _publication_inputs(workspace_tmp_dir)
    previous = public.with_name(".public.previous")
    previous.mkdir()
    (previous / "publication.json").write_text(json.dumps({"destination": str(public.resolve())}))
    public.rename(previous / "snapshot")
    _publish_snapshot(staged, public, {"legacy.csv"})
    first = _files(public)
    assert first == {"final_report.md": b"new", "evidence/audit_receipt.json": b"new", "notes.txt": b"manual"}
    _publish_snapshot(staged, public, {"legacy.csv"})
    assert _files(public) == first
    assert not previous.exists()


def test_publication_refuses_unrelated_backup_slot(workspace_tmp_dir: Path) -> None:
    staged, public = _publication_inputs(workspace_tmp_dir)
    previous = public.with_name(".public.previous")
    previous.mkdir()
    (previous / "manual.txt").write_text("preserve")
    with pytest.raises(ValueError, match="unrelated publication backup"):
        _publish_snapshot(staged, public, set())
    assert (previous / "manual.txt").read_text() == "preserve"
    assert (public / "final_report.md").read_text() == "old"


@pytest.mark.parametrize("name", [ArtifactName.BENCHMARK_FOLD_METRICS, ArtifactName.BENCHMARK_TUNED_FOLD_METRICS])
@pytest.mark.parametrize("mutation", ["duplicate", "missing", "fractional"])
def test_persisted_completed_benchmark_requires_exact_fold_coverage(
    active_artifacts_output_dir: Path, name: str, mutation: str
) -> None:
    path = active_artifacts_output_dir / "reports" / name
    frame = pd.read_csv(path)
    if mutation == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif mutation == "missing":
        frame = frame.iloc[1:]
    else:
        frame["fold"] = frame["fold"].astype(float)
        frame.loc[0, "fold"] = 1.5
    frame.to_csv(path, index=False)
    audit = run_study_audit(active_artifacts_output_dir)
    assert not audit.ok
    assert any("fold" in error and ("duplicate" in error or "exactly 10" in error) for error in audit.errors)


@pytest.mark.parametrize("name", [ArtifactName.BENCHMARK_ABLATION, ArtifactName.BENCHMARK_TUNED_ABLATION])
def test_persisted_ablation_must_match_summary_not_only_its_own_arithmetic(
    active_artifacts_output_dir: Path, name: str
) -> None:
    path = active_artifacts_output_dir / "reports" / name
    frame = pd.read_csv(path)
    frame["BER_reference"] += 0.1
    frame["BER_missing_indicator"] += 0.1
    frame.to_csv(path, index=False)
    audit = run_study_audit(active_artifacts_output_dir)
    assert not audit.ok
    assert any("BER values mismatch vs summary" in error for error in audit.errors)
