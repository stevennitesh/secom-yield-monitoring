"""Fixed-budget and combined nested benchmark entrypoints."""

from __future__ import annotations
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any
from secom.artifacts import ensure_reports_dir
from secom.common.paths import project_root_from_repo_structure
from secom.config import ArtifactName, BenchmarkClassifier, SelectorName
from secom.workflows.benchmark_common import (
    prepare_benchmark_dataset,
    normalize_benchmark_run_filters,
    build_cluster_id_map,
)
from secom.workflows.manifest import aggregate_primary_status, write_benchmark_failure


def run_original_benchmark_replication(
    input_dir: Path,
    output_dir: Path,
    *,
    classifiers_run=None,
    selectors_run=None,
    _prepared_data=None,
    _cluster_id_map=None,
) -> dict[str, Any]:
    """Run a literature-inspired fixed-budget study with genuinely nested selection."""
    try:
        return _run_original_benchmark_replication(
            input_dir=input_dir,
            output_dir=output_dir,
            classifiers_run=classifiers_run,
            selectors_run=selectors_run,
            _prepared_data=_prepared_data,
            _cluster_id_map=_cluster_id_map,
        )
    except Exception:
        with suppress(Exception):
            ensure_reports_dir(output_dir)
            write_benchmark_failure(
                manifest_path=output_dir / "reports" / ArtifactName.MANIFEST,
                project_root=project_root_from_repo_structure(),
                original_failed=True,
            )
        raise


def _run_original_benchmark_replication(**kwargs) -> dict[str, Any]:
    from secom.workflows.benchmark_tuned import _run_tuned_benchmark_replication

    return _run_tuned_benchmark_replication(**kwargs, _original=True)


def run_benchmark_replication(
    input_dir: Path,
    output_dir: Path,
    *,
    classifiers_run: list[str] | None = None,
    selectors_run: list[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run original and tuned benchmark workflows with one shared prepared dataset."""
    from secom.workflows.benchmark_tuned import run_tuned_benchmark_replication

    original_classifiers_run, original_selectors_run = normalize_benchmark_run_filters(
        classifiers_run=classifiers_run,
        selectors_run=selectors_run,
        default_classifiers=BenchmarkClassifier.TUNED_DEFAULT,
        default_selectors=SelectorName.ORIGINAL_BENCHMARK,
    )
    tuned_classifiers_run, tuned_selectors_run = normalize_benchmark_run_filters(
        classifiers_run=classifiers_run,
        selectors_run=selectors_run,
        default_classifiers=BenchmarkClassifier.TUNED_DEFAULT,
        default_selectors=SelectorName.ORIGINAL_BENCHMARK,
    )
    prepared_data = prepare_benchmark_dataset(input_dir)
    cluster_id_map = build_cluster_id_map(x_raw=prepared_data["x"])
    original_result = run_original_benchmark_replication(
        input_dir=input_dir,
        output_dir=output_dir,
        classifiers_run=original_classifiers_run,
        selectors_run=original_selectors_run,
        _prepared_data=prepared_data,
        _cluster_id_map=cluster_id_map,
    )
    tuned_result = run_tuned_benchmark_replication(
        input_dir=input_dir,
        output_dir=output_dir,
        classifiers_run=tuned_classifiers_run,
        selectors_run=tuned_selectors_run,
        _prepared_data=prepared_data,
        _cluster_id_map=cluster_id_map,
        progress=progress,
    )
    return {
        "primary_study_status": aggregate_primary_status(
            original_result["benchmark_original_status"],
            tuned_result["benchmark_tuned_status"],
        ),
        "benchmark_original_status": original_result["benchmark_original_status"],
        "benchmark_tuned_status": tuned_result["benchmark_tuned_status"],
        "selectors_run": original_selectors_run,
        "original_selectors_run": original_selectors_run,
        "tuned_selectors_run": tuned_selectors_run,
        "classifiers_run": original_classifiers_run,
        "original_classifiers_run": original_classifiers_run,
        "tuned_classifiers_run": tuned_classifiers_run,
    }
