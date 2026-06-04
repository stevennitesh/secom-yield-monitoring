"""Tests for artifact names and required artifact family boundaries."""

from __future__ import annotations

from secom.config import ArtifactName, REQUIRED_ARTIFACTS_PRIMARY, REQUIRED_ARTIFACTS_TEMPORAL


def test_artifact_names_follow_new_study_structure() -> None:
    """Assert artifact filenames follow the active benchmark-first contract."""
    assert ArtifactName.BENCHMARK_SWEEP == "benchmark_sweep.csv"
    assert ArtifactName.BENCHMARK_SUMMARY == "benchmark_summary.csv"
    assert ArtifactName.BENCHMARK_TUNED_SEARCH == "benchmark_tuned_search.csv"
    assert ArtifactName.BENCHMARK_TUNED_SUMMARY == "benchmark_tuned_summary.csv"
    assert ArtifactName.TEMPORAL_LOCKBOX == "temporal_lockbox.csv"
    assert ArtifactName.TEMPORAL_MANAGER_OUTPUTS == "temporal_manager_outputs.csv"
    assert ArtifactName.MANIFEST == "run_manifest.json"


def test_required_artifact_families_are_disjoint_except_manifest() -> None:
    """Assert primary and temporal artifact requirements stay separated."""
    primary = set(REQUIRED_ARTIFACTS_PRIMARY)
    temporal = set(REQUIRED_ARTIFACTS_TEMPORAL)
    assert ArtifactName.MANIFEST in primary
    assert ArtifactName.MANIFEST not in temporal
    assert primary.intersection(temporal) == set()
