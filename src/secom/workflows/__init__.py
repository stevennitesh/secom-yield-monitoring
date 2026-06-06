"""Workflow entry points for benchmark, temporal, audit, and full-study runs."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_WORKFLOW_EXPORTS = {
    "run_benchmark_replication": ("secom.workflows.benchmark_replication", "run_benchmark_replication"),
    "run_full_study": ("secom.workflows.full_study", "run_full_study"),
    "run_original_benchmark_replication": (
        "secom.workflows.benchmark_replication",
        "run_original_benchmark_replication",
    ),
    "run_study_audit": ("secom.workflows.audit", "run_study_audit"),
    "run_temporal_robustness": ("secom.workflows.temporal_robustness", "run_temporal_robustness"),
    "run_tuned_benchmark_replication": (
        "secom.workflows.benchmark_tuned",
        "run_tuned_benchmark_replication",
    ),
}

__all__ = list(_WORKFLOW_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load workflow entry points lazily so lightweight CLI paths stay quiet."""
    if name not in _WORKFLOW_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _WORKFLOW_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
