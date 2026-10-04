"""CLI entry point for running the full active study bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _script_path import ensure_src_on_path

ensure_src_on_path()

from secom.workflows import run_full_study

DEFAULT_INPUT_DIR = "data/raw"
DEFAULT_OUTPUT_DIR = "runs/full_study"


def parse_args() -> argparse.Namespace:
    """Parse options for the full benchmark plus temporal robustness bundle."""
    parser = argparse.ArgumentParser(description="Run the full benchmark + temporal study bundle")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--classifiers", help="Comma-separated benchmark classifiers: krr,logreg (default: krr)")
    parser.add_argument("--progress", action="store_true", help="Print study progress to stderr")
    return parser.parse_args()


def main() -> None:
    """Run all study layers and fail strict mode when the active audit fails."""
    args = parse_args()
    options = {}
    if args.classifiers:
        classifiers = [item.strip() for item in args.classifiers.split(",")]
        if not classifiers or any(item not in {"krr", "logreg"} for item in classifiers):
            raise SystemExit("--classifiers must contain krr and/or logreg")
        options["classifiers_run"] = classifiers
    if args.progress:
        options["progress"] = lambda message: print(message, file=sys.stderr, flush=True)
    result = run_full_study(Path(args.input_dir), Path(args.output_dir), **options)

    print(f"PRIMARY_STUDY_STATUS: {result['benchmark']['primary_study_status']}")
    print(f"BENCHMARK_ORIGINAL_STATUS: {result['benchmark_original_status']}")
    print(f"BENCHMARK_TUNED_STATUS: {result['benchmark_tuned_status']}")
    print(f"TEMPORAL_ROBUSTNESS_STATUS: {result['temporal']['temporal_robustness_status']}")
    print(f"FINAL_REPORT: {result['report_path'] or 'not_generated'}")

    for workflow_error in result.get("workflow_errors", []):
        print(f"WORKFLOW_ERROR: {workflow_error['step']}: {workflow_error['error']}")
    for error in result["audit"].errors:
        print(f"ERROR: {error}")
    for warning in result["audit"].warnings:
        print(f"WARNING: {warning}")
    for restriction in result["audit"].claim_restrictions:
        print(f"CLAIM_RESTRICTION: {restriction}")

    if args.strict and not result["audit"].ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
