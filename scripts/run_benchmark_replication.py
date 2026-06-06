"""CLI entry point for the original-plus-tuned benchmark bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _script_path import ensure_src_on_path

ensure_src_on_path()

from secom.workflows import run_benchmark_replication

DEFAULT_INPUT_DIR = "data/raw"
DEFAULT_OUTPUT_DIR = "runs/benchmark_replication"
PASSED_STATUS = "passed"


def _parse_csv_arg(value: str | None) -> list[str] | None:
    """Parse comma-separated CLI overrides into workflow filter lists."""
    if value is None:
        return None
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    return parsed or None


def _print_progress(message: str) -> None:
    """Print workflow progress to stderr without buffering."""
    print(message, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    """Parse options for the primary benchmark bundle command."""
    parser = argparse.ArgumentParser(
        description="Run the benchmark study bundle (original replication + tuned benchmark)"
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--classifiers",
        help="Comma-separated classifier override. Defaults to krr for both original and tuned benchmark layers.",
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the primary benchmark bundle and fail strict mode on benchmark errors."""
    args = parse_args()
    result = run_benchmark_replication(
        Path(args.input_dir),
        Path(args.output_dir),
        classifiers_run=_parse_csv_arg(args.classifiers),
        progress=_print_progress,
    )

    print(f"PRIMARY_STUDY_STATUS: {result['primary_study_status']}")
    print(f"BENCHMARK_ORIGINAL_STATUS: {result['benchmark_original_status']}")
    print(f"BENCHMARK_TUNED_STATUS: {result['benchmark_tuned_status']}")

    if args.strict and result["primary_study_status"] != PASSED_STATUS:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
