"""CLI entry point for the tuned benchmark workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _script_path import ensure_src_on_path

ensure_src_on_path()

from secom.workflows import run_tuned_benchmark_replication

DEFAULT_INPUT_DIR = "data/raw"
DEFAULT_OUTPUT_DIR = "runs/benchmark_tuned"
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
    """Parse options for the tuned benchmark command."""
    parser = argparse.ArgumentParser(description="Run the tuned benchmark study")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--classifiers",
        help="Comma-separated classifier override. Defaults to krr; use krr,logreg for the full classifier family.",
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the tuned benchmark and fail strict mode only on tuned benchmark failure."""
    args = parse_args()
    result = run_tuned_benchmark_replication(
        Path(args.input_dir),
        Path(args.output_dir),
        classifiers_run=_parse_csv_arg(args.classifiers),
        progress=_print_progress,
    )

    print(f"BENCHMARK_TUNED_STATUS: {result['benchmark_tuned_status']}")

    if args.strict and result["benchmark_tuned_status"] != PASSED_STATUS:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
