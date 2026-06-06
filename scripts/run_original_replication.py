"""CLI entry point for the faithful original benchmark replication."""

from __future__ import annotations

import argparse
from pathlib import Path

from _script_path import ensure_src_on_path

ensure_src_on_path()

from secom.workflows import run_original_benchmark_replication

DEFAULT_INPUT_DIR = "data/raw"
DEFAULT_OUTPUT_DIR = "runs/original_replication"
PASSED_STATUS = "passed"


def parse_args() -> argparse.Namespace:
    """Parse options for the faithful original benchmark replication command."""
    parser = argparse.ArgumentParser(description="Run the original 1:1 benchmark replication study")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the original replication and fail strict mode only on replication failure."""
    args = parse_args()
    result = run_original_benchmark_replication(Path(args.input_dir), Path(args.output_dir))

    print(f"BENCHMARK_ORIGINAL_STATUS: {result['benchmark_original_status']}")

    if args.strict and result["benchmark_original_status"] != PASSED_STATUS:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
