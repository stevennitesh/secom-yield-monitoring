"""CLI entry point for the secondary temporal robustness study."""

from __future__ import annotations

import argparse
from pathlib import Path

from _script_path import ensure_src_on_path

ensure_src_on_path()

from secom.workflows import run_temporal_robustness

DEFAULT_INPUT_DIR = "data/raw"
DEFAULT_OUTPUT_DIR = "runs/temporal_robustness"
STRICT_ALLOWED_STATUSES = {"passed", "warning"}


def parse_args() -> argparse.Namespace:
    """Parse options for the secondary temporal robustness stress test."""
    parser = argparse.ArgumentParser(description="Run the temporal robustness study")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run temporal stress testing and surface claim restrictions distinctly."""
    args = parse_args()
    result = run_temporal_robustness(Path(args.input_dir), Path(args.output_dir))

    print(f"TEMPORAL_ROBUSTNESS_STATUS: {result['temporal_robustness_status']}")
    if result.get("claim_restrictions"):
        for restriction in result["claim_restrictions"]:
            print(f"CLAIM_RESTRICTION: {restriction}")

    if args.strict and result["temporal_robustness_status"] not in STRICT_ALLOWED_STATUSES:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
