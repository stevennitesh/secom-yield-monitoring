"""CLI entry point for the secondary temporal robustness study."""

from __future__ import annotations

import argparse
from pathlib import Path

from _script_path import ensure_src_on_path

ensure_src_on_path()

from secom.workflows import run_temporal_robustness

from _script_status import temporal_status_from_manifest, workflow_error_line

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
    output_dir = Path(args.output_dir)
    workflow_error: str | None = None
    try:
        result = run_temporal_robustness(Path(args.input_dir), output_dir)
    except Exception as exc:
        result = temporal_status_from_manifest(output_dir)
        workflow_error = workflow_error_line("temporal", exc)

    print(f"TEMPORAL_ROBUSTNESS_STATUS: {result['temporal_robustness_status']}")
    if result.get("claim_restrictions"):
        for restriction in result["claim_restrictions"]:
            print(f"CLAIM_RESTRICTION: {restriction}")
    if workflow_error is not None:
        print(workflow_error)
        raise SystemExit(1)

    if args.strict and result["temporal_robustness_status"] not in STRICT_ALLOWED_STATUSES:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
