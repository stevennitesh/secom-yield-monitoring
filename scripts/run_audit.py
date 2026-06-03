from __future__ import annotations

import argparse
from pathlib import Path

from _script_path import ensure_src_on_path

ensure_src_on_path()

from secom.workflows import run_study_audit

DEFAULT_OUTPUT_DIR = "runs"


def parse_args() -> argparse.Namespace:
    """Parse audit options for an already-produced study output directory."""
    parser = argparse.ArgumentParser(description="Run the study audit")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run active artifact audits and mirror audit severity in the exit code."""
    args = parse_args()
    result = run_study_audit(Path(args.output_dir))

    for error in result.errors:
        print(f"ERROR: {error}")
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    for restriction in result.claim_restrictions:
        print(f"CLAIM_RESTRICTION: {restriction}")

    if args.strict and not result.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
