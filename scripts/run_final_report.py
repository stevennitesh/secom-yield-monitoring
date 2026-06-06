"""CLI entry point for rendering the canonical final report."""

from __future__ import annotations

import argparse
from pathlib import Path

from _script_path import ensure_src_on_path

ensure_src_on_path()

DEFAULT_OUTPUT_DIR = "runs/full_study"


def parse_args() -> argparse.Namespace:
    """Parse options for rendering the canonical final report."""
    parser = argparse.ArgumentParser(description="Generate the final markdown report from active artifacts")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--export-pdf", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Generate the final report from existing study artifacts."""
    args = parse_args()

    from secom.reporting import write_final_report

    try:
        out = write_final_report(Path(args.output_dir), export_pdf=args.export_pdf)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from None
    print(out)


if __name__ == "__main__":
    main()
