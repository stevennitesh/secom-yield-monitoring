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
    parser.add_argument(
        "--presentation-output-dir",
        type=Path,
        help="Fresh editorial output with verified archived execution; no model fitting",
    )
    return parser.parse_args()


def main() -> None:
    """Generate the final report from existing study artifacts."""
    args = parse_args()

    from secom.reporting import write_final_report

    try:
        if args.presentation_output_dir is not None:
            if args.export_pdf:
                raise ValueError("Presentation refresh emits Markdown and six PNGs; PDF is a separate optional render")
            from secom.common.paths import project_root_from_repo_structure
            from secom.evidence import refresh_presentation

            out = refresh_presentation(
                Path(args.output_dir), args.presentation_output_dir, project_root_from_repo_structure()
            )
        else:
            out = write_final_report(Path(args.output_dir), export_pdf=args.export_pdf)
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from None
    print(out)


if __name__ == "__main__":
    main()
