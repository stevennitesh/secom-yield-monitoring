"""Build an offline HTML reader report from an audited curated snapshot; fit no models."""

from __future__ import annotations

import argparse
from pathlib import Path
from _script_path import ensure_src_on_path

ensure_src_on_path()

from secom.html_report import build_html_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=Path("docs/results"), help="Curated report, figures and audit receipt"
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh portable HTML report folder")
    args = parser.parse_args()
    try:
        print(build_html_report(args.input_dir, args.output_dir))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
