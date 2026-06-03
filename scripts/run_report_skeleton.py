from __future__ import annotations

import argparse
from pathlib import Path

from _script_path import ensure_src_on_path

ensure_src_on_path()

from secom.reporting import write_report_skeleton

DEFAULT_OUTPUT_DIR = "runs"


def parse_args() -> argparse.Namespace:
    """Parse options for the scaffold report command."""
    parser = argparse.ArgumentParser(description="Generate the final report skeleton from active artifacts")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Generate the scaffold report used for debugging report assembly."""
    args = parse_args()
    out = write_report_skeleton(Path(args.output_dir))
    print(out)


if __name__ == "__main__":
    main()
