"""Export a validated full-study public evidence snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path
from _script_path import ensure_src_on_path

ensure_src_on_path()
from secom.common.paths import project_root_from_repo_structure
from secom.evidence import export_public_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Completed full-study run")
    parser.add_argument("--destination", type=Path, default=Path("docs/results"))
    parser.add_argument(
        "--presentation-output-dir",
        type=Path,
        help="Fresh presentation-only run; verify archived execution and retain its manifest",
    )
    args = parser.parse_args()
    print(
        export_public_snapshot(
            args.output_dir,
            args.destination,
            project_root_from_repo_structure(),
            presentation_output_dir=args.presentation_output_dir,
        )
    )


if __name__ == "__main__":
    main()
