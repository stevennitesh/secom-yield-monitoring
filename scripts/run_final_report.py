from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from secom.reporting import write_final_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the final markdown report from active artifacts"
    )
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--export-pdf", action="store_true")
    args = parser.parse_args()
    out = write_final_report(Path(args.output_dir), export_pdf=args.export_pdf)
    print(out)


if __name__ == "__main__":
    main()
