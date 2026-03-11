from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from secom.reporting import write_report_skeleton


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the final report skeleton from active artifacts")
    parser.add_argument("--output-dir", default="runs")
    args = parser.parse_args()
    out = write_report_skeleton(Path(args.output_dir))
    print(out)


if __name__ == "__main__":
    main()
