from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from secom.workflows import run_study_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the study audit")
    parser.add_argument("--output-dir", default="runs")
    args = parser.parse_args()
    run_study_audit(Path(args.output_dir))


if __name__ == "__main__":
    main()
