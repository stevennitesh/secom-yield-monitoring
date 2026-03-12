from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from secom.workflows import run_original_benchmark_replication


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the original 1:1 benchmark replication study")
    parser.add_argument("--input-dir", default="data/raw")
    parser.add_argument("--output-dir", default="runs/original_replication")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    result = run_original_benchmark_replication(Path(args.input_dir), Path(args.output_dir))
    print(f"BENCHMARK_ORIGINAL_STATUS: {result['benchmark_original_status']}")
    print(f"PRIMARY_STUDY_STATUS: {result['primary_study_status']}")
    if args.strict and result["benchmark_original_status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
