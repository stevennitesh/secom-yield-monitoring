from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from secom.workflows import run_full_study


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full benchmark + temporal study bundle")
    parser.add_argument("--input-dir", default="data/raw")
    parser.add_argument("--output-dir", default="runs/full_study")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    result = run_full_study(Path(args.input_dir), Path(args.output_dir))
    print(f"PRIMARY_STUDY_STATUS: {result['benchmark']['primary_study_status']}")
    print(f"BENCHMARK_ORIGINAL_STATUS: {result['benchmark_original_status']}")
    print(f"BENCHMARK_TUNED_STATUS: {result['benchmark_tuned_status']}")
    print(f"TEMPORAL_ROBUSTNESS_STATUS: {result['temporal']['temporal_robustness_status']}")
    print(f"REPORT_SKELETON: {result['report_path']}")
    for error in result["audit"].errors:
        print(f"ERROR: {error}")
    for warning in result["audit"].warnings:
        print(f"WARNING: {warning}")
    for restriction in result["audit"].claim_restrictions:
        print(f"CLAIM_RESTRICTION: {restriction}")
    if args.strict and not result["audit"].ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
