from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from secom.workflows import run_temporal_robustness


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the temporal robustness study")
    parser.add_argument("--input-dir", default="data/raw")
    parser.add_argument("--output-dir", default="runs/temporal_robustness")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    result = run_temporal_robustness(Path(args.input_dir), Path(args.output_dir))
    print(f"TEMPORAL_ROBUSTNESS_STATUS: {result['temporal_robustness_status']}")
    if result.get("claim_restrictions"):
        for restriction in result["claim_restrictions"]:
            print(f"CLAIM_RESTRICTION: {restriction}")
    if args.strict and result["temporal_robustness_status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
