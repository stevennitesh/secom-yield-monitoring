from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from secom.pipeline import run_01_data_contract_and_split, run_02_lane_a_replication


def main() -> None:
    parser = argparse.ArgumentParser(description="Runbook 02: Lane A replication")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--lane-a-classifier",
        choices=["krr_strict", "krr_balanced", "logreg", "all"],
        default="all",
        help="Lane A classifier mode: run all three by default, or a single classifier for fast iteration.",
    )
    args = parser.parse_args()

    _ = args.strict
    bundle = run_01_data_contract_and_split(
        input_dir=args.input_dir, output_dir=args.output_dir, project_root=PROJECT_ROOT
    )
    run_02_lane_a_replication(
        bundle=bundle,
        output_dir=args.output_dir,
        lane_a_classifier=None if args.lane_a_classifier == "all" else args.lane_a_classifier,
    )


if __name__ == "__main__":
    main()
