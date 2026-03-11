from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from secom.config import SelectorName  # noqa: E402
from secom.workflows.lane_a import LaneAThresholdMode, run_lane_a_replication  # noqa: E402
from secom.workflows.split_contract import run_split_contract  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Runbook 02: Lane A replication")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--lane-a-classifier",
        choices=["krr", "logreg", "krr_strict", "all"],
        default="all",
        help=(
            "Lane A classifier mode: run official classifiers (`krr`,`logreg`) by default, "
            "or run a single classifier for fast iteration. `krr_strict` is benchmark-only."
        ),
    )
    parser.add_argument(
        "--skip-relieff",
        action="store_true",
        help="Exclude ReliefF from Lane A runs to speed up experimentation.",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=LaneAThresholdMode.ALL,
        default=LaneAThresholdMode.GLOBAL_OOF,
        help=(
            "Threshold calibration mode. "
            "`global_oof` uses one pooled OOF threshold per config; "
            "`per_fold_train` calibrates threshold on each fold's train split "
            "and evaluates fold test at that threshold."
        ),
    )
    args = parser.parse_args()

    _ = args.strict
    bundle = run_split_contract(
        input_dir=args.input_dir, output_dir=args.output_dir, project_root=PROJECT_ROOT
    )
    lane_a_selectors = (
        [s for s in SelectorName.ACTIVE if s != SelectorName.RELIEFF]
        if args.skip_relieff
        else list(SelectorName.ACTIVE)
    )

    run_lane_a_replication(
        bundle=bundle,
        output_dir=args.output_dir,
        lane_a_classifier=None if args.lane_a_classifier == "all" else args.lane_a_classifier,
        selectors_run=lane_a_selectors,
        threshold_mode=args.threshold_mode,
    )


if __name__ == "__main__":
    main()
