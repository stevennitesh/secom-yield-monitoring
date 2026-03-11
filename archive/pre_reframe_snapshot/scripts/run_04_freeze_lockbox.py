from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from secom.workflows.freeze_lockbox import run_freeze_lockbox  # noqa: E402
from secom.workflows.lane_b import run_lane_b_stage_ab  # noqa: E402
from secom.workflows.split_contract import run_split_contract  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Runbook 04: Phase 2/3 freeze + lockbox")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    _ = args.strict
    bundle = run_split_contract(
        input_dir=args.input_dir, output_dir=args.output_dir, project_root=PROJECT_ROOT
    )
    stage3 = run_lane_b_stage_ab(bundle=bundle, output_dir=args.output_dir)
    run_freeze_lockbox(bundle=bundle, stage3=stage3, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

