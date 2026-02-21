from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from secom.workflows.split_contract import run_split_contract


def main() -> None:
    parser = argparse.ArgumentParser(description="Runbook 01: data contract and split")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    _ = args.strict
    run_split_contract(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        project_root=PROJECT_ROOT,
    )


if __name__ == "__main__":
    main()

