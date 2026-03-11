from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from secom.workflows import run_benchmark_replication


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the benchmark replication study")
    parser.add_argument("--input-dir", default="data/raw")
    parser.add_argument("--output-dir", default="runs/benchmark_replication")
    args = parser.parse_args()
    run_benchmark_replication(Path(args.input_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
