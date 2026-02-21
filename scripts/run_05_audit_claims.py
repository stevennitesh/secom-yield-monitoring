from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from secom.workflows.audit import run_artifact_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Runbook 05: artifact + claim audit")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    _ = args.strict
    result = run_artifact_audit(output_dir=args.output_dir)
    if not result.ok:
        for err in result.errors:
            print(f"ERROR: {err}")
        raise SystemExit(1)
    print("QA: PASS")


if __name__ == "__main__":
    main()

