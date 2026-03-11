from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


from secom.workflows.split_contract import run_split_contract


def test_split_contract_deterministic(synthetic_input_dir: Path, workspace_tmp_dir: Path) -> None:
    out_dir = workspace_tmp_dir / "out"
    project_root = Path(__file__).resolve().parents[1]
    bundle_a = run_split_contract(synthetic_input_dir, out_dir, project_root)
    bundle_b = run_split_contract(synthetic_input_dir, out_dir, project_root)

    assert len(bundle_a.all_data) == len(bundle_b.all_data)
    assert bundle_a.dev.shape[0] + bundle_a.lockbox.shape[0] == bundle_a.all_data.shape[0]
    assert bundle_a.dev["sorted_row_id"].max() < bundle_a.lockbox["sorted_row_id"].min()

    manifest = json.loads((out_dir / "reports" / "run_manifest.json").read_text(encoding="utf-8"))
    assert "lane_b_feasible" in manifest
    assert "outer_fold_week_ranges" in manifest
    assert manifest["strategy_doc_path"] == "docs/spec/README.md"

    expected_sha = hashlib.sha256((project_root / "docs" / "spec" / "README.md").read_bytes()).hexdigest()
    assert manifest["strategy_doc_sha256"] == expected_sha

    assert bundle_a.dev_with_weeks["week_idx"].min() == 0
    assert (bundle_a.dev_with_weeks["week_label"] == bundle_a.dev_with_weeks["week_idx"] + 1).all()

    assert manifest["lane_b_feasible"] is True
    assert manifest["outer_fold_plan_used"] == "fallback_3fold"
    assert manifest["outer_fold_week_ranges"] == [
        {"outer_fold": 1, "train_weeks": [1, 4], "test_weeks": [5, 16]},
        {"outer_fold": 2, "train_weeks": [1, 6], "test_weeks": [7, 16]},
        {"outer_fold": 3, "train_weeks": [1, 8], "test_weeks": [9, 16]},
    ]


def test_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_01_split.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Runbook 01" in result.stdout
