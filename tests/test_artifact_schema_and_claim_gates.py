from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from secom.pipeline import (
    run_01_data_contract_and_split,
    run_02_lane_a_replication,
    run_05_artifact_and_claim_audit,
)
from secom.config import SelectorName


def test_artifact_audit_lane_a_only_mode(synthetic_input_dir: Path, workspace_tmp_dir: Path) -> None:
    out_dir = workspace_tmp_dir / "out_audit_ok"
    project_root = Path(__file__).resolve().parents[1]
    bundle = run_01_data_contract_and_split(synthetic_input_dir, out_dir, project_root)
    run_02_lane_a_replication(
        bundle,
        out_dir,
        lane_a_classifier="krr_strict",
        selectors_run=[SelectorName.F_TEST, SelectorName.S2N],
    )

    manifest_path = out_dir / "reports" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["lane_b_feasible"] = False
    manifest["lane_b_infeasible_reason"] = "min_class_count_lt_5_for_inner_cv"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    result = run_05_artifact_and_claim_audit(out_dir)
    assert result.ok, result.errors


def test_artifact_audit_catches_delta_sign_error(synthetic_input_dir: Path, workspace_tmp_dir: Path) -> None:
    out_dir = workspace_tmp_dir / "out_audit_bad"
    project_root = Path(__file__).resolve().parents[1]
    bundle = run_01_data_contract_and_split(synthetic_input_dir, out_dir, project_root)
    run_02_lane_a_replication(
        bundle,
        out_dir,
        lane_a_classifier="krr_strict",
        selectors_run=[SelectorName.F_TEST, SelectorName.S2N],
    )

    manifest_path = out_dir / "reports" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["lane_b_feasible"] = False
    manifest["lane_b_infeasible_reason"] = "min_class_count_lt_5_for_inner_cv"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    ab = pd.read_csv(out_dir / "reports" / "baseline_missing_indicator_ablation.csv")
    ab.loc[:, "delta_BER"] = 999.0
    ab.to_csv(out_dir / "reports" / "baseline_missing_indicator_ablation.csv", index=False)

    result = run_05_artifact_and_claim_audit(out_dir)
    assert not result.ok
    assert any("delta_BER sign mismatch" in e for e in result.errors)
