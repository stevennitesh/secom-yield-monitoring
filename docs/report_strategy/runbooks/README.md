# SECOM Implementation Runbooks

These runbooks are execution checklists derived from the canonical strategy:

- `docs/final_end_to_end_report_strategy_merged.md`

Rule:

1. If any runbook line conflicts with the canonical strategy, the canonical strategy wins.
2. Runbooks are procedural aids for implementation order, validation gates, and artifact completeness.

Runbook sequence:

1. `01_data_contract_and_split.md`
2. `02_lane_a_replication.md`
3. `03_lane_b_stage_a_stage_b.md`
4. `04_phase2_phase3_freeze_lockbox.md`
5. `05_artifacts_and_claim_checks.md`

## API Migration Note

Workflow entrypoints were renamed to domain-oriented symbols and the legacy
pipeline compatibility module was removed.

Legacy -> Current:

1. Runbook 01 entrypoint -> `run_split_contract`
2. Runbook 02 entrypoint -> `run_lane_a_replication`
3. Runbook 03 entrypoint -> `run_lane_b_stage_ab`
4. Runbook 04 entrypoint -> `run_freeze_lockbox`
5. Runbook 05 entrypoint -> `run_artifact_audit`
