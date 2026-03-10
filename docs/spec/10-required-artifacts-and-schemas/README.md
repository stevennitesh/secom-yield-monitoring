---
spec_id: "10"
title: "Required Artifacts and Schemas"
status: "canonical"
legacy_source:
  - "docs/final_end_to_end_report_strategy_merged.md §10"
applies_to:
  - "Lane A"
  - "Lane B"
used_by_runbooks:
  - "../../report_strategy/runbooks/01_data_contract_and_split.md"
  - "../../report_strategy/runbooks/02_lane_a_replication.md"
  - "../../report_strategy/runbooks/03_lane_b_stage_a_stage_b.md"
  - "../../report_strategy/runbooks/04_phase2_phase3_freeze_lockbox.md"
  - "../../report_strategy/runbooks/05_artifacts_and_claim_checks.md"
keywords:
  - artifacts
  - schemas
  - manifest
  - row grain
---
# 10 Required Artifacts and Schemas

## Scope

The original artifact section mixed global conventions, required artifact sets, and per-file schema rules.
This directory splits those concerns by artifact family.

## Canonical Modules

- [10.1 Common Artifact Conventions](10.1-common-artifact-conventions.md)
- [10.2 Lane A Artifacts](10.2-lane-a-artifacts.md)
- [10.3 Stage B Selection Artifacts](10.3-stage-b-selection-artifacts.md)
- [10.4 Freeze, Lockbox, and MSPC Artifacts](10.4-freeze-lockbox-and-mspc-artifacts.md)
- [10.5 Feature, Drift, and Run Manifest Artifacts](10.5-feature-drift-and-run-manifest.md)
