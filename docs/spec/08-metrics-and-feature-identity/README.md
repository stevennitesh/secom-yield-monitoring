---
spec_id: "08"
title: "Metrics and Feature Identity"
status: "canonical"
legacy_source:
  - "docs/final_end_to_end_report_strategy_merged.md §8"
applies_to:
  - "Lane A"
  - "Lane B"
used_by_runbooks:
  - "../../report_strategy/runbooks/02_lane_a_replication.md"
  - "../../report_strategy/runbooks/03_lane_b_stage_a_stage_b.md"
  - "../../report_strategy/runbooks/04_phase2_phase3_freeze_lockbox.md"
keywords:
  - metrics
  - feature identity
  - feature stability
---
# 08 Metrics and Feature Identity

## Scope

The original metrics section mixed metric definitions, reporting rules, and transformed-feature identity rules.
This directory separates them into two canonical files.

## Canonical Modules

- [08.1 Metric Definitions](08.1-metric-definitions.md)
- [08.2 Feature Stability and Identity](08.2-feature-stability-and-identity.md)
