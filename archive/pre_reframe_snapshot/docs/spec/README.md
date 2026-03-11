---
title: "SECOM Canonical Spec Index"
status: "canonical"
keywords:
  - canonical
  - spec
  - navigation
  - source of truth
---
# SECOM Canonical Spec

This directory is the canonical source of truth for the SECOM yield-monitoring specification.
It replaces the single-file merged spec with modular, searchable documents organized by concern.

## Source Hierarchy

1. Files in `docs/spec/` are canonical.
2. `docs/final_end_to_end_report_strategy_merged.md` is a compatibility index and migration pointer.
3. `docs/report_strategy/runbooks/*.md` are procedural execution checklists derived from this spec.
4. Historical review and challenge documents under `docs/report_strategy/` are reference material, not canonical policy.

## Navigation

- [00 Status and Provenance](00-status-and-provenance.md)
- [01 End Goal](01-end-goal.md)
- [02 Non-Negotiable Validation and Freeze Rules](02-non-negotiable-validation-and-freeze-rules.md)
- [03 Lane B Data Partition Contract](03-lane-b-data-partition-contract.md)
- [04 Two-Lane Plan](04-two-lane-plan/README.md)
- [05 Stage B Method Ranking and Challenger Rules](05-stage-b-method-ranking-and-challenger.md)
- [06 Threshold Policy](06-threshold-policy.md)
- [07 Lockbox Protocol and Drift Gate](07-lockbox-and-drift-gate.md)
- [08 Metrics and Feature Identity](08-metrics-and-feature-identity/README.md)
- [09 Runtime and Checkpointing](09-runtime-and-checkpointing.md)
- [10 Required Artifacts and Schemas](10-required-artifacts-and-schemas/README.md)
- [11 Manager-Facing Outputs](11-manager-facing-outputs.md)
- [12 Final Report Outline](12-final-report-outline.md)
- [13 Claim Policy](13-claim-policy.md)
- [14 Pre-Registration Checklist](14-pre-registration-checklist.md)
- [Legacy Crosswalk](legacy-crosswalk.md)

## Searchability Conventions

- Filenames retain legacy section numbering where possible.
- Each file includes YAML front matter with `spec_id`, `applies_to`, and `keywords`.
- Cross-cutting rule families are split into smaller files where the original monolith was too dense.
- Runbooks should link to file paths, not section-number ranges inside the retired monolith.
