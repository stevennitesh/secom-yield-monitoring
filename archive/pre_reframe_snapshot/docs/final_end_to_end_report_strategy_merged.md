# SECOM Final End-to-End Report Strategy (Compatibility Index)

## Status

This file is no longer the canonical single-file spec.

Canonical specification content now lives under:

- `docs/spec/README.md`

This compatibility index is retained so older references to
`docs/final_end_to_end_report_strategy_merged.md` still land in a stable place.

## Canonical Source Hierarchy

1. `docs/spec/*.md` and `docs/spec/**/README.md`
2. `docs/report_strategy/runbooks/*.md` as procedural execution aids
3. Historical review and challenge docs under `docs/report_strategy/`

If any older text conflicts with `docs/spec/`, the modular `docs/spec/` file wins.

## Modular Navigation

- [Spec Index](spec/README.md)
- [Status and Provenance](spec/00-status-and-provenance.md)
- [End Goal](spec/01-end-goal.md)
- [Validation and Freeze Rules](spec/02-non-negotiable-validation-and-freeze-rules.md)
- [Lane B Data Partition Contract](spec/03-lane-b-data-partition-contract.md)
- [Two-Lane Plan](spec/04-two-lane-plan/README.md)
- [Method Ranking and Challenger Rules](spec/05-stage-b-method-ranking-and-challenger.md)
- [Threshold Policy](spec/06-threshold-policy.md)
- [Lockbox Protocol and Drift Gate](spec/07-lockbox-and-drift-gate.md)
- [Metrics and Feature Identity](spec/08-metrics-and-feature-identity/README.md)
- [Runtime and Checkpointing](spec/09-runtime-and-checkpointing.md)
- [Required Artifacts and Schemas](spec/10-required-artifacts-and-schemas/README.md)
- [Manager-Facing Outputs](spec/11-manager-facing-outputs.md)
- [Final Report Outline](spec/12-final-report-outline.md)
- [Claim Policy](spec/13-claim-policy.md)
- [Pre-Registration Checklist](spec/14-pre-registration-checklist.md)
- [Legacy Crosswalk](spec/legacy-crosswalk.md)

## Provenance

The modular spec consolidates the previously merged strategy material from:

1. `docs/report_strategy/final_end_to_end_report_strategy.md`
2. `docs/report_strategy/agreed_all_selector_multiseed_changes.md`

## Migration Note

Older runbooks cited section numbers inside this file. Those runbooks now cite the specific files under `docs/spec/` that own each rule family.
