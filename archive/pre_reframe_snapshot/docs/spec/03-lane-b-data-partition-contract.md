---
spec_id: "03"
title: "Lane B Data Partition Contract"
status: "canonical"
legacy_source:
  - "docs/final_end_to_end_report_strategy_merged.md Section 3"
applies_to:
  - "Lane B"
keywords:
  - partition
  - lockbox
  - dev split
  - timestamps
---
# 03 Lane B Data Partition Contract

## Scope

This section defines the DEV/LOCKBOX split and label conventions for Lane B.

## Timestamp Contract

1. Timestamp parsing contract is fixed:
   1. interpret dataset timestamps as day-first strings in format `DD/MM/YYYY HH:MM:SS`,
   2. parse with `errors='coerce'` (unparseable -> `NaT`),
   3. treat parsed timestamps as UTC-naive (no timezone conversion is applied).

## DEV/LOCKBOX Split Rule

1. Drop rows with unparseable timestamps (`NaT`) before sorting.
2. Add deterministic row identity before sorting:
   1. define `raw_row_id` as the 0-based row index from the raw SECOM file as read (before dropping `NaT`).
3. Sort all remaining rows by timestamp ascending.
   1. if timestamps tie, break ties by `raw_row_id` ascending,
   2. sorting must be stable.
4. Reserve lockbox by sample count, not time-span duration:
   1. `LOCKBOX = last floor(0.15 * N)` samples after sorting.
   2. `DEV = first N - floor(0.15 * N)` samples.
5. `N` is the row count after dropping `NaT`.
6. Any prior lockbox results are diagnostic only and non-final.

## Label and Metric Conventions

1. Label contract is fixed:
   1. raw dataset label `y_raw in {-1,+1}` where `-1=pass` and `+1=fail`,
   2. `y_bin = 1` for fail (positive class) and `0` for pass (negative class).
2. Metric conventions are fixed:
   1. `True+` means TPR/sensitivity on the fail class (`y_bin=1`),
   2. `True-` means TNR/specificity on the pass class (`y_bin=0`).

## Lane Scope Note

1. Lane B uses the `DEV/LOCKBOX` partition defined above for all time-aware validation and freeze.
2. Lane A replication intentionally uses the full dataset (`DEV+LOCKBOX`) for benchmark comparability and does not change Lane B's lockbox discipline.

## See Also

- [04.1 Lane A Replication](04-two-lane-plan/04.1-lane-a-replication.md)
- [04.2.1 Outer Time-Aware Folds](04-two-lane-plan/04.2.1-outer-time-aware-folds.md)
- [14 Pre-Registration Checklist](14-pre-registration-checklist.md)
