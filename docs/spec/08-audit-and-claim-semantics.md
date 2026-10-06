# 08 Audit and Claim Semantics

## Scope

This file defines how the active project distinguishes hard failures, warnings, and claim restrictions.

## Hard Errors

The following are project-level hard errors:

1. missing required active artifacts,
2. schema failures,
3. original benchmark validation failures,
4. tuned benchmark validation failures,
5. inconsistencies between manifested study status and produced active artifacts,
6. configuration, prediction/count, sample-ID or boundary lineage contradictions,
7. changed or missing manifested artifact hashes.

## Secondary Study Restrictions

The following are scoped to the temporal robustness study unless explicitly elevated:

1. drift-gated claim restrictions,
2. lockbox superiority restrictions,
3. temporal stress-test failures that do not invalidate the benchmark replication study,
4. operational framing that cannot be claimed as production readiness.

## Required Audit Output Categories

Audit outputs should distinguish:

1. original benchmark errors,
2. tuned benchmark errors,
3. shared schema errors,
4. temporal-study warnings,
5. temporal claim restrictions.

## Claim Rule

If the temporal stress-test study yields a restricted claim, the result may still be reported descriptively, but it must not invalidate the original or tuned benchmark studies by default.

## Provenance Consistency

When the manifest carries CSV hashes, any changed or missing hashed CSV is a hard audit error. Public evidence export also rejects mismatched source/spec identity or incomplete run provenance. Temporal warnings and restrictions remain visible in a passing export receipt; passing the artifact audit does not mean production validation or hosted-CI verification.

An explicitly requested presentation-only refresh is the sole narrow editorial exception: verify unchanged CSVs and the archived executed manifest/source/spec before output mutation, reject differences outside the enumerated presentation owners, and record current rendering provenance separately. This does not relax normal export rejection or scientific audit checks. Missing/changed archive members, scientific source/spec differences, overlapping/nonfresh output and incomplete execution identity block refresh. An editorial render never supplies new scientific evidence or relabels model execution; saved numeric results and all restrictions persist.

## See Also

- [01 Study Goal](01-study-goal.md)
- [02 Benchmark Replication Study](02-benchmark-replication-study.md)
- [04 Temporal Robustness Study](04-temporal-robustness-study.md)
- [06 Report Structure](06-report-structure.md)
- [07 Artifact Contracts](07-artifact-contracts.md)

## Revised Selection and Prediction Checks

Joint choices must match BER-first inner-selected family/config/thresholds, never outer minima. Prediction labels, scores, thresholds, configs, unique raw IDs, same-fold original/tuned pairing, per-fold confusion counts, and pooled summaries must agree. Missing/changed required procedure artifacts are hard errors. Disjoint temporal DEV IDs and frozen MSPC/calibration semantics are audited; low temporal class counts and unavailable training/calibration are warnings/restrictions. Retrospective later evaluation always excludes fresh confirmatory/production-readiness/superiority claims; heuristic drift PASS cannot authorize them.

## Bounded Comparator and Calibration Checks

The five required DEV KRR/calibration artifacts in spec07 participate in hard schema, hash and lineage checks whenever the temporal study is active. Recompute the declared candidate-grid coverage, effective gamma/actual selected width, per-period inner confusion counts and BER, mean inner objective, BER-first selected configuration and flags, outer prediction counts/AUC, fixed evaluation ID/label pairing, and calibration threshold/diagnostics. Candidate identity uses the stable gamma multiplier; differently sized inner/outer/full fits must not become different candidates.

For each 20%/30% KRR path, exact inner FIT/calibration/validation sets must be disjoint and lie within that path's earlier outer FIT, excluding all outer calibration/evaluation IDs. Selected LR outer-joint FIT and calibration IDs must also be disjoint from their corresponding held-out temporal prediction fold, even when score and diagnostic receipt strings have been changed consistently. LR prediction boundary receipts must match calibration receipts, and FIT/calibration must precede the outer calendar test. Final primary/challenger retain their existing retained-model calibration guards; no later raw-ID prediction consumer is required by this extension.

Deterministic inner row boundaries allow nondecreasing timestamps with disjoint IDs and explicit tie flags/counts; outer calendar evaluation remains strictly later. Tie diagnostics must not claim physical ordering or independence from raw row IDs. Fewer than ten calibration failures produces a fragility warning, not an availability minimum or permission to move boundaries. Undefined BER steps and leave-one-failure-out ranges are guarded by class counts. Fixed-score ranges describe calibration instability only, not confidence intervals or future/algorithm uncertainty.

Infeasible comparator paths explicitly cover their fixed periods with unavailable reasons and no predictions. Such temporal unavailability, sparse class counts and descriptive instability are warnings; they do not invalidate the primary benchmarks. Missing artifacts/schema, altered hashes and contradictory scientific separation receipts remain hard errors. Neither an outer minimum nor sensitivity-window comparison can select a champion or change model/window/threshold declarations. KRR remains DEV-only. Retrospective later15% restrictions, illustrative workload/cost framing and unsupported production-readiness restrictions remain in force.

The bounded tuned grid changes the executed design and source/spec identity. Implementation tests alone supply no new SECOM performance, predictive improvement or measured speedup claim. Historical runs remain unchanged and are audited against their recorded source. New scientific evidence requires a fresh authorized run; presentation-only publication follows the separate narrow provenance rule above.
