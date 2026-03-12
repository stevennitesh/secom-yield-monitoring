# 08 Audit and Claim Semantics

## Scope

This file defines how the rebuilt project distinguishes hard failures, warnings, and claim restrictions.

## Hard Errors

The following are project-level hard errors:

1. missing required active artifacts,
2. schema failures,
3. original benchmark validation failures,
4. tuned benchmark validation failures,
5. inconsistencies between manifested study status and produced active artifacts.

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

## See Also

- [01 Study Goal](01-study-goal.md)
- [02 Benchmark Replication Study](02-benchmark-replication-study.md)
- [04 Temporal Robustness Study](04-temporal-robustness-study.md)
- [06 Report Structure](06-report-structure.md)
- [07 Artifact Contracts](07-artifact-contracts.md)
