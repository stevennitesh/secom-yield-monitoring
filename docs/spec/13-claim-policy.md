---
spec_id: "13"
title: "Claim Policy"
status: "canonical"
legacy_source:
  - "docs/final_end_to_end_report_strategy_merged.md §13"
applies_to:
  - "final report"
used_by_runbooks:
  - "../report_strategy/runbooks/05_artifacts_and_claim_checks.md"
keywords:
  - claims
  - superiority
  - drift gate
  - benchmark
---
# 13 Claim Policy

## Global Precedence

1. Drift gate status governs lockbox claim eligibility per frozen supervised model.
2. If a frozen supervised model's drift gate is `HIGH_SHIFT`, superiority claims are disallowed for that model.

## Allowed Claims

1. Replication-lane improvement if CI supports it.
2. Time-aware selection is more deployment-realistic than random CV.
3. Thresholding was set without lockbox tuning.
4. Supervised advantage over MSPC only if, on the lockbox slice, the supervised model's `TPR_at_TNR90` (from `reports/final_lockbox_result.csv`) exceeds MSPC's `best_MSPC_TPR_at_TNR90` (from the `eval_scope='lockbox'` row in `reports/mspc_baseline.csv`) and that model's drift gate is not `HIGH_SHIFT`.
5. If Lane B BER is worse than Lane A BER, describe this as expected under stricter temporal validation.
6. If MSPC matches or exceeds supervised at `TNR=90%`, report this as a valid finding.
7. Benchmark-improvement claim versus `33.5%` BER is allowed only when the `classifier='krr', selector='F-test', replication_mode='strict'` mean BER 95% CI upper bound is below `0.335` (CI from `reports/lane_a_global_summary.csv` using percentile bootstrap on Lane A fold BER; `n_boot=1000`, seed `42`), and both anchor rows `(F-test,krr,strict)` and `(F-test,krr,with_missing_indicators)` exist.
8. Primary head-to-head benchmark claim versus `33.5%` must use `classifier='krr', selector='F-test', replication_mode='strict'` BER; other selectors/classifiers are supportive evidence only.

## Forbidden Claims

1. Causality from feature importance.
2. Cross-fab or long-horizon generalization without new validation.
3. Lockbox superiority after any post-lockbox tuning.
4. Supervised superiority when MSPC is equal or better at matched `TNR=90%`.
5. Any superiority claim when drift gate is `HIGH_SHIFT`.
