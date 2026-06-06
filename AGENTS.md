# AGENTS.md

## Project Mandate

This repo is being refactored around the active SECOM benchmark-first study design, not around preserving legacy workflows.

Backward compatibility is not required unless the user explicitly asks for it. If old specs, runbooks, CLI entrypoints, tests, reports, or artifact contracts conflict with the corrected study design, replace them cleanly instead of layering compatibility shims.

Optimize for:

1. correct study design
2. defensible claims
3. clean architecture
4. coherent artifacts and audits
5. performance and implementation quality

## Active Study Contract

The active study has four ordered layers:

1. original benchmark replication as the faithful literature-style study
2. tuned benchmark as the improved primary benchmark study
3. temporal robustness as secondary stress-test evidence
4. industrialization-gap analysis as required report content

Project conclusions must keep these evidence tiers separate. Temporal robustness findings can restrict operational claims, but they must not automatically invalidate the original or tuned benchmark studies.

## Canonical Sources

Use `docs/spec/` as the active source of truth, in README reading order:

1. `docs/spec/01-study-goal.md`
2. `docs/spec/02-benchmark-replication-study.md`
3. `docs/spec/03-feature-stability-and-interpretation.md`
4. `docs/spec/04-temporal-robustness-study.md`
5. `docs/spec/05-industrialization-gap-analysis.md`
6. `docs/spec/06-report-structure.md`
7. `docs/spec/07-artifact-contracts.md`
8. `docs/spec/08-audit-and-claim-semantics.md`

Historical pre-reframe snapshots and legacy report outputs are not active contracts. If recovered from git history, treat them as context only.

## Artifact And Claim Rules

Artifact names should follow the active study structure:

- original benchmark: `benchmark_*` plus `feature_stability.csv` and `feature_report.csv`
- tuned benchmark: `benchmark_tuned_*`
- temporal robustness: `temporal_*`
- shared: `run_manifest.json`, `final_report.md`, `figures/*.png`

`final_report.md` is the canonical generated report. `final_report_skeleton.md` is only a scaffold/debugging aid.

Audit and reporting logic must distinguish:

- hard errors for missing required active artifacts, schema failures, benchmark validation failures, and manifest/artifact inconsistency
- warnings for temporal-study issues
- claim restrictions for temporal drift, lockbox-superiority limits, and unsupported production-readiness claims

Do not present illustrative workload, cost, or operating-point outputs as production-validated metrics.

## Report Rules

The final report must preserve this narrative order:

1. original replication
2. tuned benchmark
3. original vs tuned comparison
4. feature stability and interpretation
5. temporal robustness
6. industrialization gaps
7. conclusions and next data requirements

Headline claims should prioritize `BER`, `TPR` / `True+`, `TNR` / `True-`, uncertainty summaries, ablation deltas, selector/classifier comparisons, and benchmark feature-stability results. Metrics such as `ROC_AUC`, `PR_AUC`, `MCC`, and `F2` are supporting diagnostics unless the active spec changes.

## Entry Points

Active commands are documented in `README.md`. Prefer focused commands for the changed surface, especially:

- `python -m pytest -q tests/<relevant_test>.py`
- `python scripts/run_benchmark_replication.py --input-dir data/raw --output-dir runs/benchmark_replication --strict`
- `python scripts/run_temporal_robustness.py --input-dir data/raw --output-dir runs/temporal_robustness --strict`
- `python scripts/run_full_study.py --input-dir data/raw --output-dir runs/full_study --strict`
- `python scripts/run_final_report.py --output-dir runs/full_study`
- `python scripts/run_audit.py --output-dir runs/<study> --strict`

For large refactors, run the relevant suite plus any still-supported entrypoint smoke checks.

## Repo-Specific Guardrails

Do not commit unless the user explicitly asks.

Generated study outputs belong under `runs/` unless the user asks to refresh tracked reference artifacts. Keep local scratch output out of tracked files.
