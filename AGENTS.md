# AGENTS.md

## Purpose and Scope

This research project demonstrates reproducible SECOM research, scientific judgment and maintainable ML engineering. The benchmark implementation and public evidence already exist. Prioritize correct study design, defensible claims, clean architecture, coherent artifacts and measured performance, in that order.

Keep work proportional. Production services, deployment, new datasets and expanded experiments require separate scope decisions. Gap analysis explains missing evidence; it does not authorize building a production system. Backward compatibility is required only when requested; replace obsolete contracts cleanly.

## Read the Relevant Owner

| Work | Owner |
| --- | --- |
| Reviewer story and runnable setup | [README](README.md) |
| Reading paths and publication decisions | [Documentation index](docs/README.md) |
| Code, checks and evidence procedures | [Development guide](docs/development.md) |
| Study, metrics, artifacts or claims | [Specifications, in reading order](docs/spec/README.md) |
| Report order and metric priorities | [Report specification](docs/spec/06-report-structure.md) |
| Artifact names and audit classifications | [Artifact contract](docs/spec/07-artifact-contracts.md), [claim semantics](docs/spec/08-audit-and-claim-semantics.md) |
| Runtime changes | [Measured performance](docs/performance.md) |

Specs own accepted scientific meaning; code and tests establish implementation. Resolve disagreements explicitly. Historical plans, prior runs, external handoffs and memory are context, not a work queue. `learning/` contains teaching demos, not study evidence.

## Scientific Boundaries

Keep the ordered evidence tiers separate: original fixed-budget replication, tuned primary benchmark, secondary temporal stress, then industrialization gaps. Temporal issues restrict operational claims without automatically invalidating benchmark results.

Headline claims use the complete joint inner-selected procedure's held-out predictions; family minima are exploratory. Prioritize BER, failure recall, pass specificity, uncertainty, ablations, selector/classifier comparisons and feature stability. AUC, MCC and F2 remain supporting diagnostics.

Both benchmarks share outer folds, nested BER-first selection and inner-held-out thresholds. Temporal studies use disjoint chronological blocks and retained fitting/calibration models; the final 15% is retrospective. Frozen rules and evaluation-label-selected diagnostics stay separate. Workload and costs are illustrative. Audit errors, temporal warnings and claim restrictions remain distinct as specified.

## Execution and Preservation

- Use the active environment and focused checks. Substantial code changes require relevant tests and supported CLI smoke checks; documentation alone needs no training.
- Full evidence runs require a fresh directory, both `krr,logreg` classifiers and `--progress --strict`; omitting classifiers defaults to KRR.
- Runtime optimization must preserve folds, grids, seeds, thresholds and exact numerical behavior.
- Do not commit unless explicitly asked.
- Raw data and full outputs stay in ignored `data/` and `runs/`. Git keeps the curated report, six figures, execution manifest and audit receipt. Detailed CSVs and ZIP archives stay local.
- Preserve completed and partial runs. Never edit generated reports or manifests to match newer documentation. `final_report.md` is canonical; the skeleton is a debugging aid.
- A new real-data evidence run or public snapshot replacement requires explicit authorization and validated export. Authorized editorial refreshes fit no models, use fresh destinations and record rendering separately from execution.
- Spec edits, including prose, change source identity. Current tests or rendering do not refresh historical scientific results. Follow the [provenance procedure](docs/development.md#documentation-changes-and-historical-provenance).
