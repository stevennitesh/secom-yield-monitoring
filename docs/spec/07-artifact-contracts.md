# 07 Artifact Contracts

## Scope

This file defines the artifact families for the active benchmark-first study.

## Benchmark Study Artifact Families

The original replication study should produce:

1. benchmark sweep
2. benchmark best config
3. benchmark fold metrics
4. benchmark summary
5. benchmark ablation
6. benchmark full-fit summary
7. feature stability
8. feature report

The tuned benchmark study should produce:

1. benchmark tuned search
2. benchmark tuned best config
3. benchmark tuned fold metrics
4. benchmark tuned summary
5. benchmark tuned ablation
6. benchmark tuned full-fit summary
7. benchmark tuned feature stability
8. benchmark tuned feature report

These artifacts are canonical for:

1. original replication benchmark metrics,
2. tuned benchmark metrics,
3. uncertainty summaries,
4. ablation results,
5. selector/classifier comparison,
6. feature-stability interpretation.

## Secondary Study Artifact Family

The temporal robustness study should produce:

1. temporal split metadata
2. temporal selector screening
3. temporal model selection
4. temporal inner-CV results
5. temporal freeze results
6. temporal lockbox results
7. temporal drift summary
8. temporal MSPC summary
9. temporal cost curves
10. temporal manager-facing outputs
11. bounded DEV KRR search
12. bounded DEV KRR held-out predictions
13. bounded DEV KRR per-period metrics
14. selected-model held-out calibration scores
15. calibration counts and threshold diagnostics

These artifacts are canonical for:

1. temporal stress-test metrics,
2. drift and claim-restriction evidence,
3. lockbox comparison outputs,
4. illustrative workload and cost framing.

## Metric Tier Mapping

### Headline Metric Sources

For each original/tuned prefix, `*_predictions.csv`, `*_procedure_fold_metrics.csv`, and `*_procedure_summary.csv` are required. They cover joint, values_only, values_and_indicators, missingness_only, and all_pass procedures on identical held-out IDs/folds. Prediction schema records raw sample ID, fold, true label, finite score, frozen threshold (sentinels allowed), prediction, study/procedure, actual selected family/input/configuration, and stable configuration identity. Summary counts and descriptive fold spread recompute from predictions; never optimize a global pooled outer threshold.

Original `benchmark_sweep.csv` now records inner search candidates/selection per outer fold, like tuned search. Best configs are descriptive modal inner selections. Existing family summaries, stability, paired ablations, and full-data in-sample fits remain exploratory/interpretive consumers. Headline figures source procedure metrics, not retrospective family minima.

### Secondary Metric Sources

Temporal robustness metrics should come from the secondary artifact family, especially:

1. temporal model selection
2. temporal lockbox results
3. temporal drift summary
4. temporal MSPC summary

### Illustrative Metric Sources

Illustrative industry-facing metrics should come from:

1. temporal cost curves
2. temporal manager-facing outputs

These outputs must not be treated as production-validated operating metrics.

## DEV Comparator and Calibration Artifacts

Required active temporal CSVs additionally include:

- `temporal_krr_search.csv`: every declared candidate and chronological inner period, stable candidate/gamma-multiplier identity, effective gamma and actual selected width, BER/AUC, threshold, recomputable confusion counts, exact FIT/calibration/validation raw IDs and timestamp boundaries, inner mean objective and selected flags for joint, values-only and combined procedures.
- `temporal_krr_predictions.csv`: available DEV procedures' held-out IDs/labels/scores/frozen thresholds/configs, path fraction and FIT/calibration boundary/count receipts.
- `temporal_krr_fold_metrics.csv`: all three procedures x 20% main/30% sensitivity x fixed periods, availability/reason, selected candidate and inner objective, held-out counts and per-period BER/AUC. Unavailable paths have no predictions.
- `temporal_calibration_scores.csv`: selected LR outer-joint/final-role and available KRR calibration scores/IDs/configs/thresholds, exact earlier FIT IDs/counts and timestamp/fraction lineage.
- `temporal_calibration_diagnostics.csv`: recomputable calibration class counts, single-example BER steps, frozen threshold, full-calibration flagged fraction, fragility/availability flags and fixed-score leave-one-failure-out ranges.

These artifacts join required schemas, manifest CSV hashes and source archives. Infeasibility may leave header-only KRR search/prediction tables; the metrics table still covers every declared path with a reason. Each 20%/30% procedure independently tunes within its earlier FIT prefix. KRR uses the shared tuned grids, StandardScaler and existing configured benchmark selector families; it adds no later15% freeze/role/MSPC/manager suite. All outer evaluation IDs and calendar periods remain those of the LR DEV study. Reports keep secondary comparison tables separate from the benchmark headlines and LR roles, with six existing figures.

Timestamp-tie receipts are required: KRR search has `inner_timestamp_tie`, selected fold metrics count `n_inner_timestamp_ties`, and calibration scores/diagnostics have `fit_calibration_timestamp_tie`. These flags recompute from boundaries. Stable `(timestamp, raw_row_id)` order establishes neither physical event order nor independence within ties. Exact FIT/calibration IDs must be disjoint; selected LR outer FIT/calibration IDs must also be disjoint from their corresponding held-out fold IDs. Their frozen boundary receipts must match predictions and precede the fixed outer calendar evaluation. Final LR role receipts retain existing calibration guards without adding a later raw-ID prediction consumer.

## Manifest Rule

The run manifest must distinguish between:

1. primary-study status,
2. benchmark original status,
3. benchmark tuned status,
4. temporal-study status,
5. temporal claim restrictions,
6. industrialization notes or gaps where applicable.

The manifest declares the original reference KRR alpha/gamma grid separately from the tuned alpha=0.1/1/10/100 and gamma multipliers=0.1/0.2/1/2 divided by each actual selected width. The original grid remains alpha=0.1/1/10 and gamma=None/0.01/0.1/1. Tuned KRR has 16 classifier configurations per selector budget versus 12 in the original; LR C, selector budgets, folds and seeds retain their existing grids. It records the DEV-only comparator scope and calibration fractions 20%/30%, keeping original/tuned primary status and temporal warnings/restrictions separate. Candidate receipts use `gamma_multiplier` as stable identity across inner/outer/full widths and numeric `gamma` as the effective fitted value; original reference receipts have no multiplier. BER-first exact ties prefer fewer features and stronger regularization before deterministic remaining order.

## Naming Rule

Artifact naming follows the active study structure. `src/secom/config.py` owns the concrete filenames and required families; `src/secom/workflows/audit.py` enforces schema and claim checks. Legacy lane numbering and historical CSVs do not define the current contract.

## Complete-Run Provenance and Evidence Export

The curated snapshot also contains a generated `index.html` reader report and `html_provenance.json`. HTML generation verifies the eight core snapshot hashes and audit status before reading results. Its separate provenance binds the HTML to those inputs, the scientific audit receipt and its two renderer/CLI owners. The scientific receipt and execution manifest remain unchanged. Export builds the companion from fresh staging files so it cannot retain stale results. The companion is reproducible from the curated snapshot alone, without local CSVs, data access or model fitting; it need not be present in the scientific source/artifact archive.

A full study requires a fresh output directory. Its manifest additionally records exact input-file hashes and observed dataset shape/counts/timestamps/missingness, LF-normalized source/spec content identity, resolved packages, seeds/grids/classifiers, UTC start/end times, modeling duration and thread settings. Git dirty status remains truthful; the base commit is never replaced by a publication placeholder. Completed CSV hashes bind the artifact set to that run. Focused study commands retain layer-specific manifests; public export requires the complete-run provenance.

The public snapshot exporter validates all artifacts and their CSV hashes, verifies the current source/spec against the run identity, and preserves the original manifest unchanged. Git tracks the technical and HTML reports, six figures, and small manifest/audit/provenance files. Detailed CSVs remain under ignored `runs/<study>/reports/`; a compressed archive of complete reports and exact normalized source is saved under `runs/<study>/evidence/`. The archive excludes raw data, and its hash appears in the public receipt. Publication revision and source execution identity are distinct concepts. Export removes only named generated CSVs and the legacy ZIP from the public snapshot, preserving manual files and local runs.

Spec clarification changes the source/spec identity even when model behavior is unchanged. Existing snapshot hashes remain evidence of the earlier execution; never rewrite them to match current prose. Audit historical artifacts against their recorded source. Ordinary full-evidence export requires matching source/spec; an explicitly requested editorial refresh uses the narrow path below. See [Development and Evidence Workflow](../development.md#documentation-changes-and-historical-provenance).

### Presentation-Only Refresh

The optional presentation destination is fresh and disjoint from the completed source run and public snapshot. It reads the unchanged CSVs in place, audits them, and verifies every CSV, the byte-identical execution manifest, exact executed source inventory and ordered spec hash against the source run's existing evidence ZIP before creating outputs. Only explicit reporting/figure/reader-vocabulary/export owners, their two CLI owners and specs06/07/08 may differ in current source; model/preprocessing/selection/split/metric/workflow/config/pin changes and specs01–05 are rejected. The normal exporter identity check remains unchanged.

The fresh editorial run contains report, shared skeleton, six figures and an unchanged manifest, without duplicated CSVs. Its compressed archive streams original CSVs, includes exact executed source under `source/`, current identifiable rendering source under `presentation_source/` and a presentation-provenance record. The small publication audit record distinguishes executed source from rendering source and enumerates changed rendering paths, source/archive identities, audit warnings/restrictions and published hashes. Old run/report/archive bytes stay unchanged. Publication preserves manual files and uses a normally created atomic directory candidate so host-readable access survives rename; failure preserves the previous snapshot. No training or numerical-result update is implied.

## See Also

- [02 Benchmark Replication Study](02-benchmark-replication-study.md)
- [03 Feature Stability and Interpretation](03-feature-stability-and-interpretation.md)
- [04 Temporal Robustness Study](04-temporal-robustness-study.md)
- [08 Audit and Claim Semantics](08-audit-and-claim-semantics.md)

## Temporal Prediction and Calibration Contract

`temporal_predictions.csv` and `temporal_procedure_fold_metrics.csv` bind the DEV joint procedure to disjoint chronological evaluation samples. Later-block rows include counts and exact conditional binomial intervals. Drift uses confirmatory_claims_allowed=False, same-model calibration score reference, and selected-indicator missingness-rate shift. MSPC uses calibration_selected_MSPC_source, frozen_threshold/counts, and observed_mean_inter_alarm_spacing. Manager outputs identify held_out_DEV_calibration and illustrative mean_weekly_policy_not_per_week_cap. All active CSVs participate in manifest hashes, audits, source archive/export, and report consumers.
