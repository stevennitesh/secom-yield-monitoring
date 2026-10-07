# 06 Report Structure

## Required Narrative Order

The canonical technical report remains `final_report.md`. A self-contained HTML companion presents its audited values and the same six figures with section navigation, accessible result tables and expandable supporting detail. It fits no models and preserves all evaluation and claim boundaries.

1. Executive summary and dataset/study scope
2. Original replication design, inner search summary, joint held-out procedure and baselines
3. Tuned benchmark design, inner search summary, joint held-out procedure and baselines
4. Original versus tuned paired comparison on identical held-out IDs/folds
5. Feature stability and descriptive interpretation for both studies
6. Temporal robustness, drift restrictions, MSPC, and illustrative workload/costs
7. Industrialization gaps
8. Conclusions and next data requirements, followed by provenance

## Metric and Headline Policy

Add an evidence-derived result summary and section navigation. Keep detailed tables in expandable appendix sections with distinct headings. Translate artifact fields and categorical values into reader-facing labels, explain the important mappings, and split wide tables into panels with shared row identifiers without dropping recorded fields. The primary benchmark chart focuses on the complete selected procedure and separates fold-mean error from pooled confusion counts; input-mode deltas remain supporting material. Clearly separate chronological test periods from the retrospective final block and state that calibration fractions refer to the earlier training region.

Lead with the recorded manufacturing pass/fail question and why raw accuracy misleads. Define balanced error, failure recall, pass specificity, calibration, cross-validation and both model families at first use. Main text, tables and charts use external vocabulary: complete selected procedure, measurements only, measurements plus missing flags, earlier fitting samples, held-out calibration and later test periods. Preserve machine identifiers, exact search tables and provenance in technical appendices. Explain bounded tuning's recall/false-alert tradeoff and limited feature engineering without declaring a hindsight winner.

The six existing figures retain their filenames and evidence owners. Comparison/delta figures use predefined held-out procedures and a positive balanced-error reduction sign. Stability labels anonymous value/missing-flag columns and exploratory family selection; examples match plotted inputs. Drift distinguishes full-DEV prevalence, selected raw-feature fitting references and same-model held-out calibration score references. Selected-indicator missingness records render as ordinary rows with named populations and units. The later LR/MSPC chart shows frozen-threshold counts; evaluation-label 90%-specificity diagnostics remain separately tabulated. Workload captions identify held-out calibration quantities and the mean-weekly policy scope. Cost captions identify frozen-rule retrospective later-block counts, the per-sample denominator and hypothetical cost-ratio meaning. Captions state sample, threshold and descriptive uncertainty limits. State the dataset's historical observation window and actual timestamp ranges of chronological test periods.

Headline metrics source the joint procedure held-out predictions: BER, TPR/True+, TNR/True-, pooled confusion counts, and descriptive fold mean/std/range. Original versus tuned uses paired fold deltas and their descriptive spread. Family minima, selector/classifier comparisons, and family input-mode ablations are explicitly exploratory nested diagnostics, not separately validated champions. ROC_AUC, PR_AUC, MCC, and F2 remain supporting diagnostics.

Compare all-pass, missingness-only, values-only, and values-plus-indicators on the same folds. Explain fixed40 versus tuned k10/20/40 and ReliefF neighbors10 versus5/10/20. Both parameter/threshold searches are nested, BER-first, with training-only preprocessing and inner OOF calibration. Declare the OOF/refit score-distribution limitation.

Do not label fold bootstrap spread as algorithm-performance CIs. Feature selection frequencies over overlapping training folds and full-data fitted coefficient summaries are descriptive. Use absolute_scaled_coefficient and stability_weighted_coefficient; do not imply causal or expected economic effects.

## Temporal and Operational Disclosure

Describe the logistic-regression role study and separate bounded DEV-only KRR comparison, fixed nonoverlapping calendar DEV tests, deterministic chronological inner tuning, FIT/calibration/future separation, and a retained model with no post-threshold refit. The last15% block is always retrospective, not fresh confirmatory. Report counts, 95% exact conditional-independent-trial TPR/TNR intervals, and sparse-class limitations. Drift heuristics cannot authorize superiority.

Distinguish frozen thresholds from retrospective evaluation-label TNR90 diagnostics. MSPC source/thresholds freeze on calibration. Report observed mean inter-alarm spacing, not ARL0. KS uses same-model held-out calibration reference; selected-indicator missingness-rate shifts describe regime associations.

Manager workload names held-out DEV calibration, is illustrative, and describes a mean-weekly policy rather than a per-week hard cap. Calibration workload does not validate future capacity/cost. Keep temporal evidence and required industrialization gaps separate from benchmark conclusions.

## Source and Historical Status

UCI describes KRR; McCann/Johnston Table2 labels Naive Bayes. Local raw measurements have590 columns versus metadata591. Scientific design changes require a new full real-data run before new performance claims. An explicitly requested presentation-only refresh may render unchanged audited evidence under the narrow provenance rules in specs07/08. Preserve every historical run and execution manifest byte; a new rendering source never becomes the executed modeling source.

## Bounded Tuning and Calibration Disclosure

Explain original 12 versus tuned 16 KRR configurations per selector budget, alpha100 coverage and dimension-relative gamma. State the regularization tie order and train-established width receipts. In the temporal section show per-period BER/AUC for main20% and independently tuned30% sensitivity paths on identical fixed tests, keeping LR roles and retrospective later-block evidence separate. Show calibration class counts, single-example BER steps and fixed-score LOFO ranges with fragility warnings and no confidence-interval claim. The existing six figures retain their benchmark/LR evidence owners; no KRR later-block or production figure is added.

Label RBF gamma accurately and explain its inverse relation to similarity range. Include the valid automatic reference setting in search counts and selected-configuration tables; distinguish it from parameters inapplicable to logistic regression. Tuned gamma equals its declared multiplier divided by the actual selected input count. Cost curves use distinct line styles and markers as well as color. The visual guide includes text counts for the frozen-rule retrospective comparison.
