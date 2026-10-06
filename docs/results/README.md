# SECOM Study in Charts

**The question:** can historical manufacturing measurements distinguish rare failures, and will the learned patterns transfer to later samples?

The [project overview](../../README.md) explains the dataset and setup. This guide follows the findings; the [technical report](final_report.md) contains exact methods and diagnostics.

## 1. Tuning Reduced False Alerts, with a Cost

| Complete training-selected procedure | Mean balanced error | Failures caught | False alerts |
| --- | ---: | ---: | ---: |
| Reference: fixed 40 inputs | 31.43% | 70 | 445 |
| Tuned: 10/20/40 inputs | 30.86% | 66 | 371 |

Balanced error averages missed-failure and false-alert rates. Both procedures use the same ten outer test folds; inner folds choose inputs, model settings and threshold using training data only. Rates average folds, while counts pool predictions over 104 failures and 1,463 passes.

![Reference and tuned procedures on shared test folds](figures/benchmark_comparison.png)

Whiskers show observed fold ranges, not confidence intervals. Tuning lowers mean error by **0.57 percentage points**, with **74 fewer false alerts and 4 fewer failures caught**. The bounded search supports a modest tradeoff. It does not establish a global optimum or exact published-study replication. Headlines evaluate the complete selection procedure; input-mode comparisons are secondary.

## 2. Repeatedly Selected Inputs Are Clues

![How often anonymous measurements and missing flags are selected](figures/feature_stability.png)

The chart shows anonymous, zero-based column identifiers and distinguishes measurement values from missing flags. A “family” combines a selector, model and input mode. The displayed families were chosen after test-summary inspection, and their training folds overlap: frequencies are exploratory associations, not causes or an independently validated champion.

Explicit feature engineering added missing-measurement flags. Ratios, interactions and historical trends were not systematically explored; kernel ridge can nevertheless learn nonlinear relationships implicitly.

## 3. Later Samples Expose Transfer Limits

Each period follows this order; calibration is the last **20% of its earlier training region**, not 20% of the full dataset:

| Earlier samples → | Separate calibration → | Later test samples |
| --- | --- | --- |
| Tune and fit the model | Choose and freeze its alert threshold | Score once, with no refit |

### Three chronological test periods

| Three disjoint chronological test periods | Logistic regression | Kernel ridge: 20% calibration |
| --- | ---: | ---: |
| Mean period balanced error | 50.58% | 47.47% |
| Pooled fraction flagged, 750 samples | 31.5% | 88.5% |
| Failures in selected calibration sets | 4 / 3 / 6 | 4 / 3 / 6 |

Fitting/calibration procedures differ, limiting direct algorithm comparisons. Sparse calibration failures make alert thresholds fragile. These tests restrict transfer claims while remaining secondary to the shuffled benchmarks.

### Final retrospective block: a separate comparison

This chart uses **235 different samples** and the retained primary logistic-regression model versus MSPC. It is separate from the 750-sample, complete-procedure comparison above.

![Failures caught and false alerts with frozen rules on the retrospective later block](figures/lockbox_vs_mspc.png)

Rules were frozen on earlier calibration. The final block contains **235 samples: 9 failures and 226 passes**. It was already exposed through benchmarking and prior reporting, so it is retrospective. Statistical process control (MSPC) is a separate baseline fitted on earlier passing samples. Label-selected 90%-specificity diagnostics remain in the technical appendix.

<details>
<summary>Supporting charts: tuning differences, drift, and workload</summary>

![Balanced-error differences; positive means lower error after tuning](figures/tuned_vs_original_delta.png)

The complete procedure is the headline; predefined input contrasts help explain it. A larger input-mode improvement does not establish a better complete procedure.

![Measurement and score shifts for the primary logistic-regression procedure](figures/temporal_drift.png)

Measurement shifts use earlier fitting samples as reference. Score shifts use held-out calibration scores from the same retained model. These diagnostics indicate change, without identifying its cause.

![Illustrative calibration workload and hypothetical error costs](figures/workload_cost_framing.png)

The cost ratio means missed-failure cost divided by false-alert cost. The 10% workload rule limits the **mean weekly flagged fraction on calibration data**; it guarantees neither every week's workload nor future capacity. These assumptions are illustrative, not validated manufacturing economics.

</details>

## 4. What Would Make the Next Study Stronger?

| Missing evidence | Question it would answer |
| --- | --- |
| Named measurements and collection timing | What is measured, and is it available before the outcome? |
| Tool/device identifiers and intervention records | Where does variation arise, and can an action improve outcomes? |
| Independent later data | Do the learned patterns and frozen rules transfer? |
| Actual review capacity and error costs | Which alert policy is useful in practice? |

Anonymous associations establish no causal, early-warning, intervention-benefit or production-readiness claim.

## Evidence

| Record | What it establishes |
| --- | --- |
| [Execution manifest](evidence/run_manifest.json) | October 4 scientific run: `runs/20261004_tuned_temporal_full_01/` |
| [Publication audit receipt](evidence/audit_receipt.json) | Separate execution/rendering identities, artifact hashes, zero errors, 21 temporal warnings and four claim restrictions |
| [Technical report](final_report.md) | Methods, result arithmetic, interpretation and detailed diagnostics |
| [Engineering checks](../engineering.md) | Data validation, prediction audits, cache equivalence and export safeguards |
| [Development procedure](../development.md#documentation-changes-and-historical-provenance) | Source matching, historical audits and deliberate refreshes |

Git keeps the report, six figures, manifest and receipt. Full CSV/source archives stay in ignored local run storage; they are not GitHub downloads. The report is rendered from saved results without fitting models; its receipt records rendering separately from scientific execution.
