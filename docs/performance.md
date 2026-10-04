# Runtime and Numerical Equivalence

On October 3, 2026, the complete two-classifier study fell from **76.1 to 20.9 minutes** of modeling on the same Windows machine: **3.63× faster**. All **26 scientific CSVs are byte-for-byte identical**, including fold metrics, search results, feature stability, frozen configurations, lockbox counts and drift evidence.

| Stage | Reference minutes | Optimized minutes | Speedup |
| --- | ---: | ---: | ---: |
| Benchmark bundle | 35.19 | 12.70 | 2.77× |
| Temporal stress test | 40.91 | 8.25 | 4.96× |
| Total modeling | 76.10 | 20.95 | 3.63× |

These are observed single-run timings, not a hardware-independent guarantee. The reference already had a bounded score cache, four ReliefF workers and one BLAS thread per process. The optimized run used one scoring process and the same BLAS limits. Both used the same raw-file hashes, folds, seeds, feature budgets, full hyperparameter grids, classifiers and 1,000 bootstrap draws. Timing covers modeling; audit and report rendering follow it.

## What Changed

- **ReliefF scoring:** profiling located nearly all fit time in upstream per-sample/per-feature Python scoring. The default adapter operates on all features with NumPy, retaining skrebate 0.8.4 feature typing, distances, neighbor order, mixed-feature ramp, class normalization and summation order. Contiguous per-feature reductions and a C-order final score matrix preserve exact floating-point results.
- **Neighbor selection:** vectorized binary hit/miss selection preserves the reference's sorting ties and combined neighbor order.
- **Threshold sweeps:** scalar comparisons retain NumPy's default asymmetric tolerance, reducing a 60-case, 1,500-sample threshold experiment from 0.890 to 0.086 seconds. Boundary and sentinel comparisons are checked against NumPy, and existing exhaustive threshold tests remain active.
- **Reuse and threads:** a bounded 128-entry ranking cache reuses only identical training measurements, labels, shape, neighbor count and backend. A 4,096-entry model-score cache binds both train and evaluation arrays, training labels, configuration and fitting implementation, storing score vectors only. This run reused 5263 fits and retained 44.1 MiB of scores. CLI thread defaults preserve explicit user settings. Both caches stay in memory.
- **Progress:** `--progress` displays temporal screening, selection, freeze and lockbox stages as well as benchmark progress.

## Real-Data ReliefF Check

The uncached benchmark uses 1,200 chronological SECOM samples, median imputation with indicators, RobustScaler and 1,032 transformed features. All scores and ranks match exactly. The reference uses four workers; the adapter uses one.

| Neighbors | Reference seconds | Optimized seconds | Speedup |
| --- | ---: | ---: | ---: |
| 5 | 9.020 | 0.620 | 14.5× |
| 10 | 7.556 | 0.689 | 11.0× |
| 20 | 7.646 | 0.822 | 9.3× |

```bash
python scripts/benchmark_relief.py --input-dir data/raw --rows 1200 --reference-jobs 4
```

## Evidence and Controls

The [runtime comparison](performance/runtime_comparison.json) includes measured durations and all matching CSV hashes. The [reference manifest](performance/reference_run_manifest.json) records the earlier source and input identities; the [optimized manifest](results/evidence/run_manifest.json) identifies the current complete run. The public [audit receipt](results/evidence/audit_receipt.json) binds the small report snapshot and records the full local archive's hash. Detailed CSVs and the source/artifact ZIP remain in ignored run storage.

The adapter fails explicitly for an unverified skrebate version or inputs outside the finite, imputed binary study contract. `SECOM_RELIEF_BACKEND=reference` selects upstream scoring for independent checks. `SECOM_RELIEF_N_JOBS` sets reference workers; the adapter uses one process. The current study retains its temporal drift warning and claim restriction.

Remaining work is repeated model fitting, kernel solves and validation. Further speedups should be measured against this baseline and preserve the scientific output; reducing the search grid or resampling would change the study.
