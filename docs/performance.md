# Runtime and Numerical Equivalence

Three records answer different questions: a historical controlled full-run comparison, sampled follow-up stages, and one observed run of the later expanded study. Changed study workloads cannot establish a controlled speedup against each other.

## Controlled Full-Run Comparison: October 3

**Modeling fell from 76.10 to 20.95 minutes (3.63x faster), with all 26 scientific CSVs byte-identical.**

| Stage | Reference minutes | Optimized minutes | Speedup |
| --- | ---: | ---: | ---: |
| Benchmark bundle | 35.19 | 12.70 | 2.77x |
| Temporal stress | 40.91 | 8.25 | 4.96x |
| Total modeling | 76.10 | 20.95 | 3.63x |

One pair on the same Windows machine; audit/report time is excluded. Inputs, folds, seeds, budgets, full grids, both classifiers and 1,000 bootstrap draws matched. Reference ReliefF used four workers; the adapter used one scoring process. Both limited BLAS to one thread.

The [comparison receipt](performance/runtime_comparison.json) records durations and matching hashes; the [reference manifest](performance/reference_run_manifest.json) records historical source/input identity. Optimized evidence remains at `runs/finalization_20261003_optimized_v3/`. Today's public manifest describes a later study and is not this timing receipt.

## Why It Became Faster

| Change | Reuse or calculation | Correctness boundary |
| --- | --- | --- |
| NumPy ReliefF scoring and neighbor selection | Replace Python feature loops | Preserve skrebate 0.8.4 typing, distances, neighbor/tie order, mixed-feature ramp, class normalization and summation order |
| Contiguous reductions and C-order score matrix | Reduce array overhead | Exact floating-point scores and ranks |
| Scalar threshold comparisons | Avoid repeated NumPy call overhead | Preserve asymmetric tolerance, boundaries and sentinels |
| Bounded in-memory ranking cache | 128 entries | Bind training measurements, labels, shape, neighbors and backend |
| Bounded model-score cache | 4,096 score-vector entries | Bind train/evaluation arrays, labels, configuration and fitting implementation |
| Thread defaults and progress | One scoring process; explicit settings respected | No altered search or sampling |

The historical run reused 5,263 fits and retained 44.1 MiB of cached scores, not a total-memory measurement. A separate 60-case, 1,500-sample threshold microbenchmark fell from 0.890 to 0.086 seconds; it does not measure whole-study speed.

## Uncached ReliefF Probe

1,200 chronological SECOM samples; median imputation with indicators, RobustScaler, 1,032 transformed columns. Reference used four workers, adapter one. Scores and ranks matched exactly.

| Neighbors | Reference seconds | Adapter seconds | Speedup |
| --- | ---: | ---: | ---: |
| 5 | 9.020 | 0.620 | 14.5x |
| 10 | 7.556 | 0.689 | 11.0x |
| 20 | 7.646 | 0.822 | 9.3x |

```bash
python scripts/benchmark_relief.py --input-dir data/raw --rows 1200 --reference-jobs 4
```

## Follow-up: Feature-Budget Preparation and Inner Metrics

October 3-4 probes compared a frozen reviewed 299-test candidate against a bounded preparation change. [The receipt](performance/study_preparation_comparison.json) records source/input hashes, versions, repetitions, cache counts, fingerprints and profiles; local source/probes remain under `.tmp/study-performance/`.

All 1,567 rows and 590 raw measurements were used with unchanged grids and F-test, Gram-Schmidt and ReliefF. Each workload began with empty caches. Timings are medians of three unprofiled repetitions with OMP/OpenBLAS/MKL set to one; loading and cProfile passes were separate, with no competing owned test/profiling process.

| Sample scope | Baseline seconds | Candidate seconds | Reduction |
| --- | ---: | ---: | ---: |
| Original: first two outer folds and full-data selector views | 18.688 | 18.057 | 3.4% timing variation |
| Tuned: one outer fold, all inner folds, both models/modes | 49.823 | 44.415 | 10.9% |
| Temporal selection: first fold, seed 42 | 10.807 | 6.633 | 38.6% |
| Temporal freeze: full development data, seed 42, five inner folds | 42.326 | 28.446 | 32.8% |

Temporal probes retained both scalers and all budget, C and ReliefF-neighbor configurations. These sampled improvements do not establish a full-study speedup.

| Improvement | Measured effect | Preserved boundary |
| --- | --- | --- |
| Prepare largest ranking once, then take exact 10/20/40 prefixes | Pipeline calls: tuned 99 to 39; selection 153 to 53; freeze 150 to 50 | Same training split, indicator mode, scaler, neighbors and seed; local views released within their owner |
| Shared selected-array storage | 7,357,120 to 4,311,520 bytes (41.4% less) | Count underlying arrays once; prepared-view storage, not peak process RSS |
| Compute only inner BER/AUC | Avoid full diagnostics and duplicate AUC during search | Same frozen threshold and single-class AUC fallback of 0.5; full outer/report metrics retained |

The tuned baseline profile spent 9.89 cumulative seconds in complete metric calculation. Fit counts, score-cache counters and retained score bytes stayed unchanged; KRR solves remained dominant.

All four output fingerprints matched, including array bytes/layout, search metrics/configurations, outer results, feature identities/stability and temporal CSV/frozen roles. Independent-fit regressions covered all five then-active temporal selectors, budgets, scalers, indicator modes, ReliefF neighbors, Gram-Schmidt ordering and KRR/logistic predictions.

Loading took 0.37-0.39 seconds; five-repeat, 1,000-draw bootstrap medians were 0.021 seconds (reference) and 0.018 (tuned). Neither justified further work. No persistent cache, concurrency, altered grid/resampling or scientific contract was added. Historical artifacts and snapshots were preserved.

## Expanded Study: October 4 Observations

The [manifest](results/evidence/run_manifest.json) records `runs/20261004_tuned_temporal_full_01/`: both models, six selectors, ten outer/three inner benchmark folds, 16 tuned KRR configurations and six predefined chronological KRR procedures. One scoring process and one BLAS thread; caches started cold.

| Stage | Seconds | Minutes |
| --- | ---: | ---: |
| Reference plus tuned benchmark | 1,101.04 | 18.35 |
| Temporal LR roles plus KRR comparisons | 282.98 | 4.72 |
| Total modeling | 1,384.02 | 23.07 |

The full command took about 23.6 minutes including audit/report; no separate reference/tuned timer exists. This single observation is not a repeated performance guarantee or a controlled comparison with the earlier 76.1-to-20.9-minute or 14.94/0.79-minute workloads.

| Resource observation | Recorded value |
| --- | ---: |
| Score-cache hits / misses | 10,991 / 36,181 |
| Retained entries / score bytes | 4,096 / 7,679,584 |
| Run storage before analysis/export | 84,455,227 bytes |

Cache bytes are not process memory. Detailed search/lineage receipts remain in ignored run archives; Git publishes the compact report/figures/provenance.

## Runtime Controls

| Control | Meaning |
| --- | --- |
| `SECOM_RELIEF_BACKEND=reference` | Independent upstream check; adapter rejects unverified versions or inputs outside the finite, imputed binary contract |
| `SECOM_RELIEF_N_JOBS` | Reference worker count; adapter uses one process |
| `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` | Default to one; preserve explicit user settings |
| `--progress` | Shows benchmark and chronological screening, selection, freeze and later-block stages |

Further optimization should target measured fitting/solve costs while preserving folds, grids, seeds, thresholds and exact outputs. Timing evidence does not remove the recorded scientific warnings or claim restrictions.
