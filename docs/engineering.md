# Engineering Checks

The study saves predictions and validates the results before generating its report. These safeguards make the analysis inspectable and repeatable.

| Concern | Implemented safeguard | Evidence |
| --- | --- | --- |
| Data integrity | Reject malformed measurements, invalid labels and nonfinite fitted scores | [Integrity tests](../tests/test_engineering_integrity.py), [input tests](../tests/test_io.py) |
| Evaluation leakage | Fit preprocessing and feature selection on training samples; choose thresholds before evaluating test samples | [Methodology tests](../tests/test_methodology.py), [study specification](spec/02-benchmark-replication-study.md) |
| Result consistency | Recompute metrics from saved predictions; check fold IDs, paired comparisons and artifact hashes | [Artifact auditor](../src/secom/workflows/audit.py), [audit tests](../tests/test_study_audit.py) |
| Safe reuse | Bind caches to data, labels and settings; test exact scores, eviction and independent returned arrays | [Score-cache tests](../tests/test_model_score_cache.py), [preparation tests](../tests/test_study_preparation_reuse.py) |
| Runtime | Measure an unchanged workload and verify numerical equivalence | [Performance measurements](performance.md) |
| Disk use and export | Keep raw data and full runs outside Git; publish a compact report; preserve the previous snapshot on raised export failures | [Export tests](../tests/test_provenance_and_export.py), [public evidence](results/README.md#evidence) |

[CI](../.github/workflows/ci.yml) runs tests and Ruff checks on Linux and Windows. Synthetic checks validate software behavior; real-data findings come from the separately recorded [study execution](results/evidence/run_manifest.json).

Historical run archives retain the exact source and specifications used for execution. Editorial report refreshes preserve saved predictions and record rendering identity separately. Export rollback handles tested exceptions; it does not provide crash-atomic updates across the public directory and local archive.

Use the [development guide](development.md) for commands and evidence-preservation procedures.
