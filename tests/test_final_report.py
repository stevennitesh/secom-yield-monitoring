"""Tests for rendering the canonical final report from active artifacts."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest
import pandas as pd

from secom.config import ArtifactName, StudyStatus
from secom.reporting import write_final_report
from tests.assertions import assert_renderable_png, assert_text_contains_all, assert_text_excludes_all


def test_wide_report_panels_preserve_values_and_distinguish_duplicate_context() -> None:
    """Splitting display tables must neither discard fields nor join different records."""
    from secom.reporting import _markdown_table

    frame = pd.DataFrame(
        {
            "context": ["same", "same"],
            "group": ["same", "same"],
            **{f"metric_{index}": [index, index + 100] for index in range(23)},
        }
    )
    original = frame.copy(deep=True)
    lines = _markdown_table(frame, list(frame.columns), headers=list(frame.columns))
    records = {"1": {}, "2": {}}
    headings = []
    for line in lines:
        if not line.startswith("| "):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells[0] == "Row":
            headings = cells
            assert len(headings) <= 9
            continue
        for heading, value in zip(headings[1:], cells[1:], strict=True):
            prior = records[cells[0]].setdefault(heading, value)
            assert prior == value
    assert records == {
        str(index + 1): {column: str(value) for column, value in row.items()} for index, row in frame.iterrows()
    }
    pd.testing.assert_frame_equal(frame, original)


def test_final_report_is_generated_from_active_artifacts(
    active_artifacts_output_dir: Path,
) -> None:
    """Final report rendering should create the canonical markdown artifact."""
    report_path = write_final_report(active_artifacts_output_dir)

    assert report_path.name == ArtifactName.FINAL_REPORT
    assert report_path.exists()


def test_final_report_contains_finished_narrative_sections(
    active_artifacts_output_dir: Path,
) -> None:
    """Final report should render finished prose instead of scaffold prompts."""
    text = write_final_report(active_artifacts_output_dir).read_text(encoding="utf-8")

    assert_text_contains_all(text, ["## What I Built", "## Provenance Appendix"])
    assert_text_excludes_all(
        text,
        [
            "Summarize the SECOM benchmark context",
            "Describe the full-dataset replication protocol",
        ],
    )


def test_final_report_rejects_audit_invalid_artifact_set(
    active_artifacts_output_dir: Path,
) -> None:
    """Canonical report generation should not publish audit-invalid claims."""
    manifest_path = active_artifacts_output_dir / "reports" / ArtifactName.MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["benchmark_tuned_status"] = "not_run"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")

    with pytest.raises(RuntimeError, match="Cannot render final report because study audit failed"):
        write_final_report(active_artifacts_output_dir)


def test_final_report_ignores_stale_temporal_artifacts_after_temporal_failure(
    active_artifacts_output_dir: Path,
) -> None:
    """Failed temporal status should not publish stale temporal table claims."""
    manifest_path = active_artifacts_output_dir / "reports" / ArtifactName.MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["temporal_robustness_status"] = StudyStatus.FAILED
    manifest["temporal_claim_restrictions"] = []
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")

    text = write_final_report(active_artifacts_output_dir).read_text(encoding="utf-8")

    assert_text_contains_all(
        text,
        [
            "Temporal robustness status: `failed`",
            "Temporal model selection artifact missing or empty.",
        ],
    )
    assert_text_excludes_all(
        text,
        [
            "Roles are chosen from chronological inner selection",
            "### Lockbox Metrics",
            "### Supervised vs MSPC",
            "The temporal study adds secondary robustness evidence without active claim restrictions",
        ],
    )


def test_report_context_uses_primary_drift_even_when_challenger_is_first(active_artifacts_output_dir: Path) -> None:
    from secom.reporting import _load_report_context

    path = active_artifacts_output_dir / "reports" / ArtifactName.TEMPORAL_DRIFT
    frame = pd.read_csv(path)
    frame.sort_values("model_scope").to_csv(path, index=False)
    ctx = _load_report_context(active_artifacts_output_dir)
    assert ctx.drift_row["model_scope"] == "primary"


def test_final_report_discloses_temporal_protocol_limits(active_artifacts_output_dir: Path) -> None:
    text = write_final_report(active_artifacts_output_dir).read_text(encoding="utf-8")
    assert_text_contains_all(
        text,
        ["fixed nonoverlapping calendar test blocks", "deterministic chronological splits", "held-out calibration"],
    )


def test_final_report_uses_required_benchmark_section_structure(
    active_artifacts_output_dir: Path,
) -> None:
    """Canonical report should expose original/tuned design, search, and results sections."""
    text = write_final_report(active_artifacts_output_dir).read_text(encoding="utf-8")

    assert_text_contains_all(
        text,
        [
            "## Original Replication Design",
            "### Appendix: Original Replication Search Summary",
            "### Original Search Space",
            "### Original Selected Configurations",
            "## Original Replication Results",
            "## Tuned Benchmark Design",
            "### Appendix: Tuned Benchmark Search Summary",
            "### Tuned Search Space",
            "### Modal Selected Configurations",
            "## Tuned Benchmark Results",
        ],
    )


def test_final_report_includes_temporal_model_selection_summary(
    active_artifacts_output_dir: Path,
) -> None:
    """Canonical report should show temporal role ranking and modal selector configs."""
    text = write_final_report(active_artifacts_output_dir).read_text(encoding="utf-8")

    assert_text_contains_all(
        text,
        [
            "### Temporal Model Selection Summary",
            "Roles are chosen from chronological inner selection",
            "Outer family ranking remains exploratory.",
            "#### Selector Ranking and Modal Configurations",
            "Modal selected inputs",
            "Modal scaler",
        ],
    )


def test_final_report_includes_drift_claim_restriction_table(
    active_artifacts_output_dir: Path,
) -> None:
    """Canonical temporal section should expose drift metrics that govern claims."""
    text = write_final_report(active_artifacts_output_dir).read_text(encoding="utf-8")

    assert_text_contains_all(
        text,
        [
            "### Drift and Claim Restrictions",
            "Confirmatory claims allowed",
            "Absolute prevalence shift",
            "Score distribution p-value",
            "Maximum measurement PSI",
        ],
    )


def test_final_report_includes_uci_original_benchmark_reference(
    active_artifacts_output_dir: Path,
) -> None:
    """Final report should include original benchmark rows and F/Pearson caveat."""
    text = write_final_report(active_artifacts_output_dir).read_text(encoding="utf-8")

    assert_text_contains_all(
        text,
        [
            "### UCI Original Benchmark Reference",
            "S2N",
            "Ttest",
            "Relief",
            "Pearson",
            "Ftest",
            "Gram Schmidt",
            "33.5 +/- 2.2",
            "local Ttest row uses a pooled two-sample t statistic",
            "Binary-label ANOVA F-test ranking and absolute Pearson correlation ranking are mathematically monotonic",
            "UCI reference table reports separate Ftest and Pearson rows",
        ],
    )


def test_final_report_distinguishes_original_and_tuned_classifier_selection(
    active_artifacts_output_dir: Path,
) -> None:
    """Final report must distinguish fixed-budget and tuned nested procedures."""
    text = write_final_report(active_artifacts_output_dir).read_text(encoding="utf-8")

    assert_text_contains_all(
        text,
        [
            "The protocol uses shuffled stratified outer folds and stratified inner cross-validation",
            "BER is the primary inner objective",
        ],
    )


def test_final_report_labels_uncertainty_as_descriptive_fold_spread(
    active_artifacts_output_dir: Path,
) -> None:
    """Final report should identify benchmark intervals as fold-bootstrap summaries."""
    text = write_final_report(active_artifacts_output_dir).read_text(encoding="utf-8")

    assert "not algorithm-performance confidence intervals" in text
    assert "Fold mean, standard deviation, and range are descriptive" in text


def test_final_report_surfaces_required_industrialization_gaps(
    active_artifacts_output_dir: Path,
) -> None:
    """Final report should keep all required industrialization limits visible."""
    text = write_final_report(active_artifacts_output_dir).read_text(encoding="utf-8")

    assert_text_contains_all(
        text,
        [
            "No stable device/tool/chamber identifier for unseen-device validation",
            "No intervention or maintenance history",
            "No explicit regime-change metadata",
            "No downstream decision or action outcome data",
            "Anonymous features limit process interpretation",
            "Single-dataset evidence only",
            "Operational framing in this report is illustrative, not production-validated",
        ],
    )


def test_final_report_surfaces_required_industrialization_next_data(
    active_artifacts_output_dir: Path,
) -> None:
    """Final report conclusions should state the required production-study inputs."""
    text = write_final_report(active_artifacts_output_dir).read_text(encoding="utf-8")

    assert_text_contains_all(
        text,
        [
            "device- or tool-level identifiers",
            "intervention logs",
            "longer-horizon cross-context validation",
            "deployment decision objectives and cost accounting",
            "verified pre-outcome timing and intervention outcomes",
        ],
    )


def test_final_report_surfaces_manifest_industrialization_notes(
    active_artifacts_output_dir: Path,
) -> None:
    """Run-specific industrialization notes from the manifest should be rendered."""
    manifest_path = active_artifacts_output_dir / "reports" / ArtifactName.MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["industrialization_notes"] = [
        "Temporal robustness not run: no feasible chronological fold plan.",
        "",
        123,
    ]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")

    text = write_final_report(active_artifacts_output_dir).read_text(encoding="utf-8")

    assert_text_contains_all(
        text,
        [
            "### Run-Specific Industrialization Notes",
            "Temporal robustness not run: no feasible chronological fold plan.",
            "123",
        ],
    )


def test_final_report_scopes_feature_selection_claims(
    active_artifacts_output_dir: Path,
) -> None:
    """Feature reporting should avoid causal or production-strength selector claims."""
    text = write_final_report(active_artifacts_output_dir).read_text(encoding="utf-8")

    assert_text_contains_all(
        text,
        [
            "Feature outputs are model-prioritization evidence from resampled benchmark artifacts, not causal proof",
            "validated process-driver identification",
            "Bars show selection frequency for exploratory families, not causal importance.",
        ],
    )
    assert_text_excludes_all(text, ["most stable and influential features"])


def test_final_report_writes_expected_figure_files(
    active_artifacts_output_dir: Path,
) -> None:
    """Final report rendering should emit the expected figure set."""
    write_final_report(active_artifacts_output_dir)

    figures_dir = active_artifacts_output_dir / "reports" / "figures"
    for name in [
        "benchmark_comparison.png",
        "tuned_vs_original_delta.png",
        "feature_stability.png",
        "temporal_drift.png",
        "lockbox_vs_mspc.png",
        "workload_cost_framing.png",
    ]:
        assert_renderable_png(figures_dir / name)


def test_final_report_pdf_export_is_optional_when_tool_missing(
    active_artifacts_output_dir: Path,
    monkeypatch,
) -> None:
    """Missing pandoc should leave markdown report generation successful."""
    monkeypatch.setattr(shutil, "which", lambda _: None)

    report_path = write_final_report(active_artifacts_output_dir, export_pdf=True)
    text = report_path.read_text(encoding="utf-8")

    assert report_path.exists()
    assert "PDF export skipped because pandoc is not available." in text
    assert not (active_artifacts_output_dir / "reports" / "final_report.pdf").exists()


def test_reader_main_text_defines_metrics_and_keeps_machine_identifiers_in_appendix(
    active_artifacts_output_dir: Path,
) -> None:
    text = write_final_report(active_artifacts_output_dir).read_text(encoding="utf-8")
    main = text.split("## Technical Appendix")[0]
    assert_text_contains_all(
        main,
        [
            "**Balanced error**",
            "**Failure recall**",
            "**Pass specificity**",
            "**Cross-validation**",
            "**Calibration**",
            "Kernel ridge regression",
            "logistic regression",
            "class-stratified",
            "complete selection procedure",
            "bounded parameter coverage",
            "engineering space is not exhausted",
        ],
    )
    assert_text_excludes_all(
        main,
        [
            "with_missing_indicators",
            "gamma_multiplier",
            "LOCKBOX",
            "True+",
            "True-",
            "claim_restriction",
            "Recorded ",
            "Configured ",
        ],
    )
    assert re.search(r"\bjoint\b", main) is None


@pytest.mark.parametrize(
    "before,after,expected",
    [
        ((70, 445), (66, 371), "fewer failures caught and fewer false alerts"),
        ((66, 371), (70, 445), "more failures caught and more false alerts"),
        ((70, 445), (70, 445), "the same number of failures caught and the same number of false alerts"),
        ((66, 445), (70, 371), "more failures caught and fewer false alerts"),
    ],
)
def test_reader_comparison_derives_observed_direction(
    active_artifacts_output_dir, monkeypatch, before, after, expected
):
    from secom import reporting

    ctx = reporting._load_report_context(active_artifacts_output_dir)
    read = reporting.read_csv_if_exists

    def contrasting(path):
        frame = read(path)
        if path.name in (ArtifactName.BENCHMARK_PROCEDURE_SUMMARY, ArtifactName.BENCHMARK_TUNED_PROCEDURE_SUMMARY):
            frame = frame.copy()
            caught, alerts = before if path.name == ArtifactName.BENCHMARK_PROCEDURE_SUMMARY else after
            frame.loc[frame.procedure.eq("joint"), ["pooled_TP", "pooled_FP"]] = [caught, alerts]
        return frame

    monkeypatch.setattr(reporting, "read_csv_if_exists", contrasting)
    main = "\n".join(reporting._render_study(ctx)).split("## Technical Appendix")[0]
    assert expected in main
    assert "This is a tradeoff: fewer false alerts accompany fewer failures caught" not in main


def test_reader_settings_counts_and_calibration_table_follow_recorded_inputs(active_artifacts_output_dir, monkeypatch):
    from dataclasses import replace
    from secom import reporting

    ctx = reporting._load_report_context(active_artifacts_output_dir)
    manifest = {**ctx.manifest, "execution": {"settings": {"classifiers": ["krr"], "original_selectors": ["ReliefF"]}}}
    later = ctx.temporal_lockbox.assign(TP=8, FN=12, FP=17, TN=63)
    ctx = replace(
        ctx,
        manifest=manifest,
        temporal_lockbox=later,
        benchmark_summary=ctx.benchmark_summary.assign(n_folds=4),
        benchmark_tuned_summary=ctx.benchmark_tuned_summary.assign(n_folds=4),
    )
    read = reporting.read_csv_if_exists

    def different_folds(path):
        frame = read(path)
        if path.name in (ArtifactName.BENCHMARK_PROCEDURE_SUMMARY, ArtifactName.BENCHMARK_TUNED_PROCEDURE_SUMMARY):
            return frame.assign(n_folds=4)
        return frame

    monkeypatch.setattr(reporting, "read_csv_if_exists", different_folds)
    text = "\n".join(reporting._render_study(ctx))
    main, appendix = text.split("## Technical Appendix", 1)
    assert "We compared 1 column-selection method using Kernel ridge." in main
    assert "Both benchmarks used 4 held-out test folds." in main
    assert "20 failures and 80 passes" in main
    assert "nine failures" not in main and "ten unseen" not in main
    assert "A family is a selector/model/input combination" in main
    compact = main.split("### Calibration counts and threshold sensitivity")[1].split("### The final later block")[0]
    assert len([line for line in compact.splitlines() if line.startswith("| ")]) == 4
    assert "Full calibration diagnostics" in compact
    assert "Recalibrated threshold min" in appendix
    assert "Final retained model" not in compact


def test_incomplete_scaffold_does_not_claim_observed_tradeoff(workspace_tmp_dir):
    from secom.artifacts import ensure_reports_dir, write_manifest
    from secom.reporting import write_report_skeleton
    from secom.workflows.manifest import initial_study_manifest

    reports = ensure_reports_dir(workspace_tmp_dir)
    write_manifest(initial_study_manifest(Path(__file__).resolve().parents[1]), reports / ArtifactName.MANIFEST)
    text = write_report_skeleton(workspace_tmp_dir).read_text(encoding="utf-8")
    assert "Benchmark comparison conclusions are unavailable" in text
    assert "complete selected-procedure comparison is unavailable" in text
    assert "modest aggregate tradeoff" not in text
    assert "two model families" not in text and "nine failures" not in text


def test_reader_complete_conclusion_summarizes_observed_results(active_artifacts_output_dir, monkeypatch):
    """Complete inputs need an evidence-specific conclusion, not a scaffold disclaimer."""
    from secom import reporting

    ctx = reporting._load_report_context(active_artifacts_output_dir)
    read = reporting.read_csv_if_exists

    def evidence(path):
        frame = read(path)
        if path.name in (ArtifactName.BENCHMARK_PROCEDURE_SUMMARY, ArtifactName.BENCHMARK_TUNED_PROCEDURE_SUMMARY):
            frame = frame.copy()
            benchmark = path.name == ArtifactName.BENCHMARK_PROCEDURE_SUMMARY
            frame.loc[frame.procedure.eq("joint"), ["mean_BER", "pooled_TP", "pooled_FP"]] = (
                [0.4, 70, 445] if benchmark else [0.395, 66, 371]
            )
        elif path.name == ArtifactName.TEMPORAL_PROCEDURE_METRICS:
            frame = frame.assign(BER=0.5, BER_available=True)
        elif path.name == ArtifactName.TEMPORAL_KRR_METRICS:
            frame = frame.assign(BER=0.48, BER_available=True, available=True, TP=10, FP=78, n_test=100)
        elif path.name == ArtifactName.TEMPORAL_CALIBRATION_DIAGNOSTICS:
            frame = frame.assign(calibration_fails=3, calibration_n=100, calibration_passes=97)
        return frame

    monkeypatch.setattr(reporting, "read_csv_if_exists", evidence)
    main = "\n".join(reporting._render_study(ctx)).split("## Technical Appendix")[0]
    conclusion = main.split("## Conclusions and Next Data Requirements")[1]
    assert "lowers mean balanced error by 0.50 percentage points (40.00% to 39.50%)" in conclusion
    assert "fewer failures caught (70 to 66) and fewer false alerts (445 to 371)" in conclusion
    assert "50.00%" in conclusion and "48.00%" in conclusion and "most later samples (88.0%)" in conclusion
    assert "do not reproduce its class-balanced performance" in conclusion
    assert "3 failed calibration examples per period" in conclusion
    assert "threshold-fragility warnings" in conclusion
    assert "pre-outcome timing" in conclusion and "no production-readiness, causal, early-warning" in conclusion
    assert "unavailable" not in conclusion and "Missing evidence" not in conclusion
    assert all(line == line.rstrip() for line in main.splitlines())


def test_reader_budget_and_fold_sentences_follow_different_settings(active_artifacts_output_dir, monkeypatch):
    from dataclasses import replace
    from secom import reporting

    ctx = reporting._load_report_context(active_artifacts_output_dir)
    ctx = replace(
        ctx,
        manifest={
            **ctx.manifest,
            "execution": {"settings": {"original_feature_budget": 12, "tuned_feature_budgets": [5, 12]}},
        },
    )
    read = reporting.read_csv_if_exists

    def different_folds(path):
        frame = read(path)
        if path.name == ArtifactName.BENCHMARK_PROCEDURE_SUMMARY:
            return frame.assign(n_folds=4)
        if path.name == ArtifactName.BENCHMARK_TUNED_PROCEDURE_SUMMARY:
            return frame.assign(n_folds=5)
        return frame

    monkeypatch.setattr(reporting, "read_csv_if_exists", different_folds)
    main = "\n".join(reporting._render_study(ctx)).split("## Technical Appendix")[0]
    assert "The reference used 4 held-out test folds; tuning used 5." in main
    assert "The reference allows up to 12 selected inputs; tuning compares budgets of 5 and 12 inputs." in main
    assert "Both benchmarks used" not in main


def test_reader_navigation_keeps_appendix_tables_without_duplicate_main_headings(active_artifacts_output_dir):
    import re

    text = write_final_report(active_artifacts_output_dir).read_text(encoding="utf-8")
    headings = re.findall(r"^## (.+)$", text, re.MULTILINE)
    assert len(headings) == len(set(headings))
    assert "[Technical tables](#technical-appendix)" in text
    assert text.count("<details>") == text.count("</details>")
    assert "<summary>Original Replication Search Summary</summary>" in text
    assert "[Full calibration diagnostics](#chronological-diagnostics)" in text
    assert text.index("### Chronological diagnostics") < text.index(
        "<summary>Temporal Robustness Stress Test</summary>"
    )
    assert "expand it to see every predefined" in text
    appendix = text.split("## Technical Appendix")[1]
    assert "Recalibrated threshold min" in appendix and "Modal scaler" in appendix


def test_executive_summary_reads_changed_evidence(active_artifacts_output_dir, monkeypatch):
    from secom import reporting

    ctx = reporting._load_report_context(active_artifacts_output_dir)
    read = reporting.read_csv_if_exists

    def changed(path):
        frame = read(path)
        if path.name in (ArtifactName.BENCHMARK_PROCEDURE_SUMMARY, ArtifactName.BENCHMARK_TUNED_PROCEDURE_SUMMARY):
            frame = frame.copy()
            original = path.name == ArtifactName.BENCHMARK_PROCEDURE_SUMMARY
            frame.loc[frame.procedure.eq("joint"), ["mean_BER", "pooled_TP", "pooled_FP"]] = (
                [0.2, 12, 8] if original else [0.4, 9, 20]
            )
        return frame

    monkeypatch.setattr(reporting, "read_csv_if_exists", changed)
    summary = "\n".join(reporting._render_study(ctx)).split("## What I Built")[0]
    assert "20.00% → 40.00%" in summary
    assert "failures caught 12 → 9" in summary
    assert "false alerts 8 → 20" in summary
    assert "31.43%" not in summary and "30.86%" not in summary


def test_drift_appendix_unpacks_all_indicator_records_without_mutating_evidence():
    """Nested diagnostics should remain complete and identify populations in ordinary rows."""
    from secom.reporting import _drift_table_lines

    frame = pd.DataFrame(
        [
            {
                "model_scope": role,
                "score_reference": "held_out_calibration_same_retained_model",
                "selected_indicator_missingness_rates": json.dumps(
                    [
                        {"feature": feature, "fit_missing_rate": 0.25, "later_missing_rate": rate}
                        for feature, rate in indicators
                    ]
                ),
            }
            for role, indicators in [("primary", [("M112", 0.5)]), ("challenger", [("M247", 0.1), ("M519", 0.75)])]
        ]
    )
    original = frame.copy(deep=True)
    text = "\n".join(_drift_table_lines(frame))
    assert "| Primary model | M112 | 25.00% | 50.00% | +25.00 |" in text
    assert "| Comparison model | M247 | 25.00% | 10.00% | -15.00 |" in text
    assert "| Comparison model | M519 | 25.00% | 75.00% | +50.00 |" in text
    assert "Held-out calibration scores from the same retained model" in text
    assert "fit_missing_rate" not in text and "[{" not in text
    pd.testing.assert_frame_equal(frame, original)


def test_report_period_ranges_use_actual_predictions_and_exclude_final_block(workspace_tmp_dir):
    from secom.reporting import _temporal_period_lines

    pd.DataFrame(
        {
            "fold": [1, 1, 2, "LOCKBOX"],
            "timestamp": ["2009-08-01 12:00", "2009-08-02 13:00", "2009-08-02 13:01", "2010-01-01"],
        }
    ).to_csv(workspace_tmp_dir / ArtifactName.TEMPORAL_PREDICTIONS, index=False)
    text = "\n".join(_temporal_period_lines(workspace_tmp_dir))
    assert "| 1 | 2009-08-01 12:00 | 2009-08-02 13:00 |" in text
    assert "| 2 | 2009-08-02 13:01 | 2009-08-02 13:01 |" in text
    assert "2010" not in text and "2008" not in text


def test_stability_example_matches_plotted_features_instead_of_first_manifest_group():
    from types import SimpleNamespace
    from secom.reporting import _feature_stability_context_lines

    summary = pd.DataFrame([{"selector": "F-test", "classifier": "krr", "replication_mode": "strict", "mean_BER": 0.3}])
    common = {"selector": "F-test", "classifier": "krr", "replication_mode": "strict", "selection_frequency": 1.0}
    report = pd.DataFrame([{**common, "feature_name_or_source_col": "M112", "feature_type": "missing_indicator"}])
    ctx = SimpleNamespace(
        manifest={
            "dataset": {
                "shared_missingness_patterns": [
                    {"features": ["M109"], "monthly_missing_rates": {}},
                    {"features": ["M112", "M247"], "monthly_missing_rates": {"2009-08": 0.25}},
                ]
            }
        },
        feature_report=report,
        benchmark_summary=summary,
        benchmark_tuned_feature_report=None,
        benchmark_tuned_summary=None,
    )
    text = "\n".join(_feature_stability_context_lines(ctx))
    assert "columns 112, 247" in text and "109" not in text
    assert "2009-08: 25.0%" in text


def test_search_breadth_counts_configurations_once_across_outer_folds(monkeypatch):
    """Distinct declared settings must not be inflated by fold count or fitted kernel widths."""
    from secom import reporting

    summaries = []
    monkeypatch.setattr(reporting, "_markdown_table", lambda frame, columns: summaries.append(frame) or [])
    rows = pd.DataFrame(
        [
            {
                "selector": "S2N",
                "classifier": "krr",
                "replication_mode": "strict",
                "fold": fold,
                "k": 10,
                "alpha": alpha,
                "gamma": 0.01 * fold,
                "gamma_multiplier": 0.1,
                "C": None,
                "n_neighbors": None,
            }
            for fold in (1, 2, 3)
            for alpha in (0.1, 1.0)
        ]
    )
    reporting._original_search_space_table(rows.assign(gamma=0.01))
    reporting._tuned_search_space_table(rows)
    assert [summary.evaluated_configs.tolist() for summary in summaries] == [[2], [2]]


def test_automatic_rbf_setting_is_counted_and_distinguished_from_missing_parameters(monkeypatch):
    """A valid default kernel setting must not look like a missing or inapplicable result."""
    from secom import reporting

    rows = pd.DataFrame(
        [
            {"classifier": "krr", "gamma": None, "gamma_multiplier": None},
            {"classifier": "krr", "gamma": 0.01, "gamma_multiplier": None},
            {"classifier": "krr", "gamma": 0.1, "gamma_multiplier": None},
            {"classifier": "krr", "gamma": 1.0, "gamma_multiplier": None},
            {"classifier": "logreg", "gamma": None, "gamma_multiplier": None},
            {"classifier": "krr", "gamma": None, "gamma_multiplier": 0.1},
        ]
    )
    original = rows.copy(deep=True)
    table = reporting._markdown_table(rows, ["classifier", "gamma", "gamma_multiplier"])
    assert "| Kernel ridge | Automatic (1 / selected input count) | n/a |" in table
    assert "| Logistic regression | n/a | n/a |" in table
    assert "| Kernel ridge | n/a | 0.100 |" in table
    pd.testing.assert_frame_equal(rows, original)

    candidates = rows.iloc[:5].assign(
        selector="S2N", replication_mode="strict", k=40, alpha=1.0, C=None, n_neighbors=None
    )
    summaries = []
    monkeypatch.setattr(reporting, "_markdown_table", lambda frame, columns: summaries.append(frame) or [])
    reporting._original_search_space_table(pd.concat([candidates, candidates], ignore_index=True))
    counts = summaries[0].set_index("classifier")
    assert counts.loc["krr", ["evaluated_configs", "gamma_values"]].tolist() == [4, 4]
    assert counts.loc["logreg", "gamma_values"] == 0
