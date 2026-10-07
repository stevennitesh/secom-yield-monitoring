"""Evidence custody, dynamic values and offline behavior of the HTML reader report."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
from html.parser import HTMLParser
from pathlib import Path

import pytest

from secom.html_report import CORE_FILES, _count, build_html_report, render_html, verified_snapshot


SOURCE = Path(__file__).resolve().parents[1] / "docs/results"


class ReportMarkup(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = []
        self.ids = []
        self.links = []
        self.external_resources = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "img":
            self.images.append(attrs)
        if tag == "a":
            self.links.append(attrs)
        if (tag == "script" and "src" in attrs) or (tag == "link" and attrs.get("rel") == "stylesheet"):
            self.external_resources.append(attrs)


def test_html_is_offline_and_preserves_curated_evidence(workspace_tmp_dir):
    source_hashes = {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() for name in CORE_FILES}
    output = workspace_tmp_dir / "html"
    page = build_html_report(SOURCE, output)
    markup = ReportMarkup()
    markup.feed(page.read_text(encoding="utf-8"))
    assert len(markup.images) == 6
    assert all(image["alt"] for image in markup.images)
    assert not markup.external_resources
    assert "31.5% of 750 samples" in page.read_text(encoding="utf-8")
    assert len(markup.ids) == len(set(markup.ids))
    assert all(link["href"][1:] in markup.ids for link in markup.links if link["href"].startswith("#"))
    for image in markup.images:
        payload = base64.b64decode(image["src"].split(",", 1)[1])
        assert payload in [(SOURCE / name).read_bytes() for name in CORE_FILES if name.endswith(".png")]
    for link in markup.links:
        if link["href"].startswith("data:"):
            payload = base64.b64decode(link["href"].split(",", 1)[1])
            assert payload in [(SOURCE / name).read_bytes() for name in (*CORE_FILES, "evidence/audit_receipt.json")]
    provenance = json.loads((output / "html_provenance.json").read_text())
    assert provenance["outputs_sha256"]["index.html"] == hashlib.sha256(page.read_bytes()).hexdigest()
    assert provenance["models_fitted"] == 0
    for name, digest in source_hashes.items():
        assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == digest
        assert (SOURCE / name).read_bytes() == (output / name).read_bytes()


def test_report_story_uses_supplied_values_and_handles_windows_newlines():
    manifest, receipt, inputs = verified_snapshot(SOURCE)
    # Fabricated display values exercise the renderer, not the scientific auditor.
    text = inputs["final_report.md"].decode("utf-8").replace("\r\n", "\n")
    text = text.replace("| 70 | 445 |", "| 70 | 450 |").replace("| 66 | 371 |", "| 65 | 400 |")
    changed = {**inputs, "final_report.md": text.replace("\n", "\r\n").encode("utf-8")}
    page = render_html(manifest, receipt, changed)
    assert "50 fewer false alerts. 5 fewer failures caught." in page
    assert "74 fewer false alerts. 4 fewer failures caught." not in page
    assert "<strong>400</strong>" in page and "<strong>65 / 104</strong>" in page


def test_decimal_formatted_counts_preserve_the_retrospective_population():
    manifest, receipt, inputs = verified_snapshot(SOURCE)
    text = inputs["final_report.md"].decode("utf-8").replace("\r\n", "\n")
    text = text.replace("| 0 | 223 | 3 | 9 |", "| 0.000 | 223.000 | 3.000 | 9.000 |")
    assert text != inputs["final_report.md"].decode("utf-8").replace("\r\n", "\n")
    page = render_html(manifest, receipt, {**inputs, "final_report.md": text.encode("utf-8")})
    assert "235 samples: 9 failures and 226 passes." in page
    for value in ("1.5", "NaN", "Infinity", "-1", "unknown"):
        with pytest.raises(ValueError, match="Invalid report count"):
            _count(value)


@pytest.mark.parametrize("failure", ["hash", "failed_audit", "extra_path", "identity"])
def test_html_rejects_invalid_evidence_before_creating_output(workspace_tmp_dir, failure):
    snapshot = workspace_tmp_dir / "snapshot"
    shutil.copytree(SOURCE, snapshot)
    path = snapshot / "evidence/audit_receipt.json"
    receipt = json.loads(path.read_text())
    if failure == "hash":
        (snapshot / "final_report.md").write_bytes(b"changed\n")
    elif failure == "failed_audit":
        receipt["audit_ok"] = False
    elif failure == "extra_path":
        receipt["files_sha256"]["../../outside"] = "untrusted"
    else:
        receipt["source_tree_sha256"] = "different"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    destination = workspace_tmp_dir / "fresh"
    with pytest.raises(ValueError):
        build_html_report(snapshot, destination)
    assert not destination.exists()


def test_html_rejects_reuse_and_overlapping_paths(workspace_tmp_dir):
    with pytest.raises(ValueError, match="overlap"):
        build_html_report(SOURCE, SOURCE / "nested")
    destination = workspace_tmp_dir / "old"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("preserve")
    with pytest.raises(ValueError, match="fresh destination"):
        build_html_report(SOURCE, destination)
    assert marker.read_text() == "preserve"
