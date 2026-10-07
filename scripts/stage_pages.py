"""Verify and stage the saved SECOM report for GitHub Pages, without fitting models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

CORE_FILES = frozenset(
    {
        "final_report.md",
        "evidence/run_manifest.json",
        *(
            f"figures/{name}.png"
            for name in (
                "benchmark_comparison",
                "tuned_vs_original_delta",
                "feature_stability",
                "lockbox_vs_mspc",
                "temporal_drift",
                "workload_cost_framing",
            )
        ),
    }
)


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def stage_pages(snapshot: Path, destination: Path) -> Path:
    source, target = snapshot.resolve(), destination.resolve()
    if source.is_relative_to(target) or target.is_relative_to(source):
        raise ValueError("Pages source and destination must not overlap")
    if target.exists():
        raise ValueError("Pages staging requires a fresh destination")

    # A fixed publication inventory keeps this script independent of ML imports
    # and prevents receipt paths or unrelated local files from expanding the site.
    receipt_bytes = (source / "evidence/audit_receipt.json").read_bytes()
    receipt = json.loads(receipt_bytes)
    if receipt.get("audit_ok") is not True or receipt.get("errors"):
        raise ValueError("Pages requires a passing scientific snapshot audit")
    if set(receipt.get("files_sha256", {})) != CORE_FILES:
        raise ValueError("Pages requires the complete curated report, manifest and six figures")
    files = {name: (source / name).read_bytes() for name in sorted(CORE_FILES)}
    hashes = {name: digest(content) for name, content in files.items()}
    if receipt["files_sha256"] != hashes:
        raise ValueError("Core evidence hash differs from the scientific snapshot receipt")
    manifest = json.loads(files["evidence/run_manifest.json"])
    if manifest["source_tree"]["sha256"] != receipt["source_tree_sha256"]:
        raise ValueError("Scientific snapshot execution identity differs from its receipt")
    files["evidence/audit_receipt.json"] = receipt_bytes
    html = (source / "index.html").read_bytes()
    provenance_bytes = (source / "html_provenance.json").read_bytes()
    provenance = json.loads(provenance_bytes)
    hashes["evidence/audit_receipt.json"] = digest(receipt_bytes)
    if provenance.get("inputs_sha256") != hashes:
        raise ValueError("HTML provenance does not match the verified scientific snapshot")
    if provenance.get("outputs_sha256") != {"index.html": digest(html)}:
        raise ValueError("HTML output hash differs from its rendering provenance")
    if provenance.get("scientific_source_tree_sha256") != manifest["source_tree"]["sha256"]:
        raise ValueError("HTML provenance references a different scientific execution")
    if provenance.get("presentation_only") is not True or provenance.get("models_fitted") != 0:
        raise ValueError("Pages requires a presentation-only report with no model fitting")

    # Publish only named, verified bytes; manual guides and unrelated files stay out.
    files.update({"index.html": html, "html_provenance.json": provenance_bytes, ".nojekyll": b""})
    target.mkdir(parents=True)
    for name, content in files.items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("docs/results"))
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Fresh directory containing only public report files"
    )
    args = parser.parse_args()
    try:
        print(stage_pages(args.input_dir, args.output_dir))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
