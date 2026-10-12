"""Offline reader report built only from a verified, curated evidence snapshot."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from html import escape
from pathlib import Path
from textwrap import wrap


FIGURES = (
    "benchmark_comparison.png",
    "tuned_vs_original_delta.png",
    "feature_stability.png",
    "lockbox_vs_mspc.png",
    "temporal_drift.png",
    "workload_cost_framing.png",
)
CORE_FILES = frozenset({"final_report.md", "evidence/run_manifest.json", *(f"figures/{name}" for name in FIGURES)})
REPOSITORY = "https://github.com/stevennitesh/secom-yield-monitoring"
REPORT_URL = "https://stevennitesh.github.io/secom-yield-monitoring/"
FEATURED_IMAGE = "images/featured.jpg"

STYLE = """
*{box-sizing:border-box}
:root{color-scheme:light;--ink:#172e3a;--muted:#526675;--teal:#086e74;--line:#d8e3e7;--paper:#f3f6f7;--reference:#365b78}
html{scroll-behavior:smooth;scroll-padding-top:5rem}
body{margin:0;background:var(--paper);color:var(--ink);font:17px/1.65 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1120px;margin:auto;padding:2.5rem 2rem}
a{color:var(--teal);text-underline-offset:4px;overflow-wrap:anywhere}
button{font:inherit;cursor:pointer}
a:focus-visible,button:focus-visible,summary:focus-visible,[tabindex]:focus-visible{outline:3px solid #da8d1d;outline-offset:4px}
.skip{position:absolute;left:1rem;top:-5rem;background:white;padding:.5rem;z-index:9}.skip:focus{top:1rem}
.hero{background:#153b49;color:white;border-radius:18px;padding:2.3rem 2.6rem}
.eyebrow,.section-number{font-size:.76rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase}
.hero .eyebrow{color:#99d9d1;margin:0 0 1.2rem}
.byline{font-size:.94rem;color:#bfe4df;margin:0 0 1.1rem}.hero-contribution{max-width:780px;color:#d7e6eb;font-size:.94rem}
h1{font-size:clamp(2.2rem,5vw,3.3rem);line-height:1.12;letter-spacing:-.035em;max-width:850px;margin:0 0 1.2rem}
.hero-intro{max-width:780px;color:#d7e6eb;font-size:1.08rem}
.hero-result{font-size:1.32rem;font-weight:700;margin-top:1.8rem}
.hero-boundary{font-size:.9rem;color:#bfd4db;max-width:850px}
.hero-actions{display:flex;flex-wrap:wrap;gap:.7rem;margin-top:1.5rem}.hero-actions a{padding:.55rem .9rem;border:1px solid #789ca7;border-radius:6px;color:white;text-decoration:none;font-size:.87rem;font-weight:600}.hero-actions a:first-child{background:#d7f0eb;color:#153b49;border-color:#d7f0eb}.hero-actions a:hover{text-decoration:underline}
p{margin:0 0 1rem;overflow-wrap:anywhere}p:last-child{margin-bottom:0}
.metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1.4rem;margin-top:1.6rem;font-variant-numeric:tabular-nums}
.metric{border-top:1px solid #62818c;padding-top:1rem}.metric strong{display:block;font-size:2rem;line-height:1.2;letter-spacing:-.025em}
.metric span{display:block;font-size:.9rem;color:#d7e6eb;margin-top:.35rem}.metric small{display:block;color:#b7ccd4;font-size:.8rem}
.contents{position:sticky;top:0;z-index:2;display:flex;flex-wrap:wrap;align-items:center;gap:.3rem 1.1rem;padding:1rem 0;background:var(--paper);border-bottom:1px solid var(--line)}
.contents a{font-size:.9rem;font-weight:600;text-decoration:none;padding:.25rem 0}.contents a:hover{text-decoration:underline}
.contents .print{margin-left:auto;color:var(--muted);background:none;border:0;font-size:.82rem;display:none}
.interactive .print{display:block}
section{min-width:0;background:white;padding:2.1rem 2.4rem;margin-top:1.5rem;border:1px solid var(--line);border-radius:14px;scroll-margin-top:1rem}
.section-number{color:var(--teal);margin-bottom:.4rem}h2{font-size:1.8rem;line-height:1.25;letter-spacing:-.02em;margin:0 0 1rem}h3{font-size:1.16rem;line-height:1.4;margin:1.8rem 0 .7rem}
.lede{font-size:1.08rem;max-width:850px}.muted,.boundary{color:var(--muted);font-size:.9rem}
.facts{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.8rem;margin:1.5rem 0}
.fact{padding:1rem;background:#edf4f5;border-radius:8px}.fact strong{display:block;font-size:1.5rem}.fact span{font-size:.8rem;color:var(--muted)}
.pipeline,.timeline{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.7rem;padding:0;list-style:none;margin:1.5rem 0;counter-reset:steps}
.pipeline li,.timeline li{counter-increment:steps;background:#edf4f5;border:1px solid var(--line);border-radius:8px;padding:1rem;font-size:.9rem}
.pipeline li:before,.timeline li:before{content:counter(steps,decimal-leading-zero);display:block;color:var(--teal);font-weight:750;font-size:.76rem;margin-bottom:.3rem}
.timeline{grid-template-columns:repeat(3,minmax(0,1fr))}.timeline strong{display:block}.timeline span{color:var(--muted);font-size:.85rem}
.note{background:#edf4f5;padding:1rem 1.15rem;border-left:3px solid var(--teal);border-radius:0 8px 8px 0;margin:1.3rem 0;font-size:.94rem}
.note.warm{background:#fff6e8;border-color:#b5761b}
.evidence-map{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.9rem;margin:1.5rem 0 0}.evidence-map article{border-top:3px solid var(--reference);padding:1rem;background:#f6f9fa;border-radius:0 0 8px 8px}.evidence-map article:nth-child(2){border-color:var(--teal)}.evidence-map article:last-child{border-color:#b5761b}.evidence-map h3{margin:0 0 .4rem;font-size:1rem}.evidence-map p{font-size:.85rem;color:var(--muted)}.evidence-map a{font-size:.85rem;font-weight:600}
.paired{background:#edf4f5;border-radius:8px;padding:1rem 1.2rem;margin:1.3rem 0}.paired p{font-size:.9rem;margin:0}.paired strong{color:var(--teal)}
.table-card{margin:1.3rem 0;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.table-label{padding:.85rem;font-size:.88rem;font-weight:650;background:#edf4f5}
.table-wrap{max-width:100%;overflow-x:auto}
.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
.scroll-hint{display:none;font-size:.75rem;font-weight:400;color:var(--muted)}
table{width:100%;border-collapse:collapse;font-size:.88rem;font-variant-numeric:tabular-nums}
caption{text-align:left;padding:.85rem;font-weight:650;background:#edf4f5}th,td{padding:.8rem .9rem;text-align:left;vertical-align:top;border-bottom:1px solid var(--line)}
thead th{background:#f0f5f6;font-size:.8rem}tbody th{font-weight:600}tbody tr:last-child>*{border-bottom:0}tbody tr:nth-child(even){background:#fafcfc}td.number{white-space:nowrap;text-align:right}
.report-figure{margin:1.7rem 0;border:1px solid var(--line);border-radius:12px;overflow:hidden;background:#f0f5f6}
.figure-heading{display:flex;justify-content:space-between;align-items:center;gap:1rem;padding:1rem 1.2rem}
.figure-heading h3{margin:0;font-size:1.05rem}.figure-number{font-size:.7rem;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.08em;margin:0 0 .2rem}
.figure-tools{display:flex;flex-wrap:wrap;gap:.7rem;align-items:center;flex-shrink:0}.figure-tools a{font-size:.875rem}
.enlarge{display:none;background:white;color:var(--teal);border:1px solid #9bb8bf;border-radius:5px;padding:.35rem .6rem;font-size:.875rem}.interactive .enlarge{display:inline-block}
.chart-panel{background:white;margin:0 .7rem .7rem;border-radius:6px;overflow:auto}.chart-panel img{display:block;width:100%;height:auto}
figcaption{padding:1rem 1.2rem;border-top:1px solid var(--line);font-size:.875rem;color:var(--muted)}
details{margin-top:1.4rem;border-top:1px solid var(--line);padding-top:1rem}summary{color:var(--teal);cursor:pointer;font-size:.95rem;font-weight:650}details[open]>summary{margin-bottom:1rem}
.definitions{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1.3rem 1.8rem;margin:0}.definitions dt{font-weight:700;margin-bottom:.35rem}.definitions dd{margin:0;font-size:.9rem;color:var(--muted)}
.downloads{display:flex;flex-wrap:wrap;gap:.6rem;margin:1.2rem 0}.downloads a{padding:.5rem .8rem;border:1px solid #aec6cd;border-radius:6px;background:#f0f5f6;text-decoration:none;font-size:.86rem}
.evidence-status{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem;margin:1.3rem 0}.evidence-status strong{font-size:1.5rem;display:block}.evidence-status span{font-size:.8rem;color:var(--muted)}
footer{padding:1.5rem .5rem;color:var(--muted);font-size:.82rem}.hash{font-size:.75rem;word-break:break-all}
dialog{width:calc(100% - 2rem);max-width:1400px;max-height:95vh;border:1px solid #aec6cd;border-radius:12px;padding:1rem;color:var(--ink)}
dialog::backdrop{background:#102e3ecc}.viewer-heading{display:flex;align-items:center;justify-content:space-between;gap:1rem}.viewer-heading h2{font-size:1.25rem;margin:0}.viewer-heading button,#zoom{padding:.4rem .7rem;border:1px solid #aec6cd;border-radius:5px;background:white;color:var(--teal)}
.viewer-help{font-size:.85rem;color:var(--muted);margin:1rem 0}.viewer-content{overflow:auto;max-height:70vh;border:1px solid var(--line);margin-top:1rem}.viewer-content img{display:block;width:100%;height:auto}.viewer-content.native img{width:auto;max-width:none}
@media(max-width:720px){body{font-size:16px}main{padding:1rem}.hero{padding:1.4rem}.hero-result{font-size:1.1rem}.metrics{gap:.8rem}.metric strong{font-size:1.3rem}.metric span,.metric small{font-size:.875rem}section{padding:1.3rem}.facts{grid-template-columns:repeat(2,minmax(0,1fr))}.pipeline{grid-template-columns:repeat(2,minmax(0,1fr))}.timeline,.definitions{grid-template-columns:1fr}.contents{gap:.3rem .9rem;padding:.65rem 0}.contents a{font-size:.875rem}.contents .print{display:none}.figure-heading{flex-direction:column;align-items:flex-start}.chart-panel img{min-width:0}.chart-panel{margin:0 .4rem .4rem}.boundary{font-size:.875rem}h2{font-size:1.45rem}th,td{padding:.6rem}.evidence-status{gap:.5rem}.evidence-status strong{font-size:1.3rem}.scroll-hint{display:block;margin-top:.3rem}.report-figure[data-portrait] .scroll-hint{display:none}.fact span,.figure-number,.hash,footer{font-size:.875rem}}
@media(max-width:720px){.evidence-map{grid-template-columns:1fr}.hero-actions a{font-size:.82rem}.paired{padding:.9rem}}
@media(max-width:480px){.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.metric:first-child{grid-column:1/-1}.metric:first-child strong{font-size:1.7rem}.hero-boundary{font-size:.875rem}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
@media print{body{background:white;font-size:11pt}main{max-width:none;padding:0}.hero{background:white;color:var(--ink);border:1px solid var(--line)}.hero .eyebrow,.hero-intro,.hero-boundary,.metric span,.metric small{color:var(--muted)}.contents,.figure-tools,dialog,.skip{display:none}section{padding:1rem;border:0;border-top:1px solid var(--line)}.chart-panel img{min-width:0}.report-figure,.facts,.timeline{break-inside:avoid}a{color:inherit}}
"""

SCRIPT = r"""
document.documentElement.classList.add('interactive');
const viewer = document.getElementById('chart-viewer');
const content = viewer.querySelector('.viewer-content');
const zoom = document.getElementById('zoom');
document.querySelector('[data-print]').addEventListener('click', () => window.print());
document.querySelectorAll('[data-enlarge]').forEach(button => button.addEventListener('click', () => {
  const figure = button.closest('figure');
  document.getElementById('viewer-title').textContent = figure.querySelector('h3').textContent;
  const image = figure.querySelector('img').cloneNode(true);
  image.src = figure.querySelector('img').currentSrc || image.src;
  content.replaceChildren(image);
  content.classList.remove('native'); zoom.setAttribute('aria-pressed','false'); zoom.textContent = 'Show native size';
  content.scrollLeft = 0; content.scrollTop = 0; viewer.showModal();
}));
zoom.addEventListener('click', () => {
  const native = content.classList.toggle('native');
  zoom.setAttribute('aria-pressed',String(native)); zoom.textContent = native ? 'Fit chart to width' : 'Show native size';
  content.scrollLeft = 0; content.scrollTop = 0;
});
viewer.addEventListener('click', event => { if (event.target === viewer) viewer.close(); });
function updateDownloads() {
  document.querySelectorAll('[data-download]').forEach(link => {
    const image = link.closest('figure').querySelector('img');
    link.href = image.currentSrc || image.src;
    link.download = link.dataset.filename.replace(/\.png$/, link.href.startsWith('data:image/svg+xml') ? '-portrait.svg' : '.png');
  });
}
document.querySelectorAll('.chart-panel img').forEach(image => image.addEventListener('load', updateDownloads));
window.addEventListener('resize', updateDownloads); updateDownloads();
function revealHash() {
  const target = document.getElementById(decodeURIComponent(location.hash.slice(1)));
  if (!target) return;
  let parent = target.closest('details'); let opened = false;
  while (parent) { if (!parent.open) { parent.open = true; opened = true; } parent = parent.parentElement.closest('details'); }
  if (opened) target.scrollIntoView();
}
window.addEventListener('hashchange',revealHash); revealHash();
let printing = [];
window.addEventListener('beforeprint', () => { printing = [...document.querySelectorAll('details')].map(d => [d,d.open]); printing.forEach(([d]) => d.open = true); });
window.addEventListener('afterprint', () => printing.forEach(([d,open]) => d.open = open));
"""


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _count(value: str) -> int:
    """Read whole counts, including the technical report's decimal formatting."""
    try:
        number = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"Invalid report count: {value}") from error
    if not number.is_finite() or number < 0 or number != number.to_integral_value():
        raise ValueError(f"Invalid report count: {value}")
    return int(number)


def verified_snapshot(snapshot: Path) -> tuple[dict, dict, dict[str, bytes]]:
    """Check every core receipt hash before reading results or mutating output."""
    receipt_bytes = (snapshot / "evidence/audit_receipt.json").read_bytes()
    receipt = json.loads(receipt_bytes)
    if receipt.get("audit_ok") is not True or receipt.get("errors"):
        raise ValueError("HTML report requires a passing publication audit")
    if set(receipt.get("files_sha256", {})) != CORE_FILES:
        raise ValueError("HTML report requires the complete curated report, manifest and six figures")
    inputs = {name: (snapshot / name).read_bytes() for name in sorted(CORE_FILES)}
    for name, content in inputs.items():
        if _sha(content) != receipt["files_sha256"][name]:
            raise ValueError(f"Curated evidence hash mismatch: {name}")
    inputs["evidence/audit_receipt.json"] = receipt_bytes
    manifest = json.loads(inputs["evidence/run_manifest.json"])
    if manifest["source_tree"]["sha256"] != receipt["source_tree_sha256"]:
        raise ValueError("HTML report execution identity differs from its audit receipt")
    return manifest, receipt, inputs


def _tables(text: str) -> list[list[dict[str, str]]]:
    """Read the canonical report's pipe tables, without interpreting Markdown as HTML."""
    result = []
    for block in re.findall(r"(?:^\|.*\|\s*$\n?)+", text, re.MULTILINE):
        rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in block.strip().splitlines()]
        if len(rows) < 2 or not all(re.fullmatch(r":?-+:?", cell) for cell in rows[1]):
            continue
        if any(len(row) != len(rows[0]) for row in rows[2:]):
            raise ValueError("Malformed canonical report table")
        result.append([dict(zip(rows[0], row, strict=True)) for row in rows[2:]])
    return result


def _table(text: str, heading: str, columns: list[str]) -> list[dict[str, str]]:
    """Require explicit headings and fields so schema changes cannot silently mislabel evidence."""
    section = text.partition(heading + "\n")[2]
    if not section:
        return []
    # Stop at the next heading at the same or a higher level.
    level = len(heading) - len(heading.lstrip("#"))
    section = re.split(rf"^#{{1,{level}}} ", section, maxsplit=1, flags=re.MULTILINE)[0]
    return next((rows for rows in _tables(section) if rows and set(columns) <= rows[0].keys()), [])


def _render_table(rows: list[dict[str, str]], columns: list[str], caption: str) -> str:
    if not rows:
        return '<p class="boundary">This result was not recorded in the supplied snapshot.</p>'
    head = "".join(f'<th scope="col">{escape(column)}</th>' for column in columns)

    def cell(value: str, index: int) -> str:
        if index == 0:
            return f'<th scope="row">{escape(value)}</th>'
        numeric = bool(re.fullmatch(r"[-+\d.,% /–→]+(?: pp| percentage points)?", value))
        style = ' class="number"' if numeric else ""
        return f"<td{style}>{escape(value)}</td>"

    body = "".join(
        "<tr>" + "".join(cell(row[column], index) for index, column in enumerate(columns)) + "</tr>" for row in rows
    )
    hint = '<span class="scroll-hint">Swipe horizontally for all columns.</span>' if len(columns) > 3 else ""
    return f'<div class="table-card"><div class="table-label" aria-hidden="true">{escape(caption)}{hint}</div><div class="table-wrap" tabindex="0" role="region" aria-label="{escape(caption)}"><table><caption class="sr-only">{escape(caption)}</caption><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div></div>'


def _uri(content: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(content).decode('ascii')}"


def _paragraphs(text: str, heading: str) -> list[str]:
    section = text.partition(heading + "\n")[2].split("\n## ", 1)[0]
    # Plain narrative paragraphs only; the values/tables have separate renderers.
    return [
        p.strip()
        for p in section.split("\n\n")
        if p.strip() and not p.lstrip().startswith(("|", "#", "!", "<", "-", "**"))
    ]


def _portrait_chart(title: str, panels: list[dict]) -> bytes:
    """Stack saved chart values with readable labels; preserve each declared axis."""
    elements = []
    y = 40

    def text(x, baseline, value, *, size=30, weight=400, anchor="start", color="#183246"):
        elements.append(
            f'<text x="{x}" y="{baseline}" font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}" fill="{color}">{escape(str(value))}</text>'
        )

    for panel in panels:
        for line in wrap(panel["title"], 28):
            text(24, y, line, size=34, weight=650)
            y += 40
        for line in wrap(panel["scope"], 34):
            text(24, y, line, size=30, color="#526575")
            y += 36
        y += 20
        maximum = float(panel["maximum"])
        if maximum <= 0:
            raise ValueError("Portrait chart requires a positive axis maximum")
        for row in panel["rows"]:
            value = float(row["value"])
            if not 0 <= value <= maximum:
                raise ValueError("Portrait chart value is outside its declared axis")
            for line in wrap(row["label"], 34):
                text(24, y, line, weight=600)
                y += 36
            text(24, y, row["display"], size=30, color="#526575")
            y += 16
            elements.append(f'<rect x="24" y="{y}" width="532" height="30" rx="3" fill="#edf3f5"/>')
            elements.append(
                f'<rect x="24" y="{y}" width="{532 * value / maximum:.4f}" height="30" rx="3" '
                f'fill="{row["color"]}" data-value="{value}" data-axis-max="{maximum}"/>'
            )
            if "range" in row:
                low, high = row["range"]
                if not 0 <= low <= value <= high <= maximum:
                    raise ValueError("Portrait chart range is outside its declared axis")
                left, right = (24 + 532 * bound / maximum for bound in (low, high))
                mid = y + 15
                elements.append(
                    f'<path d="M {left:.4f} {mid} H {right:.4f} M {left:.4f} {mid - 8} V {mid + 8} '
                    f'M {right:.4f} {mid - 8} V {mid + 8}" stroke="#183246" stroke-width="3" fill="none" '
                    f'data-low="{low}" data-high="{high}"/>'
                )
            y += 68
        for tick in panel["ticks"]:
            x = 24 + 532 * tick / maximum
            text(
                x,
                y,
                f"{tick:g}{panel.get('suffix', '')}",
                size=30,
                anchor="start" if tick == 0 else "end" if tick == maximum else "middle",
                color="#526575",
            )
        y += 24
        elements.append(f'<path d="M 24 {y} H 556" stroke="#c4cfd7" stroke-width="1"/>')
        y += 54
    description = "; ".join(
        panel["title"] + ": " + "; ".join(row["label"] + ", " + row["display"] for row in panel["rows"])
        for panel in panels
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="580" height="{y}" viewBox="0 0 580 {y}" '
        'role="img" aria-labelledby="title desc" style="font-family:system-ui,-apple-system,Segoe UI,sans-serif">'
        f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>'
        f'<rect width="580" height="{y}" fill="white"/>{"".join(elements)}</svg>'
    ).encode("utf-8")


def _benchmark_portrait(before: dict, after: dict, failures: int, passes: int) -> bytes:
    rows = [("Reference", before, "#61788e"), ("Tuned", after, "#087f78")]
    ranges = [[float(v.rstrip("%")) for v in row["Fold range"].split("–")] for _, row, _ in rows]
    maximum = max(60, max(bounds[1] for bounds in ranges) + 12)
    panels = [
        {
            "title": "Balanced error · lower is better",
            "scope": "Fold means; whiskers show observed fold ranges, not confidence intervals.",
            "maximum": maximum,
            "ticks": [0, 20, 40, maximum],
            "suffix": "%",
            "rows": [
                {
                    "label": label,
                    "value": float(row["Balanced error"].rstrip("%")),
                    "display": f"{row['Balanced error']} · range {row['Fold range']}",
                    "range": bounds,
                    "color": color,
                }
                for (label, row, color), bounds in zip(rows, ranges, strict=True)
            ],
        }
    ]
    for field, total, title, class_name in (
        ("Failures caught", failures, "Failures caught", "failures"),
        ("False alerts on passes", passes, "False alerts on passes", "passes"),
    ):
        panels.append(
            {
                "title": title,
                "scope": f"Pooled held-out counts · {total:,} {class_name} evaluated",
                "maximum": total,
                "ticks": [0, total // 2, total],
                "rows": [
                    {
                        "label": label,
                        "value": _count(row[field]),
                        "display": f"{_count(row[field]):,} / {total:,}",
                        "color": color,
                    }
                    for label, row, color in rows
                ],
            }
        )
    return _portrait_chart("Reference and tuned: the complete selected procedure", panels)


def _frozen_portrait(rows: list[dict]) -> bytes:
    panels = []
    colors = ["#61788e", "#087f78", "#929ea8"]
    for field, complement, title, class_name in (
        ("Failures caught", "Failures missed", "Failures caught", "failures"),
        ("False alerts on passes", "Passes left unflagged", "False alerts on passes", "passes"),
    ):
        values = []
        for row, color in zip(rows, colors, strict=True):
            n, total = _count(row[field]), _count(row[field]) + _count(row[complement])
            if not total:
                raise ValueError("Frozen-rule portrait requires nonempty classes")
            rate = 100 * n / total
            values.append(
                {
                    "label": row["Frozen rule"].replace("Primary model:", "Logistic regression:"),
                    "value": rate,
                    "display": f"{n:,} / {total:,} {class_name} · {rate:.1f}%",
                    "color": color,
                }
            )
        panels.append(
            {
                "title": title,
                "scope": "Same retrospective later block; thresholds frozen on earlier calibration.",
                "maximum": 100,
                "ticks": [0, 25, 50, 75, 100],
                "suffix": "%",
                "rows": values,
            }
        )
    return _portrait_chart("Frozen rules on the retrospective later block", panels)


def _feature_portrait(rows: list[dict]) -> bytes | None:
    if not rows:
        return None
    groups = dict.fromkeys(row["Study"] for row in rows)
    return _portrait_chart(
        "Selection frequency for the same plotted anonymous inputs",
        [
            {
                "title": f"{study} · selection frequency",
                "scope": "Exploratory family; overlapping training folds. Association is not causation.",
                "maximum": 100,
                "ticks": [0, 25, 50, 75, 100],
                "suffix": "%",
                "rows": [
                    {
                        "label": row["Input"],
                        "value": float(row["Selection frequency"].rstrip("%")),
                        "display": row["Selection frequency"],
                        "color": "#61788e" if study == "Reference" else "#087f78",
                    }
                    for row in rows
                    if row["Study"] == study
                ],
            }
            for study in groups
        ],
    )


def _figure(
    inputs: dict[str, bytes], report: str, name: str, title: str, number: int, *, portrait: bytes | None = None
) -> str:
    path = f"figures/{name}"
    match = re.search(rf"!\[([^\]]+)\]\({re.escape(path)}\)\s*\n\s*\n([^\n]+)", report)
    if match is None:
        alt, caption = (
            title,
            "No result for this figure was recorded in the canonical report; inspect its study status.",
        )
    else:
        alt, caption = match.groups()
    identifier = name.removesuffix(".png")
    source = f'<source media="(max-width:720px)" srcset="{_uri(portrait, "image/svg+xml")}">' if portrait else ""
    portrait_attribute = " data-portrait" if portrait else ""
    return f'''<figure class="report-figure"{portrait_attribute} aria-labelledby="{identifier}-title" aria-describedby="{identifier}-caption">
<div class="figure-heading"><div><p class="figure-number">Figure {number}</p><h3 id="{identifier}-title">{escape(title)}</h3></div>
<div class="figure-tools"><button class="enlarge" type="button" data-enlarge aria-haspopup="dialog" aria-controls="chart-viewer">Enlarge chart</button><a href="{path}" data-download data-filename="{name}" download="{name}">Download chart</a></div></div>
<div class="chart-panel" tabindex="0" role="region" aria-label="{escape(title)}"><picture>{source}<img src="{_uri(inputs[path], "image/png")}" alt="{escape(alt)}"></picture></div>
<figcaption id="{identifier}-caption">{escape(caption)}<span class="scroll-hint">Enlarge the chart and choose native size to inspect detailed labels.</span></figcaption></figure>'''


def render_html(
    manifest: dict,
    receipt: dict,
    inputs: dict[str, bytes],
    *,
    performance: dict | None = None,
    featured_image: bool = False,
) -> str:
    """Compose the reader story from displayed, audited values; fit no models."""
    report = inputs["final_report.md"].decode("utf-8").replace("\r\n", "\n")
    benchmark_columns = [
        "Procedure",
        "Balanced error",
        "Failure recall",
        "Pass specificity",
        "Fold spread (SD)",
        "Fold range",
        "Failures caught",
        "False alerts on passes",
    ]
    reference = _table(report, "## Original Replication Results", benchmark_columns)
    tuned = _table(report, "## Tuned Benchmark Results", benchmark_columns)
    try:
        before = next(row for row in reference if row["Procedure"] == "Complete selected procedure")
        after = next(row for row in tuned if row["Procedure"] == "Complete selected procedure")
        all_pass = next(row for row in reference if row["Procedure"] == "Always predict pass")
    except StopIteration as error:
        raise ValueError(
            "HTML report requires complete selected benchmark procedures and the always-pass baseline"
        ) from error
    comparison = [
        {**row, "Procedure": label}
        for row, label in (
            (before, "Reference: fixed input budget"),
            (after, "Tuned: bounded input search"),
            (all_pass, "Always predict pass"),
        )
    ]
    alerts_delta = _count(after["False alerts on passes"]) - _count(before["False alerts on passes"])
    caught_delta = _count(after["Failures caught"]) - _count(before["Failures caught"])

    def change(value, noun):
        return f"{abs(value)} {'fewer' if value < 0 else 'more'} {noun}" if value else f"The same number of {noun}"

    tradeoff = f"{change(alerts_delta, 'false alerts')}. {change(caught_delta, 'failures caught')}."
    ber_delta = Decimal(before["Balanced error"].rstrip("%")) - Decimal(after["Balanced error"].rstrip("%"))
    error_change = (
        f"Mean balanced error {'fell' if ber_delta > 0 else 'rose'} by {abs(ber_delta):.2f} percentage points"
        if ber_delta
        else "Mean balanced error was unchanged"
    )
    dataset = manifest["dataset"]
    settings = manifest.get("execution", {}).get("settings", {})
    fold_match = re.search(r"Both benchmarks used (\d+) held-out test folds", report)
    folds = fold_match.group(1) if fold_match else "the recorded"
    paired_match = re.search(
        r"On the paired test folds, tuning lowers balanced error in (\d+), raises it in (\d+), and ties in (\d+)\.",
        report,
    )
    paired = ""
    if paired_match:
        improved, worsened, tied = map(int, paired_match.groups())
        if fold_match and improved + worsened + tied != int(folds):
            raise ValueError("Paired benchmark fold counts differ from the recorded evaluation")
        paired = (
            '<div class="paired"><p><strong>'
            f"{improved} folds improved · {worsened} worsened · {tied} tied"
            "</strong><br>Matched test folds show an uneven tradeoff. This is descriptive, not a significance test.</p></div>"
        )
    sample_count, fail_count, pass_count = (int(dataset[name]) for name in ("n_samples", "n_fails", "n_passes"))
    start, end = (
        datetime.fromisoformat(dataset[name]).strftime("%B %d, %Y").replace(" 0", " ")
        for name in ("timestamp_min", "timestamp_max")
    )
    budgets = settings.get("tuned_feature_budgets", [])
    budget_text = ", ".join(map(str, budgets)) or "the recorded input budgets"
    fixed_budget = settings.get("original_feature_budget", "the recorded")
    # These tables also provide text equivalents for the temporal figures.
    periods = _table(
        report, "## Temporal Robustness Stress Test", ["Test period", "First observation", "Last observation"]
    )
    calibration = _table(
        report,
        "### Calibration counts and threshold sensitivity",
        [
            "Later test period",
            "Calibration samples / failures / passes",
            "Logistic regression flagged range",
            "Kernel ridge (20%) flagged range",
        ],
    )
    lr_summary = re.search(r"Mean per-period balanced error: ([\d.]+%)", report)
    lr_periods = _table(
        report,
        "### Later-period logistic-regression results",
        ["Later test period", "Test samples / failures", "Balanced error", "Ranking AUC"],
    )
    lr_alerts = _table(
        report,
        "### Later-period logistic-regression results",
        ["Later test period", "False alerts on passes", "Samples flagged"],
    )
    krr_periods = _table(
        report,
        "### Kernel-ridge comparison and calibration-window sensitivity",
        [
            "Procedure",
            "Period 1: error / AUC / flagged",
            "Period 2: error / AUC / flagged",
            "Period 3: error / AUC / flagged",
        ],
    )
    period_results = []
    krr_main = next(
        (row for row in krr_periods if row["Procedure"] == "Complete selected procedure (20% calibration)"), None
    )
    alerts_by_period = {row["Later test period"]: row for row in lr_alerts}
    for row in lr_periods:
        period = row["Later test period"]
        if period not in alerts_by_period:
            raise ValueError("Chronological rate and alert tables reference different periods")
        period_results.append(
            {
                "Period": period,
                "Procedure": "Logistic regression",
                "Samples / failures": row["Test samples / failures"],
                "Balanced error": row["Balanced error"],
                "Ranking AUC": row["Ranking AUC"],
                "Samples flagged": alerts_by_period[period]["Samples flagged"],
            }
        )
        if krr_main:
            values = krr_main[f"Period {period}: error / AUC / flagged"].split(" / ")
            if len(values) != 3:
                raise ValueError("Kernel-ridge period metrics require error, AUC and flagged fraction")
            period_results.append(
                {
                    "Period": period,
                    "Procedure": "Kernel ridge (20% calibration)",
                    "Samples / failures": row["Test samples / failures"],
                    **dict(zip(["Balanced error", "Ranking AUC", "Samples flagged"], values, strict=True)),
                }
            )
    krr_summary = re.search(
        r"The main kernel-ridge procedure flags ([\d.]+%) of the (\d+) later samples\. Its mean period balanced error is ([\d.]+%)",
        report,
    )
    temporal = []
    if lr_summary:
        alert_lookup = {row["Later test period"]: row for row in lr_alerts}
        lr_volume = "Not recorded"
        if lr_periods and set(alert_lookup) == {row["Later test period"] for row in lr_periods}:
            n = sum(_count(row["Test samples / failures"].split(" / ")[0]) for row in lr_periods)
            flagged = sum(
                _count(row["Failures caught"])
                + _count(alert_lookup[row["Later test period"]]["False alerts on passes"])
                for row in lr_periods
            )
            lr_volume = f"{flagged / n:.1%} of {n} samples" if n else "Unavailable"
        temporal.append(
            {
                "Procedure": "Logistic regression",
                "Mean period balanced error": lr_summary[1],
                "Pooled samples flagged": lr_volume,
            }
        )
    if krr_summary:
        temporal.append(
            {
                "Procedure": "Kernel ridge: 20% calibration",
                "Mean period balanced error": krr_summary[3],
                "Pooled samples flagged": krr_summary[1] + f" of {krr_summary[2]} samples",
            }
        )
    frozen_columns = [
        "Frozen rule",
        "Failures caught",
        "Failures missed",
        "False alerts on passes",
        "Passes left unflagged",
    ]
    frozen = _table(report, "### The final later block: retrospective, frozen thresholds", frozen_columns)
    mspc = _table(
        report,
        "#### Supervised vs MSPC",
        ["Evaluation samples", "Failures caught", "Passes unflagged", "False alerts", "Failures missed"],
    )
    frozen = [row for row in frozen if row["Frozen rule"].startswith("Primary model:")]
    for row in mspc:
        if row["Evaluation samples"] == "Final retrospective block":
            frozen.append(
                dict(
                    zip(
                        frozen_columns,
                        [
                            "Statistical process control (MSPC)",
                            row["Failures caught"],
                            row["Failures missed"],
                            row["False alerts"],
                            row["Passes unflagged"],
                        ],
                        strict=True,
                    )
                )
            )
    populations = {
        (
            _count(r["Failures caught"]) + _count(r["Failures missed"]),
            _count(r["False alerts on passes"]) + _count(r["Passes left unflagged"]),
        )
        for r in frozen
    }
    if len(populations) > 1:
        raise ValueError("Retrospective comparison populations differ")
    later_scope = "Final-block counts are unavailable."
    if populations:
        later_fails, later_passes = populations.pop()
        later_scope = f"{later_fails + later_passes} samples: {later_fails} failures and {later_passes} passes."
    costs = _table(
        report,
        "##### Cost Curves",
        [
            "Cost ratio",
            "Primary: balanced error",
            "Primary: workload limited",
            "Always predict pass",
            "Flag every sample",
        ],
    )
    feature_paragraphs = _paragraphs(report, "## Feature Stability and Interpretation")
    feature_prose = feature_paragraphs[:2]
    feature_context = [p for p in feature_paragraphs[2:] if "panel displays" in p]
    feature_values = _table(report, "### Chart values: selected inputs", ["Study", "Input", "Selection frequency"])
    drift_values = _table(
        report, "#### Chart values: distribution checks", ["Measure", "Recorded value", "Reference population"]
    )
    cost_columns = [
        "Cost ratio",
        "Primary: balanced-error threshold",
        "Primary: workload-limited threshold",
        "Always predict pass",
        "Flag every sample",
    ]
    costs = [
        {
            "Cost ratio": row["Cost ratio"],
            "Primary: balanced-error threshold": row["Primary: balanced error"],
            "Primary: workload-limited threshold": row["Primary: workload limited"],
            "Always predict pass": row["Always predict pass"],
            "Flag every sample": row["Flag every sample"],
        }
        for row in costs
    ]

    def paragraphs(items):
        return "".join(f"<p>{escape(item)}</p>" for item in items)

    performance_callout = ""
    if performance:
        reference_minutes = float(performance["reference_duration_seconds"]["total_modeling"]) / 60
        optimized_minutes = float(performance["optimized_duration_seconds"]["total_modeling"]) / 60
        identical_count = int(performance["identical_scientific_csvs"])
        if identical_count != len(performance["csv_sha256"]) or not reference_minutes > optimized_minutes > 0:
            raise ValueError("Invalid historical runtime comparison receipt")
        performance_callout = f'''<div class="note"><strong>Faster execution, unchanged scientific outputs.</strong>
<p>In the October 3, 2026 controlled comparison, I reduced modeling time from {reference_minutes:.2f} to {
            optimized_minutes:.2f} minutes on the same Windows machine. All {
            identical_count
        } scientific CSVs were byte-identical.</p>
<p class="boundary">Vectorized ReliefF scoring and bounded reuse preserved the inputs, folds, seeds and search grids. This one-pair comparison excludes audit and report time. It predates the expanded study shown here. <a href="{
            REPOSITORY
        }/blob/main/docs/performance.md">Read the engineering comparison and receipt →</a></p></div>'''

    downloads = "".join(
        f'<a href="{_uri(inputs[name], mime)}" download="{Path(name).name}">{label}</a>'
        for name, mime, label in (
            ("final_report.md", "text/markdown", "Full technical report · Markdown"),
            ("evidence/run_manifest.json", "application/json", "Execution manifest · JSON"),
            ("evidence/audit_receipt.json", "application/json", "Scientific snapshot audit · JSON"),
        )
    )
    social_image = (
        f'<meta property="og:image" content="{REPORT_URL}{FEATURED_IMAGE}">'
        '<meta property="og:image:width" content="1200"><meta property="og:image:height" content="627">'
        '<meta property="og:image:alt" content="Semiconductor Failure Classification · Steven Bangdiwala">'
        '<meta name="twitter:card" content="summary_large_image">'
        if featured_image
        else ""
    )
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="A reproducible SECOM study: bounded tuning, the failure and false-alert tradeoff, and limits of transfer to later manufacturing samples.">
<link rel="canonical" href="{REPORT_URL}"><meta property="og:type" content="article"><meta property="og:url" content="{
        REPORT_URL
    }">
<meta property="og:title" content="Semiconductor Failure Classification · Steven Bangdiwala">
<meta property="og:description" content="A Python study of semiconductor test outcomes: nested evaluation, chronological transfer and reproducible ML engineering.">{
        social_image
    }
<title>Semiconductor Failure Classification · Steven Bangdiwala</title><style>{STYLE}</style></head><body>
<a class="skip" href="#overview">Skip to report</a><main>
<header class="hero"><p class="eyebrow">SECOM · Python · Applied machine learning</p><p class="byline">Steven Bangdiwala · Independent study</p><h1>Semiconductor Failure Classification</h1>
<p class="hero-intro">Can anonymous manufacturing measurements identify failures, and do those patterns hold in later samples? This study compares a reference benchmark, bounded tuning and chronological transfer.</p>
<p class="hero-contribution"><strong>What I built:</strong> a Python pipeline for nested model evaluation, feature selection, threshold calibration, runtime acceleration and reproducible analytical reporting.</p>
<p class="hero-result">{escape(tradeoff)}</p><p class="hero-boundary">The complete training-selected procedures share {
        folds
    } held-out folds. Rates average folds; counts pool {fail_count:,} failures and {
        pass_count:,} passes. Later-sample tests are separate, and the final block is retrospective.</p>
<div class="metrics"><div class="metric"><strong>{
        escape(after["Balanced error"])
    }</strong><span>Tuned balanced error · lower is better</span><small>Reference: {
        escape(before["Balanced error"])
    }</small></div>
<div class="metric"><strong>{
        escape(after["False alerts on passes"])
    }</strong><span>False alerts on passing samples</span><small>Reference: {
        escape(before["False alerts on passes"])
    }</small></div>
<div class="metric"><strong>{escape(after["Failures caught"])} / {
        fail_count
    }</strong><span>Failures caught</span><small>Reference: {escape(before["Failures caught"])} / {
        fail_count
    }</small></div></div><div class="hero-actions"><a href="#benchmark">Explore the results</a><a href="{
        REPOSITORY
    }">View the repository</a><a href="#evidence">Inspect the evidence</a></div></header>
<nav class="contents" aria-label="Report contents"><a href="#overview">Overview</a><a href="#build">What I built</a><a href="#benchmark">Results</a><a href="#inputs">Selected inputs</a><a href="#transfer">Changing conditions</a><a href="#evidence">Methods &amp; evidence</a><button class="print" data-print type="button">Print / save PDF</button></nav>
<section id="overview"><p class="section-number">01 · The question</p><h2>Rare failures change how accuracy should be read</h2>
<p class="lede">Most recorded tests pass. An always-pass rule reaches {
        100
        * pass_count
        / sample_count:.2f}% accuracy while catching no failures. The useful test gives failure detection and false alerts equal weight.</p>
<div class="facts"><div class="fact"><strong>{
        sample_count:,}</strong><span>Recorded production entities</span></div><div class="fact"><strong>{
        dataset["n_features"]:,}</strong><span>Anonymous measurements</span></div><div class="fact"><strong>{
        fail_count:,}</strong><span>Failures · {
        100 * fail_count / sample_count:.2f}% of samples</span></div><div class="fact"><strong>{
        100 * dataset["missing_fraction"]:.2f}%</strong><span>Measurement cells missing</span></div></div>
<p><strong>Balanced error</strong> averages the missed-failure rate and the false-alert rate on passes. Lower is better. Predicting pass for everyone gives 50% balanced error.</p>
<p class="boundary">Source: <a href="https://archive.ics.uci.edu/dataset/179/secom">McCann and Johnston’s UCI SECOM dataset</a>, CC BY 4.0. Observations span {
        start
    }–{
        end
    }. A row represents a production entity with an undocumented physical unit; measurement availability before the outcome is unknown. The raw file contains {
        dataset["n_features"]
    } measurement columns, while public metadata list 591 features.</p>
<h3>Three questions, three evidence boundaries</h3>
<div class="evidence-map"><article><h3>1. Benchmark</h3><p>{
        sample_count:,} samples · shuffled folds.<br>Compare reference and tuned selection on the same held-out samples.</p><a href="#benchmark">Read the tradeoff →</a></article>
<article><h3>2. Chronological transfer</h3><p>Disjoint later periods · earlier fitting and calibration.<br>Check whether patterns transfer through time within this dataset.</p><a href="#transfer">Read the stress tests →</a></article>
<article><h3>3. Final retrospective block</h3><p>{
        escape(later_scope)
    }<br>This subset was already exposed. It supplies no fresh confirmation.</p><a href="#later-block">Read the frozen rules →</a></article></div></section>
<section id="build"><p class="section-number">02 · Engineering</p><h2>A reproducible study with explicit evaluation boundaries</h2>
<p>I built the Python orchestration, feature-selection comparisons, calibration rules, vectorized ReliefF accelerator, artifact audits and reporting. Model estimators come from scikit-learn; ReliefF builds on skrebate.</p>
{performance_callout}
<ol class="pipeline" aria-label="Implemented study workflow"><li>Validate inputs and record their hashes</li><li>Select inputs and tune within training samples</li><li>Evaluate saved held-out predictions</li><li>Audit artifacts and render the evidence</li></ol>
<p>Imputation, scaling and feature selection use training data only. Inner folds choose the selection method, model, input mode, settings and threshold. Outer folds evaluate the complete procedure once on untouched samples.</p>
<details id="methods"><summary>How the benchmark comparison is controlled</summary>
<p>The reference uses a fixed budget of up to {fixed_budget} inputs. Tuning compares {
        escape(budget_text)
    } inputs on the same outer folds and samples. Balanced error is the inner selection objective; exact ties prefer fewer inputs, then stronger regularization.</p>
<p>Benchmark thresholds use pooled inner held-out scores before the outer refit. Scores can shift after refitting; this is a calibration limitation. The bounded grid is not a global optimum, and a family chosen for its best test summary is exploratory.</p>
<p>Kernel ridge learns nonlinear similarities between samples. Logistic regression learns a weighted combination of inputs. <a href="{
        REPOSITORY
    }/tree/main/src/secom/workflows">Study code</a> · <a href="{
        REPOSITORY
    }/blob/main/docs/engineering.md">Engineering checks</a> · <a href="{
        REPOSITORY
    }/blob/main/docs/performance.md">Measured performance</a></p></details></section>
<section id="benchmark"><p class="section-number">03 · Primary benchmark</p><h2>Tuning changes the failure and false-alert tradeoff</h2>
<p class="lede">{
        escape(error_change)
    }. The result is a comparison of complete training-selected procedures, rather than a model chosen after inspecting test performance.</p>
{
        _render_table(
            comparison,
            ["Procedure", "Balanced error", "Failure recall", "Pass specificity"],
            "Class-balanced performance · each rate averages the same held-out folds",
        )
    }
<p class="boundary">Failure recall is the share of failures caught; pass specificity is the share of passes left unflagged. The search is bounded; these results do not establish a globally optimal model or an exact reproduction of the published study.</p>
{paired}
{
        _figure(
            inputs,
            report,
            FIGURES[0],
            "Reference and tuned procedures on the same test folds",
            1,
            portrait=_benchmark_portrait(before, after, fail_count, pass_count),
        )
    }
<details id="benchmark-counts"><summary>Exact counts and observed fold variation</summary>
{
        _render_table(
            comparison,
            ["Procedure", "Failures caught", "False alerts on passes"],
            f"Once-held-out predictions · {fail_count:,} failures and {pass_count:,} passes",
        )
    }
{
        _render_table(
            comparison,
            ["Procedure", "Fold spread (SD)"],
            "Balanced-error spread · descriptive standard deviation in percentage points",
        )
    }
<p class="boundary">Rates average test folds; counts pool predictions. Chart whiskers show observed minimum and maximum fold error. SD and ranges are descriptive, not algorithm-performance confidence intervals.</p></details>
<details id="input-contrasts"><summary>Supporting comparison: predefined input modes</summary><p>Measurements-only and measurements-plus-missing-flags procedures provide contrasts. Their test results do not replace the complete selected procedure as the headline. Positive chart values mean lower error after tuning.</p>
{_figure(inputs, report, FIGURES[1], "Balanced-error changes by predefined input mode", 2)}</details></section>
<section id="inputs"><p class="section-number">04 · Feature selection</p><h2>Repeatedly selected inputs are investigation clues</h2>
{paragraphs(feature_prose)}
{
        _figure(
            inputs,
            report,
            FIGURES[2],
            "Selection frequency for anonymous measurements and missing flags",
            3,
            portrait=_feature_portrait(feature_values),
        )
    }
{paragraphs(feature_context)}
<details id="feature-values"><summary>Read the selected inputs as a table</summary>
{
        _render_table(
            feature_values,
            ["Study", "Input", "Selection frequency"],
            "Same plotted inputs · exploratory families · overlapping training folds",
        )
    }</details>
<div class="note">A process engineer could use stable associations to prioritize measurements for follow-up. Named sensors, verified collection timing and intervention records are needed before drawing process or causal conclusions.</div></section>
<section id="transfer"><p class="section-number">05 · Secondary stress tests</p><h2>Later samples expose weak transfer and fragile thresholds</h2>
<p>The shuffled benchmark tests other samples from the same dataset. The chronological study asks whether earlier patterns transfer to later calendar periods. It covers weeks in the historical dataset.</p>
<ol class="timeline" aria-label="Chronological evaluation boundary"><li><strong>Earlier fitting samples</strong><span>Tune and fit using earlier labels</span></li><li><strong>Separate calibration</strong><span>Choose a threshold and retain the same fitted model</span></li><li><strong>Later test samples</strong><span>Score once with the frozen threshold; do not refit</span></li></ol>
{
        _render_table(
            temporal,
            ["Procedure", "Mean period balanced error", "Pooled samples flagged"],
            "Disjoint chronological test periods · separate from the primary benchmark",
        )
    }
<p class="boundary">Calibration uses the last 20% of each earlier training region, not 20% of the whole dataset. Model grids and preprocessing differ, so this is not an isolated algorithm comparison. The technical report also retains every input-mode comparison and an independently tuned 30% calibration sensitivity. Alert volume matters alongside failure recall.</p>
<h3>Alert volume changes sharply between periods</h3>
{
        _render_table(
            period_results,
            ["Period", "Procedure", "Samples / failures", "Balanced error", "Ranking AUC", "Samples flagged"],
            "Main procedures on identical later samples · each period is evaluated once",
        )
    }
<p class="boundary">An always-pass rule has 50% balanced error. Ranking AUC measures score ordering: 0.5 is chance ordering. It does not set an alert threshold. These separate procedures do not reproduce the shuffled benchmark's performance.</p>
<details id="chronological-detail"><summary>Calendar periods and calibration sensitivity</summary>
{
        _render_table(
            periods,
            ["Test period", "First observation", "Last observation"],
            "Observed chronological test boundaries · source timezone unspecified",
        )
    }
<p>Samples are disjoint even where boundary dates overlap. Calibration often contains only a few failures. Leaving one out and recalibrating the same fixed scores can change the threshold sharply.</p>
{
        _render_table(
            calibration,
            [
                "Later test period",
                "Calibration samples / failures / passes",
                "Logistic regression flagged range",
                "Kernel ridge (20%) flagged range",
            ],
            "Fixed-score calibration sensitivity · ranges are not confidence intervals",
        )
    }
</details><h3 id="later-block">Final retrospective block: frozen rules, a separate population</h3>
<p>{
        escape(later_scope)
    } This last block was already exposed through benchmarking and prior reports. It provides retrospective evidence rather than fresh confirmation. Only logistic regression and multivariate statistical process control (MSPC) are compared here.</p>
{_render_table(frozen, frozen_columns, "Frozen rules on the same retrospective later block")}
<p class="boundary">MSPC uses principal components fitted on earlier passing samples, with its score and thresholds selected on calibration data. The full report retains 95% exact conditional binomial intervals and separate diagnostics that choose thresholds from evaluation labels. Sparse classes and temporal dependence limit inference; no superiority claim follows.</p>
{
        _figure(
            inputs,
            report,
            FIGURES[3],
            "Failures caught and false alerts with frozen rules",
            4,
            portrait=_frozen_portrait(frozen) if len(frozen) == 3 else None,
        )
    }
<details id="drift-and-cost"><summary>Supporting evidence: distribution shifts, workload and hypothetical costs</summary>
<p>Measurement distributions use earlier fitting samples as reference. Score distributions use held-out calibration scores from the same retained model. Population stability index (PSI) describes distribution change; larger values indicate more change. These checks identify change without establishing its cause.</p>
{_figure(inputs, report, FIGURES[4], "Measurement and score distribution shifts", 5)}
<details id="drift-values"><summary>Read the distribution checks as a table</summary>
{
        _render_table(
            drift_values,
            ["Measure", "Recorded value", "Reference population"],
            "Separate reference populations · primary logistic-regression model",
        )
    }</details>
<p>The 10% workload rule limits the unweighted mean weekly flagged fraction on calibration samples. It does not guarantee every week’s workload or future capacity. Costs of frozen rules use retrospective later-block counts and hypothetical assumptions.</p>
{_figure(inputs, report, FIGURES[5], "Calibration workload and retrospective hypothetical costs", 6)}
{
        _render_table(
            costs,
            cost_columns,
            "Hypothetical cost per later-block sample · false-alert cost = 1",
        )
    }
<p class="boundary">Cost = (false alerts + cost ratio × missed failures) / later-block sample count. The ratio is missed-failure cost divided by false-alert cost. No threshold is selected from these curves, and these are not measured manufacturing economics.</p></details></section>
<section id="limits"><p class="section-number">06 · Interpretation</p><h2>What this evidence supports</h2>
<p>The study demonstrates controlled benchmark comparison, bounded model selection, reproducible prediction audits and an explicit test of chronological transfer. The modest benchmark tradeoff does not establish stable future alerting.</p>
<div class="note warm">Anonymous measurements, unknown collection timing and a single historical dataset do not establish causal effects, early warning, intervention benefit or production readiness.</div>
{
        _render_table(
            [
                {
                    "Next evidence": "Named measurements and collection timing",
                    "Question it would answer": "What is measured, and is it available before the outcome?",
                },
                {
                    "Next evidence": "Tool/device identifiers and intervention records",
                    "Question it would answer": "Where does variation arise, and can an action improve outcomes?",
                },
                {
                    "Next evidence": "Independent later manufacturing data",
                    "Question it would answer": "Do learned patterns and frozen rules transfer?",
                },
                {
                    "Next evidence": "Actual review capacity and error costs",
                    "Question it would answer": "Which alert policy is useful under observed constraints?",
                },
            ],
            ["Next evidence", "Question it would answer"],
            "Data required for stronger conclusions",
        )
    }
</section><section id="glossary"><p class="section-number">07 · Reading the measures</p><h2>Metric glossary</h2>
<dl class="definitions"><div><dt>Balanced error</dt><dd>Average missed-failure and false-alert rates. Lower is better; displayed as a percentage. An always-pass rule gives 50%.</dd></div>
<div><dt>Failure recall</dt><dd>Failures caught divided by all failures. Higher means more recorded failures detected.</dd></div><div><dt>Pass specificity</dt><dd>Passes left unflagged divided by all passes. Higher means fewer false alerts.</dd></div><div><dt>Samples flagged</dt><dd>Fraction of samples that trigger an alert, including correct failure detections and false alerts.</dd></div>
<div><dt>Fold spread</dt><dd>Observed variation across test folds. SD means standard deviation; this spread is descriptive.</dd></div><div><dt>Calibration</dt><dd>Choosing an alert threshold from held-out scores. It does not guarantee a calibrated probability or future workload.</dd></div></dl></section>
<section id="evidence"><p class="section-number">08 · Evidence trail</p><h2>Inspect the methods and reproduce the study</h2>
<p>The technical report contains the full search tables, input-mode comparisons, per-period results and intervals. The execution manifest and audit receipt bind the saved study to its inputs and scientific source.</p>
<div class="downloads">{downloads}<a href="html_provenance.json" download>HTML provenance · adjacent JSON</a></div>
<p>Charts, styles, text and the three embedded evidence downloads work offline in this single HTML. The adjacent HTML-provenance file separately binds this page to its verified inputs and renderer. Detailed CSVs and ZIP archives remain in local run storage.</p>
<p><a href="{REPOSITORY}">Repository and setup</a> · <a href="{
        REPOSITORY
    }/blob/main/docs/development.md">Evidence and regeneration procedure</a> · <a href="{
        REPOSITORY
    }/tree/main/docs/spec">Scientific specifications</a>. These source links require a network connection.</p>
<details id="identities"><summary>Technical verification, source identity and claim restrictions</summary>
<div class="evidence-status"><div><strong>{
        len(receipt["errors"])
    }</strong><span>Artifact audit errors</span></div><div><strong>{
        len(receipt["warnings"])
    }</strong><span>Temporal warnings</span></div><div><strong>{
        len(receipt["claim_restrictions"])
    }</strong><span>Claim restrictions</span></div></div>
<p>A passing artifact audit checks consistency; it does not establish industrial validation or independent reproduction. This presentation fits no models and retains the historical scientific execution.</p>
<p class="hash"><strong>Scientific execution source SHA-256:</strong> {escape(manifest["source_tree"]["sha256"])}</p>
<ul>{"".join("<li>" + escape(str(item)) + "</li>" for item in receipt["claim_restrictions"])}</ul>
<p>The scientific execution manifest and snapshot receipt are preserved. HTML generation has a separate rendering identity; later documentation or presentation changes do not refresh historical model results.</p></details></section>
<footer>Steven Bangdiwala · SECOM research report · Dataset: McCann and Johnston (2008), UCI, DOI 10.24432/C54305, CC BY 4.0. Software: MIT. Main results are benchmark estimates; temporal findings are secondary and the final block is retrospective.</footer></main>
<dialog id="chart-viewer" aria-labelledby="viewer-title" aria-describedby="viewer-help"><div class="viewer-heading"><h2 id="viewer-title">Chart</h2><form method="dialog"><button>Close chart</button></form></div><p class="viewer-help" id="viewer-help">Choose native size for detailed labels, then scroll as needed. Press Escape or Close chart to return.</p><button id="zoom" type="button" aria-pressed="false">Show native size</button><div class="viewer-content" tabindex="0" role="region" aria-label="Enlarged chart"></div></dialog>
<script>{SCRIPT}</script></body></html>
'''


def write_html_report(snapshot: Path) -> Path:
    """Add HTML and its own provenance to a fresh public staging directory."""
    manifest, receipt, inputs = verified_snapshot(snapshot)
    if (snapshot / "index.html").exists() or (snapshot / "html_provenance.json").exists():
        raise ValueError("HTML report requires fresh output files")
    renderer = Path(__file__)
    project = renderer.parents[2]
    performance_path = project / "docs/performance/runtime_comparison.json"
    performance_content = performance_path.read_bytes()
    performance = json.loads(performance_content)
    featured_path = project / "docs/images/featured.jpg"
    featured = featured_path.read_bytes() if featured_path.is_file() else None
    document = render_html(
        manifest, receipt, inputs, performance=performance, featured_image=featured is not None
    ).encode("utf-8")
    outputs = {"index.html": document}
    if featured is not None:
        outputs[FEATURED_IMAGE] = featured
    source_files = [renderer, project / "scripts/run_html_report.py"]
    provenance = {
        "presentation_only": True,
        "models_fitted": 0,
        "scientific_source_tree_sha256": manifest["source_tree"]["sha256"],
        "inputs_sha256": {name: _sha(content) for name, content in sorted(inputs.items())},
        "renderer_sha256": {
            p.relative_to(project).as_posix(): _sha(p.read_bytes().replace(b"\r\n", b"\n")) for p in source_files
        },
        "presentation_inputs_sha256": {"docs/performance/runtime_comparison.json": _sha(performance_content)},
        "outputs_sha256": {name: _sha(content) for name, content in outputs.items()},
        "scientific_receipt": "evidence/audit_receipt.json",
    }
    for name, content in outputs.items():
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (snapshot / "html_provenance.json").write_text(
        json.dumps(provenance, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return snapshot / "index.html"


def build_html_report(snapshot: Path, destination: Path) -> Path:
    """Create a portable report folder without modifying the supplied evidence."""
    source, target = snapshot.resolve(), destination.resolve()
    if source.is_relative_to(target) or target.is_relative_to(source):
        raise ValueError("HTML source and destination must not overlap")
    if target.exists():
        raise ValueError("HTML report requires a fresh destination")
    # Validate and compose the story before creating a destination.
    manifest, receipt, inputs = verified_snapshot(source)
    render_html(manifest, receipt, inputs)
    target.mkdir(parents=True)
    for name, content in inputs.items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return write_html_report(target)
