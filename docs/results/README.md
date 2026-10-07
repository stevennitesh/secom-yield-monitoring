# Report and Evidence

**[Read the SECOM study report →](https://stevennitesh.github.io/secom-yield-monitoring/)**

The HTML report is the main reading path for the question, implementation, findings and limits. Download [index.html](index.html) to read the same report offline in a browser; keep [html_provenance.json](html_provenance.json) beside it to inspect its rendering record.

## Evidence

| Record | What it establishes |
| --- | --- |
| [Canonical technical report](final_report.md) | Complete methods, result tables, intervals and diagnostics |
| [Execution manifest](evidence/run_manifest.json) | Scientific execution, input/source hashes, settings and timing |
| [Scientific snapshot audit](evidence/audit_receipt.json) | Core artifact hashes, errors, warnings and claim restrictions |
| [HTML provenance](html_provenance.json) | Separate renderer, input and HTML output hashes; no model fitting |
| [Engineering checks](../engineering.md) | Data validation, prediction audits, cache equivalence and export safeguards |
| [Development procedure](../development.md#documentation-changes-and-historical-provenance) | Source matching, historical audits and deliberate refreshes |

Detailed CSVs and complete source/artifact archives remain in ignored local run storage. Both report formats present saved results and retain rendering identities separately from scientific execution.
