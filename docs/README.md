# Documentation Map

Start with the project overview, follow the charts, then inspect the technical evidence as needed.

| Reader's question | Document |
| --- | --- |
| What problem does this solve, and what happened? | [Project overview](../README.md) |
| How should I read the results? | [Visual case study](results/README.md) |
| Can I read an organized report offline? | [HTML report](results/index.html) — open locally in a browser |
| What exactly was measured? | [Full technical report](results/final_report.md) |
| Where is the code, and how do I verify changes? | [Development guide](development.md) |
| What scientific requirements must hold? | [Study specifications](spec/README.md) |
| What made the studies faster? | [Performance evidence](performance.md) |
| What engineering safeguards support the results? | [Engineering checks](engineering.md) |

The benchmarks, chronological stress tests, artifact audit and report exporter are implemented. The [evidence record](results/README.md#evidence) identifies the scientific execution and its report rendering. `learning/` contains separate teaching examples.

The study evaluates recorded pass/fail labels. It does not establish production readiness, causal effects or early warning. Software uses the [MIT license](../LICENSE); the dataset retains its separate attribution.
