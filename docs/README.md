# Documentation Map

Start with the [hosted study report](https://stevennitesh.github.io/secom-yield-monitoring/). The [project README](../README.md) introduces the implementation and local setup.

| Reader's question | Document |
| --- | --- |
| What was investigated, built and found? | [Study report](https://stevennitesh.github.io/secom-yield-monitoring/) |
| Can I read the same report offline? | [HTML file](results/index.html) — open locally in a browser |
| What exactly was measured? | [Full technical report](results/final_report.md) |
| Where are the execution and rendering records? | [Evidence index](results/README.md#evidence) |
| Where is the code, and how do I verify changes? | [Development guide](development.md) |
| What scientific requirements must hold? | [Study specifications](spec/README.md) |
| What made the studies faster? | [Performance evidence](performance.md) |
| What engineering safeguards support the results? | [Engineering checks](engineering.md) |

The benchmarks, chronological stress tests, artifact audit and report exporter are implemented. The [evidence record](results/README.md#evidence) identifies the scientific execution and its report rendering. `learning/` contains separate teaching examples.

The study evaluates recorded pass/fail labels. It does not establish production readiness, causal effects or early warning. Software uses the [MIT license](../LICENSE); the dataset retains its separate attribution.
