# The benchmark and the tolerance study

## Research question

**How much compression does inter-agent traffic tolerate, per model, before
task accuracy degrades — and how much further does structure-preserving
compression push that frontier compared to naive compression?**

The study runs a fixed benchmark across the matrix
`{model} × {strategy} × {keep-ratio}` and reports, per model:

1. an accuracy-vs-achieved-token-ratio curve per strategy,
2. the **safe operating point**: the lowest token ratio whose accuracy stays
   within a tolerance (default 0.02) of the uncompressed baseline,
3. the headline comparison: `laconic-budgeted` vs. `naive` (whole-message,
   structure-blind pruning).

## Task families

Every task is one handoff plus a verifiable receiver-side check — accuracy is
mechanical, no LLM judge required.

| family | handoff | receiver must | measures |
|---|---|---|---|
| `extraction_relay` | verbose report with embedded facts | state specific fact values | information survival in prose |
| `tool_plan_handoff` | message carrying `tool_calls` + verbose rationale | restate the tool name and arguments exactly | structural integrity |
| `constraint_tracking` | brief with 2–4 hard constraints buried in filler | restate every constraint with exact values | loss of low-salience but load-bearing content |

The shipped dataset (`data/benchmark/*.jsonl`, 40 tasks per family) is
generated deterministically:

```bash
python -m laconic.eval.generate --seed 7 --per-family 40 --out data/benchmark
```

Anyone can regenerate it byte-identically, scale it up, or reseed it.

## Clients: offline vs. real

**Mock client (offline, default).** A deterministic perfect extractor: it
restates tool calls only if their structure still parses, and quotes report
sentences relevant to the question only if they still exist. It measures
**information survival**, not comprehension. CI runs this end-to-end; results
are labeled `client=mock`.

**Real clients.** `OpenAIClient` / `AnthropicClient` run the same matrix
against actual models (keys required). These produce the numbers the research
claims need. Nothing in this repository pre-fills them.

## Running

```bash
# offline study (seconds, free):
laconic eval --tasks data/benchmark --out results

# real-model study (costs money; do this deliberately):
python - <<'PY'
from laconic.eval import load_tasks, run_matrix, write_results, MatrixSpec, OpenAIClient
from laconic.eval.metrics import summarize, results_markdown
from pathlib import Path

tasks = [t for p in Path("data/benchmark").glob("*.jsonl") for t in load_tasks(p)]
spec = MatrixSpec(models=["gpt-4.1-mini"], keep_ratios=[0.9, 0.75, 0.6, 0.45, 0.3])
records = run_matrix(tasks, OpenAIClient(), spec)
write_results(records, "results/gpt-4.1-mini")
print(results_markdown(summarize(records)))
PY

# figures (requires laconic[plots]):
python - <<'PY'
import json
from laconic.eval.runner import RunRecord
from laconic.eval.metrics import summarize
from laconic.eval.plots import plot_accuracy_vs_ratio

records = [RunRecord(**r) for r in json.load(open("results/results.json", encoding="utf-8"))]
print(plot_accuracy_vs_ratio(summarize(records), "results/figures"))
PY
```

Outputs: `results.json`, `results.csv`, `summary.md` (the markdown table used
in the README), and one `tolerance_<model>.png` per model.

## Example offline run (mock client — information survival, not model accuracy)

Produced by `laconic eval --tasks data/benchmark --out results` on the shipped
seed-7 dataset (120 tasks, laconic v0.1.0). Reproducible in seconds:

| model | strategy | keep ratio | tasks | accuracy | mean token ratio |
|---|---|---:|---:|---:|---:|
| mock | passthrough | — | 120 | 1.000 | 1.000 |
| mock | laconic-conservative | — | 120 | 1.000 | 0.936 |
| mock | laconic-budgeted | 0.90 | 120 | 0.967 | 0.835 |
| mock | laconic-budgeted | 0.75 | 120 | 0.850 | 0.729 |
| mock | laconic-budgeted | 0.60 | 120 | 0.600 | 0.620 |
| mock | laconic-budgeted | 0.45 | 120 | 0.383 | 0.482 |
| mock | laconic-budgeted | 0.30 | 120 | 0.333 | 0.387 |
| mock | naive | 0.90 | 120 | 0.983 | 0.794 |
| mock | naive | 0.75 | 120 | 0.392 | 0.662 |
| mock | naive | 0.60 | 120 | 0.175 | 0.538 |
| mock | naive | 0.45 | 120 | 0.058 | 0.402 |
| mock | naive | 0.30 | 120 | 0.008 | 0.271 |

Per-family, the structure-preservation effect is stark: `tool_plan_handoff`
stays at **1.000 under Laconic at every ratio** (structure is untouchable by
construction) while naive falls to 0.008 at ratio 0.30. The families degrade
in the predicted order as budgets tighten: tool plans never, extraction and
constraints progressively — the tolerance curve, not a cliff.

These numbers measure **information survival** through a deterministic
extractor. They are not model accuracy; real-model runs may be kinder
(models infer from partial context) or harsher (compression artifacts
confuse them). That comparison is the open research question this harness
exists to answer.

## Interpreting results

- The **passthrough** row is the accuracy ceiling and the token baseline.
- `laconic-conservative` should sit at (or within tolerance of) the ceiling
  with a token ratio around 0.9 on this benchmark's verbose traffic.
- `laconic-budgeted` traces the tolerance curve as the budget tightens;
  `constraint_tracking` typically degrades first — that is the intended
  canary, not a bug.
- `naive` collapses on `tool_plan_handoff` at modest budgets. The horizontal
  gap between the naive and laconic curves at equal accuracy is the value of
  structure preservation.
- Estimator-tier token counts are labeled in every record (`count_tier`).
  Cross-model comparisons should use matching tiers.

## Honesty rules

- No number in the repository is invented; templates stay empty until a run
  fills them.
- Mock-client results are always labeled as information-survival, never
  presented as model accuracy.
- The benchmark is synthetic and its construction is fully disclosed (seeded
  generator in-repo). It measures compression tolerance on controlled
  traffic; it does not claim to represent any particular production
  distribution. Run the harness on your own traces for deployment decisions.
