# Laconic

**Token-efficient middleware for multi-agent LLM workflows.** Laconic reduces
the tokens spent when agents pass messages to each other, by compressing the
natural-language payload of each handoff while preserving the structural
scaffolding — tool calls, function arguments, IDs, schema fields — **losslessly**.

[![CI](https://github.com/lindameng-0/laconic/actions/workflows/ci.yml/badge.svg)](https://github.com/lindameng-0/laconic/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)

## The problem

Multi-agent workflows burn most of their budget on *inter-agent* traffic:
verbose reports, repeated context, boilerplate courtesy — re-read by a model
on every hop. General-purpose prompt compression (the LLMLingua family and
similar) is tuned for prose. Agent messages are not prose: they are structure
— tool calls, arguments, IDs, control fields — and that structure is
load-bearing. Compression that flattens it doesn't just lose nuance, it
**breaks the workflow**: a mangled `tool_calls` block fails at the API
boundary, a corrupted ID poisons every downstream step.

Laconic's job is narrower and harder than "compress a prompt": compress the
payload, never touch the structure, and never break the workflow. Its failure
mode is always *no savings*, never *corrupted handoff*.

## Honest ceiling — read this first

Laconic works entirely in the **text channel**. Users of closed models
(Claude, GPT) have no access to hidden states, KV caches, or model internals,
so there is no latent-space magic here and no order-of-magnitude claim:

- Realistic gains are a **1.3–3× factor on chatty natural-language payloads**,
  and less on structure-heavy traffic (which is protected by design).
- Dedup of repeated content is often the biggest real-world win — and it
  interacts with provider prompt caching, which Laconic is careful not to
  break (see [docs/limitations.md](docs/limitations.md)).
- Every reported saving is measured in **tokens with the target model's
  tokenizer**, never characters, and every number is labeled with how it was
  produced (`exact` / `api` / `estimate`).

## Install

```bash
pip install laconic              # core: zero heavy dependencies
pip install "laconic[openai]"    # + tiktoken for exact OpenAI token counts
pip install "laconic[anthropic]" # + Anthropic client for exact API counts
pip install "laconic[all]"       # everything (openai, anthropic, langgraph, plots)
```

> Not yet published to PyPI — install from source: `pip install -e ".[dev]"`.

## Quickstart

Profile first, compress second. The toy example runs **offline, no API keys**:

```bash
python examples/two_agent_toy/run.py
```

In code — one hook on the seam where messages cross between agents:

```python
from laconic import Session, summary_table
from laconic.integrations import CompressingHook

hook = CompressingHook(Session(target_model="gpt-4.1"))  # default: telegraphic

# wherever agent A's output becomes agent B's input:
cheap_handoff = hook.process_message(
    {"role": "assistant", "content": verbose_agent_report},
    source_agent="researcher",
    target_agent="writer",
)

print(summary_table(hook.profile))   # tokens before/after, per edge, with cost
```

Or profile an existing message log without integrating anything:

```bash
laconic profile trace.jsonl --model gpt-4.1                # where do tokens burn?
laconic profile trace.jsonl --model gpt-4.1 --strategy balanced   # what-if savings
```

LangGraph users: insert a compression node between two agents —

```python
from laconic.integrations import make_compression_node

graph.add_node("laconic", make_compression_node("gpt-4.1", "telegraphic",
                                                source_agent="planner",
                                                target_agent="executor"))
graph.add_edge("planner", "laconic")
graph.add_edge("laconic", "executor")
```

## How it works

```
raw message ──parse──▶  structural (tool calls, args, IDs)  ── untouched ──┐
                  └───▶  payload (natural language)                        │
                              │                                            ▼
                        dedup ─▶ compress ─▶ verify round-trip ─▶ rebuild message
                                                  │
                                     any failure ⇒ passthrough, recorded
```

- **Structure is safe by construction**: compressors only ever receive the
  payload text; the structural part is deep-copied at parse time and verified
  byte-identical after rebuild. Payloads embedding structure (code fences,
  JSON paragraphs, tables) are segmented and those segments protected too.
- **Strategies**: `off` (measure only) · `conservative` (filler pruning +
  token-checked abbreviations only) · `telegraphic` (**default**: conservative
  plus telegram-style function-word dropping — the closest thing to "AI
  shorthand" that stays inside models' training distribution; every content
  word, number, name, and negation survives) · `balanced` (sentence pruning
  at the model's safe keep-ratio) · `aggressive` (opt-in, clearly labeled).
- **Dedup**: repeated blocks are replaced with a short reference — only when
  the recipient already has the content earlier in its own context (safe
  default), or via an explicit rehydration tool (opt-in). Never rewrites
  message history, so provider prompt caching keeps working.
- **Profiler**: per-edge token/cost accounting, text and self-contained HTML
  reports, standalone trace analysis.

## The research: cross-model compression tolerance

Different models tolerate different compression aggressiveness on agent
traffic. `laconic.eval` runs a fixed benchmark (three task families with
verifiable ground truth, shipped in [`data/benchmark/`](data/benchmark/))
across **{model} × {strategy} × {keep-ratio}**, producing per-model
accuracy-vs-compression curves, the safe operating point per model, and the
headline comparison: structure-preserving compression vs. naive whole-message
compression.

```bash
laconic eval --tasks data/benchmark --out results   # offline, mock client
python -m laconic.eval.generate --seed 7            # regenerate the benchmark
```

The offline run uses a deterministic mock receiver and measures **information
survival** (did load-bearing content reach the receiver at all). Real-model
runs measure comprehension and require API keys — see
[docs/benchmark.md](docs/benchmark.md).

### Results

**Offline (information survival, mock client)** — reproducible in seconds
with `laconic eval`; full table and reading guide in
[docs/benchmark.md](docs/benchmark.md). Headline: on the shipped 120-task
benchmark, `tool_plan_handoff` accuracy stays at **1.000 under Laconic at
every compression ratio** (structure is untouchable by construction) and
degrades to **0.008 under naive whole-message compression** at ratio 0.30;
prose families trace a graceful tolerance curve instead of a cliff.

**Real-model results** — the repository intentionally ships **without**
claimed numbers for real models. The harness produces them
(`docs/benchmark.md`); this table stays a template until the maintainer runs
it:

| model | strategy | keep ratio | tasks | accuracy | mean token ratio | fallbacks |
|---|---|---:|---:|---:|---:|---:|
| *(run the eval against real APIs and paste `results/summary.md` here)* | | | | | | |

## Documentation

- [Architecture](docs/architecture.md) — modules and data contracts
- [Design decisions](docs/design-decisions.md) — ADRs: why text-channel-only,
  why structure-preserving, why tokens-not-characters, why reliability-first
- [Limitations](docs/limitations.md) — the honest-ceiling section
- [Related work](docs/related-work.md) — LLMLingua, CompactPrompt, TOON,
  latent communication, and where Laconic sits
- [Benchmark](docs/benchmark.md) — task families, generation, scoring
- [API reference](docs/api-reference.md)
- [Contributing](docs/contributing.md)

## Development

```bash
pip install -e ".[dev]"
pytest              # entire suite runs offline
ruff check .
```

## Citation

If you use Laconic in academic work, please cite it (see
[CITATION.cff](CITATION.cff)).

## License

[MIT](LICENSE)
