# Laconic — Revised Build Plan

This document is the working plan for the repository. It started from an external
build prompt; the changes below were made after a critique pass. Each change is
recorded so the design rationale survives (see also `docs/design-decisions.md`).

## What Laconic is

Middleware that reduces the tokens spent when agents in a multi-agent LLM workflow
pass messages to each other, for users of closed models (Claude, GPT) where only the
text channel is available. It compresses the natural-language payload of inter-agent
handoffs while preserving structural scaffolding (tool calls, arguments, IDs, schema
fields) losslessly — and never breaks the workflow.

The research contribution is a characterization of **cross-model compression
tolerance**: per model, an accuracy-vs-compression-ratio curve, the safe operating
point, and evidence that structure-preserving compression extends the safe frontier
relative to naive whole-message compression.

## Critique: changes made to the original plan

| # | Original plan | Problem | Change |
|---|---|---|---|
| 1 | Dedup "rehydrates on receive" | Tokens are only spent at the model boundary. If content is rehydrated before the receiving model reads it, nothing is saved. | Dedup has two explicit modes: **context-dedup** (replace content the recipient already has earlier in its own context window — the model can look back) and **store-dedup** (handle + a rehydration tool the receiving agent can call on demand). Docs state precisely when each pays. |
| 2 | "Be aware of and document provider prompt-caching" | Too weak. Rewriting message history breaks provider prefix-caching, and cached input tokens are ~10x cheaper — naive dedup can *increase* cost. | Dedup is **prefix-stable**: previously sent messages are never rewritten; only new content is deduplicated. `docs/limitations.md` carries a dedup-vs-provider-caching decision table. |
| 3 | Abbreviation dictionary "in the spirit of CompactPrompt" | Underspecified receiver contract: abbreviated text without a legend risks comprehension; a legend costs tokens. | Substitutions restricted to a curated legend-free list (expansions any modern model reads natively, e.g. "for example" → "e.g."). Every substitution is applied **only if it reduces token count under the target model's tokenizer**. Optional legend mode exists with documented amortization math. |
| 4 | Anthropic token counting via the API endpoint | It's a network call requiring a key — an offline profiler can't depend on it, and per-substitution checks would be absurdly slow. | Three-tier token counting, never silently mixed: `exact` (tiktoken, local), `api` (Anthropic count-tokens endpoint, opt-in), `estimate` (calibrated offline heuristic, labeled with expected error). Every stat records its tier. |
| 5 | Module named `laconic/tokenize/` | Shadows the stdlib `tokenize` module name; ruff (A005) flags it. | Renamed to `laconic/tokenizers/`. |
| 6 | "Report the accuracy delta" (mechanism unspecified) | Not actionable as written. | Three concrete levels: (a) structural round-trip check — free, always on; (b) task-level accuracy on verifiable benchmark answers; (c) optional LLM-judge faithfulness score — opt-in, costs tokens. Reports label which level produced the number. |
| 7 | "A fixed multi-agent benchmark" (no tasks defined, no data) | Not reproducible; repo needs data. | A versioned benchmark dataset ships in `data/benchmark/`: three task families with verifiable ground truth (extraction relay, tool-plan handoff, constraint tracking), produced by a committed, seeded generator (`laconic/eval/generate.py`). |
| 8 | Toy example implies real API calls | CI and first-touch UX must not require keys. | A deterministic `MockModelClient` powers the toy example and all tests; real OpenAI/Anthropic clients are opt-in. |
| 9 | Single dependency pool | LLMLingua pulls torch; tiktoken/anthropic/langgraph aren't universally wanted. | Core install depends on pydantic only. Extras: `[openai]`, `[anthropic]`, `[llmlingua]`, `[langgraph]`, `[plots]`, `[docs]`, `[dev]`, `[all]`. The offline estimator keeps the zero-extra install useful. |
| 10 | Property test: "no compressor alters structural fields" | The architecture makes that impossible *by construction* (compressors only ever receive `payload`) — the test as stated can't fail. The actual corruption risk is parse → rebuild. | The property test targets the **parser/serializer round-trip**: over a synthetic corpus, structural content is byte-identical through parse → rebuild, and passthrough rebuild equals the original message exactly. |
| 11 | (absent) | Naive-compression baseline needed for the headline comparison. | `NaiveWholeMessageCompressor` — compresses the serialized message including structure — implemented *for the eval only*, clearly marked as the thing Laconic exists to avoid. |
| 12 | (minor) | — | Pricing table carries an as-of date and is user-overridable; CHANGELOG kept; Python 3.10–3.13; CI on Ubuntu + Windows; `laconic profile trace.jsonl` works standalone so the profiler is useful before any integration. |

Unfilled fields from the original prompt, resolved:

- **Package name**: `laconic` — verified available on PyPI (404 on 2026-07-02).
- **First framework integration**: LangGraph (largest install base; its message-list state maps cleanly to a middleware hook), plus a framework-agnostic generic hook.
- **Models**: tokenizer backends and profiles for OpenAI and Anthropic; the eval harness runs offline on a mock client and accepts real clients when keys exist.

## Architecture (unchanged in spirit, tightened in contract)

```
raw message ──parse──▶ Message{structural, payload, metadata}
                          │
                          ├─ structural  ──────────────── untouched ──┐
                          └─ payload ─▶ dedup ─▶ compressor ─▶ verify ┴─▶ rebuild
                                                              │
                                              fail ⇒ passthrough + fallback recorded
```

Data contracts (enforced in code and tests):

1. A compressor only ever sees `payload`. It is structurally impossible for it to
   corrupt `structural`.
2. Every `CompressionStats` includes original tokens, compressed tokens, ratio,
   model, strategy, token-count tier, a safety flag, and whether fallback occurred.
3. If parsing or the post-rebuild verification fails, the pipeline emits the original
   message unchanged and records the fallback. Failure mode is always "no savings",
   never "broken workflow".
4. All savings are measured in tokens via the target model's tokenizer tier — never
   in characters.

## Modules

- `laconic/tokenizers/` — tiered token counting (`exact` / `api` / `estimate`).
- `laconic/message/` — `Message` model, protected-fields registry, framework parsers
  (OpenAI chat format, LangChain/LangGraph messages) with round-trip guarantee.
- `laconic/compress/` — `Compressor` interface; `Passthrough`, `Extractive`
  (filler pruning + token-checked abbreviations + novelty-based sentence pruning),
  optional `LLMLingua` adapter, and the eval-only `NaiveWholeMessage` baseline.
- `laconic/dedup/` — session content store; context-dedup and store-dedup modes;
  prefix-stable by construction.
- `laconic/adapters/` — per-model profiles: tolerance defaults, pricing (as-of date),
  tokenizer binding.
- `laconic/integrations/` — generic hook + thin LangGraph integration.
- `laconic/profiler/` — handoff records, cost estimates, text + HTML reports, JSONL
  trace reader; standalone CLI.
- `laconic/eval/` — benchmark task model, seeded generator, model clients (mock +
  real), matrix runner {model} × {strategy} × {ratio}, metrics (accuracy, safe
  operating point), plotting. **No fabricated numbers** — the harness produces them.

## Build phases

- **Phase 0** — scaffold: package, pyproject, CI, docs skeleton, license. ✔ importable.
- **Phase 1** — measurement core: tokenizers, Message + parsers, passthrough, profiler,
  toy example runs offline with a real token report.
- **Phase 2** — compression: extractive + optional LLMLingua; round-trip invariant
  tests pass; profiler shows before/after.
- **Phase 3** — dedup: content store, context/store modes, caching interactions documented.
- **Phase 4** — integration: generic hook + LangGraph, end-to-end example.
- **Phase 5** — eval harness: benchmark data, matrix runner, metrics, plots, results
  templates in README (numbers left to a real run).

## What this will not claim

- No latent-space or order-of-magnitude gains: realistic 1.3–3x on chatty payloads,
  less on structure-heavy traffic; stated plainly in README and `docs/limitations.md`.
- No invented benchmark numbers anywhere, including the README results table.
