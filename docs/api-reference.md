# API reference

The stable public surface. Everything documented here has docstrings and type
hints in source; this page is the map.

## Top level (`laconic`)

| symbol | what it is |
|---|---|
| `Session` | the pipeline: parse → dedup → compress → verify → rebuild |
| `ProcessedMessage` | `raw` (dict to send) + `stats` (`CompressionStats`) |
| `CompressionStats` | tokens before/after, ratio, tier, safety, fallback, dedup hits |
| `Message` | structural / payload / metadata decomposition |
| `WorkflowProfile` | accumulated handoff records with aggregates |
| `summary_table(profile)` | terminal report |
| `get_counter(model, allow_api=False)` | resolve a `TokenCounter` |

### `Session`

```python
Session(
    target_model: str,               # whose tokenizer & budget to use
    strategy: str = "conservative",  # off | conservative | telegraphic | balanced | aggressive
    framework: str = "openai-chat",  # or "langchain"
    compressor: Compressor | None = None,   # custom strategy (overrides `strategy`)
    dedup: bool | SessionDedup = True,
    allow_api_counting: bool = False,
    keep_ratio: float | None = None,        # explicit budget override
)

session.process(raw_message: dict, *, source_agent=None, target_agent=None)
    -> ProcessedMessage   # never raises for malformed input
```

## `laconic.tokenizers`

- `TokenCounter` — abstract; `count(text) -> int`, `tier: CountTier`.
- `CountTier` — `EXACT` / `API` / `ESTIMATE`; travels with every stat.
- `HeuristicCounter` — offline estimator (tier `estimate`).
- `get_counter(model, allow_api=False)` — registry resolution.
- Backends: `laconic.tokenizers.openai.TiktokenCounter`,
  `laconic.tokenizers.anthropic.AnthropicApiCounter`.

## `laconic.message`

- `Message`, `MessageMetadata`, `CompressionStats`.
- `get_parser(framework)` / `register_parser(cls)` — parser registry.
- `OpenAIChatParser`, `LangChainParser` — round-trip-guaranteed parsers.
- `ProtectedFieldsPolicy`, `ProtectedFieldsRegistry` — what counts as
  protected, per framework.

## `laconic.compress`

- `Compressor` — subclass to plug in a strategy; receives payload text only.
- `PassthroughCompressor` — measurement baseline.
- `ExtractiveCompressor(aggressive=False, default_budget=0.6)` — fillers,
  token-checked abbreviations, salience-aware sentence pruning under budget.
- `TelegraphicCompressor()` — conservative cleanup plus telegram-style
  function-word dropping (negations and their successors always protected).
- `laconic.compress.llmlingua.LLMLinguaCompressor` — optional adapter
  (`pip install "laconic[llmlingua]"`).
- `NaiveWholeMessageCompressor` — **eval baseline only**; deliberately unsafe.
- `segment_payload(text)` / `join_segments(segments)` — payload protection.

## `laconic.dedup`

- `SessionDedup(store=None, mode="context", min_block_tokens=25)` —
  `.process(payload, recipient=..., counter=...) -> DedupResult`;
  `.rehydrate_tool()` for store mode.
- `ContentStore` — content-addressed block store; subclass to externalize.

## `laconic.adapters`

- `ModelProfile`, `get_profile(model)`, `register_profile(profile)`.
- `keep_ratio_for(model)` — measured safe point or conservative default.
- `estimate_cost(tokens, model, cached=False)`; `PRICING_AS_OF`.

## `laconic.integrations`

- `CompressingHook(session)` — `.process_message(...)`,
  `.process_messages(..., only_new=N)` (prefix-stable), `.profile`.
- `LangGraphCompressor(target_model, strategy, only_new=1)` —
  `.compress_messages(...)`, `.as_node(source, target)`, `.profile`.
- `make_compression_node(...)` — one-call LangGraph node.

## `laconic.profiler`

- `WorkflowProfile` — `.add(stats, ...)`, `.by_edge()`, `.to_jsonl()` /
  `.from_jsonl()`, aggregates.
- `summary_table(profile)`, `to_html(profile, path)`.
- `read_trace(path)`, `profile_trace(entries, model=..., strategy=...)`.

## `laconic.eval`

- `BenchmarkTask`, `Expected`, `load_tasks(path)`, `save_tasks(tasks, path)`.
- `generate_tasks(seed=7, per_family=40)` — deterministic benchmark.
- `MockModelClient`, `OpenAIClient`, `AnthropicClient` (`ModelClient`
  protocol).
- `MatrixSpec`, `run_matrix(tasks, client, spec)`, `write_results(records, dir)`.
- `summarize(records)`, `safe_operating_points(summaries, tolerance=0.02)`,
  `results_markdown(summaries)`.
- `laconic.eval.plots.plot_accuracy_vs_ratio(summaries, out_dir)`.

## CLI

```
laconic profile TRACE.jsonl --model MODEL [--strategy S] [--framework F] [--html OUT]
laconic eval [--tasks DIR] [--out DIR] [--models ...] [--ratios ...]
laconic version
```
