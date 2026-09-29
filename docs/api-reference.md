# API reference

This page maps the public API. Handoff diagnosis is experimental: its guarantees
are deliberately narrower than semantic preservation or workflow correctness.

## `laconic`

- `Session`: message transformation and measurement.
- `ProcessedMessage`: `raw` dictionary to send and `CompressionStats`.
- `CompressionStats`: before/after tokens, ratio, count tier, dedup hits,
  fallback reason, `structure_preserved`, and `protected_spans_preserved`.
  The compatibility field `safe` describes structural status only.
- `Message`: structural fields, payload, and routing metadata.
- `WorkflowProfile`, `summary_table(profile)`, and `get_counter(model, allow_api=False)`.

### `Session`

```python
Session(
    target_model,
    strategy="telegraphic",  # off, conservative, telegraphic, balanced, aggressive
    framework="openai-chat",  # or "langchain"
    compressor=None,  # custom Compressor overrides strategy
    dedup=True,  # bool or SessionDedup
    allow_api_counting=False,
    keep_ratio=None,
    registry=None,
    counter=None,
)

session.process(
    raw_message,
    source_agent=None,
    target_agent=None,
    context_payloads=None,
)  # -> ProcessedMessage
```

`context_payloads` must be the ordered content of the recipient's actual current
history, excluding this outgoing message. Represent non-text messages with empty
strings to preserve indices. Without it, context dedup performs no replacements.

`off` is inert measurement. Parse and transformation failures fall back once
the original count is available. Initial counting failures propagate. API
counters count original/final messages; a local heuristic guides candidate search.

## `laconic.handoff`

### Source contracts and audits

- `Evidence(id, text, source)`: immutable source snapshot; `.sha256` hashes its text.
- `Requirement(id, evidence_id, quote)`: explicitly selected exact source clause.
- `HandoffContract(evidence, requirements)`: validates unique IDs, references, and
  verbatim source membership; lists become immutable tuples. `.sha256` fingerprints
  the full contract.
- `Handoff(stage, text)`: text at one observed boundary.
- `audit_handoff(contract, text, stage="handoff") -> HandoffAudit`: present and
  missing requirement IDs.
- `audit_handoffs(contract, handoffs) -> HandoffAuditReport`: per-boundary audits,
  `requirement_losses` with first missing stage/index, final missing IDs, and provenance.

Construct models with keyword arguments. Exact quote presence does not establish
that a receiver interpreted or followed the requirement.

```python
repair_handoff(
    contract,
    text,
    requirement_ids=None,
    max_added_tokens=None,
    counter=None,
)  # -> RepairResult
```

Repair appends selected missing clauses and citations in contract order.
`RepairResult` contains `text`, `restored_requirement_ids`,
`unresolved_requirement_ids`, `added_tokens`, `count_tier`, `count_model`,
`contract_sha256`, and source provenance. Unresolved IDs cover the whole
contract, including unselected requirements. The cap includes citation overhead.
Only offline counters are accepted; the default is a labeled heuristic.

### Bounded downstream replay

```python
diagnose_handoff(
    contract,
    original_text,
    candidate_text,
    validator,  # (text: str) -> ReplayOutcome
    max_evaluations=8,
    max_added_tokens=None,
    counter=None,
    validator_name=None,
    reproducibility_note=None,
)  # -> ReplayResult
```

`ReplayOutcome(success=..., metrics={}, details="")` records the caller's actual
task check. Optional metrics such as `cost_usd` and `latency_ms` are recorded
as supplied. Laconic does not infer whole-workflow savings from them.

The original must contain every declared source clause and pass the validator.
The candidate is checked next. A failed candidate can be repaired with source
quotes, followed by one greedy deletion pass. All calls count toward the budget;
exceptions or malformed outcomes stop the search. The caller owns reproducible,
isolated execution and any API cost or side effects.

`ReplayResult` includes:

- `status`: `invalid_contract`, `invalid_candidate`, `baseline_unavailable`,
  `baseline_failed`, `candidate_passed`, `repaired`, `unresolved`,
  `budget_exhausted`, or `validator_error`.
- `text`, nullable `success`, selected `added_requirement_ids`, and the reason.
- `attempts`: each `ReplayAttempt` records phase, text, text hash, token count,
  selected IDs, returned outcome, or error.
- `evaluations_used`, `max_evaluations`, `budget_exhausted`, and
  `reduction_complete`. A passing repair retains `status="repaired"` if the
  budget prevents further reduction.
- Original, candidate, and result token counts, added tokens, tier, model,
  contract/source fingerprints, validator name, guarantee, and reproducibility note.

A completed deletion pass does not prove global minimality or causal attribution.
See [handoff usage](handoffs.md) for an executable example.

## Supporting modules

- `laconic.tokenizers`: `TokenCounter.count(text)`, `CountTier`,
  `HeuristicCounter`, `get_counter`; optional `TiktokenCounter` and
  `AnthropicApiCounter` backends.
- `laconic.message`: `Message`, `MessageMetadata`, `CompressionStats`,
  `get_parser`, `register_parser`, `OpenAIChatParser`, `LangChainParser`,
  `ProtectedFieldsPolicy`, and `ProtectedFieldsRegistry`.
- `laconic.compress`: `Compressor`, `PassthroughCompressor`,
  `ExtractiveCompressor`, `TelegraphicCompressor`, `segment_payload`, and
  `join_segments`. The optional LLMLingua adapter is in
  `laconic.compress.llmlingua`; `NaiveWholeMessageCompressor` is an eval-only negative control.
- `laconic.dedup`: `ContentStore` and
  `SessionDedup(store=None, mode="context", min_block_tokens=25)`.
  Its `.process(payload, recipient=..., counter=..., context_payloads=None)`
  returns `DedupResult`; `.rehydrate_tool()` provides the store-mode callable.
- `laconic.adapters`: `ModelProfile`, `get_profile`, `register_profile`,
  `keep_ratio_for`, `estimate_cost(tokens, model, cached=False)`, and
  `PRICING_AS_OF`. Default ratios are heuristics unless measured.
- `laconic.integrations`: `CompressingHook(session)` exposes
  `.process_message(..., context_payloads=None)`, `.process_messages(..., only_new=1)`,
  and `.profile`. `LangGraphCompressor(..., only_new=1)` exposes
  `.compress_messages`, `.as_node`, and `.profile`;
  `make_compression_node` is the convenience factory.
- `laconic.profiler`: `WorkflowProfile.add`, `.by_edge`, `.to_jsonl`,
  `.from_jsonl`, `summary_table`, `to_html`, `read_trace`, and `profile_trace`.
- `laconic.eval`: `BenchmarkTask`, `Expected`, `load_tasks`, `save_tasks`,
  `generate_tasks`, `MockModelClient`, `OpenAIClient`, `AnthropicClient`,
  `MatrixSpec`, `run_matrix`, and `write_results`. Metrics include `summarize`,
  `safe_operating_points`, and `results_markdown`; plotting lives in `eval.plots`.
  Measured operating points apply only to the evaluated distribution and criterion.

## CLI

```text
laconic profile TRACE.jsonl --model MODEL [--strategy S] [--framework F] [--html OUT]
laconic eval [--tasks DIR] [--out DIR] [--models ...] [--ratios ...]
laconic audit TRACE.json [--out OUT]
laconic replay TRACE.json --validator MODULE:FUNCTION [--max-evaluations N]
               [--max-added-tokens N] [--model MODEL] [--out OUT]
laconic experiment [--seed N] [--variants N] [--max-evaluations N]
                   [--model MODEL] [--out OUT]
laconic version
```

Audit/replay traces contain a contract and observed handoffs. Replay uses the first
handoff as baseline and the last as candidate. `--validator` explicitly imports
and runs caller-owned Python code. The experiment command runs scripted
fault-injection fixtures; it does not call an LLM.
