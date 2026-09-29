# Limitations

## What has been established

The tests exercise message structure, recognized protected spans, accounting,
source-quote tracking, and bounded replay behavior. The synthetic compression
benchmark measures information survival with a deterministic receiver.
Scripted handoff experiments exercise known failure fixtures. These do not
establish improved LLM task success, production savings, or product demand.

There is no universal compression ratio or text-channel savings ceiling here.
Results depend on repetition, task, tokenizer, model, context, and accounting
scope. Structure-heavy messages may leave little eligible prose to compress.
Measure complete runs before drawing an economic conclusion.

## Structural preservation is a limited guarantee

`safe`, `structure_preserved`, and `protected_spans_preserved` describe the
returned message's structural preservation. A semantically harmful prose edit
can still pass every structural check. Segment protection covers recognized
formats; it does not discover every domain-specific invariant inside text.

`conservative` rewrites selected filler and abbreviations. `telegraphic` also
changes grammar. Guards preserve negation and selected qualifiers, but grammatical
changes can still change interpretation. `balanced` and `aggressive` can remove
sentences carrying important constraints. None is semantically lossless.
An unmeasured model profile or a default keep ratio is not a validated operating
point for your workflow.

## Exact quotes and replay checks

Handoff requirements are selected by the caller. Exact matching misses
paraphrases and does not detect contradictions, stale requirements, irrelevant
quotation, or a receiver ignoring present text. Source membership and hashes
establish provenance, not that a source is truthful or authoritative.

Repair appends cited clauses; it does not reconcile contradictory instructions.
An added-token cap includes citation overhead. A skipped clause might fit under
a different selection, so a failed proposal does not establish that no repair
exists.

Replay requires all declared source clauses in the original baseline and a
passing baseline validator. Unavailable or failing baselines support no handoff
attribution. Exceptions and invalid callback results stop the search explicitly.
A passing repaired candidate means only that the supplied validator passed on
that run. An incomplete validator can miss errors.

The caller must hold task inputs and dependencies fixed and reset mutable state
for every invocation. Determinism is required for interpreting these paired
checks; stochastic model runs need repeated evaluation outside the single-run
diagnosis. The bounded greedy deletion pass establishes neither a globally
minimal repair nor a causal root cause. `reduction_complete` reports completion
of that pass only. A passing repair can be returned with
`budget_exhausted=True` and `reduction_complete=False`.

Callbacks may make paid API calls or mutate an environment. The evaluation
budget limits callback invocations, including failed calls; it is not a money,
wall-clock, or side-effect limit. The replay library does not sandbox callbacks.

## Dedup, history, and caches

Context dedup requires `context_payloads` from the recipient's actual current
input. Reusing stale history after compaction, reset, or branching can produce
unresolvable references. The library cannot independently inspect the context
sent to a provider. Recipient identity alone is never evidence of availability.

Store dedup requires an accessible backing store and a registered rehydration
tool. Fetches add tokens and latency. Rehydrating everything before the model
reads it removes the intended input saving. The default store is in-memory and
per-process.

Generic and LangGraph hooks default to changing only the newest message.
Explicitly transforming old history can invalidate cache prefixes. Provider
cache eligibility, scope, prices, and actual hits depend on the provider and
request; no fixed cache discount or cross-agent cache boundary is assumed.
Compare actual billed usage with retrieval, output, retries, and failures included.

## Counting and fallback

Counts disclose `exact`, `api`, or `estimate` provenance. An exact tokenizer
count of serialized message text is still not a full provider request charge.
Offline estimates vary with language and content. API counters validate the
before/after message; local heuristic counts guide the inner candidate search.
Handoff repair and replay accept only offline counters.

Once the initial count succeeds, parsing errors, transformation exceptions,
protected-span or structural mismatches, failed reparsing, and token growth
return the original message with `fell_back=True`. Initial count failures
propagate rather than inventing measurements. Passthrough preserves the input;
it cannot correct an already wrong input or guarantee downstream success.

Parsers currently cover OpenAI-chat and LangChain-style dictionaries.
Multimodal content lists remain structural. Benchmark substring scoring has
known ambiguities, and the dated pricing table can drift. See the
[benchmark guide](benchmark.md) for the synthetic dataset's scope.
