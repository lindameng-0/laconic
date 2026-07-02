# Limitations

This page is deliberately blunt. If a limitation here surprises a user in
production, the docs have failed.

## The realistic ceiling

Laconic is a text-channel tool (ADR-1). On the traffic it targets:

- **Chatty natural-language payloads**: typically a 1.3–3× token reduction
  depending on strategy and how verbose the source agent is. The conservative
  default sits at the low end of that range.
- **Structure-heavy traffic** (tool calls, tool results, JSON payloads):
  little to nothing — by design. The structural part is never compressed, and
  payloads that *are* structured data pass through whole. If your workflow is
  90% tool traffic, Laconic's compressor will not help much; the profiler and
  dedup still might.
- **Dedup**: often the largest real-world saving, but only when content
  actually repeats across handoffs to the same recipient (context mode) or
  when receivers can tolerate a fetch round-trip (store mode).

There is no order-of-magnitude gain available in the text channel. Anyone
promising one is compressing something load-bearing.

## Compression is lossy where it is allowed to be

`balanced` and `aggressive` strategies drop sentences. The pruner is
redundancy-guided and never drops paragraph leads or protected segments, but
"low lexical novelty" is not "unimportant": a constraint buried in filler can
be pruned at aggressive budgets. That is exactly the failure mode the
benchmark's `constraint_tracking` family measures. If your handoffs carry
hard requirements, either keep the conservative default or run the eval
against your own traffic before turning budgets down.

## Token-count tiers

Anthropic models have no local tokenizer; without `allow_api_counting=True`
their numbers are **estimates** (tier `estimate`, roughly ±15–20% on prose).
Reports disclose this whenever estimated numbers appear. Never compare an
`estimate`-tier number against an `exact`-tier number and call the difference
a result.

## Dedup vs. provider prompt caching

Cached input tokens cost ~10× less than fresh ones on both major providers.
Interactions to be aware of:

| situation | what wins |
|---|---|
| Same growing conversation resent every turn to the same model | **Provider caching** — Laconic dedup deliberately leaves history untouched (prefix-stable) so the cache keeps hitting |
| The same large block sent to *different* agents / different contexts | **Store-mode dedup** — provider caches don't span contexts; a handle + fetch does |
| The same block repeated inside *new* messages to the same recipient | **Context-mode dedup** — the repetition is new tokens each time; a reference is cheaper |
| Short messages, nothing repeats | Neither — Laconic's dedup stays out of the way (blocks under ~25 tokens are never touched) |

What Laconic will *not* do: rewrite already-sent messages to squeeze them.
That breaks prefix caching and can increase net cost.

## Failure modes and the fallback

Any of the following causes the pipeline to deliver the **original message
unchanged** and record `fell_back=True` with a reason:

- the message doesn't parse (unknown shape, non-string role, exotic content),
- a compressor raises or returns a more expensive text,
- the rebuilt message's structural fingerprint differs in any byte,
- key order changed, or the rebuilt message fails to re-parse,
- processing increased the token count.

The cost of a fallback is zero savings on that handoff — never a corrupted
workflow. Watch the fallback rate in the profiler: a high rate usually means
the traffic is structure-heavy or a custom parser is needed.

## What the offline eval does and does not measure

The mock client is a deterministic perfect extractor: it answers from what is
literally present in the handoff it received. Offline results therefore
measure **information survival** — whether load-bearing content physically
reached the receiver — not model comprehension. A model may recover from
compression artifacts the mock cannot, and vice versa. Real-model numbers
require API keys and are produced by the same harness (`docs/benchmark.md`);
the repository ships no claimed real-model numbers.

## Other known limits

- Parsers cover the OpenAI chat format and LangChain-style dicts. Other
  formats pass through untouched until a parser is registered.
- Multimodal content (content-parts lists) is treated as fully structural —
  text parts inside it are not compressed in v0.1.
- The `contains_all` scorer uses substring matching; rare false positives are
  possible (e.g. "42" matching inside "142"). Task generation keeps values
  distinctive to make this negligible, but it is not zero.
- `SessionDedup` state is in-memory and per-process.
- Pricing tables are dated (`PRICING_AS_OF`) and will drift.
