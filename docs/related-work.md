# Related work

Where Laconic sits, stated plainly. Laconic is a *systems* contribution
(middleware + measurement) built on ideas from several research lines; the
novel part is the structure/payload split applied to inter-agent traffic and
the cross-model tolerance characterization, not any individual compression
technique.

## Token pruning: LLMLingua / LongLLMLingua / LLMLingua-2

The LLMLingua family (Jiang et al., 2023; LLMLingua-2, Pan et al., 2024) uses
a small language model to identify and drop low-information tokens, achieving
strong ratios on prose: documents, RAG context, long instructions.
LongLLMLingua adds question-aware document-level pruning for long contexts.

**Relation.** Laconic is complementary, not competitive: the optional
`LLMLinguaCompressor` adapter runs LLMLingua-2 *inside* Laconic's safety
pipeline, where it only ever sees free-text payload segments. What Laconic
adds is exactly what the LLMLingua family does not target: hard structural
guarantees on agent traffic. Evaluations of prose-tuned compression applied
to agentic/tool-use tasks report sharp task failure beyond modest ratios —
the format the agents depend on gets destroyed even when the semantic content
survives. That observed fragility is the motivating evidence for ADR-2, and
the eval harness's naive baseline reproduces the phenomenon in a controlled
way.

## Abbreviation dictionaries: CompactPrompt and similar

CompactPrompt-style approaches shrink prompts with reusable abbreviation
dictionaries and redundancy removal, without an external compression model.

**Relation.** Laconic's `ExtractiveCompressor` adopts the dictionary idea in
deliberately restricted form: a curated, legend-free list (ADR-4), each
substitution token-checked against the *target model's* tokenizer (ADR-3).
The restriction is the point — an abbreviation the receiver may not
understand, or one that tokenizes longer than its expansion, is a net loss in
this setting.

## Compact serialization: TOON and friends

Token-Oriented Object Notation and similar formats re-serialize structured
data (JSON) into representations that tokenize more cheaply, claiming
30–60% savings on data-heavy payloads.

**Relation.** Orthogonal axis: TOON compresses the *structure itself* by
changing its syntax; Laconic protects structure byte-for-byte and compresses
the *prose around it*. They could compose — a TOON-style serializer could be
registered as a (reversible) transformation for structural fields — but
Laconic v0.1 does not alter structure on principle: the receiving agent, and
any middleware between, must parse whatever format the structure arrives in,
and silent format changes are precisely the class of breakage this project
exists to prevent.

## Latent / implicit inter-agent communication

A growing research line replaces text hops between cooperating agents with
hidden-state exchange or learned latent codes, reporting large efficiency
multiples over natural-language message passing.

**Relation.** Unavailable by construction to Laconic's users: closed-model
APIs expose no hidden states, no KV caches, and no way to inject latent
vectors (ADR-1). Laconic is the pragmatic complement — how far can the
*text* channel be pushed on infrastructure people actually run today. The
honest answer (a small-integer factor, not an order of magnitude) is part of
the project's contribution: it quantifies what staying in the text channel
costs.

## Provider prompt caching

Anthropic and OpenAI both bill cache-hit input tokens at a large discount,
keyed on exact context prefixes.

**Relation.** Not compression, but the single most important deployment
interaction: middleware that rewrites message history destroys prefix-cache
hits and can *raise* net cost. Laconic's dedup and integration hooks are
prefix-stable by design (ADR-6), and `docs/limitations.md` carries the
decision table for when dedup vs. provider caching wins.

## Multi-agent framework telemetry

Frameworks (LangGraph, CrewAI, AutoGen) and observability tools report token
usage per run, and some offer history-trimming or summarization utilities.

**Relation.** Laconic's profiler differs in aiming at the *handoff* level
(per-edge accounting: which agent-to-agent seam burns the tokens) and in
being framework-agnostic (a JSONL trace suffices — no integration required).
Its summarization stance also differs: extractive-only by default, because a
summarizer that paraphrases can silently alter facts in ways that are
invisible until a downstream agent acts on them.

## The gap Laconic fills

To our knowledge no existing open tool combines: (1) hard structural
protection on agent messages, (2) per-model token-measured transformations
with provenance-labeled counting, (3) cache-aware session dedup, and (4) a
reproducible cross-model compression-tolerance benchmark for agent traffic.
That combination — rather than any single technique — is the contribution.
