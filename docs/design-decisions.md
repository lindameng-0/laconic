# Design decisions (ADRs)

Each entry records the context, the decision, and its consequences. These are
the reasons the code looks the way it does; change them only with a new ADR.

---

## ADR-1 — Text channel only (the closed-model constraint)

**Context.** The target users run large closed models (Claude, GPT) through
public APIs. Research on efficient inter-agent communication increasingly
explores latent-space channels — exchanging hidden states or learned codes
instead of text — but every such technique requires access to model internals
that API users categorically do not have.

**Decision.** Laconic operates exclusively on the text that agents already
exchange. No new representation, no fine-tuning, no assumptions about model
internals, no provider-specific private features.

**Consequences.** The achievable ceiling is bounded by what natural language
redundancy allows (see ADR-7). In exchange, the tool works today, on every
provider, with zero deployment friction — and results transfer across models,
which is precisely what the research question (cross-model tolerance) needs.

---

## ADR-2 — Structure-preserving compression

**Context.** Agent-to-agent messages are structure plus prose, and the
structure is load-bearing. Published evaluations of general-purpose prompt
compression on agent/tool tasks report abrupt task failure beyond modest
ratios: not graceful degradation, but broken JSON, lost tool names, corrupted
IDs — even when the "meaning" survives. A compressor that treats a message as
prose optimizes exactly the wrong objective.

**Decision.** Every message is decomposed into a protected structural part and
a compressible payload. Compressors receive only the payload string — the API
makes structural corruption unrepresentable rather than merely tested-against.
The payload itself is segmented, and embedded structure (code fences, JSON
paragraphs, tables) is protected as well. Messages that cannot be parsed
confidently pass through untouched.

**Consequences.** Structure-heavy traffic compresses little (that is correct
behavior, stated in the docs, not hidden). The invariant test suite pins the
guarantee. The eval harness includes a deliberately naive whole-message
compressor as a baseline to quantify what this decision buys.

---

## ADR-3 — Tokens, never characters

**Context.** All costs and context limits are denominated in model-specific
tokens. Character counts are actively misleading: an "abbreviation" that is
shorter in characters can tokenize to *more* tokens (`w.r.t.` is a classic
offender), and different models tokenize the same string differently.

**Decision.** Every saving anywhere in Laconic is computed by a
`TokenCounter` bound to the target model. Individual transformations
(each abbreviation substitution, each dedup replacement) are accepted only if
they reduce the token count under that counter. A test pins the adversarial
case where character count and token count disagree in direction.

**Consequences.** Compression is slightly slower (counting is in the inner
loop) and results are per-model, which is more honest and more useful. The
tiered-counter design (ADR-5) keeps the inner loop offline-fast.

---

## ADR-4 — Legend-free abbreviations only

**Context.** Abbreviation dictionaries (in the spirit of CompactPrompt) can
save real tokens, but they have a receiver-side contract: if the receiving
model doesn't understand an abbreviation, comprehension degrades silently;
if a legend is injected to explain it, the legend costs tokens that must be
amortized.

**Decision.** The default dictionary contains only substitutions any modern
instruction-tuned model reads natively ("for example" → "e.g.",
"in order to" → "to", filler removal). No legend is ever injected by default.
Each substitution is token-checked per ADR-3.

**Consequences.** The abbreviation stage saves modestly but never risks
comprehension. Users with high-volume domain-specific traffic can supply
custom dictionaries and legends — the amortization math is theirs to justify,
and the docs say so.

---

## ADR-5 — Tiered token counting (`exact` / `api` / `estimate`)

**Context.** OpenAI models have an exact local tokenizer (tiktoken). Current
Anthropic models do not: the exact count comes from a network endpoint that
requires a key. A profiler that silently mixes exact and approximate numbers
— or that cannot run offline — is broken for its main use case.

**Decision.** Three explicitly labeled tiers: `exact` (local tokenizer),
`api` (provider endpoint, opt-in via `allow_api_counting`), `estimate`
(calibrated offline heuristic). The tier travels with every stat, reports
disclose when estimates are present, and nothing in the default path performs
network I/O.

**Consequences.** A bare `pip install laconic` works with zero optional
dependencies, honestly labeled. Claude-side numbers are estimates unless the
user opts into API counting; the estimator's error is itself measured against
exact counters in the test suite (±25% bound on realistic prose).

---

## ADR-6 — Dedup must respect the model boundary and the provider cache

**Context.** Two subtleties naive dedup designs miss. First, tokens are only
spent when text enters a model's context — a handle that is rehydrated before
the receiving model reads it saves nothing. Second, providers price cached
input tokens at ~10× less than fresh ones, and caching is prefix-based;
rewriting message history invalidates the cache and can make "compression" a
net cost increase.

**Decision.** Dedup has two modes with explicit payoff conditions:
*context mode* (default) replaces a block only when the same recipient already
received it earlier in the session — the model resolves the reference by
looking back at its own context; *store mode* (opt-in) replaces blocks with
handles and equips the receiving agent with a rehydration tool. In both
modes, only the newly outgoing message is transformed — history is never
rewritten (prefix stability).

**Consequences.** Context-mode savings appear from the second repetition
onward, which is exactly where agent loops waste the most. The
dedup-vs-provider-caching decision table lives in `limitations.md`. The
`only_new` parameter on integration hooks exists to enforce prefix stability
at the seam.

---

## ADR-7 — Reliability over aggression, honesty over headline numbers

**Context.** A strategy that saves 30% and never breaks a workflow is worth
more than one that saves 70% and corrupts a tool call once a week — the cost
of one silent corruption in production dwarfs the token savings. Meanwhile,
the compression literature is full of order-of-magnitude claims that don't
survive contact with structured traffic.

**Decision.** The default strategy never drops content: sentence-level
pruning is opt-in, bounded by per-model safe operating points that ship
*unmeasured* (`None`) until the eval harness actually measures them — the
default budget in the absence of measurement is a gentle 0.75. The docs state
the realistic ceiling (1.3–3× on chatty payloads) up front. No benchmark
number in this repository is invented; the README results table stays a
template until a real run fills it. (The default was originally
`conservative`; ADR-8 records the move to `telegraphic`, which preserves the
never-drop-content property.)

**Consequences.** First-run savings look modest compared to marketing-driven
tools. Every number a user sees is one they can reproduce. The eval exists to
*extend* the safe frontier with evidence, not vibes.

---

## ADR-8 — Telegraphic is the default strategy

**Context.** Users consistently ask for "a more efficient language for the
AIs to talk in." Genuinely new codes are unavailable (ADR-1) and usually
tokenize worse (ADR-3); the practical headroom inside the training
distribution is telegram-style English — dropping articles, intensifiers,
politeness, and meaning-safe copulas while keeping every content word,
number, name, and negation. On the seed-7 offline benchmark this scores
information survival 1.000 at whole-message token ratio 0.886, versus 0.936
for `conservative`, and it composes with (rather than replaces) the
conservative cleanup.

**Decision.** `telegraphic` is the default `Session` strategy. Its guards are
part of the contract: negations and their following word are never dropped,
capitalized words and numbers survive, protected segments are untouched, and
the pipeline's verify-and-fallback still applies. `conservative` remains
available for users who want zero grammatical alteration.

**Consequences.** Default handoffs read like terse notes rather than full
prose. Information survival is measured at 1.000 offline, but *model
comprehension* of telegraphic prose is per-model and unverified until a real
eval run — the harness includes `laconic-telegraphic` as a matrix cell
precisely so that number can be produced. Users whose downstream agents do
grammatical inference on handoffs (rare, but possible) should switch back to
`conservative`.
