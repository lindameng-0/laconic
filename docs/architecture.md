# Architecture

Laconic has two related paths: a message transformation pipeline and a
source-backed handoff audit/replay library. Structural checks, exact-text
checks, and downstream task checks report different kinds of evidence.

## Message transformation

The parser decomposes a message into deep-copied `structural` fields, a
compressible text `payload`, and routing `metadata`. Tool calls, function
arguments, IDs, unknown fields, and other protected fields stay outside the
compressor API. Key order is retained for rebuilding the message.

```text
count original -> parse -> segment payload
                            |
           protected spans stay unchanged
           free text -> dedup -> compressor
                            |
              rebuild -> verify -> count result
                            |
              return result or original + fallback reason
```

The pipeline applies segmentation around all transformations, including custom
compressors and dedup. Recognized code fences, JSON paragraphs, and Markdown
tables are protected. It verifies their ordered contents again after rebuilding,
along with the structural fingerprint, key order, reparsing, and nonincreasing
token count. Generated dedup references are kept out of prose compression.

`strategy="off"` measures and returns the original message without parsing,
compression, or dedup side effects. Once the initial token count succeeds,
parse, transformation, and verification failures produce passthrough with a
recorded reason. Initial counting failures propagate because no valid baseline
measurement exists.

`CompressionStats.structure_preserved` and `protected_spans_preserved` describe
the returned message. The compatibility field `safe` is a structural status;
none of these flags establishes semantic equivalence or task success.

## Context and token accounting

Context dedup uses the supplied `context_payloads`: the ordered text of messages
actually present in the recipient's current context, excluding the outgoing
message. Use empty strings for non-text messages to retain indices. Recipient
names and previous calls do not establish that content survived truncation,
compaction, or a branch. Without current-context evidence, no context reference
is emitted.

`CompressingHook.process_messages` defaults to `only_new=1`. It preserves the
earlier messages and derives context evidence from the actual output list.
Explicit `only_new=None` transforms the entire list and can change an existing
cache prefix. Store dedup instead requires the caller to expose the rehydration
tool; fetches and their costs are outside the compression measurement.

Counts carry `exact`, `api`, or `estimate` provenance. For an API counter, a
local heuristic searches transformation candidates; provider counting is used
for the original and final message. Serialized-message token differences are
not complete request billing or whole-workflow savings.

## Handoff contracts and bounded replay

`handoff/contracts.py` validates immutable evidence snapshots and explicitly
selected requirement quotes. Quotes must occur verbatim in their cited source.
Audits locate each quote's first observed absence. Repair appends selected
source clauses and citations within an optional token-growth cap. Neither step
infers missing requirements, resolves contradictions, or proves comprehension.

`handoff/replay.py` takes a caller-owned validator that runs a downstream task.
It requires a source-complete original baseline to pass before comparing the
candidate. If the candidate fails, it tries source-backed additions and one
bounded greedy deletion pass. Every validator call, outcome, exception, and
text fingerprint is recorded. The call budget includes baseline and candidate;
the library does not execute generated code or shell commands itself.

A replay success applies to that validator and recorded run. The caller owns
task isolation, reproducibility, and the cost of callback execution. See
[handoff usage](handoffs.md) and [limitations](limitations.md).

## Modules and extension points

- `message/`, `compress/`, `dedup/`, and `pipeline.py`: parse, transform, verify.
- `tokenizers/` and `adapters/`: counting tiers, model profiles, dated prices.
- `integrations/`: generic hooks and a thin LangGraph adapter.
- `profiler/`: per-edge records, JSONL trace analysis, text and HTML reports.
- `handoff/`: source contracts, audits, repair proposals, bounded replay.
- `eval/`: synthetic tasks, model clients, matrix evaluation, and plotting.
- `cli.py`: profile, eval, audit, replay, experiment, and version commands.

Register parsers with `register_parser()`, adjust field policies with
`ProtectedFieldsRegistry`, and implement text transformations by subclassing
`Compressor`. `register_profile()` adds model accounting defaults; a
`ContentStore` subclass can provide external storage. Handoff replay integrates
through a validator returning `ReplayOutcome`, independent of agent framework.
