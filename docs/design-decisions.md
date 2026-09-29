# Design decisions (ADRs)

These decisions describe the current implementation. Later entries tighten
earlier assumptions; they do not retroactively establish measured model outcomes.

## ADR-1 — Operate at the text boundary

Laconic processes messages available through ordinary application interfaces.
It does not require hidden states, KV-cache access, fine-tuning, or a new learned
communication code. This makes the core framework-independent, while parser and
integration coverage still determine what it supports. Savings and behavior must
be measured on the target task; no universal text-channel ceiling is asserted.

## ADR-2 — Separate structure and verify the whole transformation

Parsers separate protected fields from prose. The pipeline segments prose around
recognized code, JSON, and tables before either dedup or compression, including
custom compressors. It verifies protected spans and rebuilt structural fields
afterward. This supports a structural preservation claim. Prose semantics and
task completion require separate checks. The naive whole-message compressor is
a negative control, not a sufficient competitive baseline.

## ADR-3 — Count tokens and label the counting scope

Transformations are assessed in tokens rather than characters. Abbreviations can
be shorter yet tokenize worse. A local counter guides candidate selection; final
before/after counts use the configured counter. These are serialized-message
measurements. Whole-request billing also depends on framing, output, caching,
tools, retries, and the provider's accounting.

## ADR-4 — Keep default abbreviations readable without a legend

The built-in dictionary uses familiar substitutions and selected filler removal.
No codebook is injected. Negation and qualifiers must not be discarded as mere
verbosity. Familiarity and token savings do not prove equivalent interpretation;
even the conservative strategy needs task validation when meaning is critical.

## ADR-5 — Separate offline search from provider counting

Every count reports `exact`, `api`, or `estimate` provenance. Normal operation
does not require network counting. With an API counter, a local heuristic handles
candidate search and the API counts the original and final message. This avoids
a remote request for each substitution. A failed initial count propagates;
without it a trustworthy fallback measurement cannot be produced. Handoff repair
and replay reject API counters.

## ADR-6 — Require current context for context dedup

Earlier recipient exposure is insufficient after a reset, fork, eviction, or
compaction. Context references therefore require the identical block in supplied
`context_payloads`, representing the actual current recipient history. Missing
context evidence disables these replacements. Store-mode references instead
require an available rehydration tool and include retrieval overhead in any
economic evaluation.

Generic hooks now default to `only_new=1`, matching the LangGraph adapter.
Explicit `only_new=None` can rewrite the whole history and its cache prefix.
Provider cache behavior and pricing must be measured rather than inferred from
recipient names or a fixed discount.

## ADR-7 — Separate integrity, provenance, and task outcomes

The legacy `safe` flag reports structural status; explicit
`structure_preserved` and `protected_spans_preserved` fields clarify its scope.
No compression strategy is advertised as semantically lossless. Unmeasured
profile ratios remain heuristics. `strategy="off"` performs measurement only,
including no dedup side effects, so it provides an interpretable baseline.

## ADR-8 — Retain telegraphic compression as an optional behavior to evaluate

`telegraphic` remains the default for compatibility. It combines conservative
cleanup with guarded function-word deletion. The guards reduce known risks but
do not establish that all grammar changes preserve meaning. Synthetic survival
results and real-model comprehension are reported separately. Use `off` for an
unchanged baseline and validate any transformation against downstream outcomes.

## ADR-9 — Begin handoff diagnosis with explicit source clauses

`HandoffContract` contains immutable evidence snapshots and caller-selected
requirements whose quotes must appear in the cited evidence. Audits record exact
presence at observed boundaries; repairs append whole source clauses and
citations within an optional token cap. This creates inspectable provenance
without claiming automatic importance ranking, semantic entailment, conflict
resolution, or proof that a receiver complied.

## ADR-10 — Bound replay and qualify every conclusion

`diagnose_handoff` invokes a caller-owned downstream validator. A source-complete
original must pass before a candidate failure is investigated. The evaluation
budget covers baseline, candidate, repair, and reduction calls. Exceptions stop
the search and remain visible in the report.

A passing source-backed repair can be reduced by one greedy deletion pass.
`reduction_complete` describes that pass, not global minimality. Results preserve
attempts, source fingerprints, supplied metrics, and reproducibility notes.
Task state isolation belongs to the caller. These checks support a scoped
replay observation; stronger causal or model-performance claims require a
separate controlled experiment.
