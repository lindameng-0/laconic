# Laconic build plan

## Current direction

Help developers inspect what happens to declared requirements across agent
handoffs, and test source-backed repairs against their own downstream task.
Message compression remains an optional transformation with explicit structural
and accounting limits.

The immediate audience is a team with a reproducible agent workflow and an
executable outcome check, such as a coding workflow with regression tests.
Success means a useful, reproducible improvement in completed work or cost per
successful task. A compression ratio alone is insufficient evidence.

## Implemented foundations

- Message parsers separate structural fields from prose; the pipeline protects
  recognized payload spans across dedup and custom transformations.
- Token counts carry provenance. API counting checks original/final messages,
  with local heuristic candidate search.
- Measurement mode is inert. Initial count errors propagate; subsequent
  transformation failures return the original message with a recorded reason.
- Context dedup requires the actual current recipient history. Generic and
  LangGraph hooks default to processing only the newest message.
- Source contracts validate exact quote membership and retain source hashes.
  Audits locate observed quote loss; repair appends cited clauses under an
  optional token-growth cap.
- Bounded replay requires a passing source-complete baseline, runs caller-owned
  validators, records every attempt, and performs at most one greedy deletion
  pass. It reports errors and incomplete reduction explicitly.
- CLI audit/replay commands and scripted fault-injection experiments make the
  mechanics reproducible. Existing profiling, compression, and synthetic
  evaluation remain available.

These are implementation milestones. They do not establish semantic
preservation, production savings, improved LLM performance, or unique novelty.
The API and limits are documented in [docs/api-reference.md](docs/api-reference.md)
and [docs/limitations.md](docs/limitations.md).

## Next gate: real workflow evidence

1. Collect representative, permissioned traces from a small number of teams
   running one well-defined workflow. Record the full original task, actual
   recipient context, source versions, and executable success criteria.
2. Reproduce observed failures before attributing them to a handoff. Separate
   missing information from misunderstood information, bad sources, and unrelated
   task failures.
3. Compare held-out tasks against unchanged and concise structured handoffs,
   artifact references, appropriate context modes, and established protected
   compression or summarization. Count source-selection and integration effort.
4. Repeat stochastic runs with controlled task state. Include cached input,
   output, retries, retrieval, validation overhead, latency, and success rate.
   Report uncertainty and failures alongside any benefit.
5. Pre-register an adoption and outcome threshold with participating teams.
   Expand only if the benefit survives those baselines and teams retain the
   integration. If simple existing controls match it, narrow or stop the product
   thesis rather than adding more compression features.

See [docs/experiments.md](docs/experiments.md) for the current experiment scope.
The first credible claim should be about an observed workflow improvement;
broader research or breakthrough claims require stronger evidence.

## Scope boundaries

No automatic source-authority selection, semantic-equivalence guarantee, global
minimal repair, proven causal root cause, universal compression ceiling, or fixed
cache discount is assumed. No real-model result is filled in without a recorded
run. Keep the library small until the real-workflow gate identifies the next
necessary capability.
