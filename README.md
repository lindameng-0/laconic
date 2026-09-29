# Laconic

**Find where agent handoffs lose requirements. Test source-backed repairs against your actual task.**

Laconic is a Python toolkit for investigating agent communication failures and
reducing unnecessary context. It combines handoff audits, bounded replay, and
structure-preserving compression. It works in the text channel without access
to model internals.

[![CI](https://github.com/lindameng-0/laconic/actions/workflows/ci.yml/badge.svg)](https://github.com/lindameng-0/laconic/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)

## Start with a failing handoff

A planner knows that a retry client must **never retry a non-idempotent request**.
The clause disappears before implementation. The resulting client retries a
request that should have stopped.

Laconic locates the first observed omission, restores the original clause with
its source, and calls your downstream checks to test the repair. Every attempt
is recorded, including failures, exceptions, token counts, and caller-supplied
cost or latency metrics.

Finding a quotation proves only that the quotation is present. Passing your
executable checks is a separate result. Neither proves universal semantic safety.

## Try it offline

Install from this checkout:

```bash
pip install -e ".[dev,langgraph]"
python examples/handoff_repair/run.py
```

The example deliberately drops an attempt limit and an idempotency requirement.
It runs a scripted retry client against behavioral checks, then shows:

```text
r0 first missing at planner
r2 first missing at planner
baseline: PASS
candidate: FAIL
repair: PASS
reduction: FAIL
reduction: FAIL
Result: repaired; restored r0, r2
Validation calls: 5/8
```

This is a deterministic protocol demonstration, **not a language-model benchmark**.
Evidence and a complete replay record are written to `results/handoff-demo/`.

```bash
laconic audit results/handoff-demo/trace.json --out results/audit.json
laconic experiment --seed 7 --variants 3 --out results/experiment.json
# Exact local token counts; may download tiktoken's vocabulary on first use:
laconic experiment --model gpt-4.1 --out results/experiment-exact.json
```

`--model` selects a tokenizer for these commands. It does not invoke a model.

## Connect your workflow

Declare important quotations from immutable source snapshots, then pass a
validator that runs your isolated downstream task and its acceptance tests:

```python
from laconic.handoff import Evidence, HandoffContract, Requirement, diagnose_handoff

clause = "Keep existing one-argument calls working."
contract = HandoffContract(
    evidence=[Evidence(id="issue-42", text=clause, source="issues/42.md")],
    requirements=[Requirement(id="compatibility", evidence_id="issue-42", quote=clause)],
)

# validate_handoff is your callback, returning ReplayOutcome(success=..., metrics=...).
result = diagnose_handoff(
    contract,
    original_text=f"Add an optional limit parameter. {clause}",
    candidate_text="Add a required limit parameter.",
    validator=validate_handoff,
    max_evaluations=8,  # includes baseline, candidate, repair and reduction
    max_added_tokens=200,  # includes source citation overhead
)
```

You choose authoritative sources and important requirements. The callback owns
external execution and must reset task state before each call. Laconic itself
does not execute generated code or send model requests. For a runnable callback,
see [the example](examples/handoff_repair/run.py).

An importable callback can also be used from the CLI:

```bash
laconic replay trace.json --validator my_workflow:validate_handoff \
  --max-evaluations 8 --max-added-tokens 200 --out results/replay.json
```

Only `candidate_passed` and `repaired` yield a successful replay exit code.
Failed baselines, invalid evidence, exceptions, exhausted budgets and unresolved
failures remain explicit. `reduction_complete` means one greedy deletion pass
finished; it does not establish global or local minimality.

See [the handoff guide](docs/handoffs.md) for the trace schema and integration contract.

## What the experiments establish

The recorded seed-7 run contains 24 synthetic fault-injection cases:

- 15 omission/revision cases were repaired and passed the scripted checks.
- 6 already-working cases, including paraphrases, were left unchanged.
- 3 contradictory implementations remained unresolved.
- Concise contracts passed every case and used fewer tokens than the compression
  strategies. Recovery does not beat preventing the omission in the first place.
- Diagnosis used 69 validator calls. That additional work is reported separately.

Cases share generated specifications and are **not independent production tasks**.
The receiver is a documented clause interpreter. No real-model reliability,
production savings, or algorithmic novelty is claimed. Read the
[experiment report](docs/experiments.md), including reproduction commands and limits.

## Compression and profiling

```python
from laconic import Session, summary_table
from laconic.integrations import CompressingHook

hook = CompressingHook(Session(target_model="gpt-4.1", strategy="conservative"))
outgoing = hook.process_messages(messages, only_new=1)
print(summary_table(hook.profile))
```

- `off` measures without changing content or dedup state.
- Structural fields and recognized protected payload spans are preserved.
  Custom compressors only receive the remaining prose.
- Context dedup requires the identical block in the recipient's **actual current
  history**. Message-list hooks derive this from the supplied history. Direct
  `Session.process` callers supply `context_payloads`; previous calls alone do
  not prove that content remains available.
- Only new messages are processed by default, preserving the supplied prefix.
- `safe`, `structure_preserved`, and `protected_spans_preserved` describe returned
  structure, not comprehension, truth, or task success.
- Counts carry `exact`, `api`, or `estimate` provenance. Serialized-message
  counts are not complete provider request usage or billing.

```bash
laconic profile trace.jsonl --model gpt-4.1
laconic profile trace.jsonl --model gpt-4.1 --strategy balanced
laconic eval --tasks data/benchmark --out results/legacy-eval
```

The original 120-task benchmark remains an information-survival regression suite.

## Development

```bash
pytest
ruff check .
ruff format --check .
```

Tests include source validation, replay budgets, exceptions, non-monotonic
validators, protected spans, current-context dedup and actual LangGraph reducers.
Exact tokenizer tests need a cached vocabulary or network access on first use.

## Documentation

- [Handoff auditing and replay](docs/handoffs.md)
- [Executable experiment and results](docs/experiments.md)
- [Architecture](docs/architecture.md)
- [Design decisions](docs/design-decisions.md)
- [Limitations](docs/limitations.md)
- [Related work](docs/related-work.md)
- [Original compression benchmark](docs/benchmark.md)
- [API reference](docs/api-reference.md)
- [Contributing](docs/contributing.md)

MIT licensed. Citation metadata: [CITATION.cff](CITATION.cff).
