# Handoff auditing and replay

Laconic separates three questions: did a declared quotation survive, does the
downstream task pass, and can restoring source clauses make a failing task pass?
Exact-text audits answer the first; your replay validator answers the others for
the recorded executions.

The library does not infer requirements, choose authoritative sources, detect
all contradictions, or guarantee semantic safety.

## Source contract and trace

`Evidence(id, text, source)` stores an immutable source snapshot, with a `sha256`
property over its exact UTF-8 text. `Requirement(id, evidence_id, quote)` pins a
clause to that source. `HandoffContract` validates IDs, references, nonempty
fields and verbatim quote membership. The complete contract is also hashed.

Choose trusted, current task specifications. Distinct versions need distinct
evidence IDs. A hash establishes identity, not truth, authorization or freshness.

`laconic audit` and `laconic replay` accept this JSON:

```json
{
  "contract": {
    "evidence": [{
      "id": "issue-42-v1",
      "source": "issues/42.md",
      "text": "Keep existing one-argument calls working."
    }],
    "requirements": [{
      "id": "compatibility",
      "evidence_id": "issue-42-v1",
      "quote": "Keep existing one-argument calls working."
    }]
  },
  "handoffs": [
    {"stage": "planner", "text": "Keep existing one-argument calls working."},
    {"stage": "implementer", "text": "Add a required limit parameter."}
  ]
}
```

Audits inspect every stage. Replay uses the first as its original baseline and
the last as its candidate. A missing quote at index zero means it was absent at
the first observed boundary; no earlier stage is blamed. Indices disambiguate
repeated stage names. Later restoration does not erase an earlier omission.

Reports contain source text and full replay inputs. Store them according to your
project's data handling rules. The CLI refuses to overwrite the input trace.

## Connect an executable validator

```bash
laconic replay trace.json --validator my_workflow:validate \
  --max-evaluations 8 --max-added-tokens 200 --out results/replay.json
```

The named Python function accepts a string and returns `ReplayOutcome`, with a
boolean `success`. Optional `metrics` records measured cost, latency, cache usage
or failed tests. Laconic preserves those values without inventing savings.

Your validator owns execution and isolation: restore the same task state before
every call, pin dependencies and model settings, and use a sandbox or test
environment for side effects. The library does not run a shell, execute generated
code, or invoke a model itself. Explicit CLI callback imports run caller-owned
Python code.

For a stochastic model, run repeated trials and a held-out evaluation separately.
One passing call does not establish reliable improvement or a causal explanation.

## Bounded search

1. Validate the contract and verify its clauses occur in the baseline.
2. Run the baseline. Stop if it fails or raises.
3. Run the candidate. Leave it unchanged if it passes, including when a
   paraphrase is semantically adequate but lacks the exact quotation.
4. Append missing source clauses with citations in contract order, respecting
   the optional token-growth cap. Whole clauses that cannot fit are skipped.
5. Test the repair. Failure remains unresolved; alternatives are untested.
6. After a passing repair, try deleting individual additions in one deterministic
   greedy pass. Keep a deletion only when the validator still passes.

Every callback counts against `max_evaluations`, including baseline, candidate,
failed attempts and exceptions. A passing repair remains successful when the
reduction budget ends: `budget_exhausted=True`, `reduction_complete=False`.
The bound limits invocation count, not elapsed time or dollars. Configure timeouts
and spending limits in your validator if it calls external systems.

`reduction_complete` only indicates that one deletion pass finished. With
non-monotonic checks, the result need not even be locally minimal. It never
proves global minimality. Missing undeclared facts, contradictions and unrelated
implementation bugs may remain unresolved.

## Interpret the result

- `candidate_passed`: the unchanged candidate passed the supplied checks.
- `repaired`: the returned source-backed repair passed those checks.
- `baseline_failed` / `baseline_unavailable`: no valid comparison was established.
- `unresolved`: the evaluated proposal failed or no declared clause was missing.
  Untested alternatives may still work.
- `budget_exhausted`: no successful candidate or repair was established before
  exhausting the callback budget.
- `validator_error`: execution raised or returned malformed data. Prior passing
  attempts remain recorded, but diagnosis stopped.
- `invalid_contract` / `invalid_candidate`: invalid input prevented diagnosis.

The first two statuses return replay exit code 0; other replay outcomes return
1. CLI input/import errors return 2. A valid audit returns 0 because a missing
quotation is a diagnostic rather than proof of task failure.

Every attempt includes its input, hash, phase, added clause IDs, tokens, outcome,
metrics and error. Sum costs over **all attempts** when assessing economic value.
Text tokens exclude provider envelopes and other workflow calls; they are not
provider usage or bills.

## Runnable example

```bash
python examples/handoff_repair/run.py
laconic audit results/handoff-demo/trace.json --out results/handoff-demo/audit.json
python -m laconic.cli replay results/handoff-demo/trace.json \
  --validator examples.handoff_repair.validator:validate --out results/handoff-demo/cli-replay.json
```

The example uses a deterministic retry-policy receiver. Read the
[experiment report](experiments.md) before interpreting its results.
