# Executable handoff experiment

## Question and method

Can Laconic identify omitted source requirements, restore them without inventing
facts, and validate the resulting downstream behavior within a bounded replay
budget? What happens when no exact quotation is missing, or a paraphrase works?

`laconic.eval.handoff_experiment` implements a transparent retry-policy receiver
and a simulated HTTP service. The receiver interprets explicit clauses into a
policy, then executes nine behavioral probes. Checks cover attempt limits,
retryable status codes, non-idempotent requests, delay limits and request IDs.
There are no network requests, model calls or generated-code execution.

For each seed, three generated specifications vary the attempt and delay limits.
Each specification gets eight controlled faults: a lost attempt limit, lost
negation, lost request identity, two lost requirements, a stale delay limit, an
equivalent paraphrase, an implementation override contradicting the requirement,
and an intact handoff. A three-stage trace records investigator, planner and
implementer boundaries.

These are **24 synthetic fault-injection cases, not 24 independent real tasks**.
Recovery partly depends on the interpreter's explicit latest-clause-wins rule.
The contradiction fixture deliberately overrides that rule and remains broken.

## Reproduce

```bash
pip install -e ".[dev,langgraph]"
laconic experiment --seed 7 --variants 3 --model gpt-4.1 --out results/handoff-seed-7.json
laconic experiment --seed 19 --variants 3 --model gpt-4.1 --out results/handoff-seed-19.json
python examples/handoff_repair/summarize.py results/handoff-seed-7.json results/handoff-seed-19.json
```

These commands select `tiktoken`'s local `o200k_base` encoding. The first use may
download its public vocabulary. Without `--model gpt-4.1`, the experiment uses
the explicitly labeled offline estimator. Neither mode calls an LLM.

## Recorded results

Both seed-7 and seed-19 runs produced the following counts per seed:

| Policy or outcome | Behavioral checks passed | Total handoff text tokens |
|---|---:|---:|
| Full original context | 24 / 24 | 13,296 |
| Concise source contract | 24 / 24 | 1,296 |
| Telegraphic compression | 24 / 24 | 10,920 |
| Budgeted compression, keep ratio 0.45 | 24 / 24 | 5,592 |
| Observed handoff with injected faults | 6 / 24 | 1,164 |

Token counts are exact for the local encoding of these **text strings**, excluding
provider envelopes, output, caches and other calls. The two seeds yield different
specification values but the same aggregate counts. This is not statistical
evidence of generalization.

Replay established successful outcomes in 21 of 24 cases per seed:

- 15 omission/revision cases were repaired: three examples of each of the five
  omission or stale-value patterns.
- 6 already-passing cases were retained without repair, including the three
  paraphrases missing an exact quotation.
- 3 contradictory implementations remained unresolved despite retaining all
  declared quotations. The tool did not call them safe.

Per seed, diagnosis consumed **69 validator calls and 16,350 input text tokens**
across those calls. Those are diagnostic costs, not savings. The count includes
baseline, candidate, repair and attempted reductions. No dollar or latency
improvement has been established.

The complete ignored `results/*.json` files preserve inputs, source fingerprints,
first-loss reports and every replay attempt. A compact checked-in summary is in
`data/experiments/handoff-summary.json` in the repository. It includes SHA-256
hashes of the full JSON reports and can be regenerated with the command above.

## What changed because of the result

Concise source contracts beat the generic compression strategies on input size
while passing the same checks. The useful role for replay is debugging and
recovering an observed failure, particularly when existing workflows already
produce imperfect handoffs. Its extra calls can cost more than sending the
original context. It should not run unconditionally on every handoff.

The contradiction and paraphrase cases demonstrate why an exact-text audit must
remain separate from task validation. A preserved quotation is neither necessary
nor sufficient for a successful downstream task.

The implementation therefore keeps missing-quote diagnostics, source-backed
restoration and validator outcomes distinct. It does not label a passing repair
as a proven root cause or a globally minimal solution.

## Remaining evidence needed

- Real agent workflows and caller-owned acceptance checks, including total
  provider usage, cache effects, generated output, tool calls and retries.
- Strong baselines: concise sender prompts, explicit contracts, native context
  management and compression within the same protected-message wrapper.
- Repeated paired trials, policy selection on calibration traces, and held-out
  tasks. Size the evaluation to the regression margin rather than treating this
  fixture count as sufficient.
- Actual adoption: teams retaining the integration because it finds failures
  they struggle to diagnose today.

No credentials were configured for real-model runs in the implementation
environment. The public validator callback and `laconic replay` command provide
the integration point for that next experiment. No breakthrough or production
effectiveness claim follows from this protocol test.
