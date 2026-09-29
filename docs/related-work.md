# Related work

Laconic combines established techniques into a small measurement, transformation,
and handoff-diagnosis library. Protected text, artifact references, and context
isolation are not new inventions here. The useful question is whether this
implementation improves a real workflow over strong existing baselines.

## Prompt compression and protected sections

The LLMLingua family provides learned token pruning, including question-aware
compression and LLMLingua-2. Its structured-prompt interface already lets callers
segment text and mark sections with `compress=False`. Laconic must not claim
that protected spans are absent from existing prompt compressors.
[Microsoft LLMLingua documentation](https://github.com/microsoft/LLMLingua#3-advanced-usage---structured-prompt-compression).

Laconic supplies message parsers, protected-span enforcement across its
transformation pipeline, accounting provenance, and an optional LLMLingua
adapter. Its structure-blind naive baseline demonstrates one failure mode; it
does not establish superiority to properly configured structured compression.

Readable abbreviation dictionaries and compact serialization address other
sources of token overhead. Laconic uses a small legend-free dictionary and
keeps structural field values unchanged. It does not establish the best
representation or compression method for a particular workload.

## Artifact references and long-running agents

Anthropic's June 13, 2025 account of its multi-agent research system describes
subagents storing outputs externally and returning lightweight references. Its
appendix also discusses summaries, external memory, and handoffs into fresh
contexts. These are precedents for avoiding repeated transmission of complete
artifacts.
[Anthropic engineering article](https://www.anthropic.com/engineering/multi-agent-research-system).

Laconic's content store and references use this general pattern. Its source
contracts add explicit caller-selected quotes, hashes, and presence tracking.
An exact-quote audit provides evidence about text survival; it cannot establish
that the evidence is sufficient, current, or understood.

## Context isolation and inherited context

LangChain's September 8, 2026 description of Deep Agents supports both
`isolated` and `fork` subagent modes. Isolated agents receive a task in a fresh
context; forked agents inherit the supervisor's conversation. The article
discusses task-dependent choices and the interaction with prompt caching.
[LangChain context-mode design](https://www.langchain.com/blog/organizing-context-in-a-multi-agent-harness).

A concise handoff with retrieval must therefore be compared with inherited
context and existing framework controls. More context can avoid repeated work;
fewer transmitted tokens do not by themselves imply lower whole-run cost.
Laconic's hooks leave earlier messages unchanged by default, but do not select
context modes or measure provider cache eligibility.

## What remains to be demonstrated

Bounded replay compares a baseline, a candidate, and source-backed additions
using a caller's downstream task validator. It makes each attempt inspectable
and limits callback count. This is an engineering mechanism, not evidence of a
new research result or a complete semantic diagnosis.

A credible comparison should include unchanged handoffs, concise structured
handoffs, artifact references, appropriate framework context modes, protected
prompt compression, and simple summarization where applicable. Evaluate
completed tasks, actual total cost, latency, retries, and missing or
misinterpreted requirements. The synthetic tests and scripted experiment
fixtures do not substitute for that study. See [experiments](experiments.md).
