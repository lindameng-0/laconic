# Laconic

Token-efficient middleware for multi-agent LLM workflows: structure-preserving
compression, session dedup, and profiling for inter-agent messages — for users
of closed models, entirely in the text channel.

Start with the [README](https://github.com/USER/laconic#readme) for the
quickstart, then:

- [Architecture](architecture.md) — the `Message` abstraction, modules, and
  the data contracts that make structural corruption impossible.
- [Design decisions](design-decisions.md) — ADRs recording *why* the design is
  what it is.
- [Limitations](limitations.md) — the honest ceiling, failure modes, and how
  the fallback handles them.
- [Related work](related-work.md) — where Laconic sits relative to LLMLingua,
  CompactPrompt, TOON, and latent-communication research.
- [Benchmark](benchmark.md) — the eval harness and the cross-model
  compression-tolerance study.
- [API reference](api-reference.md)
- [Contributing](contributing.md)

## The one-paragraph pitch

Agent-to-agent messages are structure plus prose. The structure (tool calls,
arguments, IDs, schemas) is load-bearing: compress it and the workflow breaks
outright, which is exactly what published evaluations observe when
general-purpose prompt compression is pointed at agent traffic. Laconic
splits every handoff into a protected structural part and a compressible
payload, compresses only the payload — measured in the target model's tokens,
never characters — verifies the rebuild, and passes messages through
untouched whenever anything is uncertain. On top of that sits a profiler
(find where tokens burn before changing anything) and a research harness that
measures, per model, how much compression agent traffic tolerates before task
accuracy degrades.
