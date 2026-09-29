# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow
[SemVer](https://semver.org/).

## [Unreleased]

### Changed
- Focus the next iteration on source-backed diagnosis and validator-scoped repair.
- Context dedup requires current recipient history via `context_payloads`.
  Previous calls no longer imply context availability. Generic message-list
  processing defaults to the last message.
- `safe` explicitly describes structural preservation only; stats also report
  `structure_preserved` and `protected_spans_preserved`.
- API counters measure before/after only; candidate generation uses local estimates.
- `telegraphic` is now the **default strategy** for `Session`, the LangGraph
  integration, and the toy example (ADR-8). `conservative` remains available
  for zero grammatical alteration.

### Added
- Immutable source contracts, first-loss audits, cited restoration, and bounded
  replay against caller-owned executable validators. Attempts, hashes, token
  provenance, failures, metrics and limits are recorded.
- `audit`, `replay`, and `experiment` CLI commands, an executable example and
  a synthetic behavioral fault-injection experiment with strong baselines.
- Regression tests for replay bounds, citation metadata, protected code,
  dangling references, qualifiers and actual LangGraph message reducers.
- `telegraphic` strategy / `TelegraphicCompressor`: conservative cleanup plus
  telegram-style function-word dropping (negation-protected, capitalization-
  and structure-safe). Included in the eval matrix as `laconic-telegraphic`.
  Offline (seed-7, mock client): information survival 1.000 at whole-message
  token ratio 0.89 vs 0.94 for `conservative`.

### Fixed
- `off` performs pure observation even for repeated content.
- Dedup and custom compressors cannot alter recognized protected payload spans.
- Preserve uncertainty and degree qualifiers in telegraphic compression.
- Honor an explicitly supplied empty content store.

## [0.1.0] — 2026-07-02

Initial release.

### Added
- `Message` decomposition with round-trip-guaranteed parsers (OpenAI chat
  format, LangChain-style dicts) and a configurable protected-fields registry.
- Tiered token counting: exact (tiktoken), API (Anthropic count-tokens,
  opt-in), calibrated offline estimator — provenance labeled on every number.
- Compressors: passthrough, extractive (filler pruning, token-checked
  legend-free abbreviations, novelty-guided sentence pruning under budget),
  optional LLMLingua-2 adapter, and the eval-only naive baseline.
- Payload segmentation protecting code fences, JSON paragraphs, and tables
  inside prose payloads.
- Session dedup: context mode (recipient-aware references) and store mode
  (handles + rehydration tool); prefix-stable with respect to provider
  prompt caching.
- Pipeline with verify-and-fallback: failure mode is always passthrough,
  recorded in stats.
- Profiler: per-edge token/cost accounting, text and self-contained HTML
  reports, standalone JSONL trace analysis.
- Integrations: generic message-list hook and a thin LangGraph shim.
- Eval harness: seeded benchmark generator (three task families, shipped in
  `data/benchmark/`), mock/OpenAI/Anthropic clients, matrix runner, safe
  operating point metrics, plotting.
- CLI: `laconic profile`, `laconic eval`, `laconic version`.
- Documentation: architecture, ADRs, limitations, related work, benchmark,
  API reference; CI on Ubuntu + Windows, Python 3.10–3.13.
