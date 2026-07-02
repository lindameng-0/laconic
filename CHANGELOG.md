# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow
[SemVer](https://semver.org/).

## [Unreleased]

### Added
- `telegraphic` strategy / `TelegraphicCompressor`: conservative cleanup plus
  telegram-style function-word dropping (negation-protected, capitalization-
  and structure-safe). Included in the eval matrix as `laconic-telegraphic`.
  Offline (seed-7, mock client): information survival 1.000 at whole-message
  token ratio 0.89 vs 0.94 for `conservative`.

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
