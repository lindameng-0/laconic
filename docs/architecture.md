# Architecture

## The `Message` abstraction

Every inter-agent message is decomposed at parse time:

| part | contents | who may touch it |
|---|---|---|
| `structural` | tool calls, function arguments, IDs, schema keys, control fields, unknown keys — deep-copied | nobody |
| `payload` | the natural-language content of the payload field | compressors, dedup |
| `metadata` | source/target agent, target model, framework tag | routing only, never sent to a model |

The parser also records the original key order, so a rebuild with an unchanged
payload reproduces the raw message **exactly** (`tests/test_roundtrip.py`
enforces this over a synthetic corpus of tool calls, tool results, unicode,
code fences, multimodal parts, and unknown keys).

## Data contracts

1. **A compressor only ever sees `payload`.** It is structurally impossible
   for a compressor to corrupt `structural` — the compressor API receives a
   string and returns a string; the structural dict never passes through it.
2. **Payload-embedded structure is protected too.** Before compression the
   payload is segmented (`laconic/compress/segments.py`); code fences, JSON
   paragraphs, and markdown tables are never transformed.
3. **Every `CompressionStats` carries provenance**: original tokens,
   compressed tokens, ratio, model, strategy, the token-count tier
   (`exact`/`api`/`estimate`), a safety flag, and whether fallback occurred.
4. **Verification before delivery.** After rebuild, the pipeline compares a
   canonical serialization of everything except the payload field, checks key
   order, re-parses the rebuilt message, and confirms the token count did not
   grow. Any mismatch → the original message is delivered and the fallback is
   recorded.
5. **Failure mode is "no savings", never "broken workflow".** All exceptions
   inside `Session.process` are contained.

## Module map

```
laconic/
  tokenizers/    tiered token counting: exact (tiktoken) / api (Anthropic) / estimate
  message/       Message model, protected-fields registry, framework parsers
  compress/      Compressor interface: passthrough, extractive, llmlingua (opt),
                 segments (payload protection), naive (eval-only baseline)
  dedup/         content store + session dedup (context mode / store mode)
  adapters/      per-model profiles: pricing (dated), measured safe keep-ratios
  integrations/  generic hook + thin LangGraph shim
  profiler/      handoff records, text/HTML reports, standalone trace analysis
  eval/          benchmark tasks + seeded generator, model clients (mock/real),
                 matrix runner, metrics (safe operating point), plots
  pipeline.py    Session: parse → dedup → compress → verify → rebuild
  cli.py         `laconic profile`, `laconic eval`, `laconic version`
```

## The pipeline

```
raw dict ──parse──▶ Message{structural, payload, metadata}
                       │
      no payload? ─────┤ (tool result, structured content, tiny text)
        └─▶ pass through, stats recorded
                       │
                payload ─▶ SessionDedup (per-recipient, prefix-stable)
                       │
                       ─▶ Compressor (only free-text segments)
                       │
                rebuild ─▶ verify: structural fingerprint byte-identical,
                       │           key order preserved, re-parses cleanly,
                       │           token count did not grow
                       │
              ok ─▶ deliver rebuilt message      fail ─▶ deliver original,
                                                          record fallback
```

## Where the savings land (and don't)

Tokens are spent when text enters a model's context — nowhere else. This has
two consequences the design takes seriously:

- **No rehydrate-before-read.** If a dedup handle were expanded before the
  receiving model reads the message, nothing would be saved. Context-mode
  dedup therefore only references content the *same recipient* already has in
  its context; store-mode dedup gives the receiving agent a rehydration tool
  and lets it decide.
- **Prefix stability.** Provider prompt caches bill cached input tokens at a
  fraction of the normal price. Laconic never rewrites already-sent history —
  only the newly produced handoff is processed — so cache hits on the shared
  prefix are preserved.

## Extension points

- `register_parser()` — add a message format.
- `ProtectedFieldsRegistry` — change what counts as protected per framework.
- Subclass `Compressor` — any string-to-string strategy plugs into the same
  safety pipeline; it can only ever see payload text.
- `register_profile()` — add pricing/tolerance for new models.
- Subclass `ContentStore` — back dedup with external storage.
