# Contributing

Contributions welcome — especially parsers for more frameworks, tokenizer
backends, and real-model eval results.

## Setup

```bash
git clone https://github.com/USER/laconic
cd laconic
python -m venv .venv && . .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

## Checks

```bash
pytest          # the whole suite runs offline; no API keys, no network
ruff check .    # lint
ruff format --check .
```

CI runs both on Ubuntu and Windows, Python 3.10–3.13. Please keep it green.

## Ground rules (from the ADRs — see docs/design-decisions.md)

1. **Never compress protected structure.** If your change touches parsing or
   rebuilding, the round-trip invariant tests must still pass; add corpus
   shapes for any new message form you handle.
2. **Measure in tokens, never characters.** Any new transformation must be
   token-checked through a `TokenCounter`.
3. **Fallback over failure.** New pipeline paths must degrade to passthrough,
   with the reason recorded in stats.
4. **No invented numbers.** Benchmark results come from runs of the harness;
   PRs adding real-model results should include the exact command, model IDs,
   date, and the produced `results.json`.
5. **Keep optional dependencies optional.** Heavy imports live behind lazy
   guards raising `MissingDependencyError` with the right pip extra.
6. **Docstrings + type hints on every public symbol**, and keep the README
   accurate with the code in the same PR.

## Adding a framework parser

1. Subclass `MessageParser`; set `framework`; implement `parse`/`rebuild`
   with the exact round-trip guarantee.
2. Register a `ProtectedFieldsPolicy` for the framework.
3. Add corpus shapes to `tests/conftest.py` and parametrize the round-trip
   tests.
4. Document the payload-eligibility rules in `docs/architecture.md`.

## Adding a model profile

Add a `ModelProfile` in `laconic/adapters/profiles.py` with dated pricing.
Leave `safe_keep_ratio=None` — it may only be filled by an eval run, with the
run artifacts linked in the PR.

## Releases

Bump `laconic.__version__` and `pyproject.toml` together, update
`CHANGELOG.md`, tag `vX.Y.Z`.
