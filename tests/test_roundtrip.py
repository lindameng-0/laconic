"""THE invariant test: structure survives byte-identical, always.

Over a synthetic corpus of agent messages (tool calls, tool results, unicode,
code fences, multimodal parts, unknown keys), assert:

1. ``rebuild(parse(raw))`` with an unchanged payload equals ``raw`` exactly.
2. After full pipeline processing under every strategy, everything except the
   payload field is **byte-identical** (canonical JSON comparison) and key
   order is preserved.
"""

from __future__ import annotations

import json

import pytest

from laconic.message.parsers import get_parser
from laconic.pipeline import STRATEGIES, Session


def canonical(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def structural_only(raw: dict, payload_field: str | None) -> dict:
    return {k: v for k, v in raw.items() if k != payload_field}


def test_parse_rebuild_is_identity(corpus) -> None:
    parser = get_parser("openai-chat")
    for raw in corpus:
        message = parser.parse(raw)
        rebuilt = parser.rebuild(message)
        assert rebuilt == raw, f"round-trip changed the message: {raw}"
        assert list(rebuilt.keys()) == list(raw.keys()), "key order changed"


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_pipeline_never_touches_structure(corpus, strategy: str) -> None:
    session = Session(target_model="test-model", strategy=strategy)
    parser = get_parser("openai-chat")
    for raw in corpus:
        processed = session.process(raw, source_agent="a", target_agent="b")
        out = processed.raw
        # Identify the payload field the parser would use for this message.
        message = parser.parse(raw)
        payload_field = message.payload_field
        assert canonical(structural_only(out, payload_field)) == canonical(
            structural_only(raw, payload_field)
        ), f"structural fields changed under strategy={strategy}"
        assert list(out.keys()) == list(raw.keys()), "key order changed"
        assert processed.stats.tokens_after <= processed.stats.tokens_before


def test_unparseable_messages_pass_through_unchanged() -> None:
    session = Session(target_model="test-model", strategy="aggressive")
    weird_messages = [
        {"content": "no role field at all " * 20},
        {"role": 42, "content": "role is not a string " * 20},
        {"role": "assistant", "content": {"nested": "dict content"}},
    ]
    for raw in weird_messages:
        processed = session.process(raw)
        assert processed.raw == raw
        assert processed.stats.tokens_after == processed.stats.tokens_before
        # Messages that *parse* but have no compressible payload are not
        # fallbacks; messages that fail parsing are.
        if "role" not in raw or not isinstance(raw.get("role"), str):
            assert processed.stats.fell_back


def test_tool_call_arguments_are_never_altered(corpus) -> None:
    session = Session(target_model="test-model", strategy="aggressive")
    for raw in corpus:
        if "tool_calls" not in raw:
            continue
        processed = session.process(raw)
        assert processed.raw["tool_calls"] == raw["tool_calls"]
        # The arguments JSON string must still parse.
        for call in processed.raw["tool_calls"]:
            json.loads(call["function"]["arguments"])
