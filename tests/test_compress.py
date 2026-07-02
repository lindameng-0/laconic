"""Compressor behavior: fillers, abbreviations, budgets, protected segments."""

from __future__ import annotations

from laconic.compress.extractive import ExtractiveCompressor
from laconic.compress.naive import NaiveWholeMessageCompressor
from laconic.compress.passthrough import PassthroughCompressor
from laconic.compress.segments import join_segments, segment_payload
from laconic.tokenizers.heuristic import HeuristicCounter

VERBOSE = (
    "Please note that the deployment completed successfully. "
    "It is worth noting that, for example, latency improved. "
    "Basically, the rollout is done. In order to verify, check the dashboard. "
    "The rollout is done and the deployment completed with improved latency. "
    "As previously mentioned, no incidents occurred during the window."
)

CODE_FENCED = (
    "Here is the fix that was applied to the handler.\n\n"
    "```python\ndef handler(x):\n    return x * 2  # doubled\n```\n\n"
    "Please note that the fix is already deployed to staging."
)

JSON_PARAGRAPH = (
    "The service returned the following payload.\n\n"
    '{"status": "ok", "count": 42}\n\n'
    "Please note that no retries were needed."
)


def test_passthrough_is_identity(counter: HeuristicCounter) -> None:
    assert PassthroughCompressor().compress(VERBOSE, counter=counter) == VERBOSE


def test_extractive_removes_fillers(counter: HeuristicCounter) -> None:
    out = ExtractiveCompressor().compress(VERBOSE, counter=counter)
    assert counter.count(out) < counter.count(VERBOSE)
    assert "Please note that" not in out
    assert "deployment completed successfully" in out  # information kept


def test_extractive_never_returns_more_tokens(counter: HeuristicCounter) -> None:
    compressor = ExtractiveCompressor(aggressive=True)
    for text in (VERBOSE, CODE_FENCED, JSON_PARAGRAPH, "short.", ""):
        out = compressor.compress(text, counter=counter, budget=0.5)
        assert counter.count(out) <= counter.count(text) or out == text


def test_budget_prunes_sentences(counter: HeuristicCounter) -> None:
    out = ExtractiveCompressor().compress(VERBOSE, counter=counter, budget=0.4)
    assert counter.count(out) < counter.count(VERBOSE) * 0.75
    # The opening sentence's information must survive (never drop the lead).
    assert "deployment completed successfully" in out


def test_code_fences_are_never_touched(counter: HeuristicCounter) -> None:
    out = ExtractiveCompressor(aggressive=True).compress(CODE_FENCED, counter=counter, budget=0.3)
    assert "def handler(x):\n    return x * 2  # doubled" in out


def test_json_paragraphs_are_never_touched(counter: HeuristicCounter) -> None:
    out = ExtractiveCompressor(aggressive=True).compress(
        JSON_PARAGRAPH, counter=counter, budget=0.3
    )
    assert '{"status": "ok", "count": 42}' in out


def test_segmenter_roundtrip_is_lossless() -> None:
    for text in (VERBOSE, CODE_FENCED, JSON_PARAGRAPH, "", "one\n\ntwo\n\n\nthree"):
        assert join_segments(segment_payload(text)) == text


def test_naive_baseline_destroys_structure(counter: HeuristicCounter) -> None:
    """The eval baseline must actually exhibit the failure mode it models."""
    serialized = (
        '{"role": "assistant", "content": "call the tool now", '
        '"tool_calls": [{"id": "call_ab12", "type": "function", '
        '"function": {"name": "search", "arguments": "{\\"q\\": \\"x\\"}"}}]}'
    )
    out = NaiveWholeMessageCompressor().compress(serialized, counter=counter, budget=0.4)
    import json

    try:
        json.loads(out)
        survived = True
    except json.JSONDecodeError:
        survived = False
    assert not survived, "naive compression unexpectedly preserved JSON structure"


def test_abbreviations_only_fire_when_cheaper(counter: HeuristicCounter) -> None:
    text = "This works, for example, in most cases. " * 3
    out = ExtractiveCompressor().compress(text, counter=counter)
    assert counter.count(out) <= counter.count(text)
