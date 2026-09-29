"""Pipeline behavior: strategies, fallbacks, dedup wiring, stats integrity."""

from __future__ import annotations

import pytest

from laconic.compress.base import Compressor
from laconic.compress.passthrough import PassthroughCompressor
from laconic.compress.segments import segment_payload
from laconic.dedup.session import SessionDedup
from laconic.pipeline import Session
from laconic.tokenizers.base import CountTier
from laconic.tokenizers.heuristic import HeuristicCounter


def test_remote_counter_is_only_used_for_before_and_after_measurements() -> None:
    class ApiCounter(HeuristicCounter):
        tier = CountTier.API

        def __init__(self):
            super().__init__("test-model")
            self.calls = 0

        def count(self, text):
            self.calls += 1
            return super().count(text)

    counter = ApiCounter()
    session = Session(target_model="test-model", counter=counter, dedup=False)
    result = session.process({"role": "assistant", "content": VERBOSE})
    assert not result.stats.fell_back
    assert result.stats.tier == CountTier.API
    assert counter.calls == 2


VERBOSE = (
    "Please note that the research phase is complete. It is worth noting "
    "that, for example, the dataset contains 4,812 rows. Basically, quality "
    "is acceptable. In order to proceed, the writer should draft the summary. "
    "The dataset contains 4,812 rows and quality is acceptable overall."
)


def test_strategy_off_is_pure_measurement() -> None:
    session = Session(target_model="test-model", strategy="off")
    raw = {"role": "assistant", "content": VERBOSE}
    processed = session.process(raw)
    assert processed.raw == raw
    assert processed.stats.tokens_after == processed.stats.tokens_before
    assert not processed.stats.fell_back
    repeated = session.process(raw, context_payloads=[VERBOSE])
    assert repeated.raw is raw
    assert repeated.stats.dedup_hits == 0
    assert session.dedup is not None
    assert len(session.dedup.store) == 0


def test_conservative_saves_tokens_on_verbose_payloads() -> None:
    session = Session(target_model="test-model", strategy="conservative", dedup=False)
    processed = session.process({"role": "assistant", "content": VERBOSE})
    assert processed.stats.tokens_after < processed.stats.tokens_before
    assert not processed.stats.fell_back
    assert processed.raw["content"] != VERBOSE
    assert "4,812" in processed.raw["content"]  # the payload's facts survive


def test_aggressive_saves_more_than_conservative() -> None:
    conservative = Session(target_model="test-model", strategy="conservative", dedup=False)
    aggressive = Session(target_model="test-model", strategy="aggressive", dedup=False)
    raw = {"role": "assistant", "content": VERBOSE * 4}
    saved_conservative = conservative.process(raw).stats.tokens_saved
    saved_aggressive = aggressive.process(raw).stats.tokens_saved
    assert saved_aggressive >= saved_conservative


def test_invalid_strategy_rejected() -> None:
    with pytest.raises(ValueError, match="strategy"):
        Session(target_model="m", strategy="yolo")


def test_dedup_hits_are_reported() -> None:
    session = Session(target_model="test-model", compressor=PassthroughCompressor())
    raw = {"role": "assistant", "content": VERBOSE}
    session.process(raw, target_agent="writer")
    second = session.process(raw, target_agent="writer", context_payloads=[VERBOSE])
    assert second.stats.dedup_hits == 1
    assert second.stats.tokens_after < second.stats.tokens_before


class _EvilCompressor(Compressor):
    """Misbehaving compressor: raises. The pipeline must contain it."""

    name = "evil"

    def compress(self, text, *, counter, budget=None):
        raise RuntimeError("boom")


class _BloatingCompressor(Compressor):
    """Misbehaving compressor: returns MORE tokens. Must trigger fallback."""

    name = "bloat"

    def compress(self, text, *, counter, budget=None):
        return text + " padding" * 50


@pytest.mark.parametrize("compressor_cls", [_EvilCompressor, _BloatingCompressor])
def test_misbehaving_compressors_trigger_passthrough(compressor_cls) -> None:
    session = Session(target_model="test-model", compressor=compressor_cls(), dedup=False)
    raw = {"role": "assistant", "content": VERBOSE}
    processed = session.process(raw)
    assert processed.raw == raw
    assert processed.stats.fell_back
    assert processed.stats.fallback_reason
    assert processed.stats.tokens_after == processed.stats.tokens_before


def test_custom_keep_ratio_is_respected() -> None:
    gentle = Session(target_model="test-model", strategy="balanced", keep_ratio=0.95, dedup=False)
    harsh = Session(target_model="test-model", strategy="balanced", keep_ratio=0.4, dedup=False)
    raw = {"role": "assistant", "content": VERBOSE * 6}
    assert harsh.process(raw).stats.tokens_after <= gentle.process(raw).stats.tokens_after


class _RecordingCompressor(Compressor):
    name = "recording"

    def __init__(self):
        self.seen = []

    def compress(self, text, *, counter, budget=None):
        self.seen.append(text)
        return text


@pytest.mark.parametrize("dedup", [True, SessionDedup(mode="store")])
def test_protected_spans_never_reach_compressor_or_dedup(dedup) -> None:
    code = (
        "```python\nfrom pathlib import Path\n\n"
        'config = {"environment": "production", "region": "us-west-2", '
        '"services": ["payments", "billing", "invoicing", "subscription"]}\n\n'
        "print(config)\n```"
    )
    payload = f"Here is the exact deployment script.\n\n{code}\n\nEnd of script."
    compressor = _RecordingCompressor()
    session = Session("test-model", compressor=compressor, dedup=dedup)
    raw = {"role": "assistant", "content": payload, "id": "original"}
    for _ in range(2):
        result = session.process(raw, context_payloads=[payload])
        protected = [
            s.text for s in segment_payload(result.raw["content"]) if s.kind == "protected"
        ]
        assert protected == [code]
        assert result.stats.structure_preserved
        assert result.stats.protected_spans_preserved
        assert not result.stats.fell_back
    assert all("config" not in text and "```" not in text for text in compressor.seen)


class _RemovingBoundaryCompressor(Compressor):
    name = "removing-boundary"

    def compress(self, text, *, counter, budget=None):
        return text.strip()


def test_protected_span_boundary_damage_falls_back() -> None:
    payload = (
        "Here is the original script that must remain executable.\n"
        "```python\nprint('preserve this implementation exactly')\n```"
    )
    session = Session("test-model", compressor=_RemovingBoundaryCompressor(), dedup=False)
    raw = {"role": "assistant", "content": payload}
    result = session.process(raw)
    assert result.raw is raw
    assert result.stats.fell_back
    assert result.stats.fallback_reason == "protected payload span mismatch"
    assert result.stats.safe  # Returned original retains its structure, not semantic proof.


def test_repeated_calls_without_context_do_not_deduplicate() -> None:
    session = Session("test-model", compressor=PassthroughCompressor())
    raw = {"role": "assistant", "content": VERBOSE}
    for _ in range(2):
        result = session.process(raw, target_agent="writer")
        assert result.raw == raw
        assert result.stats.dedup_hits == 0


def test_original_content_not_assumed_present_after_compression() -> None:
    session = Session("test-model", strategy="conservative")
    raw = {"role": "assistant", "content": VERBOSE}
    first = session.process(raw, target_agent="writer")
    assert first.raw["content"] != VERBOSE
    second = session.process(raw, target_agent="writer", context_payloads=[first.raw["content"]])
    assert second.stats.dedup_hits == 0
