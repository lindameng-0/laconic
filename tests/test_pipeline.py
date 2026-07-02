"""Pipeline behavior: strategies, fallbacks, dedup wiring, stats integrity."""

from __future__ import annotations

import pytest

from laconic.compress.base import Compressor
from laconic.pipeline import Session

VERBOSE = (
    "Please note that the research phase is complete. It is worth noting "
    "that, for example, the dataset contains 4,812 rows. Basically, quality "
    "is acceptable. In order to proceed, the writer should draft the summary. "
    "The dataset contains 4,812 rows and quality is acceptable overall."
)


def test_strategy_off_is_pure_measurement() -> None:
    session = Session(target_model="test-model", strategy="off", dedup=False)
    raw = {"role": "assistant", "content": VERBOSE}
    processed = session.process(raw)
    assert processed.raw == raw
    assert processed.stats.tokens_after == processed.stats.tokens_before
    assert not processed.stats.fell_back


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
    session = Session(target_model="test-model", strategy="off")
    raw = {"role": "assistant", "content": VERBOSE}
    session.process(raw, target_agent="writer")
    second = session.process(raw, target_agent="writer")
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
