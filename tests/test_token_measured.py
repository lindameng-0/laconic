"""Design principle #1: savings are measured in tokens, never characters.

The adversarial case: a string that is *shorter in characters* but *more
expensive in tokens*. Any code path that measured with ``len()`` would get the
comparison backwards; these tests pin the correct direction.
"""

from __future__ import annotations

from laconic.compress.extractive import ExtractiveCompressor
from laconic.message.model import CompressionStats
from laconic.pipeline import Session
from laconic.tokenizers.base import CountTier
from laconic.tokenizers.heuristic import HeuristicCounter

# Under the heuristic (and under real BPE tokenizers), a run of single-letter
# words costs one token each, while one long word costs far fewer tokens than
# its character count suggests.
FEWER_CHARS_MORE_TOKENS = "a b c d e f g h i j k l"  # 23 chars, ~12 tokens
MORE_CHARS_FEWER_TOKENS = "internationalization"  # 20 chars, ~4 tokens


def test_the_adversarial_pair_behaves_as_designed(counter: HeuristicCounter) -> None:
    assert len(FEWER_CHARS_MORE_TOKENS) > len(MORE_CHARS_FEWER_TOKENS)
    assert counter.count(FEWER_CHARS_MORE_TOKENS) > counter.count(MORE_CHARS_FEWER_TOKENS)


def test_compressor_no_worse_uses_tokens_not_chars(counter: HeuristicCounter) -> None:
    compressor = ExtractiveCompressor()
    # _no_worse must reject a candidate that is shorter in chars but more
    # expensive in tokens.
    kept = compressor._no_worse(
        original=MORE_CHARS_FEWER_TOKENS, candidate=FEWER_CHARS_MORE_TOKENS, counter=counter
    )
    assert kept == MORE_CHARS_FEWER_TOKENS


def test_stats_token_counts_come_from_the_counter(corpus) -> None:
    session = Session(target_model="test-model", strategy="conservative")
    for raw in corpus[:20]:
        processed = session.process(raw)
        stats = processed.stats
        assert stats.tokens_before == session._count_message(raw)
        assert stats.tokens_after == session._count_message(processed.raw)
        assert stats.tier == CountTier.ESTIMATE  # unknown model → labeled estimate


def test_stats_ratio_and_saved_derive_from_tokens() -> None:
    stats = CompressionStats(
        model="m",
        strategy="s",
        tokens_before=200,
        tokens_after=150,
        tier=CountTier.EXACT,
    )
    assert stats.ratio == 0.75
    assert stats.tokens_saved == 50
