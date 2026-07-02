"""Naive whole-message compression — the eval baseline Laconic exists to avoid.

.. warning::
    **Never use this in a real workflow.** This compressor deliberately treats
    the entire serialized message — tool calls, arguments, IDs and all — as
    prose and prunes it to a target ratio. It exists so the eval harness can
    quantify how quickly workflows break when structure is not protected,
    reproducing the failure mode reported for general-purpose prompt
    compression applied to agent traffic.
"""

from __future__ import annotations

import re

from laconic.compress.base import Compressor
from laconic.tokenizers.base import TokenCounter

_WORD_SPLIT = re.compile(r"\s+")

_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "there",
        "here",
        "not",
        "we",
        "you",
        "they",
        "i",
    ]
)


class NaiveWholeMessageCompressor(Compressor):
    """Structure-blind token pruning over the full serialized message.

    Stage 1 drops stopwords; stage 2 drops every k-th remaining word until the
    budget is met. Both stages happily destroy JSON syntax, key names, and
    IDs — which is the point of the baseline.
    """

    name = "naive-whole-message"
    reversible = False

    def compress(
        self,
        text: str,
        *,
        counter: TokenCounter,
        budget: float | None = None,
    ) -> str:
        if budget is None or not text.strip():
            return text
        target = max(1, int(counter.count(text) * budget))
        words = _WORD_SPLIT.split(text)

        # Stage 1: stopword removal.
        kept = [w for w in words if w.lower().strip(".,;:") not in _STOPWORDS]
        candidate = " ".join(kept)
        # Stage 2: uniform decimation until under budget.
        while counter.count(candidate) > target and len(kept) > 8:
            kept = [w for i, w in enumerate(kept) if i % 5 != 2]
            candidate = " ".join(kept)
        return candidate
