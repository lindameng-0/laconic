"""Offline token estimation.

Used when no exact tokenizer is available locally (notably for Anthropic
models, whose exact counter is an API endpoint — see
``docs/design-decisions.md``, ADR-5). Every number produced here is labeled
``CountTier.ESTIMATE`` and must never be presented as exact.

The estimator is calibrated against modern BPE vocabularies (cl100k/o200k
class): common short words are ~1 token, long words split into subwords,
digit runs split every ~3 characters, and most punctuation marks stand alone.
Expected error is roughly ±15–20% on English prose and less on structured
text; the eval harness reports estimator error against exact counters when
one is available.
"""

from __future__ import annotations

import re

from laconic.tokenizers.base import CountTier, TokenCounter

_TOKEN_RE = re.compile(
    r"""
    (?P<word>[A-Za-z]+(?:'[A-Za-z]+)?)   # words, incl. contractions
    | (?P<number>\d+)                    # digit runs
    | (?P<space>\s+)                     # whitespace runs
    | (?P<punct>.)                       # any other single character
    """,
    re.VERBOSE,
)

# Constants fitted by grid search against tiktoken (o200k_base) on mixed
# prose / structured samples; max observed error ~12% on the fitting set.
# Average characters per token inside a long word:
_SUBWORD_CHARS = 7
# Digit runs split roughly every 3 characters:
_DIGIT_CHARS = 3
# Punctuation often merges with neighboring tokens, so it costs < 1 on average:
_PUNCT_WEIGHT = 0.7


class HeuristicCounter(TokenCounter):
    """Calibrated offline token estimator (tier: ``estimate``).

    Args:
        model: The model the estimate is attributed to (for labeling only —
            the same heuristic is used regardless).
    """

    tier = CountTier.ESTIMATE

    def __init__(self, model: str = "unknown") -> None:
        self.model = model

    def count(self, text: str) -> int:
        if not text:
            return 0
        tokens = 0.0
        for match in _TOKEN_RE.finditer(text):
            if match.lastgroup == "word":
                length = len(match.group())
                tokens += 1 + max(0, (length - 1) // _SUBWORD_CHARS)
            elif match.lastgroup == "number":
                length = len(match.group())
                tokens += 1 + max(0, (length - 1) // _DIGIT_CHARS)
            elif match.lastgroup == "space":
                # Whitespace usually merges into neighboring tokens; long runs
                # (indentation, blank lines) cost extra.
                tokens += len(match.group()) // 4
            else:
                tokens += _PUNCT_WEIGHT
        return max(1, round(tokens))
