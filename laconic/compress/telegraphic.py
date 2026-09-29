"""Telegraphic compression: the closest thing to "AI shorthand" that works.

Drops function words the way telegrams did — "The revenue in Q1 was 85.0
percent" becomes "Revenue in Q1 85.0 percent" — while keeping every content
word, number, name, and negation. This is the most aggressive strategy that
still lives inside the models' training distribution: terse English is
something LLMs have seen a great deal of; invented symbol codes are not, and
they usually tokenize *worse* (see ADR-3 and ``docs/related-work.md``).

Reliability posture: word-level dropping is riskier than sentence-level
pruning — grammar carries meaning more often than it seems. Safeguards:

- Negations (not, never, no, without...) are always kept, and the word
  following a kept negation is never dropped.
- Copulas (is/are/was/were) are dropped only when not adjacent to a negation.
- Capitalized words, numbers, and everything inside protected segments are
  untouchable.
- Like every compressor, it only ever sees payload text, and the pipeline's
  verify-and-fallback still applies.

Whether a given *model* reads telegraphic handoffs as accurately as full
prose is exactly the kind of per-model question the eval harness measures —
this strategy ships as an eval subject first and a production option second.
"""

from __future__ import annotations

import re

from laconic.compress.base import Compressor
from laconic.compress.segments import Segment, join_segments, segment_payload
from laconic.tokenizers.base import TokenCounter

#: Function words a telegram would omit. Deliberately excludes negations,
#: quantifiers, and modals whose loss can invert meaning (not, no, never,
#: must, all, none, only...).
_DROPPABLE = frozenset(
    [
        "a",
        "an",
        "the",
        "please",
        "kindly",
    ]
)

# Degree, frequency, certainty, and scope modifiers are content. Conservative
# filler cleanup can also remove some of them, so reject that cleanup when
# its qualifier sequence changes before applying the telegraphic pass.
_QUALIFIERS = re.compile(
    r"\b(?:very|really|quite|rather|fairly|somewhat|broadly|generally|basically|"
    r"essentially|actually|certainly|definitely|simply|just)\b",
    re.IGNORECASE,
)

#: Copulas droppable only when meaning-safe (no adjacent negation).
_COPULAS = frozenset(["is", "are", "was", "were"])

#: Words that must survive, and that protect their successor too.
_NEGATIONS = frozenset(
    [
        "not",
        "no",
        "never",
        "none",
        "nor",
        "cannot",
        "can't",
        "don't",
        "doesn't",
        "didn't",
        "won't",
        "isn't",
        "aren't",
        "wasn't",
        "without",
    ]
)

_TOKEN_SPLIT = re.compile(r"(\s+)")
_WORD = re.compile(r"^[A-Za-z']+$")


def _telegraph_sentenceward(text: str) -> str:
    """Drop function words from one free-text block, telegraph style."""
    parts = _TOKEN_SPLIT.split(text)
    out: list[str] = []
    protect_next = False
    for part in parts:
        if not part or part.isspace():
            out.append(part)
            continue
        core = part.strip(".,;:!?()\"'")
        lowered = core.lower()
        is_word = bool(_WORD.match(core))

        if protect_next:
            out.append(part)
            protect_next = bool(is_word and lowered in _NEGATIONS)
            continue
        if is_word and lowered in _NEGATIONS:
            out.append(part)
            protect_next = True
            continue
        if (
            is_word
            and core.islower()  # capitalized words (names, sentence heads) survive
            and (lowered in _DROPPABLE or lowered in _COPULAS)
            and core == part  # punctuation-bearing tokens survive intact
        ):
            continue  # drop it
        out.append(part)

    collapsed = re.sub(r"[ \t]{2,}", " ", "".join(out))
    return re.sub(r" ([.,;:!?])", r"\1", collapsed)


class TelegraphicCompressor(Compressor):
    """Function-word dropping over free-text payload segments.

    Attempts conservative cleanup first, retaining it only when the sequence
    of degree, frequency, certainty, and scope modifiers survives. Then drops
    articles, courtesy words, and eligible copulas. No budget parameter: the
    transformation is all-or-nothing per word class, so its achieved ratio is
    a property of the traffic (roughly 0.8–0.9 of payload tokens on verbose
    prose, more when the prose is chattier). Use the eval harness to learn
    whether your target model tolerates it before enabling it in production.
    """

    name = "telegraphic"
    reversible = False

    def __init__(self) -> None:
        # Deferred import: extractive imports the same segment machinery.
        from laconic.compress.extractive import ExtractiveCompressor

        self._cleanup = ExtractiveCompressor(aggressive=False)

    def compress(
        self,
        text: str,
        *,
        counter: TokenCounter,
        budget: float | None = None,
    ) -> str:
        if not text.strip():
            return text
        cleaned = self._cleanup.compress(text, counter=counter)
        if _QUALIFIERS.findall(cleaned.lower()) != _QUALIFIERS.findall(text.lower()):
            cleaned = text
        segments = segment_payload(cleaned)
        out: list[Segment] = []
        for segment in segments:
            if segment.kind == "protected":
                out.append(segment)
                continue
            out.append(Segment(kind="text", text=_telegraph_sentenceward(segment.text)))
        return self._no_worse(text, join_segments(out), counter)
