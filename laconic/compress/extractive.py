"""Extractive compression: deterministic, model-free, structure-aware.

Three stages, each measured in tokens against the target model's counter:

1. **Whitespace normalization** — collapse runs of blank lines and trailing
   spaces (free savings, zero risk).
2. **Filler pruning + token-checked abbreviations** — remove curated
   information-free phrases; apply a legend-free abbreviation dictionary,
   keeping each substitution only if it lowers the token count.
3. **Novelty-based sentence pruning** (only when a ``budget`` is set) — drop
   the most redundant sentences until the budget is met. The first sentence
   of every paragraph is always kept; sentences are never reordered.

Protected segments inside the payload (code fences, JSON paragraphs, tables)
are never touched — the segmenter runs before any stage.
"""

from __future__ import annotations

import re

from laconic.compress.abbreviations import iter_abbreviations, strip_fillers
from laconic.compress.base import Compressor
from laconic.compress.segments import Segment, join_segments, segment_payload
from laconic.tokenizers.base import TokenCounter

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")
_WORD_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
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
        "being",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "there",
        "here",
        "not",
        "no",
        "yes",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "will",
        "would",
        "can",
        "could",
        "should",
        "may",
        "might",
        "into",
        "over",
        "under",
        "about",
        "we",
        "you",
        "they",
        "he",
        "she",
        "i",
        "our",
        "your",
        "their",
        "his",
        "her",
        "them",
        "us",
    ]
)


def _content_words(sentence: str) -> frozenset[str]:
    return frozenset(w for w in _WORD_RE.findall(sentence.lower()) if w not in _STOPWORDS)


_HAS_NUMBER = re.compile(r"\d")
_REQUIREMENT_LANGUAGE = re.compile(
    r"\b(must|require[sd]?|requirement|constraint|deadline|never|always|"
    r"exceed|exclude[sd]?|at (?:most|least)|no later|before|hard)\b",
    re.IGNORECASE,
)
# A capitalized word that is not sentence-initial: crude named-entity signal.
_MID_SENTENCE_PROPER = re.compile(r"(?<=[a-z,;] )[A-Z][a-z]+")


def _salience(sentence: str) -> float:
    """Load-bearing-content score in [0, ~1.5].

    Numbers, requirement language, and named entities are what downstream
    agents act on; sentences carrying them are pruned last. Pure lexical
    novelty would do the opposite — verbose filler is worded diversely while
    facts repeat the entities they describe.
    """
    score = 0.0
    if _HAS_NUMBER.search(sentence):
        score += 0.6
    if _REQUIREMENT_LANGUAGE.search(sentence):
        score += 0.6
    if _MID_SENTENCE_PROPER.search(sentence):
        score += 0.3
    return score


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)  # trailing spaces
    text = re.sub(r"\n{3,}", "\n\n", text)  # runs of blank lines
    text = re.sub(r"[ \t]{2,}", " ", text)  # interior space runs
    return text


class ExtractiveCompressor(Compressor):
    """Deterministic extractive compressor (no external model required).

    Args:
        aggressive: Enable sentence pruning even without an explicit budget,
            using :attr:`default_budget`.
        default_budget: Keep-ratio used when ``aggressive`` is set and no
            budget is passed to :meth:`compress`.
    """

    name = "extractive"
    reversible = False

    def __init__(self, aggressive: bool = False, default_budget: float = 0.6) -> None:
        if not 0.0 < default_budget <= 1.0:
            raise ValueError("default_budget must be in (0, 1]")
        self.aggressive = aggressive
        self.default_budget = default_budget

    def compress(
        self,
        text: str,
        *,
        counter: TokenCounter,
        budget: float | None = None,
    ) -> str:
        if not text.strip():
            return text
        if budget is None and self.aggressive:
            budget = self.default_budget

        segments = segment_payload(text)
        out_segments: list[Segment] = []
        for segment in segments:
            if segment.kind == "protected":
                out_segments.append(segment)
                continue
            cleaned = self._lossless_pass(segment.text, counter)
            out_segments.append(Segment(kind="text", text=cleaned))

        result = join_segments(out_segments)

        if budget is not None:
            target = max(1, int(counter.count(text) * budget))
            if counter.count(result) > target:
                result = self._prune_sentences(out_segments, target, counter)

        return self._no_worse(text, result, counter)

    def _lossless_pass(self, text: str, counter: TokenCounter) -> str:
        """Whitespace + fillers + token-checked abbreviations."""
        candidate = _normalize_whitespace(text)
        candidate = strip_fillers(candidate)
        for pattern, replacement in iter_abbreviations():
            substituted = pattern.sub(replacement, candidate)
            # Keep the substitution only if it is cheaper in *tokens*.
            if substituted != candidate and counter.count(substituted) < counter.count(candidate):
                candidate = substituted
        candidate = _normalize_whitespace(candidate)
        return self._no_worse(text, candidate, counter)

    def _prune_sentences(
        self,
        segments: list[Segment],
        target_tokens: int,
        counter: TokenCounter,
    ) -> str:
        """Drop low-value sentences from text segments until under budget.

        Drop priority = redundancy − salience:

        - *Redundancy*: lexical overlap of a sentence's content words with the
          other currently kept sentences.
        - *Salience*: load-bearing signals — numbers, requirement/constraint
          language, named entities. In agent traffic these carry the facts a
          downstream agent acts on; lexical novelty alone actively favors
          verbose filler (which is worded diversely) over facts (which repeat
          the entities they describe), so salience must outrank novelty.

        Paragraph-leading sentences and protected segments are never dropped;
        order is preserved.
        """
        # Flatten into (segment_index, sentence_index, sentence, droppable).
        entries: list[dict] = []
        for seg_index, segment in enumerate(segments):
            if segment.kind == "protected":
                continue
            sentences = _SENTENCE_SPLIT.split(segment.text)
            for sent_index, sentence in enumerate(sentences):
                entries.append(
                    {
                        "seg": seg_index,
                        "idx": sent_index,
                        "text": sentence,
                        "dropped": False,
                        "droppable": sent_index > 0 and bool(sentence.strip()),
                    }
                )

        def rebuild() -> str:
            parts: list[Segment] = []
            for seg_index, segment in enumerate(segments):
                if segment.kind == "protected":
                    parts.append(segment)
                    continue
                kept = [e["text"] for e in entries if e["seg"] == seg_index and not e["dropped"]]
                parts.append(Segment(kind="text", text=" ".join(kept)))
            return join_segments(parts)

        for entry in entries:
            entry["words"] = _content_words(entry["text"])
            entry["salience"] = _salience(entry["text"])

        current = rebuild()
        while counter.count(current) > target_tokens:
            best = None
            best_score = float("-inf")
            for entry in entries:
                if entry["dropped"] or not entry["droppable"]:
                    continue
                words: frozenset[str] = entry["words"]
                if not words:
                    score = 10.0  # empty/space-only sentences go first
                else:
                    # Redundancy: how much of this sentence's content already
                    # appears in the *other* kept sentences.
                    others: set[str] = set()
                    for other in entries:
                        if other is not entry and not other["dropped"]:
                            others |= other["words"]
                    redundancy = len(words & others) / len(words)
                    score = redundancy - entry["salience"]
                if score > best_score:
                    best_score = score
                    best = entry
            if best is None:
                break  # nothing left that may be dropped
            best["dropped"] = True
            current = rebuild()
        return current
