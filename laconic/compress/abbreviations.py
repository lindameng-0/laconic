"""Curated, legend-free abbreviation dictionary.

Design contract (ADR-4 in ``docs/design-decisions.md``):

- Only expansions that any modern instruction-tuned model reads natively are
  included, so **no legend is injected** and no comprehension risk is added.
- A substitution is applied only when it **reduces the token count under the
  target model's tokenizer** — a shorter string is not necessarily cheaper.
- Substitutions never fire inside words (word-boundary anchored) and never
  inside protected segments (the segmenter runs first).

Users can extend or replace the dictionary; entries are (pattern, replacement)
where the pattern is matched case-insensitively at word boundaries.
"""

from __future__ import annotations

import re

#: (expansion phrase, abbreviation). Ordered: longer phrases first so they win.
ABBREVIATIONS: list[tuple[str, str]] = [
    ("as soon as possible", "ASAP"),
    ("that is to say", "i.e."),
    ("in other words", "i.e."),
    ("for example", "e.g."),
    ("for instance", "e.g."),
    ("with respect to", "w.r.t."),
    ("with regard to", "regarding"),
    ("in order to", "to"),
    ("in the event that", "if"),
    ("in the case that", "if"),
    ("due to the fact that", "because"),
    ("owing to the fact that", "because"),
    ("despite the fact that", "although"),
    ("at this point in time", "now"),
    ("at the present time", "now"),
    ("a large number of", "many"),
    ("a small number of", "few"),
    ("the majority of", "most"),
    ("is able to", "can"),
    ("are able to", "can"),
    ("was able to", "could"),
    ("has the ability to", "can"),
    ("it is important to note that", ""),
    ("it is worth noting that", ""),
    ("et cetera", "etc."),
    ("versus", "vs."),
    ("approximately", "~"),
]

#: Sentence-initial filler that carries no information in agent traffic.
FILLER_PATTERNS: list[str] = [
    r"please note that\s+",
    r"it should be noted that\s+",
    r"as previously mentioned,?\s+",
    r"as mentioned earlier,?\s+",
    r"as you can see,?\s+",
    r"needless to say,?\s+",
    r"i would like to point out that\s+",
    r"i want to emphasize that\s+",
    r"to be honest,?\s+",
    r"basically,?\s+",
    r"essentially,?\s+",
    r"certainly!\s*",
    r"of course!\s*",
    r"sure thing!\s*",
    r"great question!\s*",
]

_COMPILED_ABBREVIATIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE), replacement)
    for phrase, replacement in ABBREVIATIONS
]

_COMPILED_FILLERS: list[re.Pattern[str]] = [
    re.compile(pattern, re.IGNORECASE) for pattern in FILLER_PATTERNS
]


_SENTENCE_START_LOWER = re.compile(r"(^|[.!?]\s+)([a-z])")


def _recapitalize(text: str) -> str:
    """Uppercase sentence-initial letters left lowercase by phrase removal.

    Without this, downstream sentence splitting (which keys on a capital
    after end punctuation) silently merges adjacent sentences — and the
    pruner then drops facts fused to filler.
    """
    return _SENTENCE_START_LOWER.sub(lambda m: m.group(1) + m.group(2).upper(), text)


def strip_fillers(text: str) -> str:
    """Remove information-free filler phrases."""
    for pattern in _COMPILED_FILLERS:
        text = pattern.sub("", text)
    return _recapitalize(text)


def iter_abbreviations() -> list[tuple[re.Pattern[str], str]]:
    """The compiled abbreviation patterns, longest-phrase-first."""
    return list(_COMPILED_ABBREVIATIONS)
