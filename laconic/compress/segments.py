"""Payload segmentation: protect embedded structure inside prose.

Even after a message's structural *fields* are set aside, the payload text
itself can embed structure — fenced code blocks, JSON documents, tables.
Compressing those corrupts meaning as surely as compressing a tool call, so
payloads are segmented and only free-text segments are ever transformed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

_FENCE_RE = re.compile(r"^```.*?^```[ \t]*$", re.MULTILINE | re.DOTALL)

SegmentKind = Literal["text", "protected"]


@dataclass(frozen=True)
class Segment:
    """A contiguous slice of payload text, either compressible or protected."""

    kind: SegmentKind
    text: str


def _paragraph_is_structured(paragraph: str) -> bool:
    stripped = paragraph.strip()
    if not stripped:
        return False
    if stripped[0] in "{[":
        try:
            json.loads(stripped)
            return True
        except (json.JSONDecodeError, ValueError):
            return False
    # Markdown tables.
    lines = stripped.splitlines()
    return len(lines) >= 2 and all(line.lstrip().startswith("|") for line in lines[:2])


def segment_payload(text: str) -> list[Segment]:
    """Split payload text into compressible and protected segments.

    Protected: fenced code blocks, paragraphs that parse as JSON, markdown
    tables. Everything else is free text.
    """
    segments: list[Segment] = []
    cursor = 0
    for match in _FENCE_RE.finditer(text):
        if match.start() > cursor:
            segments.extend(_split_text_block(text[cursor : match.start()]))
        segments.append(Segment(kind="protected", text=match.group()))
        cursor = match.end()
    if cursor < len(text):
        segments.extend(_split_text_block(text[cursor:]))
    return segments


def _split_text_block(block: str) -> list[Segment]:
    """Split a fence-free block on blank lines, protecting structured paragraphs."""
    segments: list[Segment] = []
    parts = re.split(r"(\n[ \t]*\n)", block)  # keep separators so join is lossless
    for part in parts:
        if not part:
            continue
        if _paragraph_is_structured(part):
            segments.append(Segment(kind="protected", text=part))
        else:
            segments.append(Segment(kind="text", text=part))
    return segments


def join_segments(segments: list[Segment]) -> str:
    """Reassemble segments into payload text (inverse of :func:`segment_payload`)."""
    return "".join(segment.text for segment in segments)
