"""Payload compressors. All operate on payload text only, never structure."""

from laconic.compress.base import Compressor
from laconic.compress.extractive import ExtractiveCompressor
from laconic.compress.naive import NaiveWholeMessageCompressor
from laconic.compress.passthrough import PassthroughCompressor
from laconic.compress.segments import Segment, join_segments, segment_payload
from laconic.compress.telegraphic import TelegraphicCompressor

__all__ = [
    "Compressor",
    "ExtractiveCompressor",
    "NaiveWholeMessageCompressor",
    "PassthroughCompressor",
    "Segment",
    "TelegraphicCompressor",
    "join_segments",
    "segment_payload",
]
