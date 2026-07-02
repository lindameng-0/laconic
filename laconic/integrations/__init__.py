"""Framework integrations. Thin shims — the logic lives in the core pipeline."""

from laconic.integrations.generic import CompressingHook
from laconic.integrations.langgraph import LangGraphCompressor, make_compression_node

__all__ = [
    "CompressingHook",
    "LangGraphCompressor",
    "make_compression_node",
]
