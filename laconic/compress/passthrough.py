"""No-op compressor: the measurement baseline."""

from __future__ import annotations

from laconic.compress.base import Compressor
from laconic.tokenizers.base import TokenCounter


class PassthroughCompressor(Compressor):
    """Returns text unchanged. Used to measure a workflow before compressing it."""

    name = "passthrough"
    reversible = True

    def compress(
        self,
        text: str,
        *,
        counter: TokenCounter,
        budget: float | None = None,
    ) -> str:
        return text
