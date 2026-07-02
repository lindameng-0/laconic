"""Compressor interface.

A compressor receives **only** the payload text of a message — never the
structural part — so corrupting tool calls or schemas is impossible by
construction. Its contract:

- ``compress(text, counter, budget)`` returns text whose token count (per
  ``counter``) is less than or equal to the original's; if it cannot improve,
  it returns the input unchanged.
- ``budget`` is a target *keep ratio* in ``(0, 1]``: ``0.6`` asks to keep at
  most ~60% of the original tokens. ``None`` means "save what you can
  losslessly-in-meaning, without a hard target".
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from laconic.tokenizers.base import TokenCounter


class Compressor(ABC):
    """Base class for payload compressors.

    Attributes:
        name: Strategy name recorded in :class:`~laconic.message.CompressionStats`.
        reversible: Whether the transformation is mechanically reversible.
            (Extractive compression is not; dedup handles are.)
    """

    name: str = "base"
    reversible: bool = False

    @abstractmethod
    def compress(
        self,
        text: str,
        *,
        counter: TokenCounter,
        budget: float | None = None,
    ) -> str:
        """Return a token-cheaper version of ``text``.

        Implementations must measure with ``counter`` (tokens, never
        ``len()``) and must return the input unchanged when they cannot
        reduce its token count.
        """

    def _no_worse(self, original: str, candidate: str, counter: TokenCounter) -> str:
        """Return ``candidate`` only if it does not cost more tokens."""
        if counter.count(candidate) <= counter.count(original):
            return candidate
        return original
