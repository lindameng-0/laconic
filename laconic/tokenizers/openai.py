"""Exact token counting for OpenAI models via tiktoken (tier: ``exact``)."""

from __future__ import annotations

from laconic.exceptions import MissingDependencyError
from laconic.tokenizers.base import CountTier, TokenCounter

_FALLBACK_ENCODING = "o200k_base"


class TiktokenCounter(TokenCounter):
    """Counts tokens with the local tiktoken tokenizer.

    Args:
        model: OpenAI model name (e.g. ``"gpt-4.1"``). If tiktoken does not
            know the model, the ``o200k_base`` encoding is used and the counter
            remains tier ``exact`` for o200k-family models.

    Raises:
        MissingDependencyError: If tiktoken is not installed.
    """

    tier = CountTier.EXACT

    def __init__(self, model: str) -> None:
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover - exercised via registry tests
            raise MissingDependencyError("Exact OpenAI token counting", "openai") from exc
        self.model = model
        try:
            self._encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self._encoding = tiktoken.get_encoding(_FALLBACK_ENCODING)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text, disallowed_special=()))
