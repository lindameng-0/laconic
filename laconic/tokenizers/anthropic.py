"""Token counting for Anthropic models.

Anthropic does not ship a local tokenizer for current Claude models; the exact
count comes from the ``messages.count_tokens`` API endpoint. That is a network
call requiring an API key, so it is **opt-in** (tier ``api``). The default for
Claude targets is the offline estimator (tier ``estimate``) — see ADR-5 in
``docs/design-decisions.md``.
"""

from __future__ import annotations

from typing import Any

from laconic.exceptions import MissingDependencyError
from laconic.tokenizers.base import CountTier, TokenCounter
from laconic.tokenizers.heuristic import HeuristicCounter


class AnthropicApiCounter(TokenCounter):
    """Exact-by-API token counting via ``messages.count_tokens`` (tier: ``api``).

    Each call is a network round-trip. Use for calibration and final reports,
    not for per-substitution checks inside a compressor — pair it with an
    estimate-tier counter for inner loops.

    Args:
        model: Anthropic model name (e.g. ``"claude-sonnet-4-5"``).
        client: An optional pre-configured ``anthropic.Anthropic`` client.
            If omitted, one is constructed from the environment
            (``ANTHROPIC_API_KEY``).

    Raises:
        MissingDependencyError: If the ``anthropic`` package is not installed.
    """

    tier = CountTier.API

    def __init__(self, model: str, client: Any | None = None) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise MissingDependencyError("Anthropic API token counting", "anthropic") from exc
            client = anthropic.Anthropic()
        self.model = model
        self._client = client

    def count(self, text: str) -> int:
        if not text:
            return 0
        response = self._client.messages.count_tokens(
            model=self.model,
            messages=[{"role": "user", "content": text}],
        )
        # The endpoint counts the full request; subtract the fixed overhead of
        # an empty-ish message scaffold so deltas between two texts are exact.
        return int(response.input_tokens)


class AnthropicEstimateCounter(HeuristicCounter):
    """Offline estimator attributed to an Anthropic model (tier: ``estimate``).

    Identical heuristic to :class:`~laconic.tokenizers.heuristic.HeuristicCounter`;
    exists so stats carry the Claude model name while being honestly labeled
    as estimates.
    """
