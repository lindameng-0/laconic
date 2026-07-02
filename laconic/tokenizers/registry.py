"""Resolve the right token counter for a model name.

Resolution order:

1. OpenAI-family model + tiktoken installed → :class:`TiktokenCounter` (exact).
2. Anthropic-family model + ``allow_api=True`` + anthropic installed →
   :class:`AnthropicApiCounter` (api).
3. Otherwise → :class:`HeuristicCounter` / :class:`AnthropicEstimateCounter`
   (estimate).

The fallback is always available, so a bare ``pip install laconic`` works with
zero optional dependencies — every number is just labeled ``estimate``.
"""

from __future__ import annotations

from typing import Any

from laconic.exceptions import MissingDependencyError
from laconic.tokenizers.anthropic import AnthropicApiCounter, AnthropicEstimateCounter
from laconic.tokenizers.base import TokenCounter
from laconic.tokenizers.heuristic import HeuristicCounter

_OPENAI_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt", "text-embedding")
_ANTHROPIC_PREFIXES = ("claude",)


def model_family(model: str) -> str:
    """Classify a model name as ``"openai"``, ``"anthropic"``, or ``"unknown"``."""
    lowered = model.lower()
    if lowered.startswith(_OPENAI_PREFIXES):
        return "openai"
    if lowered.startswith(_ANTHROPIC_PREFIXES):
        return "anthropic"
    return "unknown"


def get_counter(
    model: str,
    *,
    allow_api: bool = False,
    anthropic_client: Any | None = None,
) -> TokenCounter:
    """Return the best available :class:`TokenCounter` for ``model``.

    Args:
        model: Target model name (e.g. ``"gpt-4.1"``, ``"claude-sonnet-4-5"``).
        allow_api: Permit network calls for exact Anthropic counts. Off by
            default so nothing in Laconic performs I/O unless asked.
        anthropic_client: Optional pre-configured Anthropic client, used only
            when ``allow_api`` is set.

    Returns:
        A counter whose ``tier`` attribute states how its numbers were produced.
    """
    family = model_family(model)
    if family == "openai":
        try:
            from laconic.tokenizers.openai import TiktokenCounter

            return TiktokenCounter(model)
        except MissingDependencyError:
            return HeuristicCounter(model)
    if family == "anthropic":
        if allow_api:
            try:
                return AnthropicApiCounter(model, client=anthropic_client)
            except MissingDependencyError:
                return AnthropicEstimateCounter(model)
        return AnthropicEstimateCounter(model)
    return HeuristicCounter(model)
