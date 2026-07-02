"""Token-counting interface.

Every token number reported anywhere in Laconic flows through a
:class:`TokenCounter`. Savings are **never** computed from character counts —
a shorter string can tokenize to more tokens (design principle #1).

Counters are tiered, and the tier travels with every number so that exact and
estimated counts are never silently mixed:

- ``exact`` — a local tokenizer identical to the provider's (e.g. tiktoken).
- ``api`` — the provider's token-counting endpoint (network call, exact).
- ``estimate`` — a calibrated offline heuristic (expected error ~±15–20%).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class CountTier(str, Enum):
    """Provenance of a token count. Exact and estimated counts are never mixed."""

    EXACT = "exact"
    API = "api"
    ESTIMATE = "estimate"


@dataclass(frozen=True)
class TokenCount:
    """A token count together with its provenance.

    Attributes:
        tokens: Number of tokens.
        tier: How the number was produced (see :class:`CountTier`).
        model: The model whose tokenizer (or estimate calibration) was used.
    """

    tokens: int
    tier: CountTier
    model: str


class TokenCounter(ABC):
    """Counts tokens for one target model.

    Subclasses must set :attr:`model` and :attr:`tier` and implement
    :meth:`count`.
    """

    model: str
    tier: CountTier

    @abstractmethod
    def count(self, text: str) -> int:
        """Return the number of tokens in ``text`` for :attr:`model`."""

    def count_with_tier(self, text: str) -> TokenCount:
        """Return the count bundled with its provenance."""
        return TokenCount(tokens=self.count(text), tier=self.tier, model=self.model)
