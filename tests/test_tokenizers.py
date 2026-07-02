"""Token counting: tiers, estimator behavior, and the tokens-not-characters rule."""

from __future__ import annotations

import pytest

from laconic.tokenizers.base import CountTier
from laconic.tokenizers.heuristic import HeuristicCounter
from laconic.tokenizers.registry import get_counter, model_family


def test_model_family_classification() -> None:
    assert model_family("gpt-4.1") == "openai"
    assert model_family("o3-mini") == "openai"
    assert model_family("claude-sonnet-4-5") == "anthropic"
    assert model_family("mistral-large") == "unknown"


def test_registry_returns_exact_for_openai() -> None:
    pytest.importorskip("tiktoken", reason="exact-counter test needs tiktoken")
    counter = get_counter("gpt-4.1")
    assert counter.tier == CountTier.EXACT


def test_registry_returns_estimate_for_anthropic_by_default() -> None:
    counter = get_counter("claude-sonnet-4-5")
    assert counter.tier == CountTier.ESTIMATE
    assert counter.model == "claude-sonnet-4-5"


def test_heuristic_basic_properties() -> None:
    counter = HeuristicCounter()
    assert counter.count("") == 0
    assert counter.count("word") == 1
    short = counter.count("The cat sat on the mat.")
    longer = counter.count("The cat sat on the mat. " * 10)
    assert 0 < short < longer


def test_heuristic_within_tolerance_of_exact() -> None:
    """The estimator must stay within ±25% of tiktoken on realistic prose."""
    pytest.importorskip("tiktoken", reason="calibration test needs tiktoken")
    exact = get_counter("gpt-4.1")
    estimate = HeuristicCounter()
    prose = (
        "Please note that the quarterly review has been completed and the "
        "revenue figures came in at approximately 12.4 percent above plan, "
        "which is broadly consistent with the projections we discussed. "
        "Deployment is scheduled for 2026-07-15 in the Rotterdam region."
    ) * 4
    e = exact.count(prose)
    h = estimate.count(prose)
    assert abs(h - e) / e < 0.25, f"estimator off by {abs(h - e) / e:.1%} (exact={e}, est={h})"


def test_count_with_tier_labels_provenance() -> None:
    counter = HeuristicCounter("some-model")
    result = counter.count_with_tier("hello world")
    assert result.tier == CountTier.ESTIMATE
    assert result.model == "some-model"
    assert result.tokens == counter.count("hello world")
