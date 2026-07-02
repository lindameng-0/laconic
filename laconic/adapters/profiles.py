"""Per-model profiles: pricing and compression-tolerance operating points.

Pricing is **as of 2026-06** (``PRICING_AS_OF``), in USD per million tokens,
and will drift — override via :func:`register_profile` or the ``pricing``
argument of the profiler when it matters.

``safe_keep_ratio`` is the keep-ratio at which task accuracy measured by the
eval harness stays within tolerance of the uncompressed baseline. It ships as
``None`` (unmeasured) for every model: **these numbers must come from running
``laconic.eval`` against real APIs, not from this file's defaults.** Until a
measurement exists, the pipeline uses ``DEFAULT_CONSERVATIVE_KEEP_RATIO``.
"""

from __future__ import annotations

from dataclasses import dataclass

PRICING_AS_OF = "2026-06"

#: Keep-ratio used for budgeted strategies when no measured operating point
#: exists for the target model. Deliberately gentle (reliability > aggression).
DEFAULT_CONSERVATIVE_KEEP_RATIO = 0.75


@dataclass(frozen=True)
class ModelProfile:
    """Pricing and tolerance profile for one model.

    Attributes:
        model: Model name prefix this profile matches.
        family: ``"openai"`` / ``"anthropic"`` / ``"unknown"``.
        input_price: USD per million input tokens (``None`` = unknown).
        cached_input_price: USD per million *cache-hit* input tokens. The gap
            between this and ``input_price`` is why dedup must be
            prefix-stable (see ``laconic/dedup/session.py``).
        output_price: USD per million output tokens.
        safe_keep_ratio: Measured safe operating point from the eval harness;
            ``None`` until an actual run has produced it.
    """

    model: str
    family: str
    input_price: float | None = None
    cached_input_price: float | None = None
    output_price: float | None = None
    safe_keep_ratio: float | None = None


_PROFILES: list[ModelProfile] = [
    # OpenAI — prices as of PRICING_AS_OF.
    ModelProfile("gpt-4.1-mini", "openai", 0.40, 0.10, 1.60),
    ModelProfile("gpt-4.1-nano", "openai", 0.10, 0.025, 0.40),
    ModelProfile("gpt-4.1", "openai", 2.00, 0.50, 8.00),
    ModelProfile("gpt-4o-mini", "openai", 0.15, 0.075, 0.60),
    ModelProfile("gpt-4o", "openai", 2.50, 1.25, 10.00),
    ModelProfile("o3", "openai", 2.00, 0.50, 8.00),
    # Anthropic — prices as of PRICING_AS_OF.
    ModelProfile("claude-opus-4", "anthropic", 15.00, 1.50, 75.00),
    ModelProfile("claude-sonnet-4", "anthropic", 3.00, 0.30, 15.00),
    ModelProfile("claude-haiku-4", "anthropic", 1.00, 0.10, 5.00),
    ModelProfile("claude-3-5-haiku", "anthropic", 0.80, 0.08, 4.00),
]


def get_profile(model: str) -> ModelProfile:
    """Return the profile whose name is the longest prefix of ``model``.

    Unknown models get a profile with no pricing and no measured tolerance.
    """
    lowered = model.lower()
    best: ModelProfile | None = None
    for profile in _PROFILES:
        if lowered.startswith(profile.model) and (
            best is None or len(profile.model) > len(best.model)
        ):
            best = profile
    if best is not None:
        return best
    family = "unknown"
    if lowered.startswith(("gpt-", "o1", "o3", "o4")):
        family = "openai"
    elif lowered.startswith("claude"):
        family = "anthropic"
    return ModelProfile(model=model, family=family)


def register_profile(profile: ModelProfile) -> None:
    """Add or override a model profile (longest-prefix match wins)."""
    _PROFILES.insert(0, profile)


def keep_ratio_for(model: str) -> float:
    """The keep-ratio budgeted strategies should use for ``model``:
    the measured safe operating point when one exists, else the conservative
    default."""
    profile = get_profile(model)
    if profile.safe_keep_ratio is not None:
        return profile.safe_keep_ratio
    return DEFAULT_CONSERVATIVE_KEEP_RATIO


def estimate_cost(tokens: int, model: str, *, cached: bool = False) -> float | None:
    """USD cost of ``tokens`` input tokens for ``model``; ``None`` if unknown."""
    profile = get_profile(model)
    price = profile.cached_input_price if cached else profile.input_price
    if price is None:
        return None
    return tokens * price / 1_000_000
