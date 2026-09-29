"""The central ``Message`` abstraction and compression statistics.

Every inter-agent message is decomposed into:

- ``structural`` — protected fields that pass through byte-identical (tool
  calls, function arguments, IDs, schema keys, control fields).
- ``payload`` — the natural-language content eligible for compression.
- ``metadata`` — routing information (source/target agent, target model,
  framework tag).

Data contract: a compressor only ever receives ``payload``. It is structurally
impossible for a compressor to corrupt ``structural`` (design principle #3).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from laconic.tokenizers.base import CountTier


class MessageMetadata(BaseModel):
    """Routing and provenance information for a handoff."""

    source_agent: str | None = None
    target_agent: str | None = None
    target_model: str | None = None
    framework: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """A parsed inter-agent message, split into protected and compressible parts.

    Attributes:
        structural: All protected fields of the raw message, deep-copied and
            never modified by any compressor.
        payload: The natural-language text eligible for compression. Empty when
            the message has no compressible part (e.g. a pure tool-call message).
        payload_field: The key in the raw message the payload was extracted
            from, and where it is reinserted on rebuild.
        key_order: Original key order of the raw message, so a rebuild with an
            unchanged payload reproduces the raw message exactly.
        metadata: Routing information; never sent to any model.
    """

    model_config = ConfigDict(protected_namespaces=())

    structural: dict[str, Any] = Field(default_factory=dict)
    payload: str = ""
    payload_field: str | None = None
    key_order: list[str] = Field(default_factory=list)
    metadata: MessageMetadata = Field(default_factory=MessageMetadata)

    @property
    def has_payload(self) -> bool:
        """Whether this message carries any compressible text."""
        return bool(self.payload_field) and bool(self.payload)


class CompressionStats(BaseModel):
    """The record every processed handoff produces (design principle #5).

    Token counts always come from a :class:`~laconic.tokenizers.TokenCounter`;
    ``tier`` states their provenance so estimates are never mistaken for exact
    numbers.
    """

    model_config = ConfigDict(protected_namespaces=())

    model: str
    strategy: str
    tokens_before: int
    tokens_after: int
    tier: CountTier
    safe: bool = Field(
        default=True,
        description=(
            "Backward-compatible structural status for the returned message; "
            "does not imply semantic equivalence or downstream task success."
        ),
    )
    structure_preserved: bool = True
    protected_spans_preserved: bool = True
    fell_back: bool = False
    fallback_reason: str | None = None
    dedup_hits: int = 0

    @property
    def ratio(self) -> float:
        """Compressed/original token ratio; 1.0 means no savings."""
        if self.tokens_before == 0:
            return 1.0
        return self.tokens_after / self.tokens_before

    @property
    def tokens_saved(self) -> int:
        """Tokens removed by processing (never negative in fallback)."""
        return self.tokens_before - self.tokens_after
