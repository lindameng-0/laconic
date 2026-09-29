"""The processing pipeline: parse → dedup → compress → verify → rebuild.

The pipeline preserves message fields and recognized protected payload spans.
Failures pass the original message through, with the fallback recorded in stats.
These checks do not establish semantic equivalence or downstream task success.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from laconic.adapters.profiles import keep_ratio_for
from laconic.compress.base import Compressor
from laconic.compress.extractive import ExtractiveCompressor
from laconic.compress.passthrough import PassthroughCompressor
from laconic.compress.segments import Segment, join_segments, segment_payload
from laconic.compress.telegraphic import TelegraphicCompressor
from laconic.dedup.session import SessionDedup
from laconic.exceptions import LaconicError
from laconic.message.model import CompressionStats
from laconic.message.parsers import get_parser
from laconic.message.protected import ProtectedFieldsRegistry
from laconic.tokenizers.base import CountTier, TokenCounter
from laconic.tokenizers.heuristic import HeuristicCounter
from laconic.tokenizers.registry import get_counter

#: Built-in strategy names accepted by :class:`Session`.
STRATEGIES = ("off", "conservative", "telegraphic", "balanced", "aggressive")


@dataclass(frozen=True)
class ProcessedMessage:
    """A processed handoff: the message to send, plus its stats."""

    raw: dict[str, Any]
    stats: CompressionStats


def _structural_fingerprint(message_raw: dict[str, Any], payload_field: str | None) -> str:
    """Canonical serialization of everything except the payload field."""
    structural = {k: v for k, v in message_raw.items() if k != payload_field}
    return json.dumps(structural, sort_keys=True, ensure_ascii=False, default=str)


class Session:
    """A compression session for one workflow run.

    Holds the dedup state, the target-model token counter, and the chosen
    strategy. One :class:`Session` per workflow execution; recipients are
    tracked inside it.

    Args:
        target_model: The model that will *read* the processed messages —
            savings are measured with its tokenizer and the budget comes from
            its profile.
        strategy: ``"off"`` (measure only), ``"conservative"`` (lossless-in-
            meaning cleanup only), ``"telegraphic"`` (the default: conservative
            plus telegram-style function-word dropping — content words,
            numbers, names, negations, and structure always survive; see
            ADR-8), ``"balanced"`` (budgeted pruning at the model's safe
            keep-ratio), ``"aggressive"`` (budgeted at
            ``aggressive_keep_ratio``) — or pass a custom ``compressor``.
        framework: Message format tag (``"openai-chat"`` or ``"langchain"``).
        compressor: Custom :class:`~laconic.compress.Compressor`; overrides
            ``strategy``.
        dedup: Enable session dedup (context mode). Pass a configured
            :class:`~laconic.dedup.SessionDedup` for store mode.
        allow_api_counting: Permit network calls for exact Anthropic counts.
        keep_ratio: Explicit keep-ratio budget; overrides the profile's.
        registry: Custom protected-fields registry.
    """

    aggressive_keep_ratio = 0.5

    def __init__(
        self,
        target_model: str,
        strategy: str = "telegraphic",
        framework: str = "openai-chat",
        compressor: Compressor | None = None,
        dedup: bool | SessionDedup = True,
        allow_api_counting: bool = False,
        keep_ratio: float | None = None,
        registry: ProtectedFieldsRegistry | None = None,
        counter: TokenCounter | None = None,
    ) -> None:
        if compressor is None and strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES} (got {strategy!r})")
        self.target_model = target_model
        self.strategy_name = compressor.name if compressor else strategy
        self.framework = framework
        self.counter = counter or get_counter(target_model, allow_api=allow_api_counting)
        # Candidate generation must not call a remote counter for every word
        # substitution. Provider counting validates the before/after pair only.
        self._search_counter = (
            HeuristicCounter(target_model) if self.counter.tier == CountTier.API else self.counter
        )
        self.parser = get_parser(framework, registry=registry)

        if compressor is not None:
            self.compressor: Compressor = compressor
            self._budget = keep_ratio
        elif strategy == "off":
            self.compressor = PassthroughCompressor()
            self._budget = None
        elif strategy == "conservative":
            self.compressor = ExtractiveCompressor(aggressive=False)
            self._budget = None
        elif strategy == "telegraphic":
            self.compressor = TelegraphicCompressor()
            self._budget = None
        elif strategy == "balanced":
            self.compressor = ExtractiveCompressor(aggressive=True)
            self._budget = keep_ratio if keep_ratio is not None else keep_ratio_for(target_model)
        else:  # aggressive
            self.compressor = ExtractiveCompressor(aggressive=True)
            self._budget = keep_ratio if keep_ratio is not None else self.aggressive_keep_ratio

        if isinstance(dedup, SessionDedup):
            self.dedup: SessionDedup | None = dedup
        elif dedup:
            self.dedup = SessionDedup()
        else:
            self.dedup = None

    def process(
        self,
        raw: dict[str, Any],
        *,
        source_agent: str | None = None,
        target_agent: str | None = None,
        context_payloads: Sequence[str] | None = None,
    ) -> ProcessedMessage:
        """Process one outgoing handoff message.

        Returns the message to actually send plus its
        :class:`~laconic.message.CompressionStats`. Parse and transformation
        errors fall back to passthrough. A failed initial token count propagates
        to the caller because no valid measurement is available.

        ``context_payloads`` must contain the recipient's actual current
        history, in message order, excluding ``raw``. Without it context dedup
        does not replace blocks. The caller must not supply stale history.
        """
        tokens_before = self._count_message(raw)
        try:
            return self._process_inner(
                raw,
                tokens_before=tokens_before,
                source_agent=source_agent,
                target_agent=target_agent,
                context_payloads=context_payloads,
            )
        except LaconicError as exc:
            return self._fallback(raw, tokens_before, f"{type(exc).__name__}: {exc}")
        except Exception as exc:
            return self._fallback(raw, tokens_before, f"unexpected {type(exc).__name__}: {exc}")

    def _process_inner(
        self,
        raw: dict[str, Any],
        *,
        tokens_before: int,
        source_agent: str | None,
        target_agent: str | None,
        context_payloads: Sequence[str] | None,
    ) -> ProcessedMessage:
        if self.strategy_name == "off":
            return ProcessedMessage(
                raw=raw, stats=self._stats(tokens_before, tokens_before, safe=True)
            )
        message = self.parser.parse(raw)
        message.metadata.source_agent = source_agent
        message.metadata.target_agent = target_agent
        message.metadata.target_model = self.target_model

        if not message.has_payload:
            # Nothing compressible (tool call, structured content, tiny text).
            return ProcessedMessage(
                raw=raw,
                stats=self._stats(tokens_before, tokens_before, safe=True),
            )

        segments = segment_payload(message.payload)
        protected = [segment.text for segment in segments if segment.kind == "protected"]
        output: list[Segment] = []
        dedup_hits = 0
        for segment in segments:
            if segment.kind == "protected" or not segment.text.strip():
                output.append(segment)
                continue
            text = segment.text
            hits = 0
            if self.dedup is not None:
                result = self.dedup.process(
                    text,
                    recipient=target_agent or "default",
                    counter=self._search_counter,
                    context_payloads=context_payloads,
                )
                text, hits = result.text, result.hits
                dedup_hits += hits
            # A generated reference is a protocol token, not compressible prose.
            if not hits:
                text = self.compressor.compress(
                    text, counter=self._search_counter, budget=self._budget
                )
            output.append(Segment(kind="text", text=text))
        payload = join_segments(output)

        protected_after = [
            segment.text for segment in segment_payload(payload) if segment.kind == "protected"
        ]
        if protected_after != protected:
            return self._fallback(raw, tokens_before, "protected payload span mismatch")

        rebuilt = self.parser.rebuild(message, payload=payload)

        # Verification: structure must be byte-identical, and the rebuilt
        # message must re-parse cleanly.
        before_fp = _structural_fingerprint(raw, message.payload_field)
        after_fp = _structural_fingerprint(rebuilt, message.payload_field)
        if before_fp != after_fp or list(rebuilt.keys()) != list(raw.keys()):
            return self._fallback(raw, tokens_before, "structural round-trip mismatch")
        self.parser.parse(rebuilt)  # raises ParseError -> fallback in process()

        tokens_after = self._count_message(rebuilt)
        if tokens_after > tokens_before:
            return self._fallback(raw, tokens_before, "processing increased token count")

        stats = self._stats(tokens_before, tokens_after, safe=True, dedup_hits=dedup_hits)
        return ProcessedMessage(raw=rebuilt, stats=stats)

    def _count_message(self, raw: dict[str, Any]) -> int:
        return self.counter.count(json.dumps(raw, ensure_ascii=False, default=str))

    def _stats(
        self,
        tokens_before: int,
        tokens_after: int,
        *,
        safe: bool,
        dedup_hits: int = 0,
        fell_back: bool = False,
        fallback_reason: str | None = None,
    ) -> CompressionStats:
        return CompressionStats(
            model=self.target_model,
            strategy=self.strategy_name,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tier=self.counter.tier,
            safe=safe,
            structure_preserved=safe,
            protected_spans_preserved=safe,
            fell_back=fell_back,
            fallback_reason=fallback_reason,
            dedup_hits=dedup_hits,
        )

    def _fallback(self, raw: dict[str, Any], tokens_before: int, reason: str) -> ProcessedMessage:
        return ProcessedMessage(
            raw=raw,
            stats=self._stats(
                tokens_before,
                tokens_before,
                safe=True,  # passthrough is always safe
                fell_back=True,
                fallback_reason=reason,
            ),
        )
