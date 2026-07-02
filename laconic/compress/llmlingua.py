"""Optional adapter for the LLMLingua prompt-compression library.

LLMLingua performs learned token pruning with a small local model. It is tuned
for prose (documents, RAG context) — exactly why Laconic exists is that agent
traffic is *not* prose — but on long free-text payloads it can outperform the
extractive compressor. Inside Laconic it only ever sees payload text, and only
the free-text segments of it, so the structural fragility documented in
``docs/related-work.md`` does not apply here.

This module degrades gracefully: importing it never fails; constructing the
compressor without ``llmlingua`` installed raises
:class:`~laconic.exceptions.MissingDependencyError` with the pip extra to use.
"""

from __future__ import annotations

from typing import Any

from laconic.compress.base import Compressor
from laconic.compress.segments import Segment, join_segments, segment_payload
from laconic.exceptions import MissingDependencyError
from laconic.tokenizers.base import TokenCounter


class LLMLinguaCompressor(Compressor):
    """Adapter around ``llmlingua.PromptCompressor`` (optional dependency).

    Args:
        model_name: Compression model passed to LLMLingua. The default is
            LLMLingua-2's multilingual BERT-class model, small enough for CPU.
        device: Device map (``"cpu"`` default; ``"cuda"`` if available).
        default_budget: Keep-ratio used when no budget is passed.
    """

    name = "llmlingua"
    reversible = False

    def __init__(
        self,
        model_name: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        device: str = "cpu",
        default_budget: float = 0.6,
    ) -> None:
        try:
            from llmlingua import PromptCompressor
        except ImportError as exc:
            raise MissingDependencyError("LLMLingua compression", "llmlingua") from exc
        self.default_budget = default_budget
        self._compressor: Any = PromptCompressor(
            model_name=model_name,
            use_llmlingua2=True,
            device_map=device,
        )

    def compress(
        self,
        text: str,
        *,
        counter: TokenCounter,
        budget: float | None = None,
    ) -> str:
        if not text.strip():
            return text
        rate = budget if budget is not None else self.default_budget

        segments = segment_payload(text)
        out: list[Segment] = []
        for segment in segments:
            if segment.kind == "protected" or not segment.text.strip():
                out.append(segment)
                continue
            result = self._compressor.compress_prompt(segment.text, rate=rate)
            out.append(Segment(kind="text", text=result["compressed_prompt"]))
        candidate = join_segments(out)
        return self._no_worse(text, candidate, counter)
