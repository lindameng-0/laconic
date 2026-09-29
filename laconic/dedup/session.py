"""Session-scoped deduplication of repeated payload blocks.

Where the savings actually land (this is the part naive designs get wrong):
tokens are only spent when text enters a model's context. Rehydrating a handle
*before* the receiving model reads it saves nothing. So dedup has two explicit
modes, each with a precise claim about when it pays:

- **context-dedup** (default): a block is replaced by a short reference only
  when the caller supplies the identical block in the recipient's currently
  active context. Previous calls and recipient names are not evidence that
  content is still available after a reset, branch, or context compaction.
- **store-dedup** (opt-in): blocks are replaced by handles unconditionally and
  the receiving agent is given a ``rehydrate`` tool to fetch content on
  demand. Saves tokens whenever the receiver doesn't actually need the block;
  costs an extra tool round-trip when it does.

Prefix stability (ADR-6): dedup never rewrites previously sent messages —
only the *new outgoing* payload is transformed. Rewriting history would break
provider prompt-caching, whose cached input tokens are ~10x cheaper, and could
turn "savings" into a net cost increase.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from laconic.compress.segments import Segment, join_segments, segment_payload
from laconic.dedup.store import ContentStore, content_hash, make_handle
from laconic.tokenizers.base import TokenCounter

_BLOCK_SPLIT = re.compile(r"\n[ \t]*\n")


def _reference_text(handle: str, location: tuple[int, int]) -> str:
    message, paragraph = location
    return f"[ref {handle}: see earlier message {message}, paragraph {paragraph}]"


def _handle_text(handle: str) -> str:
    return f"[ref {handle}: call rehydrate('{handle}') for the full content]"


@dataclass
class DedupResult:
    """Outcome of deduplicating one payload."""

    text: str
    hits: int


class SessionDedup:
    """Deduplicates repeated blocks across a session's handoffs.

    Args:
        store: Content store backing rehydration (shared across the session).
        mode: ``"context"`` (default, safe) or ``"store"`` (requires giving the
            receiving agent the :meth:`rehydrate` tool).
        min_block_tokens: Blocks cheaper than this are never deduplicated —
            below ~25 tokens the reference text costs as much as the content.
    """

    def __init__(
        self,
        store: ContentStore | None = None,
        mode: str = "context",
        min_block_tokens: int = 25,
    ) -> None:
        if mode not in ("context", "store"):
            raise ValueError("mode must be 'context' or 'store'")
        self.store = store if store is not None else ContentStore()
        self.mode = mode
        self.min_block_tokens = min_block_tokens

    def process(
        self,
        payload: str,
        *,
        recipient: str,
        counter: TokenCounter,
        context_payloads: Sequence[str] | None = None,
    ) -> DedupResult:
        """Replace repeated blocks in ``payload`` for ``recipient``.

        ``context_payloads`` is the ordered content of messages actually in
        the recipient's current context, excluding this outgoing message.
        Use an empty string for non-text messages to preserve message indices.
        No context references are emitted without this evidence. ``recipient``
        is retained for compatibility, but never implies context membership.

        Protected segments are never replaced. Store-mode references require
        the caller to expose :meth:`rehydrate_tool` to the receiving agent.
        """
        locations = {
            block: (message_index, paragraph_index)
            for message_index, content in enumerate(context_payloads or (), start=1)
            for paragraph_index, block in enumerate(_BLOCK_SPLIT.split(content), start=1)
        }
        out: list[Segment] = []
        hits = 0
        for segment in segment_payload(payload):
            if segment.kind == "protected":
                out.append(segment)
                continue
            result = self._process_text(segment.text, counter=counter, locations=locations)
            out.append(Segment(kind="text", text=result.text))
            hits += result.hits
        return DedupResult(text=join_segments(out), hits=hits)

    def _process_text(
        self,
        payload: str,
        *,
        counter: TokenCounter,
        locations: dict[str, tuple[int, int]],
    ) -> DedupResult:
        blocks = _BLOCK_SPLIT.split(payload)
        separators = _BLOCK_SPLIT.findall(payload)
        out: list[str] = []
        hits = 0

        for block in blocks:
            digest = content_hash(block)
            replaceable = counter.count(block) >= self.min_block_tokens
            present = (
                block in locations if self.mode == "context" else make_handle(digest) in self.store
            )
            if replaceable and present:
                handle = make_handle(digest)
                reference = (
                    _reference_text(handle, locations[block])
                    if self.mode == "context"
                    else _handle_text(handle)
                )
                if counter.count(reference) < counter.count(block):
                    out.append(reference)
                    hits += 1
                else:
                    out.append(block)
            else:
                out.append(block)
            if replaceable:
                self.store.put(block)

        # Reassemble with the original separators (lossless when hits == 0).
        rebuilt: list[str] = []
        for index, block in enumerate(out):
            rebuilt.append(block)
            if index < len(separators):
                rebuilt.append(separators[index])
        return DedupResult(text="".join(rebuilt), hits=hits)

    def rehydrate_tool(self) -> Callable[[str], str]:
        """A ``rehydrate(handle) -> str`` callable to expose to receiving agents
        (store-dedup mode). Register it as a tool/function in your framework."""

        def rehydrate(handle: str) -> str:
            """Return the full content for a Laconic content handle."""
            return self.store.get(handle)

        return rehydrate
