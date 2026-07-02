"""Session-scoped deduplication of repeated payload blocks.

Where the savings actually land (this is the part naive designs get wrong):
tokens are only spent when text enters a model's context. Rehydrating a handle
*before* the receiving model reads it saves nothing. So dedup has two explicit
modes, each with a precise claim about when it pays:

- **context-dedup** (default, safe): a block is replaced by a short reference
  *only when the same recipient already received the identical block earlier
  in the session* — the model can look back at its own context to resolve the
  reference. Saves tokens on every repetition after the first.
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
from collections.abc import Callable
from dataclasses import dataclass

from laconic.dedup.store import ContentStore, content_hash, make_handle
from laconic.tokenizers.base import TokenCounter

_BLOCK_SPLIT = re.compile(r"\n[ \t]*\n")


def _reference_text(handle: str) -> str:
    return f"[ref {handle}: unchanged from earlier in this conversation]"


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
        self.store = store or ContentStore()
        self.mode = mode
        self.min_block_tokens = min_block_tokens
        # digest -> True, per recipient: what each recipient has already seen.
        self._seen_by_recipient: dict[str, set[str]] = {}

    def process(
        self,
        payload: str,
        *,
        recipient: str,
        counter: TokenCounter,
    ) -> DedupResult:
        """Replace repeated blocks in ``payload`` for ``recipient``.

        Only blocks the reference is actually cheaper than (in tokens, per
        ``counter``) are replaced. The original payload is stored before any
        replacement so rehydration is always possible.
        """
        seen = self._seen_by_recipient.setdefault(recipient, set())
        blocks = _BLOCK_SPLIT.split(payload)
        separators = _BLOCK_SPLIT.findall(payload)
        out: list[str] = []
        hits = 0

        for block in blocks:
            digest = content_hash(block)
            replaceable = counter.count(block) >= self.min_block_tokens
            if replaceable and self._should_replace(digest, seen):
                handle = make_handle(digest)
                reference = (
                    _reference_text(handle) if self.mode == "context" else _handle_text(handle)
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
                seen.add(digest)

        # Reassemble with the original separators (lossless when hits == 0).
        rebuilt: list[str] = []
        for index, block in enumerate(out):
            rebuilt.append(block)
            if index < len(separators):
                rebuilt.append(separators[index])
        return DedupResult(text="".join(rebuilt), hits=hits)

    def _should_replace(self, digest: str, seen: set[str]) -> bool:
        if self.mode == "context":
            return digest in seen  # recipient has it earlier in context
        return make_handle(digest) in self.store  # store mode: any prior sighting

    def rehydrate_tool(self) -> Callable[[str], str]:
        """A ``rehydrate(handle) -> str`` callable to expose to receiving agents
        (store-dedup mode). Register it as a tool/function in your framework."""

        def rehydrate(handle: str) -> str:
            """Return the full content for a Laconic content handle."""
            return self.store.get(handle)

        return rehydrate
