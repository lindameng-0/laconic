"""Content-addressed store for deduplicated blocks."""

from __future__ import annotations

import hashlib

from laconic.exceptions import UnknownHandleError

_HANDLE_PREFIX = "lc"
_HANDLE_LEN = 10


def content_hash(text: str) -> str:
    """Stable short hash of a content block (sha256, first 10 hex chars)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:_HANDLE_LEN]


def make_handle(digest: str) -> str:
    """Format a digest as a handle string, e.g. ``lc:3f2a9c81d0``."""
    return f"{_HANDLE_PREFIX}:{digest}"


class ContentStore:
    """In-memory content-addressed store, scoped to one session.

    Blocks are stored under their content hash; :meth:`get` rehydrates a
    handle back to the original text. The store is deliberately simple — for
    multi-process workflows, subclass and back it with your own storage.
    """

    def __init__(self) -> None:
        self._blocks: dict[str, str] = {}

    def put(self, text: str) -> str:
        """Store ``text``; return its handle (idempotent)."""
        digest = content_hash(text)
        self._blocks[digest] = text
        return make_handle(digest)

    def get(self, handle: str) -> str:
        """Rehydrate a handle.

        Raises:
            UnknownHandleError: If the handle is not in this store.
        """
        digest = handle.split(":", 1)[-1]
        try:
            return self._blocks[digest]
        except KeyError as exc:
            raise UnknownHandleError(f"unknown content handle: {handle}") from exc

    def __contains__(self, handle: str) -> bool:
        return handle.split(":", 1)[-1] in self._blocks

    def __len__(self) -> int:
        return len(self._blocks)
