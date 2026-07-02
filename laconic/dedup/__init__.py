"""Session-scoped dedup: context references and store handles."""

from laconic.dedup.session import DedupResult, SessionDedup
from laconic.dedup.store import ContentStore, content_hash, make_handle

__all__ = [
    "ContentStore",
    "DedupResult",
    "SessionDedup",
    "content_hash",
    "make_handle",
]
