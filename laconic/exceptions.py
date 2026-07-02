"""Exception types for Laconic.

The pipeline is designed so that *no* exception escapes into a user's workflow:
errors during parsing, compression, or verification trigger a passthrough
fallback that is recorded in stats. These exception types exist for internal
signalling and for users who call low-level APIs directly.
"""

from __future__ import annotations


class LaconicError(Exception):
    """Base class for all Laconic errors."""


class ParseError(LaconicError):
    """A raw message could not be split into structural/payload parts."""


class RebuildMismatchError(LaconicError):
    """A rebuilt message failed the structural round-trip verification."""


class MissingDependencyError(LaconicError, ImportError):
    """An optional dependency is required for the requested feature.

    Raised with an actionable message naming the pip extra to install.
    """

    def __init__(self, feature: str, extra: str) -> None:
        super().__init__(
            f"{feature} requires an optional dependency. "
            f'Install it with: pip install "laconic[{extra}]"'
        )
        self.feature = feature
        self.extra = extra


class UnknownHandleError(LaconicError):
    """A dedup handle was not found in the content store."""
