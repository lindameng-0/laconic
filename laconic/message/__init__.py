"""Message decomposition: structural (protected) vs payload (compressible)."""

from laconic.message.model import CompressionStats, Message, MessageMetadata
from laconic.message.parsers import (
    LangChainParser,
    MessageParser,
    OpenAIChatParser,
    get_parser,
    register_parser,
)
from laconic.message.protected import (
    DEFAULT_REGISTRY,
    ProtectedFieldsPolicy,
    ProtectedFieldsRegistry,
    looks_structured,
)

__all__ = [
    "DEFAULT_REGISTRY",
    "CompressionStats",
    "LangChainParser",
    "Message",
    "MessageMetadata",
    "MessageParser",
    "OpenAIChatParser",
    "ProtectedFieldsPolicy",
    "ProtectedFieldsRegistry",
    "get_parser",
    "looks_structured",
    "register_parser",
]
