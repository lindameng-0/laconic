"""Framework message parsers with a round-trip guarantee.

The invariant these parsers must uphold (and that ``tests/test_roundtrip.py``
enforces over a synthetic corpus):

    ``rebuild(parse(raw))`` with an unchanged payload **equals** ``raw``
    exactly — same keys, same order, structural content byte-identical.

If a message cannot be parsed confidently, :class:`~laconic.exceptions.ParseError`
is raised and the pipeline passes the message through uncompressed. Failure
mode is always "no savings", never "broken workflow".
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any

from laconic.exceptions import ParseError
from laconic.message.model import Message, MessageMetadata
from laconic.message.protected import DEFAULT_REGISTRY, ProtectedFieldsRegistry


class MessageParser(ABC):
    """Splits a raw framework message into ``structural`` and ``payload``."""

    framework: str

    def __init__(self, registry: ProtectedFieldsRegistry | None = None) -> None:
        self.registry = registry or DEFAULT_REGISTRY

    @abstractmethod
    def parse(self, raw: dict[str, Any]) -> Message:
        """Decompose ``raw`` into a :class:`Message`.

        Raises:
            ParseError: If the message shape is not recognized.
        """

    @abstractmethod
    def rebuild(self, message: Message, payload: str | None = None) -> dict[str, Any]:
        """Reassemble a raw message from ``message``, substituting ``payload``
        (or the original payload when ``None``)."""


class OpenAIChatParser(MessageParser):
    """Parser for the OpenAI chat-completions message format.

    Payload is the ``content`` field when it is a plain string on an eligible
    role. Everything else — ``tool_calls``, ``tool_call_id``, ``name``,
    multimodal content parts, unknown keys — is structural and passes through
    untouched.
    """

    framework = "openai-chat"

    def parse(self, raw: dict[str, Any]) -> Message:
        if not isinstance(raw, dict):
            raise ParseError(f"expected a dict message, got {type(raw).__name__}")
        if "role" not in raw or not isinstance(raw["role"], str):
            raise ParseError("message has no string 'role' field")

        policy = self.registry.get(self.framework)
        role = raw["role"]
        structural = {k: copy.deepcopy(v) for k, v in raw.items()}
        payload = ""
        payload_field: str | None = None

        value = raw.get(policy.payload_field)
        if policy.payload_eligible(role, value):
            payload = structural.pop(policy.payload_field)
            payload_field = policy.payload_field

        return Message(
            structural=structural,
            payload=payload,
            payload_field=payload_field,
            key_order=list(raw.keys()),
            metadata=MessageMetadata(framework=self.framework),
        )

    def rebuild(self, message: Message, payload: str | None = None) -> dict[str, Any]:
        text = message.payload if payload is None else payload
        out: dict[str, Any] = {}
        for key in message.key_order:
            if key == message.payload_field:
                out[key] = text
            elif key in message.structural:
                out[key] = copy.deepcopy(message.structural[key])
        # Keys not present at parse time cannot appear; keys added to
        # structural after parse (none, by contract) are ignored.
        return out


class LangChainParser(OpenAIChatParser):
    """Parser for LangChain/LangGraph message dicts.

    Operates on the dict form of ``BaseMessage`` (``message.model_dump()`` or
    the ``{"type": ..., "content": ...}`` shape). The LangGraph integration
    converts message objects to dicts before calling this parser and back
    after, keeping this module free of optional dependencies.
    """

    framework = "langchain"

    def parse(self, raw: dict[str, Any]) -> Message:
        if not isinstance(raw, dict):
            raise ParseError(f"expected a dict message, got {type(raw).__name__}")
        role = raw.get("type") or raw.get("role")
        if not isinstance(role, str):
            raise ParseError("message has no string 'type'/'role' field")

        policy = self.registry.get(self.framework)
        structural = {k: copy.deepcopy(v) for k, v in raw.items()}
        payload = ""
        payload_field: str | None = None

        value = raw.get(policy.payload_field)
        # Messages carrying tool calls keep their content protected too:
        # models often interleave rationale with calls, and the safest default
        # is not to touch a message that triggers tools.
        has_tool_calls = bool(raw.get("tool_calls"))
        if not has_tool_calls and policy.payload_eligible(role, value):
            payload = structural.pop(policy.payload_field)
            payload_field = policy.payload_field

        return Message(
            structural=structural,
            payload=payload,
            payload_field=payload_field,
            key_order=list(raw.keys()),
            metadata=MessageMetadata(framework=self.framework),
        )


_PARSERS: dict[str, type[MessageParser]] = {
    OpenAIChatParser.framework: OpenAIChatParser,
    LangChainParser.framework: LangChainParser,
}


def get_parser(framework: str, registry: ProtectedFieldsRegistry | None = None) -> MessageParser:
    """Return a parser instance for ``framework``.

    Raises:
        ParseError: If no parser is registered for the framework.
    """
    try:
        parser_cls = _PARSERS[framework]
    except KeyError as exc:
        known = ", ".join(sorted(_PARSERS))
        raise ParseError(f"no parser for framework '{framework}' (known: {known})") from exc
    return parser_cls(registry=registry)


def register_parser(parser_cls: type[MessageParser]) -> None:
    """Register a custom :class:`MessageParser` subclass by its framework tag."""
    _PARSERS[parser_cls.framework] = parser_cls
