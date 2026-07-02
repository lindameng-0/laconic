"""Protected-fields policy: what may never be compressed.

A field is *protected* when altering it can break the workflow even if the
meaning survives: tool calls, function arguments, IDs, JSON keys, schema
fields, control fields. The registry is configurable per framework, and the
default posture is conservative — anything not explicitly payload-eligible is
protected (design principle #3).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# Content that *looks* structured is protected even when it sits in a payload
# position: JSON documents, XML-ish blobs.
_JSON_LIKE = re.compile(r"^\s*[\[{]")
_TAG_HEAVY = re.compile(r"^\s*<[a-zA-Z][^>]*>")


def looks_structured(text: str) -> bool:
    """Heuristic: is this string structured data rather than prose?

    Strings that parse as JSON objects/arrays, or open with an XML/HTML tag,
    are treated as structural and never compressed.
    """
    if _JSON_LIKE.match(text):
        try:
            json.loads(text)
            return True
        except (json.JSONDecodeError, ValueError):
            # Not valid JSON — could still be prose starting with '['.
            return bool(_JSON_LIKE.match(text.strip()[:1]))
    return bool(_TAG_HEAVY.match(text))


@dataclass
class ProtectedFieldsPolicy:
    """Per-framework policy for splitting messages into structural vs payload.

    Attributes:
        framework: Framework tag this policy applies to.
        payload_field: The single key whose string value may be compressed.
        payload_roles: Roles whose payload is eligible; other roles pass
            through untouched. ``None`` means all roles are eligible.
        never_compress_structured_payload: When the payload string itself looks
            like structured data (JSON, XML), keep it protected.
        min_payload_chars: Payloads shorter than this are not worth touching
            and pass through.
    """

    framework: str
    payload_field: str = "content"
    payload_roles: frozenset[str] | None = None
    never_compress_structured_payload: bool = True
    min_payload_chars: int = 80

    def payload_eligible(self, role: str, value: object) -> bool:
        """Decide whether ``value`` (the payload field of a message with
        ``role``) may be handed to a compressor."""
        if not isinstance(value, str):
            return False
        if len(value) < self.min_payload_chars:
            return False
        if self.payload_roles is not None and role not in self.payload_roles:
            return False
        return not (self.never_compress_structured_payload and looks_structured(value))


@dataclass
class ProtectedFieldsRegistry:
    """Registry of :class:`ProtectedFieldsPolicy` by framework tag."""

    policies: dict[str, ProtectedFieldsPolicy] = field(default_factory=dict)

    def register(self, policy: ProtectedFieldsPolicy) -> None:
        """Add or replace the policy for a framework."""
        self.policies[policy.framework] = policy

    def get(self, framework: str) -> ProtectedFieldsPolicy:
        """Return the policy for ``framework``, or a conservative default."""
        if framework in self.policies:
            return self.policies[framework]
        return ProtectedFieldsPolicy(framework=framework)


#: Default registry used by the pipeline. Tool-role messages usually carry
#: machine-readable results, so they are excluded from compression by default
#: for the OpenAI chat format; LangChain-style messages behave the same way.
DEFAULT_REGISTRY = ProtectedFieldsRegistry()
DEFAULT_REGISTRY.register(
    ProtectedFieldsPolicy(
        framework="openai-chat",
        payload_field="content",
        payload_roles=frozenset({"user", "assistant", "system", "developer"}),
    )
)
DEFAULT_REGISTRY.register(
    ProtectedFieldsPolicy(
        framework="langchain",
        payload_field="content",
        payload_roles=frozenset({"human", "ai", "system"}),
    )
)
