"""Framework-agnostic integration: wrap any message-passing seam.

Most agent frameworks reduce to "a list of message dicts gets handed to the
next model call". :class:`CompressingHook` sits on that seam. The concrete
framework integrations stay thin by delegating here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from laconic.pipeline import Session
from laconic.profiler.records import WorkflowProfile


class CompressingHook:
    """Processes outgoing handoffs and accumulates a profile.

    Args:
        session: The configured :class:`~laconic.pipeline.Session`.
        profile: Optional shared :class:`~laconic.profiler.WorkflowProfile`;
            one is created if omitted.

    Example::

        hook = CompressingHook(Session(target_model="gpt-4.1"))
        cheap_messages = hook.process_messages(
            messages, source_agent="planner", target_agent="executor"
        )
        ...
        print(summary_table(hook.profile))
    """

    def __init__(self, session: Session, profile: WorkflowProfile | None = None) -> None:
        self.session = session
        self.profile = profile or WorkflowProfile()

    def process_message(
        self,
        raw: dict[str, Any],
        *,
        source_agent: str | None = None,
        target_agent: str | None = None,
        context_payloads: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Process one outgoing message; record stats; return what to send."""
        processed = self.session.process(
            raw,
            source_agent=source_agent,
            target_agent=target_agent,
            context_payloads=context_payloads,
        )
        self.profile.add(processed.stats, source_agent=source_agent, target_agent=target_agent)
        return processed.raw

    def process_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        source_agent: str | None = None,
        target_agent: str | None = None,
        only_new: int | None = 1,
    ) -> list[dict[str, Any]]:
        """Process a message list before a model call.

        Args:
            messages: The full message list about to be sent.
            only_new: Only the last ``only_new`` messages are processed
                (default 1); the actual untouched history provides context
                evidence. Explicit ``None`` processes the full list, using
                only already-emitted output as evidence for each message.
        """
        if only_new is not None and only_new < 0:
            raise ValueError("only_new must be non-negative or None")
        if only_new is None:
            head: list[dict[str, Any]] = []
            tail = list(messages)
        else:
            split = max(0, len(messages) - only_new)
            head = list(messages[:split])
            tail = list(messages[split:])
        output = head
        context = [
            message.get("content") if isinstance(message.get("content"), str) else ""
            for message in head
        ]
        for message in tail:
            processed = self.process_message(
                message,
                source_agent=source_agent,
                target_agent=target_agent,
                context_payloads=context,
            )
            output.append(processed)
            content = processed.get("content")
            context.append(content if isinstance(content, str) else "")
        return output
