"""LangGraph integration (thin by design — the logic lives in the core).

LangGraph state usually carries a ``messages`` list. The seam Laconic hooks
is the edge between agents: wrap a node's outgoing update with
:func:`compress_state_messages`, or insert :func:`make_compression_node` as an
explicit graph node between two agents.

The module imports without LangChain installed; only object↔dict conversion
needs ``langchain-core`` and is guarded. Plain dict messages work with no
optional dependencies at all.
"""

from __future__ import annotations

from typing import Any

from laconic.exceptions import MissingDependencyError
from laconic.integrations.generic import CompressingHook
from laconic.pipeline import Session
from laconic.profiler.records import WorkflowProfile


def _to_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return message
    if hasattr(message, "model_dump"):  # langchain-core BaseMessage
        return message.model_dump()
    raise TypeError(f"cannot convert {type(message).__name__} to a message dict")


def _from_dict(raw: dict[str, Any], like: Any) -> Any:
    """Rebuild a message object of the same class as ``like`` from ``raw``."""
    if isinstance(like, dict):
        return raw
    try:
        return type(like)(**{k: v for k, v in raw.items() if k != "type"})
    except Exception as exc:
        raise MissingDependencyError("LangGraph message conversion", "langgraph") from exc


class LangGraphCompressor:
    """Compresses the ``messages`` channel of a LangGraph state.

    Args:
        target_model: Model that reads the compressed messages.
        strategy: See :class:`~laconic.pipeline.Session`.
        only_new: How many trailing messages to process per step (default 1 —
            the newly produced handoff). Keeps the prefix stable for provider
            caching.
        **session_kwargs: Forwarded to :class:`~laconic.pipeline.Session`.
    """

    def __init__(
        self,
        target_model: str,
        strategy: str = "conservative",
        only_new: int = 1,
        **session_kwargs: Any,
    ) -> None:
        session = Session(
            target_model=target_model,
            strategy=strategy,
            framework="langchain",
            **session_kwargs,
        )
        self.hook = CompressingHook(session)
        self.only_new = only_new

    @property
    def profile(self) -> WorkflowProfile:
        """Accumulated token measurements for this graph run."""
        return self.hook.profile

    def compress_messages(
        self,
        messages: list[Any],
        *,
        source_agent: str | None = None,
        target_agent: str | None = None,
    ) -> list[Any]:
        """Compress the trailing ``only_new`` messages of a message list.

        Accepts LangChain message objects or plain dicts; returns the same
        types it was given.
        """
        if not messages:
            return messages
        split = max(0, len(messages) - self.only_new)
        head = list(messages[:split])
        tail = []
        for message in messages[split:]:
            raw = _to_dict(message)
            processed = self.hook.process_message(
                raw, source_agent=source_agent, target_agent=target_agent
            )
            tail.append(message if processed is raw else _from_dict(processed, message))
        return head + tail

    def as_node(self, source_agent: str | None = None, target_agent: str | None = None) -> Any:
        """Return a LangGraph-compatible node: ``state -> {"messages": [...]}``.

        Insert between two agents::

            graph.add_node("laconic", compressor.as_node("planner", "executor"))
            graph.add_edge("planner", "laconic")
            graph.add_edge("laconic", "executor")
        """

        def node(state: dict[str, Any]) -> dict[str, Any]:
            messages = state.get("messages", [])
            return {
                "messages": self.compress_messages(
                    messages, source_agent=source_agent, target_agent=target_agent
                )
            }

        return node


def make_compression_node(
    target_model: str,
    strategy: str = "conservative",
    source_agent: str | None = None,
    target_agent: str | None = None,
    **session_kwargs: Any,
) -> Any:
    """One-call convenience: a ready-to-insert LangGraph compression node."""
    compressor = LangGraphCompressor(target_model=target_model, strategy=strategy, **session_kwargs)
    return compressor.as_node(source_agent=source_agent, target_agent=target_agent)
