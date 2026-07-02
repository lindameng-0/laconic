"""Model clients for the eval harness.

The runner renders every (possibly compressed) handoff into the receiver's
user turn as JSON text, followed by the task question — the same rendering
for every client, so mock and real runs are comparable.

Two kinds of client:

- :class:`MockModelClient` — deterministic, offline, free. It behaves as a
  *perfect extractor*: it answers strictly from what is literally present in
  the handoff text it received. What it measures is therefore **information
  survival** through compression — did the load-bearing content reach the
  receiver at all — not model comprehension. CI runs entirely on this client,
  and results produced with it are labeled ``mock`` everywhere.
- :class:`OpenAIClient` / :class:`AnthropicClient` — thin adapters over the
  real APIs for measuring actual model behavior. These produce the numbers
  the research claims need; they require keys and are never invoked by tests.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from laconic.eval.tasks import first_json_object
from laconic.exceptions import MissingDependencyError


class ModelClient(Protocol):
    """Minimal completion interface the runner needs."""

    def complete(self, messages: list[dict[str, Any]], model: str) -> str:
        """Return the assistant's text answer for a chat exchange."""
        ...


#: Rendering contract shared by the runner and the mock client: the receiver's
#: user turn is ``HANDOFF_HEADER + handoff JSON + TASK_HEADER + question``.
HANDOFF_HEADER = "Another agent sent you this handoff message (JSON):"
TASK_HEADER = "Your task:"


class MockModelClient:
    """Deterministic offline receiver — a perfect extractor, nothing more.

    Answering rules (purely mechanical):

    - Parse the handoff JSON embedded in the user turn. If it carries
      ``tool_calls``, restate the first call as ``{"name": ...,
      "arguments": {...}}`` — but only if its structure still parses.
      Corrupted structure yields a visible failure, exactly as a real
      executor would produce one.
    - Otherwise, quote every sentence of the handoff content that shares
      enough rare words with the question, plus every sentence flagged as a
      requirement. Content that compression pruned away cannot be quoted.
    """

    name = "mock"

    def complete(self, messages: list[dict[str, Any]], model: str) -> str:
        user_text = self._last_user_text(messages)
        handoff_text, _, question = user_text.rpartition(TASK_HEADER)
        if not handoff_text:
            handoff_text, question = user_text, user_text
        handoff_text = handoff_text.replace(HANDOFF_HEADER, "", 1)

        handoff = first_json_object(handoff_text)
        if handoff is not None and handoff.get("tool_calls"):
            return self._restate_tool_call(handoff["tool_calls"])
        if handoff is not None and isinstance(handoff.get("content"), str):
            content = handoff["content"]
        else:
            # Structure did not survive (naive compression): fall back to
            # treating the handoff portion as prose.
            content = handoff_text
        return self._extract(content, question)

    @staticmethod
    def _last_user_text(messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                return message["content"]
        return ""

    @staticmethod
    def _restate_tool_call(tool_calls: Any) -> str:
        if not isinstance(tool_calls, list) or not tool_calls:
            return "ERROR: tool call structure is malformed"
        call = tool_calls[0]
        function = call.get("function", {}) if isinstance(call, dict) else {}
        name = function.get("name")
        raw_args = function.get("arguments", "")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            return f"ERROR: tool call arguments are not valid JSON: {raw_args!r}"
        if not isinstance(name, str) or not isinstance(args, dict):
            return "ERROR: tool call structure is malformed"
        return json.dumps({"name": name, "arguments": args})

    @staticmethod
    def _extract(content: str, question: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", content)
        stop = frozenset(
            [
                "a",
                "an",
                "the",
                "and",
                "or",
                "of",
                "to",
                "in",
                "on",
                "at",
                "by",
                "for",
                "with",
                "from",
                "as",
                "is",
                "are",
                "was",
                "were",
                "be",
                "i",
                "you",
                "we",
                "they",
                "it",
                "this",
                "that",
                "these",
                "those",
                "what",
                "which",
                "who",
                "how",
                "much",
                "many",
                "exact",
                "exactly",
                "include",
                "including",
                "keep",
                "all",
                "every",
                "from",
                "received",
                "another",
                "agent",
                "sent",
                "message",
                "json",
                "your",
                "task",
                "numbers",
                "report",
            ]
        )
        question_words = {w for w in re.findall(r"[a-z0-9.]+", question.lower()) if w not in stop}
        picked: list[str] = []
        for sentence in sentences:
            lowered = sentence.lower()
            words = set(re.findall(r"[a-z0-9.]+", lowered))
            if "requirement" in lowered or len(words & question_words) >= 3:
                picked.append(sentence)
        return " ".join(picked) if picked else "the received content does not say"


class OpenAIClient:
    """Real OpenAI chat-completions client (requires the ``openai`` package + key)."""

    def __init__(self, client: Any | None = None, temperature: float = 0.0) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise MissingDependencyError("OpenAI eval client", "openai") from exc
            client = OpenAI()
        self._client = client
        self.temperature = temperature

    def complete(self, messages: list[dict[str, Any]], model: str) -> str:
        response = self._client.chat.completions.create(
            model=model, messages=messages, temperature=self.temperature
        )
        return response.choices[0].message.content or ""


class AnthropicClient:
    """Real Anthropic messages client (requires ``laconic[anthropic]`` + key)."""

    def __init__(self, client: Any | None = None, max_tokens: int = 1024) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise MissingDependencyError("Anthropic eval client", "anthropic") from exc
            client = anthropic.Anthropic()
        self._client = client
        self.max_tokens = max_tokens

    def complete(self, messages: list[dict[str, Any]], model: str) -> str:
        system = "\n".join(m["content"] for m in messages if m.get("role") == "system")
        chat = [m for m in messages if m.get("role") != "system"]
        response = self._client.messages.create(
            model=model,
            system=system or None,
            messages=chat,
            max_tokens=self.max_tokens,
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
