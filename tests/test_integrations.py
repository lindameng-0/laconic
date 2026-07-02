"""Integration hooks: generic message-list seam and the LangGraph shim."""

from __future__ import annotations

from laconic.integrations.generic import CompressingHook
from laconic.integrations.langgraph import LangGraphCompressor
from laconic.pipeline import Session

VERBOSE = (
    "Please note that the summary below is complete. It is worth noting "
    "that, for example, conversion improved by 2.4 points in the checkout "
    "flow. Basically, the experiment succeeded. In order to ship it, we "
    "need approval from the growth team before Friday."
)


def test_hook_processes_and_profiles() -> None:
    hook = CompressingHook(Session(target_model="test-model"))
    out = hook.process_message(
        {"role": "assistant", "content": VERBOSE},
        source_agent="analyst",
        target_agent="pm",
    )
    assert out["role"] == "assistant"
    assert len(hook.profile.records) == 1
    assert hook.profile.tokens_saved > 0


def test_only_new_keeps_prefix_stable() -> None:
    """Messages before the split point must pass through *unprocessed* —
    rewriting them would break provider prompt caching."""
    hook = CompressingHook(Session(target_model="test-model"))
    history = [
        {"role": "user", "content": VERBOSE + " (turn 1)"},
        {"role": "assistant", "content": VERBOSE + " (turn 2)"},
        {"role": "assistant", "content": VERBOSE + " (turn 3, new)"},
    ]
    out = hook.process_messages(history, only_new=1)
    assert out[0] is history[0]  # untouched, same object
    assert out[1] is history[1]
    assert len(hook.profile.records) == 1  # only the new message was processed


def test_langgraph_compressor_with_dict_messages() -> None:
    compressor = LangGraphCompressor(target_model="test-model", only_new=1)
    messages = [
        {"type": "human", "content": "start"},
        {"type": "ai", "content": VERBOSE},
    ]
    out = compressor.compress_messages(messages, source_agent="planner", target_agent="executor")
    assert len(out) == 2
    assert out[0] is messages[0]
    assert compressor.profile.tokens_saved > 0


def test_langgraph_node_shape() -> None:
    compressor = LangGraphCompressor(target_model="test-model")
    node = compressor.as_node("a", "b")
    result = node({"messages": [{"type": "ai", "content": VERBOSE}]})
    assert "messages" in result
    assert len(result["messages"]) == 1


def test_tool_call_messages_pass_through_langgraph() -> None:
    compressor = LangGraphCompressor(target_model="test-model")
    message = {
        "type": "ai",
        "content": VERBOSE,
        "tool_calls": [{"name": "search", "args": {"q": "x"}, "id": "1"}],
    }
    out = compressor.compress_messages([message])
    assert out[0] == message  # tool-call messages are protected wholesale
