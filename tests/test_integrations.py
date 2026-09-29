"""Integration hooks: generic message-list seam and the LangGraph shim."""

from __future__ import annotations

from laconic.compress.passthrough import PassthroughCompressor
from laconic.integrations.generic import CompressingHook
from laconic.integrations.langgraph import LangGraphCompressor, make_compression_node
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


def test_generic_default_preserves_history_and_uses_only_active_context() -> None:
    hook = CompressingHook(Session("test-model", compressor=PassthroughCompressor()))
    original = {"role": "assistant", "content": VERBOSE}
    first = hook.process_messages([original], target_agent="writer")
    assert first == [original]
    repeated = hook.process_messages([original, dict(original)], target_agent="writer")
    assert repeated[0] is original
    assert "[ref lc:" in repeated[1]["content"]
    assert "earlier message 1, paragraph 1" in repeated[1]["content"]
    reset = hook.process_messages([original], target_agent="writer")
    assert reset == [original]


def test_explicit_full_history_processing_does_not_invent_previous_messages() -> None:
    hook = CompressingHook(Session("test-model", compressor=PassthroughCompressor()))
    original = {"role": "assistant", "content": VERBOSE}
    for _ in range(2):
        result = hook.process_messages([original, dict(original)], only_new=None)
        assert result[0] == original
        assert "[ref lc:" in result[1]["content"]


def test_langgraph_node_does_not_share_context_between_fresh_runs() -> None:
    node = make_compression_node("test-model", compressor=PassthroughCompressor())
    original = {"type": "ai", "content": VERBOSE}
    for _ in range(2):
        result = node({"messages": [original]})
        assert result["messages"] == [original]
    result = node({"messages": [original, dict(original)]})
    assert result["messages"][0] is original
    assert "[ref lc:" in result["messages"][1]["content"]


def test_real_langgraph_reducer_preserves_ids_and_history() -> None:
    import pytest

    pytest.importorskip("langgraph")
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.graph import END, START, MessagesState, StateGraph

    graph = StateGraph(MessagesState)
    graph.add_node("compress", make_compression_node("test-model", strategy="conservative"))
    graph.add_edge(START, "compress")
    graph.add_edge("compress", END)
    app = graph.compile()
    result = app.invoke({"messages": [HumanMessage(content="start"), AIMessage(content=VERBOSE)]})
    messages = result["messages"]
    assert len(messages) == 2
    assert messages[0].content == "start"
    assert messages[1].content != VERBOSE
    assert len({message.id for message in messages}) == 2
    assert all(message.id for message in messages)
