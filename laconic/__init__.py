"""Laconic — token-efficient middleware for multi-agent LLM workflows.

Compresses the natural-language payload of inter-agent handoffs while
preserving structural scaffolding (tool calls, arguments, IDs, schemas)
losslessly. Text-channel only, honestly bounded, measured in tokens.

Quickstart::

    from laconic import Session, summary_table
    from laconic.integrations import CompressingHook

    hook = CompressingHook(Session(target_model="gpt-4.1"))
    outgoing = hook.process_message(
        {"role": "assistant", "content": long_agent_report},
        source_agent="researcher",
        target_agent="writer",
    )
    print(summary_table(hook.profile))
"""

from laconic.message.model import CompressionStats, Message
from laconic.pipeline import ProcessedMessage, Session
from laconic.profiler.records import WorkflowProfile
from laconic.profiler.report import summary_table
from laconic.tokenizers.registry import get_counter

__version__ = "0.1.0"

__all__ = [
    "CompressionStats",
    "Message",
    "ProcessedMessage",
    "Session",
    "WorkflowProfile",
    "__version__",
    "get_counter",
    "summary_table",
]
