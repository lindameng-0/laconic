"""Standalone trace analysis: profile a message log without touching your code.

"Show me where my agent workflow burns tokens" should not require integrating
anything. Dump your inter-agent messages to a JSONL file (one raw message per
line, optionally wrapped as ``{"source": ..., "target": ..., "message": {...}}``)
and run::

    laconic profile trace.jsonl --model gpt-4.1

This replays the trace through a measurement session (and optionally a what-if
compression strategy) and prints the profile.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from laconic.pipeline import Session
from laconic.profiler.records import WorkflowProfile


def read_trace(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL trace file into a list of entries.

    Each line is either a raw message dict (``{"role": ..., "content": ...}``)
    or an envelope ``{"source": ..., "target": ..., "message": {...}}``.
    """
    entries: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(data, dict):
                raise ValueError(f"{path}:{line_number}: expected an object per line")
            entries.append(data)
    return entries


def profile_trace(
    entries: list[dict[str, Any]],
    *,
    model: str,
    strategy: str = "off",
    framework: str = "openai-chat",
) -> WorkflowProfile:
    """Replay trace entries through a :class:`~laconic.pipeline.Session`.

    With ``strategy="off"`` this is pure measurement; any other strategy
    produces a what-if profile showing what that strategy would have saved.
    """
    session = Session(target_model=model, strategy=strategy, framework=framework)
    profile = WorkflowProfile()
    for entry in entries:
        if "message" in entry and isinstance(entry["message"], dict):
            raw = entry["message"]
            source = entry.get("source")
            target = entry.get("target")
        else:
            raw = entry
            source = None
            target = None
        processed = session.process(raw, source_agent=source, target_agent=target)
        profile.add(processed.stats, source_agent=source, target_agent=target)
    return profile
