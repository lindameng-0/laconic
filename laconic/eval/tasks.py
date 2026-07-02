"""Benchmark task model.

A task is one inter-agent handoff with a verifiable outcome: a *sender* has
produced a handoff message; a *receiver* must answer a question (or emit a
structured action) whose ground truth is known. Compression is applied to the
handoff; accuracy is scored on the receiver's answer. This makes the accuracy
delta of any compression strategy directly measurable — no LLM judge needed.

Three task families:

- ``extraction_relay`` — verbose report with embedded facts; the receiver is
  asked for specific fact values. Measures information survival in prose.
- ``tool_plan_handoff`` — a message carrying ``tool_calls`` structure plus a
  verbose rationale; the receiver must restate the tool name and arguments.
  Measures whether structure survives (it must, under Laconic; it visibly
  does not under naive compression).
- ``constraint_tracking`` — instructions with hard constraints buried in
  filler; the receiver must restate every constraint. Measures loss of
  load-bearing but low-salience content — where aggressive pruning fails
  first.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

TaskFamily = Literal["extraction_relay", "tool_plan_handoff", "constraint_tracking"]


class Expected(BaseModel):
    """Ground truth for scoring a receiver's answer.

    ``kind``:
        - ``contains_all`` — every string in ``values`` must appear in the
          answer (case-insensitive).
        - ``json_fields`` — the answer must contain a JSON object whose
          ``fields`` entries all match exactly.
    """

    kind: Literal["contains_all", "json_fields"]
    values: list[str] = Field(default_factory=list)
    fields: dict[str, Any] = Field(default_factory=dict)


class BenchmarkTask(BaseModel):
    """One handoff-and-verify benchmark task."""

    task_id: str
    family: TaskFamily
    handoff_message: dict[str, Any]
    receiver_question: str
    expected: Expected

    def score(self, answer: str) -> bool:
        """Return whether ``answer`` satisfies the ground truth."""
        if self.expected.kind == "contains_all":
            lowered = answer.lower()
            return all(value.lower() in lowered for value in self.expected.values)
        # json_fields: find the first JSON object in the answer and compare.
        obj = first_json_object(answer)
        if obj is None:
            return False
        return all(obj.get(key) == value for key, value in self.expected.fields.items())


def first_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first parseable JSON object embedded in ``text``.

    String-aware brace matching, so braces inside JSON string values do not
    derail the scan.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for end in range(start, len(text)):
            char = text[end]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : end + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
        start = text.find("{", start + 1)
    return None


def load_tasks(path: str | Path) -> list[BenchmarkTask]:
    """Load benchmark tasks from a JSONL file."""
    tasks: list[BenchmarkTask] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                tasks.append(BenchmarkTask.model_validate_json(line))
    return tasks


def save_tasks(tasks: list[BenchmarkTask], path: str | Path) -> None:
    """Write benchmark tasks to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for task in tasks:
            fh.write(json.dumps(task.model_dump(mode="json"), ensure_ascii=False))
            fh.write("\n")
