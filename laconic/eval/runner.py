"""The matrix runner: {model} × {strategy} × {keep-ratio} over the benchmark.

For every cell, each task's handoff is processed by the strategy, rendered
into the receiver's user turn (the same rendering for every client — see
``laconic/eval/clients.py``), answered by the model client, and scored against
ground truth. Results are written as JSON and CSV; nothing is ever
interpolated or invented.

Strategies:

- ``passthrough`` — the accuracy/token baseline.
- ``laconic-conservative`` — lossless-in-meaning extractive cleanup.
- ``laconic-budgeted`` — extractive with sentence pruning at each keep-ratio.
- ``naive`` — whole-message, structure-blind pruning at each keep-ratio
  (the baseline Laconic exists to beat; never use it in production).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from laconic.compress.naive import NaiveWholeMessageCompressor
from laconic.eval.clients import HANDOFF_HEADER, TASK_HEADER, ModelClient
from laconic.eval.tasks import BenchmarkTask
from laconic.pipeline import Session
from laconic.tokenizers.base import TokenCounter
from laconic.tokenizers.registry import get_counter

_SYSTEM_PROMPT = (
    "You are the receiving agent in a two-agent workflow. Answer the task "
    "using only the handoff you were given. Be precise with names, numbers, "
    "and structured values."
)


class RunRecord(BaseModel):
    """Outcome of one task under one (model, strategy, ratio) cell."""

    task_id: str
    family: str
    model: str
    client: str
    strategy: str
    keep_ratio: float | None
    tokens_before: int
    tokens_after: int
    count_tier: str
    fell_back: bool
    correct: bool
    answer: str


@dataclass
class MatrixSpec:
    """What to run.

    Attributes:
        models: Model names to evaluate (receiver side).
        keep_ratios: Budgets for the budgeted strategies.
        include_naive: Include the structure-blind baseline.
        include_conservative: Include the unbudgeted lossless-ish strategy.
    """

    models: list[str] = field(default_factory=lambda: ["mock"])
    keep_ratios: list[float] = field(default_factory=lambda: [0.9, 0.75, 0.6, 0.45, 0.3])
    include_naive: bool = True
    include_conservative: bool = True
    include_telegraphic: bool = True


def _render_receiver_messages(handoff_text: str, question: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"{HANDOFF_HEADER}\n{handoff_text}\n\n{TASK_HEADER} {question}",
        },
    ]


def _laconic_handoff(
    task: BenchmarkTask,
    model: str,
    budget: float | None,
    counter: TokenCounter,
    *,
    telegraphic: bool = False,
) -> tuple[str, int, int, bool]:
    """Process a handoff through the Laconic pipeline; return rendering + stats."""
    if telegraphic:
        session = Session(target_model=model, strategy="telegraphic", counter=counter)
    elif budget is None:
        session = Session(target_model=model, strategy="conservative", counter=counter)
    else:
        session = Session(
            target_model=model, strategy="balanced", keep_ratio=budget, counter=counter
        )
    processed = session.process(task.handoff_message, target_agent="receiver")
    text = json.dumps(processed.raw, ensure_ascii=False)
    return (
        text,
        processed.stats.tokens_before,
        processed.stats.tokens_after,
        processed.stats.fell_back,
    )


def _naive_handoff(
    task: BenchmarkTask, budget: float, counter: TokenCounter
) -> tuple[str, int, int, bool]:
    """Structure-blind baseline: prune the serialized message as prose."""
    original = json.dumps(task.handoff_message, ensure_ascii=False)
    compressor = NaiveWholeMessageCompressor()
    mangled = compressor.compress(original, counter=counter, budget=budget)
    return mangled, counter.count(original), counter.count(mangled), False


def _passthrough_handoff(task: BenchmarkTask, counter: TokenCounter) -> tuple[str, int, int, bool]:
    text = json.dumps(task.handoff_message, ensure_ascii=False)
    tokens = counter.count(text)
    return text, tokens, tokens, False


def run_matrix(
    tasks: list[BenchmarkTask],
    client: ModelClient,
    spec: MatrixSpec | None = None,
    *,
    allow_api_counting: bool = False,
) -> list[RunRecord]:
    """Run the full study matrix and return one record per (cell, task).

    ``client`` answers for every model in the spec — pass the mock client with
    ``models=["mock"]`` for the offline information-survival study, or a real
    client with real model names for the model-comprehension study.
    """
    spec = spec or MatrixSpec()
    records: list[RunRecord] = []
    client_name = getattr(client, "name", type(client).__name__)

    for model in spec.models:
        counter = get_counter(model, allow_api=allow_api_counting)
        cells: list[tuple[str, float | None]] = [("passthrough", None)]
        if spec.include_conservative:
            cells.append(("laconic-conservative", None))
        if spec.include_telegraphic:
            cells.append(("laconic-telegraphic", None))
        for ratio in spec.keep_ratios:
            cells.append(("laconic-budgeted", ratio))
            if spec.include_naive:
                cells.append(("naive", ratio))

        for strategy, ratio in cells:
            for task in tasks:
                if strategy == "passthrough":
                    text, before, after, fell_back = _passthrough_handoff(task, counter)
                elif strategy == "laconic-conservative":
                    text, before, after, fell_back = _laconic_handoff(task, model, None, counter)
                elif strategy == "laconic-telegraphic":
                    text, before, after, fell_back = _laconic_handoff(
                        task, model, None, counter, telegraphic=True
                    )
                elif strategy == "laconic-budgeted":
                    text, before, after, fell_back = _laconic_handoff(task, model, ratio, counter)
                else:  # naive
                    assert ratio is not None
                    text, before, after, fell_back = _naive_handoff(task, ratio, counter)

                messages = _render_receiver_messages(text, task.receiver_question)
                answer = client.complete(messages, model)
                records.append(
                    RunRecord(
                        task_id=task.task_id,
                        family=task.family,
                        model=model,
                        client=client_name,
                        strategy=strategy,
                        keep_ratio=ratio,
                        tokens_before=before,
                        tokens_after=after,
                        count_tier=counter.tier.value,
                        fell_back=fell_back,
                        correct=task.score(answer),
                        answer=answer,
                    )
                )
    return records


def write_results(records: list[RunRecord], out_dir: str | Path) -> tuple[Path, Path]:
    """Write records as ``results.json`` and ``results.csv``; return the paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "results.json"
    csv_path = out / "results.csv"

    json_path.write_text(
        json.dumps([r.model_dump(mode="json") for r in records], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    fields = list(RunRecord.model_fields)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(record.model_dump(mode="json"))
    return json_path, csv_path
