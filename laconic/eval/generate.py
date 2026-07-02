"""Seeded benchmark generator.

The benchmark that ships in ``data/benchmark/`` is produced by this module
with a fixed seed, so anyone can regenerate it byte-identically::

    python -m laconic.eval.generate --seed 7 --per-family 40 --out data/benchmark

Synthetic-but-principled: facts, tool plans, and constraints are embedded in
verbose connective prose of the kind agents actually emit, so extractive
pruning has something real to remove and something real to lose.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from laconic.eval.tasks import BenchmarkTask, Expected, save_tasks

_NAMES = [
    "Meridian Corp",
    "Halvorsen Ltd",
    "Okabe Systems",
    "Brightwell Group",
    "Cascadia Partners",
    "Novi Analytics",
    "Tern Logistics",
    "Peralta Foods",
]
_CITIES = [
    "Rotterdam",
    "Osaka",
    "Denver",
    "Tallinn",
    "Fortaleza",
    "Nairobi",
    "Adelaide",
    "Kraków",
]
_METRICS = ["revenue", "headcount", "churn rate", "gross margin", "uptime", "backlog"]
_QUARTERS = ["Q1 2025", "Q2 2025", "Q3 2025", "Q4 2025", "Q1 2026"]

_FILLER_SENTENCES = [
    "It is worth noting that the broader context here has been discussed at length in prior syncs.",
    "As previously mentioned, the team remains generally optimistic about the overall direction.",
    "Please note that this summary is provided for convenience from the latest available data.",
    "For what it's worth, several stakeholders have expressed broadly similar sentiments before.",
    "Basically, the situation continues to evolve and we will keep monitoring it going forward.",
    "There are, of course, a number of caveats that apply as they usually do in these reports.",
    "In order to keep everyone aligned, this recap intentionally errs on the side of completeness.",
    "The methodology follows the same approach we have used in previous reporting cycles.",
]

_TOOLS = [
    ("search_inventory", {"warehouse": _CITIES, "min_stock": [10, 25, 50, 100]}),
    (
        "create_ticket",
        {"priority": ["low", "medium", "high"], "queue": ["ops", "billing", "infra"]},
    ),
    ("schedule_job", {"cron": ["0 6 * * *", "*/15 * * * *", "0 0 * * 1"], "retries": [1, 2, 3]}),
    ("fetch_report", {"period": _QUARTERS, "format": ["csv", "json"]}),
]

_CONSTRAINTS = [
    "the response must be written in {lang}",
    "the total budget must not exceed {n} thousand dollars",
    "delivery must happen before {q}",
    "the vendor {name} must be excluded from consideration",
    "all figures must be reported in {unit}",
    "the summary must be at most {n} bullet points",
]
_LANGS = ["English", "German", "Japanese", "Spanish"]
_UNITS = ["euros", "US dollars", "metric tons", "units per week"]


def _prose_wrap(rng: random.Random, facts: list[str]) -> str:
    """Interleave fact sentences with verbose filler, agent-report style."""
    sentences: list[str] = []
    fillers = list(_FILLER_SENTENCES)
    rng.shuffle(fillers)
    filler_iter = iter(fillers * 3)
    for fact in facts:
        sentences.append(next(filler_iter))
        sentences.append(fact)
        if rng.random() < 0.6:
            sentences.append(next(filler_iter))
    return " ".join(sentences)


def _make_extraction_task(rng: random.Random, index: int) -> BenchmarkTask:
    company = rng.choice(_NAMES)
    metric_a, metric_b = rng.sample(_METRICS, 2)
    value_a = f"{rng.randint(2, 98)}.{rng.randint(0, 9)}"
    value_b = f"{rng.randint(101, 987)}"
    quarter = rng.choice(_QUARTERS)
    facts = [
        f"For {company}, the {metric_a} in {quarter} was {value_a} percent.",
        f"The {metric_b} for {company} in {quarter} came in at {value_b}.",
    ]
    payload = _prose_wrap(rng, facts)
    return BenchmarkTask(
        task_id=f"extraction_relay-{index:03d}",
        family="extraction_relay",
        handoff_message={"role": "assistant", "content": payload},
        receiver_question=(
            f"From the report you received: what was the {metric_a} (in percent) and "
            f"the {metric_b} for {company} in {quarter}? Include the exact numbers."
        ),
        expected=Expected(kind="contains_all", values=[value_a, value_b]),
    )


def _make_tool_plan_task(rng: random.Random, index: int) -> BenchmarkTask:
    tool_name, arg_space = rng.choice(_TOOLS)
    args = {key: rng.choice(values) for key, values in arg_space.items()}
    call_id = f"call_{rng.randrange(16**8):08x}"
    rationale_facts = [
        f"The plan is to invoke the {tool_name} operation once with the parameters attached.",
        "The executor should not modify any parameter values before invoking the tool.",
    ]
    payload = _prose_wrap(rng, rationale_facts)
    message = {
        "role": "assistant",
        "content": payload,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": tool_name, "arguments": json.dumps(args)},
            }
        ],
    }
    return BenchmarkTask(
        task_id=f"tool_plan_handoff-{index:03d}",
        family="tool_plan_handoff",
        handoff_message=message,
        receiver_question=(
            "Restate the planned tool invocation as a single JSON object with keys "
            '"name" and "arguments" (arguments as an object), exactly as specified '
            "in the plan you received."
        ),
        expected=Expected(kind="json_fields", fields={"name": tool_name, "arguments": args}),
    )


def _make_constraint_task(rng: random.Random, index: int) -> BenchmarkTask:
    n_constraints = rng.randint(2, 4)
    templates = rng.sample(_CONSTRAINTS, n_constraints)
    constraints: list[str] = []
    keywords: list[str] = []
    for template in templates:
        lang = rng.choice(_LANGS)
        name = rng.choice(_NAMES)
        unit = rng.choice(_UNITS)
        quarter = rng.choice(_QUARTERS)
        number = rng.randint(3, 900)
        constraint = template.format(lang=lang, n=number, name=name, unit=unit, q=quarter)
        constraints.append(constraint)
        for token in (lang, name, unit, quarter, str(number)):
            if (
                ("{" + "lang}" in template and token == lang)
                or ("{name}" in template and token == name)
                or ("{unit}" in template and token == unit)
                or ("{q}" in template and token == quarter)
                or ("{n}" in template and token == str(number))
            ):
                keywords.append(token)
    facts = [f"Hard requirement: {c}." for c in constraints]
    payload = _prose_wrap(rng, facts)
    return BenchmarkTask(
        task_id=f"constraint_tracking-{index:03d}",
        family="constraint_tracking",
        handoff_message={"role": "assistant", "content": payload},
        receiver_question=(
            "List every hard requirement stated in the brief you received, keeping "
            "all specific values (names, numbers, languages, units, deadlines) exact."
        ),
        expected=Expected(kind="contains_all", values=keywords),
    )


def generate_tasks(seed: int = 7, per_family: int = 40) -> list[BenchmarkTask]:
    """Generate the full benchmark deterministically from ``seed``."""
    rng = random.Random(seed)
    tasks: list[BenchmarkTask] = []
    for index in range(per_family):
        tasks.append(_make_extraction_task(rng, index))
    for index in range(per_family):
        tasks.append(_make_tool_plan_task(rng, index))
    for index in range(per_family):
        tasks.append(_make_constraint_task(rng, index))
    return tasks


def main(argv: list[str] | None = None) -> int:
    """CLI: regenerate the benchmark dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--per-family", type=int, default=40)
    parser.add_argument("--out", type=Path, default=Path("data/benchmark"))
    args = parser.parse_args(argv)

    tasks = generate_tasks(seed=args.seed, per_family=args.per_family)
    by_family: dict[str, list[BenchmarkTask]] = {}
    for task in tasks:
        by_family.setdefault(task.family, []).append(task)
    for family, family_tasks in by_family.items():
        path = args.out / f"{family}.jsonl"
        save_tasks(family_tasks, path)
        print(f"wrote {len(family_tasks)} tasks -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
