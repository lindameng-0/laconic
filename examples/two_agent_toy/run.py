"""Minimal two-agent workflow with Laconic in the middle.

Runs fully offline by default (a scripted researcher and a mock writer), so
you can see real token numbers in under a second with zero API keys::

    python examples/two_agent_toy/run.py
    python examples/two_agent_toy/run.py --strategy aggressive --model gpt-4.1
    python examples/two_agent_toy/run.py --html report.html

The "workflow" is deliberately tiny: a researcher agent produces a verbose
report (three times, with overlapping content — exactly what agent loops do),
and a writer agent consumes each handoff. Laconic sits on the seam.
"""

from __future__ import annotations

import argparse

from laconic import Session, summary_table
from laconic.integrations import CompressingHook
from laconic.profiler.report import to_html

# A scripted "researcher": verbose, repetitive, structure-bearing — the kind
# of traffic multi-agent workflows actually produce.
REPORT_HEADER = (
    "Please note that this is the consolidated research summary for the "
    "checkout-conversion project, provided for your convenience. It is worth "
    "noting that the methodology follows the same approach as previous cycles."
)

FINDINGS = [
    (
        "Finding 1: conversion in the redesigned checkout improved by 2.4 "
        "percentage points for returning users in Q1 2026. Basically, the "
        "effect is concentrated in mobile sessions. In order to validate "
        "this, we re-ran the analysis with the holiday window excluded and "
        "the improvement held at 2.1 points."
    ),
    (
        "Finding 2: cart abandonment dropped from 68 percent to 61 percent. "
        "As previously mentioned, the largest driver is the removal of the "
        "forced account-creation step. It should be noted that support "
        "tickets about checkout errors fell 34 percent in the same period."
    ),
    (
        "Finding 3: the payment-provider fallback logic added 180ms of p95 "
        "latency. For example, sessions routed through the secondary "
        "provider show a measurably lower completion rate of 88.2 percent "
        "versus 93.5 percent on the primary."
    ),
]

SHARED_APPENDIX = (
    "Appendix — experiment configuration: cohort definition follows the "
    "standard mid-market segmentation, traffic was split 50/50 with sticky "
    "assignment, the observation window covers 2026-01-05 through 2026-03-29, "
    "and statistical testing uses sequential analysis with alpha 0.05. All "
    "dashboards live in the analytics workspace under 'checkout-2026'."
)


def researcher_report(finding: str) -> str:
    """One verbose handoff: header + finding + the same appendix every time."""
    return f"{REPORT_HEADER}\n\n{finding}\n\n{SHARED_APPENDIX}"


def writer_agent(handoff_content: str) -> str:
    """A stand-in for the receiving model call: 'writes' a one-liner."""
    first_sentence = handoff_content.split(". ")[0][:80]
    return f"[writer drafted a paragraph based on: {first_sentence}...]"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        default="conservative",
        choices=["off", "conservative", "telegraphic", "balanced", "aggressive"],
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1",
        help="target model for token counting (tiktoken exact if installed)",
    )
    parser.add_argument("--html", help="write an HTML report to this path")
    args = parser.parse_args()

    hook = CompressingHook(Session(target_model=args.model, strategy=args.strategy))

    for finding in FINDINGS:
        raw_handoff = {"role": "assistant", "content": researcher_report(finding)}
        cheap_handoff = hook.process_message(
            raw_handoff, source_agent="researcher", target_agent="writer"
        )
        draft = writer_agent(cheap_handoff["content"])
        print(draft)

    print()
    print(summary_table(hook.profile))

    if args.html:
        path = to_html(hook.profile, args.html)
        print(f"\nHTML report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
