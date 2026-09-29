"""Executable fault-injection study for handoff diagnosis and repair.

This is a deterministic protocol experiment, NOT an LLM evaluation. A small
retry-policy interpreter consumes handoffs and executes requests against a
scripted service. Behavioral assertions (attempt limits, idempotency, request
identity and backoff) grade its actions, rather than scoring quoted keywords.
The fixtures deliberately include failures that verbatim repair cannot fix.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from laconic.handoff.contracts import (
    Evidence,
    Handoff,
    HandoffContract,
    Requirement,
    audit_handoffs,
)
from laconic.handoff.replay import ReplayOutcome, diagnose_handoff
from laconic.pipeline import Session
from laconic.tokenizers.base import TokenCounter
from laconic.tokenizers.heuristic import HeuristicCounter


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 1
    retry_codes: tuple[int, ...] = (500,)
    idempotent_only: bool = False
    delay_cap_ms: int = 1000
    preserve_request_id: bool = False

    def execute(self, statuses: list[int], *, idempotent: bool) -> list[dict[str, Any]]:
        """Execute a bounded fake HTTP exchange; no network or generated code."""
        events = []
        for index, status in enumerate(statuses[: self.attempts]):
            events.append(
                {
                    "status": status,
                    "request_id": "request-1"
                    if self.preserve_request_id
                    else f"request-{index + 1}",
                    "delay_ms": 0 if index == 0 else min(100 * 2 ** (index - 1), self.delay_cap_ms),
                }
            )
            if status not in self.retry_codes or (self.idempotent_only and not idempotent):
                break
        return events


def interpret_retry_policy(text: str) -> RetryPolicy:
    """A deliberately limited, transparent receiver; no language model is used.

    Latest explicit clauses win. The override models an implementation defect
    outside the source contract, so adding missing quotations cannot repair it.
    """
    attempts = re.findall(r"at most (\d+) attempts", text, re.IGNORECASE)
    codes = re.findall(r"Retry status codes ([\d, ]+)", text, re.IGNORECASE)
    caps = re.findall(r"(?:delay at|backoff ceiling:) (\d+) milliseconds", text, re.IGNORECASE)
    idempotent_only = bool(re.search(r"Never retry (?:a )?non-idempotent request", text, re.I))
    if "OVERRIDE retry_non_idempotent = true" in text:
        idempotent_only = False
    return RetryPolicy(
        attempts=int(attempts[-1]) if attempts else 1,
        retry_codes=tuple(int(v.strip()) for v in codes[-1].split(",")) if codes else (500,),
        idempotent_only=idempotent_only,
        delay_cap_ms=int(caps[-1]) if caps else 1000,
        preserve_request_id=bool(
            re.search(r"Preserve (?:the )?caller's request identifier across retries", text, re.I)
        ),
    )


@dataclass(frozen=True)
class WorkflowCase:
    case_id: str
    fault: str
    contract: HandoffContract
    original: str
    concise: str
    candidate: str
    expected: RetryPolicy
    stages: tuple[Handoff, ...]

    def validate(self, text: str) -> ReplayOutcome:
        """Check execution semantics, including boundary and negative cases."""
        actual = interpret_retry_policy(text)
        expected = self.expected
        probes = [
            ([503] * 12, True),
            ([429] * 12, True),
            ([503, 200], True),
            ([429, 503, 200], True),
            ([400, 200], True),
            ([500, 200], True),
            ([503, 200], False),
            ([429, 200], False),
            ([200], True),
        ]
        failures = [
            f"statuses={statuses}, idempotent={idempotent}"
            for statuses, idempotent in probes
            if actual.execute(statuses, idempotent=idempotent)
            != expected.execute(statuses, idempotent=idempotent)
        ]
        return ReplayOutcome(
            success=not failures,
            metrics={"checks": float(len(probes)), "failed_checks": float(len(failures))},
            details="; ".join(failures) if failures else "All retry execution checks passed.",
        )


FAULTS = (
    "lost_attempt_limit",
    "lost_negation",
    "lost_request_identity",
    "lost_two_requirements",
    "stale_delay_limit",
    "equivalent_paraphrase",
    "contradictory_implementation",
    "intact",
)


def generate_cases(seed: int = 7, variants: int = 3) -> list[WorkflowCase]:
    if variants < 1:
        raise ValueError("variants must be positive")
    rng = random.Random(seed)
    cases = []
    for variant in range(variants):
        attempts = rng.randint(3, 6)
        delay = rng.choice((50, 75, 125, 150))
        quotes = [
            f"Use at most {attempts} attempts, counting the initial request.",
            "Retry status codes 429, 503 and no other status codes.",
            "Never retry a non-idempotent request.",
            f"Cap the retry delay at {delay} milliseconds.",
            "Preserve the caller's request identifier across retries.",
        ]
        source = "\n".join(quotes)
        evidence = Evidence(
            id=f"spec-{variant}", text=source, source=f"fixture://retry/{seed}/{variant}"
        )
        contract = HandoffContract(
            evidence=[evidence],
            requirements=[
                Requirement(id=f"r{index}", evidence_id=evidence.id, quote=quote)
                for index, quote in enumerate(quotes)
            ],
        )
        filler = (
            "The investigation reviewed deployment history and the surrounding service. "
            "Please note that this background explains the reasoning behind the chosen approach. "
            "The previous discussion examined alternatives and recorded implementation notes. "
        )
        original = "\n\n".join(f"{filler * 3}\n{quote}" for quote in quotes)
        expected = RetryPolicy(attempts, (429, 503), True, delay, True)
        for fault in FAULTS:
            selected = list(quotes)
            if fault == "lost_attempt_limit":
                selected.pop(0)
            elif fault == "lost_negation":
                selected.pop(2)
            elif fault == "lost_request_identity":
                selected.pop(4)
            elif fault == "lost_two_requirements":
                selected = [q for i, q in enumerate(quotes) if i not in (0, 2)]
            elif fault == "stale_delay_limit":
                selected[3] = f"Cap the retry delay at {delay * 4} milliseconds."
            elif fault == "equivalent_paraphrase":
                selected[3] = f"Backoff ceiling: {delay} milliseconds."
            elif fault == "contradictory_implementation":
                selected.append("OVERRIDE retry_non_idempotent = true")
            candidate = "\n".join(selected)
            cases.append(
                WorkflowCase(
                    case_id=f"seed-{seed}-variant-{variant}-{fault}",
                    fault=fault,
                    contract=contract,
                    original=original,
                    concise=source,
                    candidate=candidate,
                    expected=expected,
                    stages=(
                        Handoff(stage="investigator", text=original),
                        Handoff(stage="planner", text=candidate),
                        Handoff(stage="implementer", text=candidate),
                    ),
                )
            )
    return cases


def run_experiment(
    *,
    seed: int = 7,
    variants: int = 3,
    counter: TokenCounter | None = None,
    max_evaluations: int = 8,
) -> dict[str, Any]:
    counter = counter or HeuristicCounter("offline")
    records = []
    for case in generate_cases(seed, variants):
        policies = {"full_context": case.original, "concise_contract": case.concise}
        for strategy in ("telegraphic", "balanced"):
            session = Session(
                target_model=counter.model,
                strategy=strategy,
                counter=counter,
                dedup=False,
                keep_ratio=0.45,
            )
            policies[strategy] = session.process(
                {"role": "assistant", "content": case.original}
            ).raw["content"]
        policies["observed_handoff"] = case.candidate
        replay = diagnose_handoff(
            case.contract,
            case.original,
            case.candidate,
            case.validate,
            max_evaluations=max_evaluations,
            counter=counter,
            validator_name="scripted-retry-execution-v1",
            reproducibility_note="Deterministic receiver and service; no model calls or billing.",
        )
        records.append(
            {
                "case_id": case.case_id,
                "fault": case.fault,
                "policies": {
                    name: {
                        "tokens": counter.count(text),
                        "outcome": case.validate(text).model_dump(),
                    }
                    for name, text in policies.items()
                },
                "audit": audit_handoffs(case.contract, case.stages).model_dump(mode="json"),
                "replay": replay.model_dump(mode="json"),
            }
        )
    return {
        "experiment": "retry-handoff-fault-injection-v1",
        "seed": seed,
        "variants": variants,
        "cases": len(records),
        "tokenizer": counter.model,
        "count_tier": counter.tier.value,
        "receiver": "deterministic retry-policy interpreter (not an LLM)",
        "limitations": [
            "Known omissions and mutations are injected into synthetic handoffs.",
            "The explicit contract is supplied by the fixture author; discovery is not evaluated.",
            "Behavioral checks cover a scripted retry service, not production coding tasks.",
            "Replay is extra work; text token sizes exclude provider envelopes and are not bills.",
            "A passing repair establishes sufficiency for this validator, not semantic safety.",
            "Concise contracts are a strong baseline; recovery need not beat them in size.",
        ],
        "records": records,
    }


def write_experiment(result: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def summarize_experiment(result: dict[str, Any]) -> str:
    """Show failures and replay overhead alongside the successful controls."""
    records = result["records"]
    lines = [
        f"Scripted handoff experiment: {len(records)} cases, seed {result['seed']}",
        f"Receiver: {result['receiver']}",
        f"Token counts: {result['count_tier']} ({result['tokenizer']}); not API billing",
        "",
    ]
    for name in records[0]["policies"]:
        successes = sum(r["policies"][name]["outcome"]["success"] for r in records)
        tokens = sum(r["policies"][name]["tokens"] for r in records)
        lines.append(f"{name}: {successes}/{len(records)} passed; {tokens:,} total text tokens")
    passing = sum(r["replay"]["success"] is True for r in records)
    recovered = sum(r["replay"]["status"] == "repaired" for r in records)
    calls = sum(r["replay"]["evaluations_used"] for r in records)
    replay_tokens = sum(a["tokens"] for r in records for a in r["replay"]["attempts"])
    lines.extend(
        [
            f"after diagnosis: {passing}/{len(records)} passed; {recovered} repaired",
            f"diagnostic overhead: {calls} validator calls; {replay_tokens:,} input text tokens",
            "",
            "Statuses by injected fault:",
        ]
    )
    for fault in FAULTS:
        statuses: dict[str, int] = {}
        for record in records:
            if record["fault"] == fault:
                status = record["replay"]["status"]
                statuses[status] = statuses.get(status, 0) + 1
        lines.append(f"  {fault}: " + ", ".join(f"{k}={v}" for k, v in statuses.items()))
    lines.extend(["", "Limits:", *(f"- {item}" for item in result["limitations"])])
    return "\n".join(lines)
