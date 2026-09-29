"""Bounded experiments with caller-owned downstream task validators.

This module never runs generated code, shells, or model APIs. The caller's
validator may do expensive work and must reset task state between evaluations.
Passing a replay checks that validator on that run; it does not establish
semantic equivalence, a globally minimal repair, or a causal root cause.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictBool

from laconic.handoff.contracts import HandoffContract, audit_handoff, repair_handoff
from laconic.tokenizers.base import CountTier, TokenCounter
from laconic.tokenizers.heuristic import HeuristicCounter

ReplayStatus = Literal[
    "invalid_contract",
    "invalid_candidate",
    "baseline_unavailable",
    "baseline_failed",
    "candidate_passed",
    "repaired",
    "unresolved",
    "budget_exhausted",
    "validator_error",
]

_GUARANTEE = (
    "Success applies only to the supplied validator in the recorded replay. "
    "Source checks establish exact quote membership, not source truth or semantic equivalence. "
    "The search does not prove a root cause or a globally minimal repair."
)
_REPRODUCIBILITY = (
    "For attribution, use a deterministic validator with a fixed task and dependencies, "
    "and reset mutable state before every call. The caller owns that isolation. "
    "A single passing stochastic run is not evidence of a stable improvement."
)


class ReplayOutcome(BaseModel):
    """The result of actually running the caller's downstream check.

    ``metrics`` can include total ``cost_usd``, ``latency_ms``, or other measured
    quantities. Laconic records them as supplied, without estimating a saving.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    success: StrictBool
    metrics: dict[str, FiniteFloat] = Field(default_factory=dict)
    details: str | dict[str, Any] = ""


class ReplayAttempt(BaseModel):
    """An actual validator invocation, including unsuccessful invocations."""

    index: int
    phase: Literal["baseline", "candidate", "repair", "reduction"]
    text: str
    text_sha256: str
    added_requirement_ids: tuple[str, ...] = ()
    tokens: int
    outcome: ReplayOutcome | None = None
    error: str | None = None


class ReplayResult(BaseModel):
    """Inspectable evidence from a bounded search, not a semantic guarantee."""

    status: ReplayStatus
    text: str | None
    success: bool | None = None
    added_requirement_ids: tuple[str, ...] = ()
    attempts: list[ReplayAttempt] = Field(default_factory=list)
    max_evaluations: int
    evaluations_used: int = 0
    budget_exhausted: bool = False
    reduction_complete: bool = False
    original_tokens: int | None = None
    candidate_tokens: int | None = None
    result_tokens: int | None = None
    added_tokens: int = 0
    count_tier: str
    model: str
    validator_name: str
    reason: str
    guarantee: str = _GUARANTEE
    reproducibility_note: str = _REPRODUCIBILITY
    contract_sha256: str | None = None
    source_provenance: list[dict[str, str]] = Field(default_factory=list)


def diagnose_handoff(
    contract: HandoffContract,
    original_text: str | None,
    candidate_text: str,
    validator: Callable[[str], ReplayOutcome],
    *,
    max_evaluations: int = 8,
    max_added_tokens: int | None = None,
    counter: TokenCounter | None = None,
    validator_name: str | None = None,
    reproducibility_note: str | None = None,
) -> ReplayResult:
    """Compare a source-complete baseline, a candidate, and source-backed repairs.

    Baseline and candidate calls count toward ``max_evaluations``. When the
    baseline passes and the candidate fails, append missing source quotes,
    then greedily remove additions in source order while the validator passes.
    An addition limit can exclude a feasible repair; failure is inconclusive
    outside the evaluated proposals. No callback runs for an invalid contract
    or a baseline that omits a declared source requirement.

    Validator exceptions or malformed return values stop the search. A passing
    repair remains ``status='repaired'`` if the call budget prevents further
    reduction; ``budget_exhausted`` and ``reduction_complete`` describe that
    limitation separately. Even a completed single deletion pass is not a
    global minimality proof, especially for non-monotonic validators.
    """
    if isinstance(max_evaluations, bool) or not isinstance(max_evaluations, int):
        raise ValueError("max_evaluations must be a positive integer")
    if max_evaluations < 1:
        raise ValueError("max_evaluations must be a positive integer")
    if max_added_tokens is not None and (
        isinstance(max_added_tokens, bool)
        or not isinstance(max_added_tokens, int)
        or max_added_tokens < 0
    ):
        raise ValueError("max_added_tokens must be a nonnegative integer or None")

    counter = counter or HeuristicCounter()
    if counter.tier == CountTier.API:
        raise ValueError("diagnose_handoff requires an offline token counter")
    attempts: list[ReplayAttempt] = []
    baseline_tokens = counter.count(original_text) if isinstance(original_text, str) else None
    candidate_tokens = counter.count(candidate_text) if isinstance(candidate_text, str) else None
    provenance: list[dict[str, str]] = []
    contract_fingerprint: str | None = None
    name = validator_name or getattr(validator, "__qualname__", type(validator).__name__)
    note = _REPRODUCIBILITY
    if reproducibility_note:
        note += " Caller context: " + reproducibility_note

    def result(
        status: ReplayStatus,
        reason: str,
        *,
        text: str | None = None,
        success: bool | None = None,
        added: tuple[str, ...] = (),
        exhausted: bool = False,
        minimized: bool = False,
    ) -> ReplayResult:
        if text is None and isinstance(candidate_text, str):
            text = candidate_text
        result_tokens = counter.count(text) if text is not None else None
        return ReplayResult(
            status=status,
            text=text,
            success=success,
            added_requirement_ids=added,
            attempts=attempts,
            max_evaluations=max_evaluations,
            evaluations_used=len(attempts),
            budget_exhausted=exhausted,
            reduction_complete=minimized,
            original_tokens=baseline_tokens,
            candidate_tokens=candidate_tokens,
            result_tokens=result_tokens,
            added_tokens=max(0, (result_tokens or 0) - (candidate_tokens or 0)),
            count_tier=counter.tier.value,
            model=counter.model,
            validator_name=name,
            reason=reason,
            reproducibility_note=note,
            contract_sha256=contract_fingerprint,
            source_provenance=provenance,
        )

    try:
        # Revalidate rather than trusting a previously constructed container.
        if not isinstance(contract, HandoffContract):
            raise ValueError("expected a HandoffContract")
        contract = HandoffContract(evidence=contract.evidence, requirements=contract.requirements)
        contract_fingerprint = contract.sha256
        for evidence in contract.evidence:
            provenance.append(
                {
                    "id": evidence.id,
                    "source": evidence.source,
                    "sha256": hashlib.sha256(evidence.text.encode("utf-8")).hexdigest(),
                }
            )
    except (TypeError, ValueError, AttributeError) as exc:
        return result("invalid_contract", f"Source contract could not be verified: {exc}")

    if not isinstance(candidate_text, str):
        return result("invalid_candidate", "Candidate must be text; no validator was called.")
    if not isinstance(original_text, str):
        return result(
            "baseline_unavailable", "No original baseline text was supplied; no conclusion."
        )
    baseline_audit = audit_handoff(contract, original_text, stage="baseline")
    if baseline_audit.missing_requirement_ids:
        return result(
            "invalid_contract",
            "Original baseline omits declared source requirements: "
            + ", ".join(baseline_audit.missing_requirement_ids),
        )
    if not callable(validator):
        return result("validator_error", "Validator must be callable; no validator was called.")

    def evaluate(
        text: str,
        phase: Literal["baseline", "candidate", "repair", "reduction"],
        added: tuple[str, ...] = (),
    ) -> ReplayAttempt:
        # Only this function invokes caller code, and even a raised call counts.
        if len(attempts) >= max_evaluations:
            raise RuntimeError("internal replay budget exceeded")
        attempt = ReplayAttempt(
            index=len(attempts),
            phase=phase,
            text=text,
            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            added_requirement_ids=added,
            tokens=counter.count(text),
        )
        attempts.append(attempt)
        try:
            outcome = validator(text)
            if not isinstance(outcome, ReplayOutcome):
                raise TypeError("validator must return ReplayOutcome")
            # Snapshot mutable details and check objects made through model_construct.
            attempt.outcome = ReplayOutcome.model_validate(outcome.model_dump()).model_copy(
                deep=True
            )
        except Exception as exc:
            attempt.error = f"{type(exc).__name__}: {exc}"
        return attempt

    baseline = evaluate(original_text, "baseline")
    if baseline.error:
        return result(
            "validator_error", "Baseline validator raised or returned an invalid outcome."
        )
    if not baseline.outcome or not baseline.outcome.success:
        return result(
            "baseline_failed", "Original baseline failed; handoff attribution is unavailable."
        )
    if len(attempts) >= max_evaluations:
        return result(
            "budget_exhausted", "Budget ended before candidate evaluation.", exhausted=True
        )

    candidate = evaluate(candidate_text, "candidate")
    if candidate.error:
        return result(
            "validator_error", "Candidate validator raised or returned an invalid outcome."
        )
    if candidate.outcome and candidate.outcome.success:
        return result(
            "candidate_passed",
            "Candidate passed the supplied validator; no repair needed.",
            success=True,
            minimized=True,
        )

    missing = audit_handoff(contract, candidate_text).missing_requirement_ids
    if not missing:
        return result(
            "unresolved", "Candidate failed but no declared source quote is missing.", success=False
        )
    proposal = repair_handoff(
        contract,
        candidate_text,
        requirement_ids=missing,
        max_added_tokens=max_added_tokens,
        counter=counter,
    )
    active = tuple(proposal.restored_requirement_ids)
    if proposal.text == candidate_text or not active:
        return result(
            "unresolved",
            "No source-backed addition fits the configured token limit.",
            success=False,
        )
    if len(attempts) >= max_evaluations:
        return result(
            "budget_exhausted", "Budget ended before a repair could be evaluated.", exhausted=True
        )

    full_repair = evaluate(proposal.text, "repair", active)
    if full_repair.error:
        return result("validator_error", "Repair validator raised or returned an invalid outcome.")
    if not full_repair.outcome or not full_repair.outcome.success:
        return result(
            "unresolved",
            "The evaluated source-backed repair failed. Other subsets or repairs remain untested.",
            success=False,
        )

    best_text = proposal.text
    # Empty additions already failed as the candidate; do not pay to repeat it.
    for requirement_id in active:
        if len(active) == 1:
            break
        reduced_ids = tuple(item for item in active if item != requirement_id)
        reduced = repair_handoff(
            contract, candidate_text, requirement_ids=reduced_ids, counter=counter
        )
        if max_added_tokens is not None and reduced.added_tokens > max_added_tokens:
            continue
        if len(attempts) >= max_evaluations:
            return result(
                "repaired",
                "A repair passed; the budget ended before greedy reduction completed.",
                text=best_text,
                success=True,
                added=active,
                exhausted=True,
            )
        trial = evaluate(reduced.text, "reduction", reduced_ids)
        if trial.error:
            return result(
                "validator_error",
                "Reduction validator raised or returned an invalid outcome; search stopped. "
                "Earlier passing attempts remain recorded, but no completed diagnosis is claimed.",
                text=best_text,
                added=active,
            )
        if trial.outcome and trial.outcome.success:
            active = reduced_ids
            best_text = reduced.text

    return result(
        "repaired",
        "A source-backed repair passed the supplied validator. One greedy deletion pass completed; "
        "global minimality and semantic equivalence are unproven.",
        text=best_text,
        success=True,
        added=active,
        minimized=True,
    )
