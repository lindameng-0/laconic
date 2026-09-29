"""Replay budgets, source validation, and actual caller-owned task checks.

The deterministic worker below is a test fixture, not evidence that a model
understands a restored clause or that a production workflow is improved.
"""

from __future__ import annotations

from typing import Any

import pytest

from laconic.handoff.contracts import Evidence, HandoffContract, Requirement
from laconic.handoff.replay import ReplayOutcome, diagnose_handoff
from laconic.tokenizers.base import CountTier
from laconic.tokenizers.heuristic import HeuristicCounter

COMPAT = "Keep existing one-argument calls working."
LOGGING = "Include a log entry for each request."


def test_completed_reduction_pass_is_not_a_minimality_claim() -> None:
    quotes = ("Requirement alpha.", "Requirement beta.", "Requirement gamma.")
    contract = _contract(*quotes)

    def validator(text: str) -> ReplayOutcome:
        present = frozenset(i for i, quote in enumerate(quotes) if quote in text)
        return ReplayOutcome(success=present in ({0, 1, 2}, {0, 2}, {2}))

    result = diagnose_handoff(contract, "\n".join(quotes), "Candidate", validator)
    assert result.success
    assert result.reduction_complete
    assert result.added_requirement_ids == ("r0", "r2")
    assert "global minimality" in result.reason
    assert validator(quotes[2]).success  # A smaller passing set exists outside this one pass.


def _contract(*quotes: str) -> HandoffContract:
    return HandoffContract(
        evidence=[Evidence(id="issue", text="\n".join(quotes), source="issue.txt")],
        requirements=[
            Requirement(id=f"r{index}", evidence_id="issue", quote=quote)
            for index, quote in enumerate(quotes)
        ],
    )


def _compatibility_check(text: str) -> ReplayOutcome:
    """Simulate a worker's two known implementations, then execute the old API."""

    def compatible(query: str, limit: int = 10) -> list[str]:
        return [query][:limit]

    def breaking(query: str, limit: int) -> list[str]:
        return [query][:limit]

    implementation = compatible if COMPAT in text else breaking
    try:
        passed = implementation("old caller") == ["old caller"]
        details = "Existing caller succeeded."
    except TypeError as exc:
        passed = False
        details = str(exc)
    return ReplayOutcome(success=passed, metrics={"cost_usd": 0, "latency_ms": 0}, details=details)


def test_restores_compatibility_clause_and_removes_irrelevant_addition() -> None:
    contract = _contract(COMPAT, LOGGING)
    result = diagnose_handoff(
        contract,
        "Add a limit argument.\n" + COMPAT + "\n" + LOGGING,
        "Add a required limit argument.",
        _compatibility_check,
        validator_name="legacy caller",
        reproducibility_note="Fixed in-process worker fixture, no API calls.",
    )
    assert result.status == "repaired"
    assert result.success is True
    assert result.added_requirement_ids == ("r0",)
    assert result.text is not None and COMPAT in result.text and LOGGING not in result.text
    assert result.reduction_complete
    assert not result.budget_exhausted
    assert result.evaluations_used == len(result.attempts) == 5
    assert [attempt.phase for attempt in result.attempts] == [
        "baseline",
        "candidate",
        "repair",
        "reduction",
        "reduction",
    ]
    assert result.attempts[0].outcome.success is True
    assert result.attempts[1].outcome.success is False
    assert "required positional argument" in result.attempts[1].outcome.details
    assert result.attempts[2].outcome.success is True
    assert result.attempts[2].outcome.metrics["cost_usd"] == 0
    assert result.added_tokens > 0
    assert result.count_tier == "estimate"
    assert result.contract_sha256 == contract.sha256
    assert result.source_provenance[0]["sha256"] == contract.evidence[0].sha256
    assert all(len(attempt.text_sha256) == 64 for attempt in result.attempts)


def test_joint_requirements_are_kept_when_both_are_needed() -> None:
    contract = _contract(COMPAT, LOGGING)

    def validator(text: str) -> ReplayOutcome:
        return ReplayOutcome(success=COMPAT in text and LOGGING in text)

    result = diagnose_handoff(contract, f"{COMPAT}\n{LOGGING}", "Implement feature.", validator)
    assert result.status == "repaired"
    assert result.added_requirement_ids == ("r0", "r1")
    assert result.evaluations_used == 5
    assert result.reduction_complete
    assert [attempt.outcome.success for attempt in result.attempts[-2:]] == [False, False]


def test_candidate_pass_does_not_trigger_repairs() -> None:
    result = diagnose_handoff(_contract(COMPAT), COMPAT, COMPAT, _compatibility_check)
    assert result.status == "candidate_passed"
    assert result.success is True
    assert result.evaluations_used == 2
    assert not result.added_requirement_ids


def test_failed_baseline_stops_without_attribution() -> None:
    calls: list[str] = []

    def validator(text: str) -> ReplayOutcome:
        calls.append(text)
        return ReplayOutcome(success=False, details="Independent fixture failure.")

    result = diagnose_handoff(_contract(COMPAT), COMPAT, "Missing clause", validator)
    assert result.status == "baseline_failed"
    assert result.success is None
    assert calls == [COMPAT]
    assert result.evaluations_used == 1


@pytest.mark.parametrize("baseline", [None, 42])
def test_unavailable_baseline_never_invokes_validator(baseline: Any) -> None:
    def validator(text: str) -> ReplayOutcome:
        pytest.fail("validator must not run without a baseline")

    result = diagnose_handoff(_contract(COMPAT), baseline, "Candidate", validator)
    assert result.status == "baseline_unavailable"
    assert result.success is None
    assert result.evaluations_used == 0


def test_baseline_must_include_all_contract_requirements_before_callbacks() -> None:
    def validator(text: str) -> ReplayOutcome:
        pytest.fail("source-incomplete baseline must be rejected before replay")

    result = diagnose_handoff(_contract(COMPAT, LOGGING), COMPAT, "Candidate", validator)
    assert result.status == "invalid_contract"
    assert "r1" in result.reason
    assert result.evaluations_used == 0


def test_unverified_source_contract_is_rejected_before_callbacks() -> None:
    contract = HandoffContract.model_construct(
        evidence=(Evidence(id="issue", text="Unrelated evidence", source="issue.txt"),),
        requirements=(Requirement(id="r0", evidence_id="issue", quote=COMPAT),),
    )

    def validator(text: str) -> ReplayOutcome:
        pytest.fail("unverified source quote must be rejected before replay")

    result = diagnose_handoff(contract, COMPAT, "Candidate", validator)
    assert result.status == "invalid_contract"
    assert result.success is None
    assert result.evaluations_used == 0


def test_nontext_candidate_is_blocked() -> None:
    def validator(text: str) -> ReplayOutcome:
        pytest.fail("nontext candidate must be rejected before replay")

    result = diagnose_handoff(_contract(COMPAT), COMPAT, {"text": COMPAT}, validator)
    assert result.status == "invalid_candidate"
    assert result.text is None
    assert result.evaluations_used == 0


def test_empty_candidate_can_be_repaired() -> None:
    result = diagnose_handoff(_contract(COMPAT), COMPAT, "", _compatibility_check)
    assert result.status == "repaired"
    assert result.success is True
    assert result.added_requirement_ids == ("r0",)


@pytest.mark.parametrize("budget", [1, 2, 3, 4, 5, 8])
def test_replay_budget_includes_baseline_candidate_and_failed_reductions(budget: int) -> None:
    calls: list[str] = []

    def validator(text: str) -> ReplayOutcome:
        calls.append(text)
        return ReplayOutcome(success=COMPAT in text and LOGGING in text)

    result = diagnose_handoff(
        _contract(COMPAT, LOGGING),
        f"{COMPAT}\n{LOGGING}",
        "Candidate",
        validator,
        max_evaluations=budget,
    )
    assert len(calls) == result.evaluations_used == len(result.attempts)
    assert len(calls) <= budget
    if budget < 3:
        assert result.status == "budget_exhausted"
        assert result.success is None
        assert result.budget_exhausted
    elif budget < 5:
        assert result.status == "repaired"
        assert result.success is True
        assert result.budget_exhausted
        assert not result.reduction_complete
    else:
        assert result.status == "repaired"
        assert result.success is True
        assert result.reduction_complete
        assert not result.budget_exhausted


@pytest.mark.parametrize("error_call", [0, 1, 2, 3])
def test_validator_exception_is_recorded_and_stops_immediately(error_call: int) -> None:
    calls = 0

    def validator(text: str) -> ReplayOutcome:
        nonlocal calls
        index = calls
        calls += 1
        if index == error_call:
            raise RuntimeError("worker is unavailable")
        return ReplayOutcome(success=COMPAT in text)

    result = diagnose_handoff(
        _contract(COMPAT, LOGGING), f"{COMPAT}\n{LOGGING}", "Candidate", validator
    )
    assert result.status == "validator_error"
    assert result.success is None
    assert calls == error_call + 1
    assert result.evaluations_used == calls
    assert result.attempts[-1].error == "RuntimeError: worker is unavailable"
    assert result.attempts[-1].outcome is None


def test_malformed_validator_return_is_an_explicit_error() -> None:
    result = diagnose_handoff(_contract(COMPAT), COMPAT, "Candidate", lambda text: True)
    assert result.status == "validator_error"
    assert result.evaluations_used == 1
    assert "must return ReplayOutcome" in result.attempts[0].error
    assert result.success is None


def test_repair_failure_does_not_claim_no_feasible_subset_exists() -> None:
    other_needed_fact = "Use the archived parser implementation."

    def validator(text: str) -> ReplayOutcome:
        return ReplayOutcome(success=other_needed_fact in text)

    result = diagnose_handoff(
        _contract(COMPAT), f"{COMPAT}\n{other_needed_fact}", "Candidate", validator
    )
    assert result.status == "unresolved"
    assert result.success is False
    assert result.evaluations_used == 3
    assert result.attempts[-1].phase == "repair"
    assert "Other subsets or repairs remain untested" in result.reason
    assert not result.added_requirement_ids


def test_no_missing_source_clause_cannot_be_repaired_by_inventing_information() -> None:
    def validator(text: str) -> ReplayOutcome:
        return ReplayOutcome(success="Additional fact" in text)

    result = diagnose_handoff(_contract(COMPAT), COMPAT + " Additional fact", COMPAT, validator)
    assert result.status == "unresolved"
    assert result.evaluations_used == 2
    assert "no declared source quote is missing" in result.reason


def test_token_cap_prevents_over_budget_repair_from_being_evaluated() -> None:
    result = diagnose_handoff(
        _contract(COMPAT), COMPAT, "Candidate", _compatibility_check, max_added_tokens=0
    )
    assert result.status == "unresolved"
    assert result.evaluations_used == 2
    assert result.added_tokens == 0


def test_report_scopes_success_and_reproducibility_without_semantic_claims() -> None:
    result = diagnose_handoff(
        _contract(COMPAT),
        COMPAT,
        "Candidate",
        _compatibility_check,
        reproducibility_note="worker=v1; fixture=legacy-call; seed=0",
    )
    assert "only to the supplied validator" in result.guarantee
    assert "not source truth or semantic equivalence" in result.guarantee
    assert "globally minimal" in result.guarantee
    assert "deterministic validator" in result.reproducibility_note
    assert "worker=v1" in result.reproducibility_note
    serialized = result.model_dump(mode="json")
    assert serialized["attempts"][1]["outcome"]["success"] is False


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_evaluation_budget_is_rejected(budget: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        diagnose_handoff(
            _contract(COMPAT), COMPAT, "Candidate", _compatibility_check, max_evaluations=budget
        )


def test_api_token_counter_is_rejected_before_counting_or_callbacks() -> None:
    class NeverCallCounter(HeuristicCounter):
        tier = CountTier.API

        def count(self, text: str) -> int:
            pytest.fail("API counter must not be called")

    with pytest.raises(ValueError, match="offline token counter"):
        diagnose_handoff(
            _contract(COMPAT),
            COMPAT,
            "Candidate",
            _compatibility_check,
            counter=NeverCallCounter(),
        )
