"""Source-backed, exact-text checks at agent handoff boundaries.

These checks establish only *verbatim evidence survival*. A present quote may
be ignored, contradicted, or misunderstood by the receiver. Neither auditing
nor appending a source clause proves semantic preservation or task compliance.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from laconic.tokenizers.base import CountTier, TokenCounter
from laconic.tokenizers.heuristic import HeuristicCounter

GUARANTEE = "verbatim evidence survival"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")


class Evidence(_FrozenModel):
    """An immutable source snapshot, identified independently of its contents."""

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source: str = Field(min_length=1)

    @field_validator("id", "text", "source")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evidence fields must not be blank")
        return value

    @property
    def sha256(self) -> str:
        """SHA-256 of the exact source text encoded as UTF-8."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


class Requirement(_FrozenModel):
    """A caller-selected clause that must survive exactly, including negation.

    Source membership is checked by :class:`HandoffContract`. Selection is
    explicit: this library cannot infer which clauses are important or current.
    """

    id: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)

    @field_validator("id", "evidence_id", "quote")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("requirement fields must not be blank")
        return value


class HandoffContract(_FrozenModel):
    """Validated evidence snapshots and source-backed required clauses.

    Lists are accepted on input and stored as tuples, so the validated
    collection cannot subsequently be mutated. Conflicting versions remain
    separate evidence IDs; this class never chooses which version is authoritative.
    """

    evidence: tuple[Evidence, ...] = Field(min_length=1)
    requirements: tuple[Requirement, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_references(self) -> HandoffContract:
        evidence_by_id = {item.id: item for item in self.evidence}
        if len(evidence_by_id) != len(self.evidence):
            raise ValueError("evidence IDs must be unique")
        if len({item.id for item in self.requirements}) != len(self.requirements):
            raise ValueError("requirement IDs must be unique")
        for requirement in self.requirements:
            evidence = evidence_by_id.get(requirement.evidence_id)
            if evidence is None:
                raise ValueError(
                    f"requirement {requirement.id!r} cites unknown evidence "
                    f"{requirement.evidence_id!r}"
                )
            if requirement.quote not in evidence.text:
                raise ValueError(
                    f"requirement {requirement.id!r} quote does not occur verbatim "
                    f"in evidence {evidence.id!r}"
                )
        return self

    @property
    def sha256(self) -> str:
        """Fingerprint the complete contract, including sources and clause order."""
        encoded = json.dumps(
            self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class Handoff(_FrozenModel):
    stage: str = Field(min_length=1)
    text: str

    @field_validator("stage")
    @classmethod
    def nonblank_stage(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("stage must not be blank")
        return value


class EvidenceProvenance(_FrozenModel):
    id: str
    source: str
    sha256: str


class HandoffAudit(_FrozenModel):
    stage: str
    present_requirement_ids: tuple[str, ...]
    missing_requirement_ids: tuple[str, ...]


class RequirementLoss(_FrozenModel):
    requirement_id: str
    evidence_id: str
    first_missing_stage: str | None
    first_missing_index: int | None
    final_present: bool


class HandoffAuditReport(_FrozenModel):
    guarantee: Literal["verbatim evidence survival"] = GUARANTEE
    contract_sha256: str
    evidence: tuple[EvidenceProvenance, ...]
    handoffs: tuple[HandoffAudit, ...]
    requirement_losses: tuple[RequirementLoss, ...]
    final_missing_requirement_ids: tuple[str, ...]


class RepairProvenance(_FrozenModel):
    requirement_id: str
    evidence_id: str
    source: str
    evidence_sha256: str
    quote: str


class RepairResult(_FrozenModel):
    guarantee: Literal["verbatim evidence survival"] = GUARANTEE
    contract_sha256: str
    text: str
    restored_requirement_ids: tuple[str, ...]
    unresolved_requirement_ids: tuple[str, ...]
    added_tokens: int
    count_tier: CountTier
    count_model: str
    provenance: tuple[RepairProvenance, ...]


def audit_handoff(contract: HandoffContract, text: str, stage: str = "handoff") -> HandoffAudit:
    """Check case-sensitive, byte-for-byte-equivalent text membership.

    No whitespace normalization, paraphrase matching, entailment, or contradiction
    detection is performed. A quote inside a negated assertion still survives.
    """
    handoff = Handoff(stage=stage, text=text)
    return HandoffAudit(
        stage=handoff.stage,
        present_requirement_ids=tuple(
            r.id for r in contract.requirements if r.quote in handoff.text
        ),
        missing_requirement_ids=tuple(
            r.id for r in contract.requirements if r.quote not in handoff.text
        ),
    )


def audit_handoffs(contract: HandoffContract, handoffs: Iterable[Handoff]) -> HandoffAuditReport:
    """Locate each clause's first observed absence and its final presence.

    An absence at index zero means the clause was already missing at the first
    observed boundary; it does not attribute the loss to an unseen earlier stage.
    Repeated stage names are disambiguated by zero-based indices.
    """
    audits = tuple(audit_handoff(contract, h.text, h.stage) for h in handoffs)
    if not audits:
        raise ValueError("at least one handoff is required")
    losses: list[RequirementLoss] = []
    for requirement in contract.requirements:
        first_missing = next(
            (
                i
                for i, audit in enumerate(audits)
                if requirement.id in audit.missing_requirement_ids
            ),
            None,
        )
        losses.append(
            RequirementLoss(
                requirement_id=requirement.id,
                evidence_id=requirement.evidence_id,
                first_missing_index=first_missing,
                first_missing_stage=audits[first_missing].stage
                if first_missing is not None
                else None,
                final_present=requirement.id in audits[-1].present_requirement_ids,
            )
        )
    return HandoffAuditReport(
        contract_sha256=contract.sha256,
        evidence=tuple(
            EvidenceProvenance(id=e.id, source=e.source, sha256=e.sha256) for e in contract.evidence
        ),
        handoffs=audits,
        requirement_losses=tuple(losses),
        final_missing_requirement_ids=audits[-1].missing_requirement_ids,
    )


def repair_handoff(
    contract: HandoffContract,
    text: str,
    requirement_ids: Iterable[str] | None = None,
    max_added_tokens: int | None = None,
    counter: TokenCounter | None = None,
) -> RepairResult:
    """Append missing source clauses in contract order, within a token-growth cap.

    Clauses and their citations are added whole. Clauses that do not fit are
    skipped, allowing later, shorter clauses to fit. Existing text is retained
    exactly. ``unresolved_requirement_ids`` covers the entire contract, including
    requirements not selected for repair. No conflicts are resolved and no facts
    are inferred. API counters are rejected; the default is an offline estimate.

    The cap measures ``max(0, count(result) - count(original))`` using the supplied
    counter and includes citation overhead. It is not a provider billing claim.
    """
    if max_added_tokens is not None and (
        isinstance(max_added_tokens, bool)
        or not isinstance(max_added_tokens, int)
        or max_added_tokens < 0
    ):
        raise ValueError("max_added_tokens must be a nonnegative integer or None")
    if isinstance(requirement_ids, (str, bytes)):
        raise TypeError("requirement_ids must be an iterable of IDs, not a string")
    selected = (
        {r.id for r in contract.requirements} if requirement_ids is None else set(requirement_ids)
    )
    unknown = selected - {r.id for r in contract.requirements}
    if unknown:
        raise ValueError(f"unknown requirement IDs: {sorted(unknown)!r}")
    counter = counter if counter is not None else HeuristicCounter("handoff")
    if counter.tier == CountTier.API:
        raise ValueError("repair_handoff requires an offline token counter")
    original_audit = audit_handoff(contract, text)
    original_tokens = counter.count(text)
    repaired = text
    # Generated citation metadata is not a restored source clause. Otherwise a
    # source label equal to another quote could suppress that quote's repair.
    # Keep bodies separate: joining them could invent a contiguous quotation
    # across an inserted citation that does not occur in the delivered text.
    clause_bodies = [text]
    evidence_by_id = {e.id: e for e in contract.evidence}
    for requirement in contract.requirements:
        if requirement.id not in selected or any(
            requirement.quote in body for body in clause_bodies
        ):
            continue
        evidence = evidence_by_id[requirement.evidence_id]
        citation = json.dumps(
            {"requirement": requirement.id, "evidence": evidence.id, "source": evidence.source},
            ensure_ascii=False,
        )
        candidate = f"{repaired}\n\n[Source clause {citation}]\n{requirement.quote}"
        added = max(0, counter.count(candidate) - original_tokens)
        if max_added_tokens is None or added <= max_added_tokens:
            repaired = candidate
            clause_bodies.append(requirement.quote)
    present_ids = {
        r.id for r in contract.requirements if any(r.quote in body for body in clause_bodies)
    }
    restored = tuple(
        r.id
        for r in contract.requirements
        if r.id in original_audit.missing_requirement_ids and r.id in present_ids
    )
    return RepairResult(
        contract_sha256=contract.sha256,
        text=repaired,
        restored_requirement_ids=restored,
        unresolved_requirement_ids=tuple(
            r.id for r in contract.requirements if r.id not in present_ids
        ),
        added_tokens=max(0, counter.count(repaired) - original_tokens),
        count_tier=counter.tier,
        count_model=counter.model,
        provenance=tuple(
            RepairProvenance(
                requirement_id=r.id,
                evidence_id=r.evidence_id,
                source=evidence_by_id[r.evidence_id].source,
                evidence_sha256=evidence_by_id[r.evidence_id].sha256,
                quote=r.quote,
            )
            for r in contract.requirements
            if r.id in restored
        ),
    )
