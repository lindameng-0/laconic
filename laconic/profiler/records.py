"""Profiler data model: one record per handoff, aggregated per workflow run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from laconic.adapters.profiles import estimate_cost
from laconic.message.model import CompressionStats


class HandoffRecord(BaseModel):
    """Everything measured about one inter-agent handoff."""

    model_config = ConfigDict(protected_namespaces=())

    index: int
    source_agent: str | None = None
    target_agent: str | None = None
    stats: CompressionStats

    @property
    def edge(self) -> str:
        """The workflow edge this handoff traveled, e.g. ``planner->executor``.

        ASCII on purpose: this string ends up on terminals whose encoding
        (e.g. cp1252 on Windows) cannot print an arrow character.
        """
        return f"{self.source_agent or '?'}->{self.target_agent or '?'}"

    @property
    def cost_before(self) -> float | None:
        """Estimated USD input cost of the unprocessed message."""
        return estimate_cost(self.stats.tokens_before, self.stats.model)

    @property
    def cost_after(self) -> float | None:
        """Estimated USD input cost of the processed message."""
        return estimate_cost(self.stats.tokens_after, self.stats.model)


class WorkflowProfile(BaseModel):
    """Aggregated measurement of one workflow run."""

    records: list[HandoffRecord] = Field(default_factory=list)

    def add(
        self,
        stats: CompressionStats,
        *,
        source_agent: str | None = None,
        target_agent: str | None = None,
    ) -> HandoffRecord:
        """Append a handoff record."""
        record = HandoffRecord(
            index=len(self.records),
            source_agent=source_agent,
            target_agent=target_agent,
            stats=stats,
        )
        self.records.append(record)
        return record

    # -- aggregates ---------------------------------------------------------

    @property
    def tokens_before(self) -> int:
        return sum(r.stats.tokens_before for r in self.records)

    @property
    def tokens_after(self) -> int:
        return sum(r.stats.tokens_after for r in self.records)

    @property
    def tokens_saved(self) -> int:
        return self.tokens_before - self.tokens_after

    @property
    def overall_ratio(self) -> float:
        if self.tokens_before == 0:
            return 1.0
        return self.tokens_after / self.tokens_before

    @property
    def fallback_count(self) -> int:
        return sum(1 for r in self.records if r.stats.fell_back)

    @property
    def count_tiers(self) -> set[str]:
        """Which token-count tiers appear in this profile (never mix silently)."""
        return {r.stats.tier.value for r in self.records}

    def by_edge(self) -> dict[str, dict[str, Any]]:
        """Aggregate tokens per workflow edge — where the tokens actually burn."""
        edges: dict[str, dict[str, Any]] = {}
        for record in self.records:
            agg = edges.setdefault(
                record.edge,
                {"handoffs": 0, "tokens_before": 0, "tokens_after": 0, "fallbacks": 0},
            )
            agg["handoffs"] += 1
            agg["tokens_before"] += record.stats.tokens_before
            agg["tokens_after"] += record.stats.tokens_after
            agg["fallbacks"] += int(record.stats.fell_back)
        return edges

    # -- persistence --------------------------------------------------------

    def to_jsonl(self, path: str | Path) -> None:
        """Write one JSON record per handoff."""
        path = Path(path)
        with path.open("w", encoding="utf-8") as fh:
            for record in self.records:
                fh.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False))
                fh.write("\n")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> WorkflowProfile:
        """Load a profile previously written by :meth:`to_jsonl`."""
        profile = cls()
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    profile.records.append(HandoffRecord.model_validate_json(line))
        return profile
