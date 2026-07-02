"""Profiler: records, aggregation, reports, trace analysis."""

from __future__ import annotations

import json
from pathlib import Path

from laconic.pipeline import Session
from laconic.profiler.records import WorkflowProfile
from laconic.profiler.report import summary_table, to_html
from laconic.profiler.trace import profile_trace, read_trace

VERBOSE = (
    "Please note that the analysis is complete. It is worth noting that the "
    "top finding, for example, concerns retention. Basically, retention "
    "dropped 3.1 points in Q2 2026 for the mid-market cohort specifically."
)


def _profile() -> WorkflowProfile:
    session = Session(target_model="test-model", strategy="conservative")
    profile = WorkflowProfile()
    for target in ("writer", "writer", "reviewer"):
        processed = session.process(
            {"role": "assistant", "content": VERBOSE},
            source_agent="researcher",
            target_agent=target,
        )
        profile.add(processed.stats, source_agent="researcher", target_agent=target)
    return profile


def test_aggregates() -> None:
    profile = _profile()
    assert len(profile.records) == 3
    assert profile.tokens_before > profile.tokens_after
    assert 0 < profile.overall_ratio <= 1
    edges = profile.by_edge()
    assert set(edges) == {"researcher->writer", "researcher->reviewer"}
    assert edges["researcher->writer"]["handoffs"] == 2


def test_summary_table_contents() -> None:
    table = summary_table(_profile())
    assert "researcher->writer" in table
    assert "test-model" in table
    assert "estimate" in table  # tier disclosure for unknown models


def test_html_report(tmp_path: Path) -> None:
    out = to_html(_profile(), tmp_path / "report.html")
    html = out.read_text(encoding="utf-8")
    assert "Laconic profile" in html
    assert "researcher-&gt;writer" in html  # html-escaped edge label


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    profile = _profile()
    path = tmp_path / "profile.jsonl"
    profile.to_jsonl(path)
    loaded = WorkflowProfile.from_jsonl(path)
    assert loaded.tokens_before == profile.tokens_before
    assert [r.edge for r in loaded.records] == [r.edge for r in profile.records]


def test_trace_profiling(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    lines = [
        json.dumps({"role": "assistant", "content": VERBOSE}),
        json.dumps(
            {
                "source": "planner",
                "target": "executor",
                "message": {"role": "assistant", "content": VERBOSE},
            }
        ),
    ]
    trace.write_text("\n".join(lines), encoding="utf-8")
    entries = read_trace(trace)
    assert len(entries) == 2
    profile = profile_trace(entries, model="test-model", strategy="off")
    assert profile.tokens_before == profile.tokens_after  # 'off' measures only
    whatif = profile_trace(entries, model="test-model", strategy="conservative")
    assert whatif.tokens_after < whatif.tokens_before
