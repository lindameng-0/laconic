"""Eval harness: deterministic generation, mock scoring, matrix integrity.

These tests also encode the study's *sanity expectations* on the offline
information-survival metric: passthrough must score ~perfectly, Laconic must
keep tool-plan tasks intact at every ratio, and the naive baseline must break
them. If those relationships ever fail to hold, the harness itself is broken.
"""

from __future__ import annotations

from pathlib import Path

from laconic.eval.clients import MockModelClient
from laconic.eval.generate import generate_tasks
from laconic.eval.metrics import results_markdown, safe_operating_points, summarize
from laconic.eval.runner import MatrixSpec, run_matrix, write_results
from laconic.eval.tasks import load_tasks, save_tasks


def test_generation_is_deterministic() -> None:
    a = generate_tasks(seed=7, per_family=5)
    b = generate_tasks(seed=7, per_family=5)
    assert [t.model_dump() for t in a] == [t.model_dump() for t in b]
    c = generate_tasks(seed=8, per_family=5)
    assert [t.model_dump() for t in a] != [t.model_dump() for t in c]


def test_task_families_and_shapes() -> None:
    tasks = generate_tasks(seed=7, per_family=4)
    families = {t.family for t in tasks}
    assert families == {"extraction_relay", "tool_plan_handoff", "constraint_tracking"}
    for task in tasks:
        assert task.handoff_message.get("role")
        if task.family == "tool_plan_handoff":
            assert task.handoff_message.get("tool_calls")


def test_save_load_roundtrip(tmp_path: Path) -> None:
    tasks = generate_tasks(seed=7, per_family=3)
    path = tmp_path / "tasks.jsonl"
    save_tasks(tasks, path)
    loaded = load_tasks(path)
    assert [t.model_dump() for t in loaded] == [t.model_dump() for t in tasks]


def _small_matrix() -> tuple[list, list]:
    tasks = generate_tasks(seed=7, per_family=6)
    spec = MatrixSpec(models=["mock"], keep_ratios=[0.6, 0.3])
    records = run_matrix(tasks, MockModelClient(), spec)
    return tasks, records


def test_matrix_shape() -> None:
    tasks, records = _small_matrix()
    # cells: passthrough + conservative + (budgeted + naive) * 2 ratios = 6
    assert len(records) == len(tasks) * 6


def test_passthrough_accuracy_is_high() -> None:
    _, records = _small_matrix()
    passthrough = [r for r in records if r.strategy == "passthrough"]
    accuracy = sum(r.correct for r in passthrough) / len(passthrough)
    assert accuracy >= 0.95, (
        f"passthrough accuracy {accuracy:.2f} — the mock extractor or the "
        f"generator drifted; failures: "
        f"{[(r.task_id, r.answer[:80]) for r in passthrough if not r.correct][:5]}"
    )


def test_laconic_preserves_tool_plans_naive_breaks_them() -> None:
    _, records = _small_matrix()

    def accuracy(strategy: str, family: str, ratio: float | None = None) -> float:
        cell = [
            r
            for r in records
            if r.strategy == strategy
            and r.family == family
            and (ratio is None or r.keep_ratio == ratio)
        ]
        return sum(r.correct for r in cell) / len(cell)

    # Structure preservation: Laconic keeps tool plans perfect at every ratio.
    assert accuracy("laconic-budgeted", "tool_plan_handoff") == 1.0
    # The naive baseline destroys them at aggressive ratios.
    assert accuracy("naive", "tool_plan_handoff", 0.3) < 0.5


def test_budgeted_compression_actually_compresses() -> None:
    _, records = _small_matrix()
    budgeted = [r for r in records if r.strategy == "laconic-budgeted" and r.keep_ratio == 0.3]
    mean_ratio = sum(r.tokens_after / r.tokens_before for r in budgeted) / len(budgeted)
    assert mean_ratio < 0.9


def test_results_files_and_markdown(tmp_path: Path) -> None:
    _, records = _small_matrix()
    json_path, csv_path = write_results(records, tmp_path)
    assert json_path.exists() and csv_path.exists()
    summaries = summarize(records)
    markdown = results_markdown(summaries)
    assert "| model | strategy |" in markdown
    assert "passthrough" in markdown
    points = safe_operating_points(summaries)
    assert any(p.strategy.startswith("laconic") for p in points)
