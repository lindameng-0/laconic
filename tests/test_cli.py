"""CLI smoke tests (offline)."""

from __future__ import annotations

import json
from pathlib import Path

from laconic.cli import main

VERBOSE = (
    "Please note that the migration finished. It is worth noting that, for "
    "example, 14 tables moved cleanly. Basically, downtime was 90 seconds. "
    "In order to close the ticket, attach the runbook link please."
)


def test_version(capsys) -> None:
    assert main(["version"]) == 0
    from laconic import __version__

    assert capsys.readouterr().out.strip() == __version__


def test_profile_command(tmp_path: Path, capsys) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps({"role": "assistant", "content": VERBOSE}) + "\n", encoding="utf-8")
    html = tmp_path / "report.html"
    code = main(
        [
            "profile",
            str(trace),
            "--model",
            "test-model",
            "--strategy",
            "conservative",
            "--html",
            str(html),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "Laconic profile" in out
    assert html.exists()


def test_eval_command(tmp_path: Path, capsys) -> None:
    from laconic.eval.generate import generate_tasks
    from laconic.eval.tasks import save_tasks

    tasks_dir = tmp_path / "benchmark"
    save_tasks(generate_tasks(seed=7, per_family=2), tasks_dir / "tasks.jsonl")
    out_dir = tmp_path / "results"
    code = main(["eval", "--tasks", str(tasks_dir), "--out", str(out_dir), "--ratios", "0.5"])
    assert code == 0
    assert (out_dir / "results.json").exists()
    assert (out_dir / "results.csv").exists()
    assert (out_dir / "summary.md").exists()
    assert "passthrough" in capsys.readouterr().out
