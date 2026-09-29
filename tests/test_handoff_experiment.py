"""Executable acceptance checks for the scripted receiver and diagnosis loop."""

from __future__ import annotations

import json
import sys
import types

import pytest

from laconic.cli import main
from laconic.eval.handoff_experiment import generate_cases, run_experiment, summarize_experiment
from laconic.handoff import audit_handoff, diagnose_handoff


def test_retry_execution_distinguishes_omissions_conflicts_and_paraphrases() -> None:
    cases = {c.fault: c for c in generate_cases(19, 1)}
    for case in cases.values():
        assert case.validate(case.original).success
        assert case.validate(case.concise).success
    for fault in ("lost_negation", "lost_attempt_limit", "stale_delay_limit"):
        case = cases[fault]
        assert not case.validate(case.candidate).success
        repaired = diagnose_handoff(case.contract, case.original, case.candidate, case.validate)
        assert repaired.status == "repaired"
        assert case.validate(repaired.text).success
    conflict = cases["contradictory_implementation"]
    assert not audit_handoff(conflict.contract, conflict.candidate).missing_requirement_ids
    result = diagnose_handoff(
        conflict.contract, conflict.original, conflict.candidate, conflict.validate
    )
    assert result.status == "unresolved"
    paraphrase = cases["equivalent_paraphrase"]
    assert audit_handoff(paraphrase.contract, paraphrase.candidate).missing_requirement_ids
    result = diagnose_handoff(
        paraphrase.contract, paraphrase.original, paraphrase.candidate, paraphrase.validate
    )
    assert result.status == "candidate_passed"
    assert result.text == paraphrase.candidate


def test_experiment_records_strong_baseline_and_failed_repairs() -> None:
    result = run_experiment(seed=31, variants=1)
    assert result["cases"] == 8
    assert "not an LLM" in result["receiver"]
    assert all(r["policies"]["concise_contract"]["outcome"]["success"] for r in result["records"])
    assert any(r["replay"]["status"] == "unresolved" for r in result["records"])
    assert "diagnostic overhead" in summarize_experiment(result)
    assert "not API billing" in summarize_experiment(result)


def _trace_file(tmp_path):
    case = next(c for c in generate_cases(19, 1) if c.fault == "lost_negation")
    path = tmp_path / "trace.json"
    path.write_text(
        json.dumps(
            {
                "contract": case.contract.model_dump(mode="json"),
                "handoffs": [h.model_dump(mode="json") for h in case.stages],
            }
        ),
        encoding="utf-8",
    )
    return case, path


def test_audit_and_replay_cli_use_caller_validator(tmp_path, monkeypatch, capsys) -> None:
    case, path = _trace_file(tmp_path)
    assert main(["audit", str(path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["final_missing_requirement_ids"] == ["r2"]
    module = types.ModuleType("test_workflow_validator")
    module.validate = case.validate
    monkeypatch.setitem(sys.modules, module.__name__, module)
    output = tmp_path / "result.json"
    assert (
        main(
            [
                "replay",
                str(path),
                "--validator",
                "test_workflow_validator:validate",
                "--out",
                str(output),
                "--max-evaluations",
                "3",
            ]
        )
        == 0
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "repaired"
    assert result["evaluations_used"] == 3
    assert result["source_provenance"]


def test_replay_refuses_overwriting_evidence_before_importing_validator(tmp_path, capsys) -> None:
    _, path = _trace_file(tmp_path)
    original = path.read_bytes()
    assert (
        main(
            [
                "replay",
                str(path),
                "--validator",
                "missing.module:validate",
                "--out",
                str(path),
            ]
        )
        == 2
    )
    assert path.read_bytes() == original
    assert "overwrite" in capsys.readouterr().err


def test_audit_rejects_invalid_source_quotes(tmp_path, capsys) -> None:
    _, path = _trace_file(tmp_path)
    trace = json.loads(path.read_text(encoding="utf-8"))
    trace["contract"]["requirements"][0]["quote"] = "An invented requirement."
    path.write_text(json.dumps(trace), encoding="utf-8")
    assert main(["audit", str(path)]) == 2
    assert "verbatim" in capsys.readouterr().err


def test_experiment_cli_and_budget_validation(tmp_path, capsys) -> None:
    output = tmp_path / "experiment.json"
    assert main(["experiment", "--variants", "1", "--out", str(output)]) == 0
    assert json.loads(output.read_text())["cases"] == 8
    assert "not an LLM" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["experiment", "--max-evaluations", "0"])
