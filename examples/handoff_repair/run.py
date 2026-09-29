"""A complete, offline handoff diagnosis with executable retry checks.

Run from the repository root: python examples/handoff_repair/run.py
No model, network, or paid API is used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow the checkout example to run before an editable package installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from laconic.eval.handoff_experiment import generate_cases
from laconic.handoff import audit_handoffs, diagnose_handoff


def main() -> None:
    case = next(case for case in generate_cases() if case.fault == "lost_two_requirements")
    report = audit_handoffs(case.contract, case.stages)
    result = diagnose_handoff(
        case.contract, case.original, case.candidate, case.validate, max_evaluations=8
    )
    print("Laconic handoff diagnosis (scripted receiver; no language model)")
    for loss in report.requirement_losses:
        if loss.first_missing_index is not None:
            print(f"{loss.requirement_id} first missing at {loss.first_missing_stage}")
    for attempt in result.attempts:
        outcome = attempt.outcome
        print(f"{attempt.phase}: {'PASS' if outcome and outcome.success else 'FAIL'}")
    print(f"Result: {result.status}; restored {', '.join(result.added_requirement_ids)}")
    print(f"Validation calls: {result.evaluations_used}/{result.max_evaluations}")
    print(result.guarantee)
    output = Path("results/handoff-demo")
    output.mkdir(parents=True, exist_ok=True)
    trace = {
        "contract": case.contract.model_dump(mode="json"),
        "handoffs": [h.model_dump(mode="json") for h in case.stages],
    }
    (output / "trace.json").write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    (output / "replay.json").write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"Evidence and replay record: {output}")


if __name__ == "__main__":
    main()
