"""Importable validator for the seed-7 demo trace, not a general task checker."""

from laconic.eval.handoff_experiment import generate_cases
from laconic.handoff import ReplayOutcome

_CASE = next(case for case in generate_cases() if case.fault == "lost_two_requirements")


def validate(text: str) -> ReplayOutcome:
    """Execute the demo's fixed retry-service probes from a clean state."""
    return _CASE.validate(text)
