"""The research harness: cross-model compression-tolerance study."""

from laconic.eval.clients import AnthropicClient, MockModelClient, ModelClient, OpenAIClient
from laconic.eval.generate import generate_tasks
from laconic.eval.metrics import (
    CellSummary,
    SafeOperatingPoint,
    results_markdown,
    safe_operating_points,
    summarize,
)
from laconic.eval.runner import MatrixSpec, RunRecord, run_matrix, write_results
from laconic.eval.tasks import BenchmarkTask, Expected, load_tasks, save_tasks

__all__ = [
    "AnthropicClient",
    "BenchmarkTask",
    "CellSummary",
    "Expected",
    "MatrixSpec",
    "MockModelClient",
    "ModelClient",
    "OpenAIClient",
    "RunRecord",
    "SafeOperatingPoint",
    "generate_tasks",
    "load_tasks",
    "results_markdown",
    "run_matrix",
    "safe_operating_points",
    "save_tasks",
    "summarize",
    "write_results",
]
