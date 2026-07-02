"""Metrics over run records: accuracy curves and safe operating points."""

from __future__ import annotations

from dataclasses import dataclass

from laconic.eval.runner import RunRecord

#: Accuracy may drop at most this much below the passthrough baseline for a
#: cell to count as "safe".
DEFAULT_TOLERANCE = 0.02


@dataclass(frozen=True)
class CellSummary:
    """Aggregate for one (model, strategy, keep_ratio) cell."""

    model: str
    strategy: str
    keep_ratio: float | None
    tasks: int
    accuracy: float
    mean_token_ratio: float
    fallback_rate: float


def summarize(records: list[RunRecord]) -> list[CellSummary]:
    """Aggregate records into per-cell summaries."""
    cells: dict[tuple[str, str, float | None], list[RunRecord]] = {}
    for record in records:
        cells.setdefault((record.model, record.strategy, record.keep_ratio), []).append(record)

    summaries: list[CellSummary] = []
    for (model, strategy, ratio), cell_records in sorted(
        cells.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or 1.0)
    ):
        n = len(cell_records)
        accuracy = sum(r.correct for r in cell_records) / n
        token_ratio = (
            sum(
                (r.tokens_after / r.tokens_before) if r.tokens_before else 1.0 for r in cell_records
            )
            / n
        )
        fallback_rate = sum(r.fell_back for r in cell_records) / n
        summaries.append(
            CellSummary(
                model=model,
                strategy=strategy,
                keep_ratio=ratio,
                tasks=n,
                accuracy=accuracy,
                mean_token_ratio=token_ratio,
                fallback_rate=fallback_rate,
            )
        )
    return summaries


def baseline_accuracy(summaries: list[CellSummary], model: str) -> float | None:
    """Passthrough accuracy for ``model``, or ``None`` if not in the run."""
    for cell in summaries:
        if cell.model == model and cell.strategy == "passthrough":
            return cell.accuracy
    return None


@dataclass(frozen=True)
class SafeOperatingPoint:
    """The most aggressive cell for a (model, strategy) that stayed safe."""

    model: str
    strategy: str
    keep_ratio: float | None
    accuracy: float
    baseline: float
    mean_token_ratio: float


def safe_operating_points(
    summaries: list[CellSummary],
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[SafeOperatingPoint]:
    """Per (model, strategy): the lowest measured *token* ratio whose accuracy
    stays within ``tolerance`` of that model's passthrough baseline.

    Cells for models with no passthrough baseline are skipped — a safe point
    is only meaningful relative to a measured baseline.
    """
    points: list[SafeOperatingPoint] = []
    pairs = sorted({(c.model, c.strategy) for c in summaries if c.strategy != "passthrough"})
    for model, strategy in pairs:
        baseline = baseline_accuracy(summaries, model)
        if baseline is None:
            continue
        safe_cells = [
            c
            for c in summaries
            if c.model == model and c.strategy == strategy and c.accuracy >= baseline - tolerance
        ]
        if not safe_cells:
            continue
        best = min(safe_cells, key=lambda c: c.mean_token_ratio)
        points.append(
            SafeOperatingPoint(
                model=model,
                strategy=strategy,
                keep_ratio=best.keep_ratio,
                accuracy=best.accuracy,
                baseline=baseline,
                mean_token_ratio=best.mean_token_ratio,
            )
        )
    return points


def results_markdown(summaries: list[CellSummary], tolerance: float = DEFAULT_TOLERANCE) -> str:
    """Render summaries as the markdown results table used in the README/docs."""
    lines = [
        "| model | strategy | keep ratio | tasks | accuracy | mean token ratio | fallbacks |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for cell in summaries:
        ratio = "—" if cell.keep_ratio is None else f"{cell.keep_ratio:.2f}"
        lines.append(
            f"| {cell.model} | {cell.strategy} | {ratio} | {cell.tasks} "
            f"| {cell.accuracy:.3f} | {cell.mean_token_ratio:.3f} "
            f"| {cell.fallback_rate:.1%} |"
        )
    lines.append("")
    for point in safe_operating_points(summaries, tolerance):
        ratio = "—" if point.keep_ratio is None else f"{point.keep_ratio:.2f}"
        lines.append(
            f"- **{point.model} / {point.strategy}** safe operating point: "
            f"keep ratio {ratio} -> {point.mean_token_ratio:.3f} of baseline tokens "
            f"at accuracy {point.accuracy:.3f} (baseline {point.baseline:.3f})"
        )
    return "\n".join(lines)
