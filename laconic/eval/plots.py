"""Plotting for the study's headline figures (requires ``laconic[plots]``).

Figure 1 — per model: accuracy vs. achieved token ratio, one line per
strategy. The gap between the ``laconic-budgeted`` and ``naive`` lines is the
structure-preservation effect; where each line crosses the tolerance band is
its safe frontier.
"""

from __future__ import annotations

from pathlib import Path

from laconic.eval.metrics import CellSummary, baseline_accuracy
from laconic.exceptions import MissingDependencyError

_STRATEGY_STYLES = {
    "laconic-conservative": {"color": "#2a7de1", "marker": "s"},
    "laconic-budgeted": {"color": "#1a9e5c", "marker": "o"},
    "naive": {"color": "#c0392b", "marker": "x"},
}


def plot_accuracy_vs_ratio(
    summaries: list[CellSummary],
    out_dir: str | Path,
    tolerance: float = 0.02,
) -> list[Path]:
    """Write one accuracy-vs-token-ratio figure per model; return the paths."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise MissingDependencyError("Plotting", "plots") from exc

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for model in sorted({c.model for c in summaries}):
        baseline = baseline_accuracy(summaries, model)
        fig, ax = plt.subplots(figsize=(7, 4.5))

        for strategy, style in _STRATEGY_STYLES.items():
            cells = sorted(
                (c for c in summaries if c.model == model and c.strategy == strategy),
                key=lambda c: c.mean_token_ratio,
            )
            if not cells:
                continue
            ax.plot(
                [c.mean_token_ratio for c in cells],
                [c.accuracy for c in cells],
                label=strategy,
                **style,
            )

        if baseline is not None:
            ax.axhline(baseline, color="#888", linestyle="--", linewidth=1, label="passthrough")
            ax.axhspan(baseline - tolerance, baseline, color="#888", alpha=0.12)

        ax.set_xlabel("achieved token ratio (compressed / original)")
        ax.set_ylabel("task accuracy")
        ax.set_title(f"Compression tolerance — {model}")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.invert_xaxis()  # more compression to the right
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(alpha=0.25)
        fig.tight_layout()

        path = out / f"tolerance_{model.replace('/', '_')}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths
