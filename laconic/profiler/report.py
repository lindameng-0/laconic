"""Report rendering: CLI summary table and self-contained HTML report."""

from __future__ import annotations

import html
from pathlib import Path

from laconic.adapters.profiles import PRICING_AS_OF, estimate_cost
from laconic.profiler.records import WorkflowProfile


def _fmt_cost(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.4f}"


def _table(rows: list[list[str]], headers: list[str]) -> str:
    """Render a plain-text table (no third-party dependency)."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep = "  ".join("-" * w for w in widths)
    body = ["  ".join(c.ljust(widths[i]) for i, c in enumerate(row)) for row in rows]
    return "\n".join([line, sep, *body])


def summary_table(profile: WorkflowProfile) -> str:
    """Human-readable summary of a workflow run for the terminal."""
    if not profile.records:
        return "No handoffs recorded."

    model = profile.records[0].stats.model
    strategy = profile.records[0].stats.strategy
    tiers = ", ".join(sorted(profile.count_tiers))

    edge_rows = []
    for edge, agg in sorted(profile.by_edge().items(), key=lambda kv: -kv[1]["tokens_before"]):
        saved = agg["tokens_before"] - agg["tokens_after"]
        pct = 100 * saved / agg["tokens_before"] if agg["tokens_before"] else 0.0
        edge_rows.append(
            [
                edge,
                str(agg["handoffs"]),
                f"{agg['tokens_before']:,}",
                f"{agg['tokens_after']:,}",
                f"{saved:,} ({pct:.1f}%)",
                str(agg["fallbacks"]),
            ]
        )

    cost_before = estimate_cost(profile.tokens_before, model)
    cost_after = estimate_cost(profile.tokens_after, model)

    # Terminal output stays ASCII: Windows consoles often run cp1252 and
    # crash on arrows and dashes.
    lines = [
        f"Laconic profile - model: {model}, strategy: {strategy}, token counts: {tiers}",
        "",
        _table(
            edge_rows,
            ["edge", "handoffs", "tok before", "tok after", "saved", "fallbacks"],
        ),
        "",
        f"total: {profile.tokens_before:,} -> {profile.tokens_after:,} tokens "
        f"({profile.tokens_saved:,} saved, ratio {profile.overall_ratio:.3f}), "
        f"{profile.fallback_count} fallback(s)",
        f"est. input cost: {_fmt_cost(cost_before)} -> {_fmt_cost(cost_after)} "
        f"(pricing as of {PRICING_AS_OF}; input-side only)",
    ]
    if "estimate" in profile.count_tiers:
        lines.append(
            "note: some counts are offline estimates (~15-20% error) - install the "
            "model's tokenizer extra or enable API counting for exact numbers"
        )
    return "\n".join(lines)


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Laconic profile</title>
<style>
 body {{ font: 14px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 60rem;
        color: #1a1a2e; padding: 0 1rem; }}
 h1 {{ font-size: 1.4rem; }} h2 {{ font-size: 1.1rem; margin-top: 2rem; }}
 table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
 th, td {{ text-align: left; padding: .4rem .6rem; border-bottom: 1px solid #ddd; }}
 th {{ background: #f4f4f8; }}
 .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
 .bar {{ background: #e8e8f0; border-radius: 3px; overflow: hidden; height: 10px; }}
 .bar > div {{ background: #5568d3; height: 100%; }}
 .note {{ color: #666; font-size: .85rem; }}
 .fallback {{ color: #b3261e; }}
</style></head><body>
<h1>Laconic profile</h1>
<p>model <b>{model}</b> · strategy <b>{strategy}</b> · token counts: {tiers}</p>
<p><b>{tokens_before:,}</b> → <b>{tokens_after:,}</b> tokens
 (<b>{tokens_saved:,}</b> saved, ratio {ratio:.3f}) ·
 est. input cost {cost_before} → {cost_after}
 <span class="note">(pricing as of {pricing_as_of}, input-side only)</span></p>
<h2>Tokens by edge</h2>
<table><tr><th>edge</th><th class="num">handoffs</th><th class="num">before</th>
<th class="num">after</th><th class="num">saved</th><th style="width:30%">share of spend</th></tr>
{edge_rows}
</table>
<h2>Handoffs</h2>
<table><tr><th class="num">#</th><th>edge</th><th class="num">before</th>
<th class="num">after</th><th class="num">ratio</th><th>tier</th><th>notes</th></tr>
{handoff_rows}
</table>
<p class="note">{estimate_note}</p>
</body></html>
"""


def to_html(profile: WorkflowProfile, path: str | Path) -> Path:
    """Write a self-contained HTML report; returns the path written."""
    path = Path(path)
    if not profile.records:
        path.write_text("<p>No handoffs recorded.</p>", encoding="utf-8")
        return path

    model = profile.records[0].stats.model
    strategy = profile.records[0].stats.strategy
    max_edge = max(agg["tokens_before"] for agg in profile.by_edge().values()) or 1

    edge_rows = []
    for edge, agg in sorted(profile.by_edge().items(), key=lambda kv: -kv[1]["tokens_before"]):
        saved = agg["tokens_before"] - agg["tokens_after"]
        width = 100 * agg["tokens_before"] / max_edge
        edge_rows.append(
            f"<tr><td>{html.escape(edge)}</td>"
            f"<td class='num'>{agg['handoffs']}</td>"
            f"<td class='num'>{agg['tokens_before']:,}</td>"
            f"<td class='num'>{agg['tokens_after']:,}</td>"
            f"<td class='num'>{saved:,}</td>"
            f"<td><div class='bar'><div style='width:{width:.0f}%'></div></div></td></tr>"
        )

    handoff_rows = []
    for record in profile.records:
        stats = record.stats
        note = ""
        if stats.fell_back:
            reason = html.escape(stats.fallback_reason or "")
            note = f"<span class='fallback'>fallback: {reason}</span>"
        elif stats.dedup_hits:
            note = f"{stats.dedup_hits} dedup hit(s)"
        handoff_rows.append(
            f"<tr><td class='num'>{record.index}</td>"
            f"<td>{html.escape(record.edge)}</td>"
            f"<td class='num'>{stats.tokens_before:,}</td>"
            f"<td class='num'>{stats.tokens_after:,}</td>"
            f"<td class='num'>{stats.ratio:.3f}</td>"
            f"<td>{stats.tier.value}</td><td>{note}</td></tr>"
        )

    estimate_note = ""
    if "estimate" in profile.count_tiers:
        estimate_note = (
            "Some token counts are offline estimates (±15–20%). Install the "
            "model's tokenizer extra or enable API counting for exact numbers."
        )

    path.write_text(
        _HTML_TEMPLATE.format(
            model=html.escape(model),
            strategy=html.escape(strategy),
            tiers=", ".join(sorted(profile.count_tiers)),
            tokens_before=profile.tokens_before,
            tokens_after=profile.tokens_after,
            tokens_saved=profile.tokens_saved,
            ratio=profile.overall_ratio,
            cost_before=_fmt_cost(estimate_cost(profile.tokens_before, model)),
            cost_after=_fmt_cost(estimate_cost(profile.tokens_after, model)),
            pricing_as_of=PRICING_AS_OF,
            edge_rows="\n".join(edge_rows),
            handoff_rows="\n".join(handoff_rows),
            estimate_note=estimate_note,
        ),
        encoding="utf-8",
    )
    return path
