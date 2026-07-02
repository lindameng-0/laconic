"""The ``laconic`` command-line interface.

Subcommands:

- ``laconic profile TRACE.jsonl --model MODEL [--strategy S] [--html OUT]`` —
  profile a message trace without integrating anything.
- ``laconic eval --tasks DIR --out DIR [--models ...]`` — run the offline
  (mock-client) study matrix and write results + summary.
- ``laconic version`` — print the version.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_profile(args: argparse.Namespace) -> int:
    from laconic.profiler.report import summary_table, to_html
    from laconic.profiler.trace import profile_trace, read_trace

    entries = read_trace(args.trace)
    profile = profile_trace(
        entries, model=args.model, strategy=args.strategy, framework=args.framework
    )
    print(summary_table(profile))
    if args.html:
        path = to_html(profile, args.html)
        print(f"\nHTML report written to {path}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from laconic.eval.clients import MockModelClient
    from laconic.eval.metrics import results_markdown, summarize
    from laconic.eval.runner import MatrixSpec, run_matrix, write_results
    from laconic.eval.tasks import load_tasks

    tasks = []
    tasks_dir = Path(args.tasks)
    for path in sorted(tasks_dir.glob("*.jsonl")):
        tasks.extend(load_tasks(path))
    if not tasks:
        print(f"no benchmark tasks found in {tasks_dir}", file=sys.stderr)
        return 1

    spec = MatrixSpec(models=args.models, keep_ratios=args.ratios)
    print(
        f"running offline study: {len(tasks)} tasks x "
        f"{2 + 2 * len(args.ratios)} strategy cells (mock client — measures "
        f"information survival, not model comprehension)"
    )
    records = run_matrix(tasks, MockModelClient(), spec)
    json_path, csv_path = write_results(records, args.out)
    summary = results_markdown(summarize(records))
    (Path(args.out) / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"\nresults: {json_path}, {csv_path}")
    return 0


def _cmd_version(_args: argparse.Namespace) -> int:
    from laconic import __version__

    print(__version__)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``laconic`` console script."""
    parser = argparse.ArgumentParser(
        prog="laconic",
        description="Token-efficient middleware for multi-agent LLM workflows.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_profile = sub.add_parser("profile", help="profile a JSONL message trace")
    p_profile.add_argument("trace", help="JSONL file: one message (or envelope) per line")
    p_profile.add_argument("--model", required=True, help="target model for token counting")
    p_profile.add_argument(
        "--strategy",
        default="off",
        help="'off' measures as-is; any Session strategy shows what-if savings",
    )
    p_profile.add_argument("--framework", default="openai-chat")
    p_profile.add_argument("--html", help="also write a self-contained HTML report here")
    p_profile.set_defaults(func=_cmd_profile)

    p_eval = sub.add_parser("eval", help="run the offline (mock) study matrix")
    p_eval.add_argument("--tasks", default="data/benchmark", help="benchmark JSONL directory")
    p_eval.add_argument("--out", default="results", help="output directory")
    p_eval.add_argument("--models", nargs="+", default=["mock"])
    p_eval.add_argument("--ratios", nargs="+", type=float, default=[0.9, 0.75, 0.6, 0.45, 0.3])
    p_eval.set_defaults(func=_cmd_eval)

    p_version = sub.add_parser("version", help="print the version")
    p_version.set_defaults(func=_cmd_version)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
