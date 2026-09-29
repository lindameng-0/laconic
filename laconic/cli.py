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
import importlib
import json
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def _read_handoff_trace(path: str):
    from laconic.handoff import Handoff, HandoffContract

    class Trace(BaseModel):
        model_config = ConfigDict(extra="forbid")
        contract: HandoffContract
        handoffs: list[Handoff] = Field(min_length=1)

    # Explicit types resolve local annotations even with postponed evaluation.
    Trace.model_rebuild(_types_namespace={"HandoffContract": HandoffContract, "Handoff": Handoff})
    return Trace.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _write_json(data: dict, output: str | None, *, input_path: str | None = None) -> None:
    rendered = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if output:
        path = Path(output)
        if input_path and path.resolve() == Path(input_path).resolve():
            raise ValueError("output must not overwrite the input evidence trace")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


def _check_output(args: argparse.Namespace) -> None:
    if args.out and Path(args.out).resolve() == Path(args.trace).resolve():
        raise ValueError("output must not overwrite the input evidence trace")


def _cmd_audit(args: argparse.Namespace) -> int:
    from laconic.handoff import audit_handoffs

    _check_output(args)
    trace = _read_handoff_trace(args.trace)
    report = audit_handoffs(trace.contract, trace.handoffs)
    _write_json(report.model_dump(mode="json"), args.out, input_path=args.trace)
    # A missing quotation is a diagnostic, not a proven semantic failure.
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    from laconic.handoff import diagnose_handoff
    from laconic.tokenizers.registry import get_counter

    _check_output(args)
    trace = _read_handoff_trace(args.trace)
    module_name, separator, function_name = args.validator.partition(":")
    if not separator or not module_name or not function_name or ":" in function_name:
        raise ValueError("validator must have the form importable.module:function")
    validator = getattr(importlib.import_module(module_name), function_name)
    if not callable(validator):
        raise ValueError("validator must name a callable")
    result = diagnose_handoff(
        trace.contract,
        trace.handoffs[0].text if len(trace.handoffs) > 1 else None,
        trace.handoffs[-1].text,
        validator,
        max_evaluations=args.max_evaluations,
        max_added_tokens=args.max_added_tokens,
        counter=get_counter(args.model),
        validator_name=args.validator,
    )
    _write_json(result.model_dump(mode="json"), args.out, input_path=args.trace)
    return 0 if result.success is True else 1


def _cmd_experiment(args: argparse.Namespace) -> int:
    from laconic.eval.handoff_experiment import run_experiment, summarize_experiment
    from laconic.tokenizers.registry import get_counter

    result = run_experiment(
        seed=args.seed,
        variants=args.variants,
        counter=get_counter(args.model),
        max_evaluations=args.max_evaluations,
    )
    _write_json(result, args.out)
    if args.out:
        print(summarize_experiment(result))
        print(f"Full experiment record: {args.out}")
    return 0


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
        f"{3 + 2 * len(args.ratios)} strategy cells (mock client - measures "
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
        description="Audit, replay, and optimize agent handoffs with source-backed evidence.",
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

    p_audit = sub.add_parser("audit", help="trace where declared source clauses disappear")
    p_audit.add_argument("trace", help="JSON object containing contract and handoffs")
    p_audit.add_argument("--out", help="write the audit JSON (default: stdout)")
    p_audit.set_defaults(func=_cmd_audit)

    p_replay = sub.add_parser("replay", help="test source-backed repairs with your task validator")
    p_replay.add_argument("trace", help="JSON trace; first handoff is baseline, last is candidate")
    p_replay.add_argument(
        "--validator",
        required=True,
        help="importable.module:function returning ReplayOutcome; runs caller-owned Python code",
    )
    p_replay.add_argument("--max-evaluations", type=_positive_int, default=8)
    p_replay.add_argument("--max-added-tokens", type=_nonnegative_int)
    p_replay.add_argument(
        "--model", default="offline", help="local tokenizer target; no model call"
    )
    p_replay.add_argument("--out", help="write the full replay record as JSON (default: stdout)")
    p_replay.set_defaults(func=_cmd_replay)

    p_experiment = sub.add_parser("experiment", help="run scripted handoff fault-injection checks")
    p_experiment.add_argument("--seed", type=int, default=7)
    p_experiment.add_argument("--variants", type=_positive_int, default=3)
    p_experiment.add_argument("--max-evaluations", type=_positive_int, default=8)
    p_experiment.add_argument("--model", default="offline", help="tokenizer target; no model call")
    p_experiment.add_argument("--out", help="write full JSON; print a compact summary")
    p_experiment.set_defaults(func=_cmd_experiment)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, OSError, ImportError, AttributeError) as exc:
        print(f"laconic: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
