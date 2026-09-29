"""Create a small, reproducible snapshot from complete experiment JSON files."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/experiments/handoff-summary.json"))
    args = parser.parse_args()
    if args.out.resolve() in {path.resolve() for path in args.reports}:
        parser.error("summary output must not overwrite an experiment report")
    runs = []
    for path in args.reports:
        raw = path.read_bytes()
        report = json.loads(raw)
        if report.get("experiment") != "retry-handoff-fault-injection-v1":
            parser.error(f"unsupported experiment: {path}")
        records = report["records"]
        runs.append(
            {
                "seed": report["seed"],
                "variants": report["variants"],
                "cases": len(records),
                "tokenizer": report["tokenizer"],
                "count_tier": report["count_tier"],
                "report_sha256": hashlib.sha256(raw).hexdigest(),
                "policy_results": {
                    name: {
                        "passed": sum(r["policies"][name]["outcome"]["success"] for r in records),
                        "total_text_tokens": sum(r["policies"][name]["tokens"] for r in records),
                    }
                    for name in records[0]["policies"]
                },
                "replay_statuses": dict(Counter(r["replay"]["status"] for r in records)),
                "validator_calls": sum(r["replay"]["evaluations_used"] for r in records),
                "diagnostic_input_tokens": sum(
                    attempt["tokens"] for r in records for attempt in r["replay"]["attempts"]
                ),
                "result_text_tokens": sum(r["replay"]["result_tokens"] for r in records),
            }
        )
    summary = {
        "experiment": "retry-handoff-fault-injection-v1",
        "receiver": "Scripted retry-policy interpreter; no language model or paid calls.",
        "interpretation": (
            "Synthetic faults share generated specifications and latest-clause-wins semantics. "
            "These are protocol checks, not independent production tasks or model success rates. "
            "Text token totals are not bills. Diagnostic overhead includes all validator calls."
        ),
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(runs)} run summaries to {args.out}")


if __name__ == "__main__":
    main()
