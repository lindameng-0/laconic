"""Shared fixtures: a synthetic corpus of agent messages for invariant tests."""

from __future__ import annotations

import json
import random
from typing import Any

import pytest

from laconic.tokenizers.heuristic import HeuristicCounter

_VERBOSE_TEXT = (
    "Please note that the quarterly review has been completed and it is worth "
    "noting that, for example, the revenue figures came in at approximately "
    "12.4 percent above plan. As previously mentioned, the team remains "
    "confident. In order to proceed we need sign-off from finance. "
    "Basically, the outlook is stable and the backlog is well understood. "
    "It should be noted that delivery must happen before Q3 2026."
)


def make_corpus(seed: int = 0, size: int = 200) -> list[dict[str, Any]]:
    """Generate a corpus of synthetic agent messages of varied shapes."""
    rng = random.Random(seed)
    corpus: list[dict[str, Any]] = []
    for index in range(size):
        shape = rng.randrange(6)
        if shape == 0:  # plain assistant prose
            corpus.append({"role": "assistant", "content": _VERBOSE_TEXT})
        elif shape == 1:  # tool-call message with rationale
            corpus.append(
                {
                    "role": "assistant",
                    "content": _VERBOSE_TEXT,
                    "tool_calls": [
                        {
                            "id": f"call_{rng.randrange(16**8):08x}",
                            "type": "function",
                            "function": {
                                "name": rng.choice(["search", "create_ticket", "run_job"]),
                                "arguments": json.dumps(
                                    {"query": "backlog", "limit": rng.randint(1, 50)}
                                ),
                            },
                        }
                    ],
                }
            )
        elif shape == 2:  # tool result (JSON content — protected)
            corpus.append(
                {
                    "role": "tool",
                    "tool_call_id": f"call_{rng.randrange(16**8):08x}",
                    "content": json.dumps({"status": "ok", "rows": rng.randint(0, 9)}),
                }
            )
        elif shape == 3:  # short message (below compression threshold)
            corpus.append({"role": "user", "content": "Proceed with step 2."})
        elif shape == 4:  # unicode + code fence payload
            corpus.append(
                {
                    "role": "assistant",
                    "content": (
                        _VERBOSE_TEXT
                        + "\n\n```python\ndef handler(x):\n    return x * 2\n```\n\n"
                        + "Résumé: naïve façade — 校正済み. "
                        + _VERBOSE_TEXT
                    ),
                    "name": "agent_α",
                }
            )
        else:  # structured content parts (multimodal shape — protected)
            corpus.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look at this"},
                        {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                    ],
                }
            )
        corpus[-1]["_index"] = index  # unknown keys must survive round-trip too
    return corpus


@pytest.fixture(scope="session")
def corpus() -> list[dict[str, Any]]:
    return make_corpus()


@pytest.fixture()
def counter() -> HeuristicCounter:
    return HeuristicCounter("test-model")
