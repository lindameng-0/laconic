"""Dedup semantics: context mode, store mode, prefix stability, break-even."""

from __future__ import annotations

from laconic.dedup.session import SessionDedup
from laconic.dedup.store import ContentStore
from laconic.exceptions import UnknownHandleError
from laconic.tokenizers.heuristic import HeuristicCounter

BLOCK = (
    "The full deployment checklist covers database migrations, cache warmup, "
    "feature flag rollout order, canary thresholds, and the rollback playbook "
    "for every service in the dependency graph of the payments platform."
)


def test_context_dedup_first_send_is_untouched(counter: HeuristicCounter) -> None:
    dedup = SessionDedup()
    result = dedup.process(BLOCK, recipient="writer", counter=counter)
    assert result.text == BLOCK
    assert result.hits == 0


def test_context_dedup_replaces_repeat_for_same_recipient(counter: HeuristicCounter) -> None:
    dedup = SessionDedup()
    dedup.process(BLOCK, recipient="writer", counter=counter)
    result = dedup.process(BLOCK, recipient="writer", counter=counter, context_payloads=[BLOCK])
    assert result.hits == 1
    assert "[ref lc:" in result.text
    assert counter.count(result.text) < counter.count(BLOCK)


def test_context_dedup_does_not_leak_across_recipients(counter: HeuristicCounter) -> None:
    """A different recipient has NOT seen the block — it must arrive in full."""
    dedup = SessionDedup()
    dedup.process(BLOCK, recipient="writer", counter=counter)
    result = dedup.process(BLOCK, recipient="reviewer", counter=counter)
    assert result.text == BLOCK
    assert result.hits == 0


def test_small_blocks_are_never_deduplicated(counter: HeuristicCounter) -> None:
    small = "OK, proceeding."
    dedup = SessionDedup()
    dedup.process(small, recipient="w", counter=counter)
    result = dedup.process(small, recipient="w", counter=counter)
    assert result.text == small


def test_store_mode_rehydration_roundtrip(counter: HeuristicCounter) -> None:
    store = ContentStore()
    dedup = SessionDedup(store=store, mode="store")
    dedup.process(BLOCK, recipient="a", counter=counter)
    result = dedup.process(BLOCK, recipient="b", counter=counter)  # store mode: any sighting
    assert result.hits == 1
    import re

    match = re.search(r"lc:[0-9a-f]{10}", result.text)
    assert match, f"no handle found in: {result.text}"
    rehydrate = dedup.rehydrate_tool()
    assert rehydrate(match.group()) == BLOCK


def test_multi_block_payload_dedups_per_block(counter: HeuristicCounter) -> None:
    other = "Unrelated fresh analysis of the churn cohort for the current quarter follows here."
    payload_one = f"{BLOCK}\n\n{other}"
    payload_two = f"{BLOCK}\n\nCompletely new content in the second paragraph this time around."
    dedup = SessionDedup()
    dedup.process(payload_one, recipient="w", counter=counter)
    result = dedup.process(
        payload_two, recipient="w", counter=counter, context_payloads=[payload_one]
    )
    assert result.hits == 1
    assert "Completely new content" in result.text
    assert BLOCK not in result.text


def test_unknown_handle_raises() -> None:
    store = ContentStore()
    try:
        store.get("lc:0000000000")
        raise AssertionError("expected UnknownHandleError")
    except UnknownHandleError:
        pass


def test_dedup_is_lossless_when_nothing_repeats(counter: HeuristicCounter) -> None:
    dedup = SessionDedup()
    text = "First paragraph here.\n\nSecond paragraph there.\n\n\nThird, oddly spaced."
    result = dedup.process(text, recipient="w", counter=counter)
    assert result.text == text


def test_previous_calls_are_not_evidence_of_active_context(counter: HeuristicCounter) -> None:
    dedup = SessionDedup()
    dedup.process(BLOCK, recipient="writer", counter=counter)
    for context in (None, [], ["A different, compacted context."]):
        result = dedup.process(BLOCK, recipient="writer", counter=counter, context_payloads=context)
        assert result.text == BLOCK
        assert result.hits == 0


def test_references_identify_actual_context_location(counter: HeuristicCounter) -> None:
    dedup = SessionDedup()
    result = dedup.process(
        BLOCK,
        recipient="writer",
        counter=counter,
        context_payloads=["", f"A short introduction.\n\n{BLOCK}"],
    )
    assert result.hits == 1
    assert "earlier message 2, paragraph 2" in result.text


def test_dedup_preserves_code_with_blank_lines(counter: HeuristicCounter) -> None:
    payload = (
        f"Execute this script.\n\n```python\n# setup\n\nvalue = {BLOCK!r}\n\nprint(value)\n```"
    )
    dedup = SessionDedup()
    for _ in range(2):
        result = dedup.process(
            payload, recipient="executor", counter=counter, context_payloads=[payload]
        )
        assert result.text == payload
        assert result.hits == 0


def test_store_mode_uses_explicitly_supplied_empty_store(counter: HeuristicCounter) -> None:
    store = ContentStore()
    dedup = SessionDedup(store=store, mode="store")
    dedup.process(BLOCK, recipient="writer", counter=counter)
    assert dedup.store is store
    assert len(store) == 1
