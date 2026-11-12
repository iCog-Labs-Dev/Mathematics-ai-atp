"""The request journal is ordered, bounded and opt-in."""

import json

from maths_ai.hybrid_reasoner.command_journal import JOURNAL_ENV, CommandJournal, default_journal


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_events_are_written_in_order_with_bounded_strings(tmp_path):
    journal = CommandJournal(tmp_path / "j.jsonl")
    journal.event({"stage": "prepared", "sequence": 4, "request": "goal.tactic {}", "phase": "operation"})
    journal.event({"stage": "completed", "sequence": 4, "outcome": "TacticSuccess"})
    journal.event({"stage": "prepared", "sequence": 5, "request": "x" * 5000})
    records = lines(journal.path)
    assert [(r["journal_seq"], r["stage"], r.get("sequence")) for r in records] == [
        (1, "prepared", 4), (2, "completed", 4), (3, "prepared", 5)]
    assert records[2]["request"].endswith("[2952 more]")


def test_journal_stops_at_its_bound_and_says_so(tmp_path):
    journal = CommandJournal(tmp_path / "j.jsonl", max_bytes=600)
    for index in range(20):
        journal.event({"stage": "prepared", "request": f"tactic {index}"})
    records = lines(journal.path)
    assert records[-1]["kind"] == "truncated"
    assert journal.path.stat().st_size <= 600 + 64
    journal.event({"stage": "prepared", "request": "after"})
    assert lines(journal.path) == records


def test_journal_is_off_without_the_environment_variable(tmp_path, monkeypatch):
    monkeypatch.delenv(JOURNAL_ENV, raising=False)
    assert default_journal() is None
    monkeypatch.setenv(JOURNAL_ENV, str(tmp_path / "process.jsonl"))
    assert default_journal() is default_journal()
