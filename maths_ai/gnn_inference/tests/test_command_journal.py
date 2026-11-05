"""The retained-execution command journal is ordered, bounded and opt-in."""

import json

from maths_ai.hybrid_reasoner.command_journal import JOURNAL_ENV, CommandJournal, default_journal


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_command_is_written_before_its_outcome(tmp_path):
    journal = CommandJournal(tmp_path / "j.jsonl")
    first = journal.command(phase="operation", command="unfold Nat.pow")
    journal.outcome(first, "accepted", validation="3:scope")
    second = journal.command(phase="replay", command="x" * 5000)
    records = lines(journal.path)
    assert [(r["kind"], r["seq"]) for r in records] == [("command", 1), ("outcome", 1), ("command", 2)]
    assert records[2]["command"].endswith("[2952 more]")


def test_journal_stops_at_its_bound_and_says_so(tmp_path):
    journal = CommandJournal(tmp_path / "j.jsonl", max_bytes=600)
    for index in range(20):
        journal.command(phase="operation", command=f"tactic {index}")
    records = lines(journal.path)
    assert records[-1]["kind"] == "truncated"
    assert journal.path.stat().st_size <= 600 + 64
    journal.command(phase="operation", command="after")
    assert lines(journal.path) == records


def test_journal_is_off_without_the_environment_variable(tmp_path, monkeypatch):
    monkeypatch.delenv(JOURNAL_ENV, raising=False)
    assert default_journal() is None
    monkeypatch.setenv(JOURNAL_ENV, str(tmp_path / "process.jsonl"))
    assert default_journal() is default_journal()
