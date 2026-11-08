"""Failure evidence: bounded session histories, archived bundles and pool capture."""

import asyncio
from dataclasses import dataclass
import json

from pantograph_async import PantographRemoteInternalError, RequestEvent
from pantograph_async.evidence import snapshot_for

from maths_ai.hybrid_reasoner.command_journal import JOURNAL_ENV
from maths_ai.hybrid_reasoner.failure_evidence import (
    FailureArchive, RequestHistory, execution_phase, failure_record, failure_summary,
)
from maths_ai.hybrid_reasoner.search_results import record_view

from maths_ai.gnn_inference.tests.test_rl_live_collection import _create_pool, _Result


def event(sequence, operation="goal.tactic", stage="prepared", request="goal.tactic {}", **fields):
    return RequestEvent(session_id="s", pid=1, sequence=sequence, operation=operation, stage=stage,
                        request=request, request_truncated=False, **fields)


REPL_RECORD = json.dumps({
    "kind": "pantograph-command-failure", "version": 1, "repl_command_sequence": 9,
    "operation": "goal.tactic", "state_id": 1, "goal_index": 0, "phase": "observe",
    "exception_category": "error", "internal_exception_id": None,
    "detail": "unknown free variable", "detail_truncated": False, "render_status": "rendered"})


def failed_snapshot():
    error = PantographRemoteInternalError("Unexpected REPL command failure", command="goal.tactic",
                                          payload={"error": "internal", "recovery": "session-unusable"})
    request = b'goal.tactic {"stateId":1,"goalId":0,"tactic":"simp"}\n'
    return snapshot_for(session_id="s", pid=1, sequence=3, operation="goal.tactic", request=request,
                        body={"stateId": 1, "goalId": 0}, stage="decoded", error=error,
                        stderr_tail=REPL_RECORD + "\n").finalized(REPL_RECORD + "\n")


@dataclass
class Server:
    last_failure: object = None
    request_history: object = None


@dataclass
class Seed:
    def as_dict(self):
        return {"target": "True", "locals": []}


@dataclass
class Item:
    seed: Seed


def test_history_labels_phases_bounds_itself_and_feeds_the_journal(tmp_path, monkeypatch):
    monkeypatch.setenv(JOURNAL_ENV, str(tmp_path / "journal.jsonl"))
    history = RequestHistory(max_events=3)
    history(event(1, "goal.start"))
    with execution_phase("replay"):
        history(event(2))
    for sequence in (3, 4):
        history(event(sequence))
    events = history.events()
    assert [e["sequence"] for e in events] == [2, 3, 4] and history.dropped == 1
    assert events[0]["phase"] == "replay" and events[1]["phase"] is None
    journal = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text().splitlines()]
    assert [line["sequence"] for line in journal] == [1, 2, 3, 4]


def test_failure_record_is_plain_and_says_what_is_missing():
    history = RequestHistory()
    history(event(1, "goal.start", request='goal.start {"expr":"True"}'))
    history(event(3))
    snapshot = failed_snapshot()
    record = failure_record(context={"batch_index": 2}, theorem=Item(Seed()),
                            error=PantographRemoteInternalError("x"), server=Server(snapshot, history))
    json.dumps(record)
    assert record["replayable"] and not record["missing"]
    assert record["snapshot"]["association"] == "matched"
    assert "unknown free variable" in record["summary"] and "observe" in record["summary"]
    partial = failure_record(context={}, theorem=object(), error=RuntimeError("y"), server=Server())
    assert not partial["replayable"]
    assert set(partial["missing"]) == {"seed", "failure snapshot", "request history"}
    assert failure_summary(None) == "no failure snapshot"


def test_archive_is_atomic_bounded_and_reports_the_limit(tmp_path):
    archive = FailureArchive(tmp_path / "failures", max_bundles=2)
    record = {"summary": "s", "replayable": True, "context": {"batch_index": 0}}
    first, second, third = (archive.write(dict(record)) for _ in range(3))
    assert first["stored"] and second["stored"] and not third["stored"]
    bundle = json.loads((tmp_path / "failures" / first["path"]).read_text())
    assert bundle["failure_id"] == first["failure_id"]
    assert not list((tmp_path / "failures").rglob("*.tmp"))
    ledger = [json.loads(l) for l in (tmp_path / "failures" / "ledger.jsonl").read_text().splitlines()]
    assert [l["kind"] for l in ledger] == ["failure", "failure", "capture_limit"]
    assert archive.uncaptured == 1


def test_a_replaced_worker_leaves_an_archived_failure_on_its_record(tmp_path):
    async def run():
        pool, _model, servers = await _create_pool(workers=1)
        pool.failure_archive = FailureArchive(tmp_path / "failures")

        async def search(reasoner, theorem, greedy, deadline):
            if theorem == "crash":
                reasoner.server.last_failure = failed_snapshot()
                reasoner.server.request_history = RequestHistory()
                reasoner.server.request_history(event(1, "goal.start"))
                raise PantographRemoteInternalError("Unexpected REPL command failure")
            return _Result(theorem)

        pool.search_factory = search
        async with pool:
            result = await pool.collect_round(["crash", "solve-next"], timeout_s=1,
                                              model_generation=0, proposals=[None] * 2)
        return result

    result = asyncio.run(run())
    failed = result.records[0]
    assert failed.outcome == "error" and failed.failure["stored"]
    assert failed.failure["timeout_owner"] is None
    bundle = json.loads((tmp_path / "failures" / failed.failure["path"]).read_text())
    assert bundle["context"]["replacement"] == "replaced" and bundle["context"]["batch_index"] == 0
    assert bundle["snapshot"]["repl_records"][0]["detail"] == "unknown free variable"
    assert record_view(failed)["failure"] == failed.failure
    assert result.records[1].failure is None and result.stats["workers_replaced"] == 1
