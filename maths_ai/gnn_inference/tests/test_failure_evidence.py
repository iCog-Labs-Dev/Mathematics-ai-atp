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
                            error=PantographRemoteInternalError("x"), server=Server(snapshot, history),
                            identity={"environment_fingerprint": "sha256:x"},
                            cleanup_error=RuntimeError("close failed"))
    json.dumps(record)
    assert record["replayable"] and not record["missing"] and record["final"]
    assert record["cleanup_error"]["message"] == "close failed"
    assert record["snapshot"]["association"] == "matched"
    assert "unknown free variable" in record["summary"] and "observe" in record["summary"]
    unbound = failure_record(context={}, theorem=Item(Seed()), error=RuntimeError("y"),
                             server=Server(snapshot, history))
    assert not unbound["replayable"] and unbound["missing"] == ["runtime identity"]
    partial = failure_record(context={}, theorem=object(), error=RuntimeError("y"), server=Server())
    assert not partial["replayable"]
    assert {"seed", "failure snapshot", "request history"} <= set(partial["missing"])
    assert failure_summary(None) == "no failure snapshot"


def _total_bytes(root):
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def test_archive_is_atomic_bounded_and_reports_the_limit(tmp_path):
    archive = FailureArchive(tmp_path / "failures", max_bundles=2)
    record = {"summary": "s", "replayable": True, "context": {"batch_index": 0}}
    first, second, third = (archive.write(dict(record)) for _ in range(3))
    assert first["stored"] and second["stored"] and not third["stored"]
    bundle = json.loads((tmp_path / "failures" / first["path"]).read_text())
    assert bundle["failure_id"] == first["failure_id"] and bundle["status"] == "final"
    assert not list((tmp_path / "failures").rglob("*.tmp"))
    ledger = [json.loads(l) for l in (tmp_path / "failures" / "ledger.jsonl").read_text().splitlines()]
    assert [l["kind"] for l in ledger] == ["failure", "finalized", "failure", "finalized", "capture_limit"]
    assert archive.uncaptured == 1


def test_reopening_an_archive_keeps_its_bundles_ids_and_bounds(tmp_path):
    root = tmp_path / "failures"
    first = FailureArchive(root, max_bundles=1).write({"summary": "first", "context": {}})
    original = (root / first["path"]).read_bytes()
    reopened = FailureArchive(root, max_bundles=1)
    second = reopened.write({"summary": "second", "context": {}})
    assert not second["stored"] and second["failure_id"] != first["failure_id"]
    assert (root / first["path"]).read_bytes() == original
    # A larger count bound on reopening continues numbering without collision.
    third = FailureArchive(root, max_bundles=5).write({"summary": "third", "context": {}})
    assert third["stored"] and third["path"] != first["path"]
    assert (root / first["path"]).read_bytes() == original


def test_the_byte_bound_covers_bundles_and_ledger(tmp_path):
    root = tmp_path / "failures"
    archive = FailureArchive(root, max_bytes=4000)
    for index in range(20):
        archive.write({"summary": "x" * 200, "context": {"batch_index": index}})
    assert _total_bytes(root) <= 4000 and archive.uncaptured > 0
    reopened = FailureArchive(root, max_bytes=4000)
    reopened.write({"summary": "after restart", "context": {}})
    assert _total_bytes(root) <= 4000
    kinds = [json.loads(l)["kind"] for l in (root / "ledger.jsonl").read_text().splitlines()]
    assert kinds.count("capture_limit") == 1


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


def _crashing_pool_test(tmp_path, error, *, close_error=None, replacement_error=None):
    async def run():
        pool, _model, servers = await _create_pool(workers=1)
        pool.failure_archive = FailureArchive(tmp_path / "failures")

        async def search(reasoner, theorem, greedy, deadline):
            reasoner.server.last_failure = failed_snapshot()
            reasoner.server.request_history = RequestHistory()
            reasoner.server.request_history(event(1, "goal.start"))
            if close_error is not None:
                async def broken_close():
                    raise close_error
                reasoner.server.aclose = broken_close
            raise error

        if replacement_error is not None:
            async def failing_factory(worker_id):
                raise replacement_error
            original_replace = pool._replacement_worker

            async def replacement(worker):
                pool.server_factory = failing_factory
                return await original_replace(worker)
            pool._replacement_worker = replacement
        pool.search_factory = search
        try:
            async with pool:
                return await pool.collect_round(["crash"], timeout_s=1, model_generation=0, proposals=[None])
        except BaseException as exc:
            return exc

    outcome = asyncio.run(run())
    ledger = [json.loads(l) for l in (tmp_path / "failures" / "ledger.jsonl").read_text().splitlines()]
    bundles = [json.loads(p.read_text()) for p in (tmp_path / "failures").glob("attempts/*/failure.json")]
    return outcome, ledger, bundles


def test_a_fatal_integrity_failure_is_archived_and_still_propagates(tmp_path):
    from pantograph_async import PantographProtocolError

    outcome, ledger, bundles = _crashing_pool_test(tmp_path, PantographProtocolError("validation failed"))
    assert isinstance(outcome, PantographProtocolError)
    assert len(bundles) == 1 and bundles[0]["context"]["replacement"] == "not replaced (fatal)"
    assert bundles[0]["error"]["type"] == "PantographProtocolError"
    assert [l["kind"] for l in ledger] == ["failure", "finalized"]


def test_a_failed_old_client_close_keeps_both_errors(tmp_path):
    outcome, ledger, bundles = _crashing_pool_test(
        tmp_path, PantographRemoteInternalError("boom"), close_error=OSError("close failed"))
    assert isinstance(outcome, BaseException)
    assert len(bundles) == 1 and bundles[0]["context"]["replacement"] == "cleanup failed"
    assert bundles[0]["error"]["type"] == "PantographRemoteInternalError"
    assert bundles[0]["cleanup_error"]["message"].endswith("close failed")


def test_a_failed_replacement_start_is_archived(tmp_path):
    outcome, ledger, bundles = _crashing_pool_test(
        tmp_path, PantographRemoteInternalError("boom"), replacement_error=RuntimeError("no REPL"))
    assert len(bundles) == 1 and bundles[0]["context"]["replacement"] == "replacement failed"
    assert bundles[0]["cleanup_error"]["type"] == "WorkerStartupError"
