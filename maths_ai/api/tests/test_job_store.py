"""Job lifecycle, atomic results and restart reconciliation."""

from __future__ import annotations

import asyncio

import pytest

from maths_ai.api.job_store import (
    ADMITTING,
    CANCELLED,
    CANCELLING,
    COMPLETED,
    INTERRUPTED,
    RUNNING,
    JobStateError,
    JobStore,
)


def _run(coroutine):
    return asyncio.run(coroutine)


def test_lifecycle_and_a_committed_completion_wins_over_cancellation(tmp_path):
    async def run():
        store = JobStore(tmp_path)
        await store.open()
        job_id = await store.create({"dataset_profile": "smoke"}, {"repeats": 1})
        await store.transition(job_id, ADMITTING)
        await store.transition(job_id, RUNNING)
        await store.transition(job_id, CANCELLING)
        # The last chunk finished while the cancellation was settling.
        job = await store.finish(job_id, COMPLETED, {"complete": True}, partial=False)
        assert job["state"] == COMPLETED and job["finished_at"]
        with pytest.raises(JobStateError):
            await store.transition(job_id, CANCELLED)
        assert (await store.result(job_id)) == {"complete": True}

    _run(run())


def test_chunks_persist_with_progress(tmp_path):
    async def run():
        store = JobStore(tmp_path)
        await store.open()
        job_id = await store.create({}, {})
        await store.record_chunk(job_id, 0, [{"index": 0, "outcome": "solved"}], {"solved": 1.0},
                                 {"attempts_completed": 1})
        await store.record_chunk(job_id, 0, [{"index": 1, "outcome": "unsolved"}], {"solved": 1.0},
                                 {"attempts_completed": 2})
        attempts, stats = await store.attempts(job_id)
        job = await store.get(job_id)
        return attempts, stats, job

    attempts, stats, job = _run(run())
    assert [a["index"] for a in attempts] == [0, 1]
    assert stats == {0: {"solved": 1.0}}
    assert job["progress"] == {"attempts_completed": 2}


def test_restart_interrupts_unfinished_jobs_and_drops_uncommitted_results(tmp_path):
    async def first():
        store = JobStore(tmp_path)
        await store.open()
        queued = await store.create({}, {})
        running = await store.create({}, {})
        await store.transition(running, ADMITTING)
        done = await store.create({}, {})
        await store.transition(done, ADMITTING)
        await store.transition(done, RUNNING)
        await store.finish(done, COMPLETED, {"complete": True}, partial=False)
        return queued, running, done

    queued, running, done = _run(first())
    # A crash after publishing a file but before committing its reference.
    orphan = tmp_path / "results" / f"{queued}.json"
    orphan.write_text("{}")

    async def second():
        store = JobStore(tmp_path)
        interrupted = await store.open()
        return store, interrupted, [await store.get(i) for i in (queued, running, done)]

    store, interrupted, jobs = _run(second())
    assert sorted(interrupted) == sorted([queued, running])
    assert [job["state"] for job in jobs] == [INTERRUPTED, INTERRUPTED, COMPLETED]
    assert jobs[0]["error_category"] == "service-restart"
    assert not orphan.exists()
    assert _run(store.result(done)) == {"complete": True}
