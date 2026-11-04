"""Incremental admission: bounded steps, absolute deadline, same report as preflight."""

from __future__ import annotations

import asyncio
import time

from maths_ai.gnn_inference.atp_lean_gnn.seed_admission import (
    AdmissionReport,
    AdmissionRow,
    AdmissionSource,
    AdmissionSession,
    TheoremItem,
    preflight,
    file_candidates,
)
from maths_ai.gnn_inference.tests.test_rl_training_driver import _factory, _ReplayClient
from maths_ai.lean_replay import seed_from_text


def _items(names):
    return AdmissionSource("rows", {"rows": [AdmissionRow(TheoremItem(seed=seed_from_text(goal=name), tactic_label="", size=1)) for name in names]})


NAMES = ["p", "q", "span_bad", "r", "s"]


def test_steps_admit_in_order_and_match_one_shot_preflight():
    async def run():
        stepped = AdmissionReport()
        client = _ReplayClient(rejected=("span",))
        session = AdmissionSession(
            _items(NAMES), _factory(client), max_admitted=10, time_budget_s=60.0, report=stepped
        )
        first = await session.step(2, interval_s=30)
        assert [i.seed.target for i in first] == ["p", "q"]
        assert session.rows_seen == 2 and not session.done
        rest = []
        while not session.done:
            rest.extend(await session.step(2, interval_s=30))
        await session.close()

        whole = AdmissionReport()
        admitted = await preflight(
            _items(NAMES), _factory(_ReplayClient(rejected=("span",))),
            max_admitted=10, time_budget_s=60.0, report=whole,
        )
        return first + rest, stepped, admitted, whole

    stepped_items, stepped, whole_items, whole = asyncio.run(run())
    assert [i.seed.target for i in stepped_items] == [i.seed.target for i in whole_items] == ["p", "q", "r", "s"]
    assert stepped.admitted == whole.admitted == 4
    assert sum(stepped.quarantined.values()) == sum(whole.quarantined.values()) == 1


def test_time_paused_between_steps_counts_against_the_deadline():
    async def run():
        report = AdmissionReport()
        session = AdmissionSession(
            _items(NAMES), _factory(_ReplayClient()), max_admitted=10, time_budget_s=30, report=report
        )
        await session.step(1, interval_s=30)
        session._deadline = time.monotonic() + 0.05
        await asyncio.sleep(0.08)  # another job ran in between
        later = await session.step(10)
        return session, report, later

    session, report, later = asyncio.run(run())
    assert later == [] and session.done
    assert report.stopped_by == "time-budget"


def test_pool_full_stops_the_session():
    async def run():
        report = AdmissionReport()
        session = AdmissionSession(
            _items(NAMES), _factory(_ReplayClient()), max_admitted=2, time_budget_s=60.0, report=report
        )
        while not session.done:
            await session.step(3, interval_s=30)
        return session, report

    session, report = asyncio.run(run())
    assert len(session.admitted) == 2
    assert report.stopped_by == "pool-full"


def test_rejected_source_rows_consume_step_capacity(tmp_path):
    import json

    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(
        [json.dumps({"goal": "True", "hypotheses": ["not a declaration"]})] * 100
        + [json.dumps({"goal": "True"})]
    ))

    async def run():
        report = AdmissionReport()
        session = AdmissionSession(
            file_candidates(path, max_scan_rows=200, max_state_chars=None, report=report),
            _factory(_ReplayClient()), max_admitted=10, time_budget_s=30, report=report,
        )
        assert await session.step(1, interval_s=30) == []
        assert session.rows_seen == report.scanned == 1
        assert session._client is None
        await session.close()

    asyncio.run(run())


def test_blocked_source_setup_can_be_closed_without_waiting_for_a_row(tmp_path):
    import os

    source = tmp_path / "blocked.jsonl"
    os.mkfifo(source)

    async def run():
        report = AdmissionReport()
        session = AdmissionSession(
            file_candidates(source, max_scan_rows=10, max_state_chars=None, report=report),
            _factory(_ReplayClient()), max_admitted=10, time_budget_s=30, report=report,
        )
        started = time.monotonic()
        assert await session.step(1, interval_s=0.05) == []
        assert time.monotonic() - started < 0.5
        pid = session._rows.process.pid
        await session.close()
        from pathlib import Path
        assert not Path(f"/proc/{pid}").exists()

    asyncio.run(run())
