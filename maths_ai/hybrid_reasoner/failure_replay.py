"""Replay a failure bundle without a model, retriever, GPU or PLN.

The bundle's request history is re-issued through typed client methods on a
fresh session. Recorded states are mapped to the fresh states their replayed
requests return; recorded IDs are never used as handles. Recorded tactic
rejections must be rejected again. The replay reproduces the failure only if
the final request fails with the same signature: operation, error type, and the
REPL record's phase, exception category and exception identifier.
"""

from __future__ import annotations

import json
from typing import Any

from pantograph_async import (
    PantographClient,
    PantographEnv,
    PantographError,
    PantographTacticRejected,
)
from pantograph_async.compatibility import companion_base

from .failure_evidence import FAILURE_SCHEMA_VERSION

__all__ = ["ReplayUnsupported", "failure_signature", "replay_failure"]

SKIPPED = {"protocol.describe", "options.set", "options.print", "stat"}


class ReplayUnsupported(ValueError):
    """The bundle cannot be replayed exactly; the reason says what is missing."""


def failure_signature(snapshot: dict[str, Any] | None) -> dict[str, Any] | None:
    if not snapshot:
        return None
    records = snapshot.get("repl_records") or []
    record = records[-1] if records and snapshot.get("association") == "matched" else None
    return {
        "operation": snapshot.get("operation"),
        "error_type": snapshot.get("error_type"),
        "phase": record and record.get("phase"),
        "exception_category": record and record.get("exception_category"),
        "internal_exception_id": record and record.get("internal_exception_id"),
    }


def _body(event: dict[str, Any]) -> dict[str, Any]:
    _, _, payload = event["request"].partition(" ")
    return json.loads(payload)


def _steps(bundle: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    events = bundle.get("requests") or []
    finished = {e["sequence"]: e for e in events if e.get("stage") in ("completed", "failed")}
    return [(e, finished.get(e["sequence"])) for e in events if e.get("stage") == "prepared"]


async def replay_failure(bundle: dict[str, Any], env: PantographEnv, *, allow_mismatch: bool = False) -> dict[str, Any]:
    """Re-issue the bundle's session history; report whether the failure recurred."""
    if bundle.get("schema_version") != FAILURE_SCHEMA_VERSION:
        raise ReplayUnsupported(f"Unsupported failure schema {bundle.get('schema_version')!r}.")
    if not bundle.get("replayable"):
        raise ReplayUnsupported(f"Bundle is not replayable; missing: {bundle.get('missing')}.")
    recorded = (bundle.get("identity") or {}).get("companion_commit")
    current = companion_base()["source_commit"]
    if recorded != current and not allow_mismatch:
        return {"status": "environment-mismatch", "recorded_companion": recorded, "current_companion": current}
    expected = failure_signature(bundle.get("snapshot"))
    states: dict[int, Any] = {}
    replayed = 0
    client = await PantographClient.create(env)
    try:
        async with client.state_scope() as scope:
            for prepared, finished in _steps(bundle):
                operation = prepared["operation"]
                if operation in SKIPPED:
                    continue
                body = _body(prepared)
                final = finished is None or finished.get("stage") == "failed"
                outcome = (finished or {}).get("outcome")
                try:
                    if operation == "goal.start":
                        result = await scope.start_goal(body["expr"], levels=body.get("levels", ()))
                    elif operation == "goal.tactic":
                        source = states[body["stateId"]]
                        try:
                            result = await scope.apply_tactic(source, body["tactic"], goal_index=body.get("goalId", 0))
                        except PantographTacticRejected:
                            if final or outcome == "TacticSuccess":
                                raise
                            result = None
                    elif operation == "goal.inspect":
                        result = await scope.inspect(states[body["stateId"]], body["goals"])
                    elif operation == "goal.certify":
                        result = await scope.certify(states[body["stateId"]], max_heartbeats=body.get("maxHeartbeats"))
                    elif operation == "goal.delete":
                        for state_id in body["stateIds"]:
                            if state_id in states:
                                await scope.release(states.pop(state_id))
                        result = None
                    else:
                        raise ReplayUnsupported(f"Replay does not support {operation}.")
                except KeyError as exc:
                    raise ReplayUnsupported(f"Recorded state {exc} was never replayed.") from exc
                except PantographError:
                    if not final:
                        raise
                    break
                replayed += 1
                allocated = (finished or {}).get("allocated_state_id")
                if allocated is not None and result is not None:
                    states[allocated] = result
                if final:
                    break
    except PantographError:
        pass
    finally:
        await client.aclose()
    observed = failure_signature(client.last_failure.to_dict() if client.last_failure else None)
    return {
        "status": "reproduced" if observed is not None and observed == expected else "not-reproduced",
        "expected": expected, "observed": observed, "replayed_requests": replayed,
        "repl_detail": (client.last_failure.repl_failure.detail
                        if client.last_failure and client.last_failure.repl_failure else None),
    }
