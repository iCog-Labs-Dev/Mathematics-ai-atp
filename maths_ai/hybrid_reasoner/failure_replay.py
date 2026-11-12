"""Replay a failure bundle without a model, retriever, GPU or PLN.

The bundle's request history is re-issued through typed client methods on a
fresh session. Recorded states are mapped to the fresh states their replayed
requests return; recorded IDs are never used as handles.

The target is the request the terminal snapshot names (its client request
sequence). Every earlier request must end as it did before: a success, an
ordinary tactic rejection, a rejected goal start or a resource rollback. A
different outcome is a divergence, not a reproduction. The failure is
reproduced only if the target fails with the same operation, error type, and
the REPL record's phase, exception category and exception identifier.

The replay runtime must match the recorded one (environment fingerprint over
toolchain, Lake manifest, REPL executable, imports and compiled files, plus
effective options and client and companion identities). A mismatch is reported
without replaying unless explicitly allowed, and then it is reported with the
result.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from pantograph_async import (
    PantographClient,
    PantographEnv,
    PantographError,
    PantographTacticRejected,
)

from .failure_evidence import FAILURE_SCHEMA_VERSION, runtime_identity

__all__ = ["ReplayUnsupported", "failure_signature", "replay_failure"]

SKIPPED = {"protocol.describe", "options.set", "options.print", "stat"}
IDENTITY_FIELDS = ("environment_fingerprint", "imports", "effective_options", "client_version", "companion_commit")


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


def _steps(bundle: dict[str, Any], target: int) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    events = bundle.get("requests") or []
    finished = {e["sequence"]: e for e in events if e.get("stage") in ("completed", "failed")}
    steps = [(e, finished.get(e["sequence"])) for e in events
             if e.get("stage") == "prepared" and e["sequence"] <= target]
    if not steps or steps[-1][0]["sequence"] != target:
        raise ReplayUnsupported(f"The history does not contain the target request {target}.")
    for prepared, done in steps[:-1]:
        if done is None:
            raise ReplayUnsupported(f"Request {prepared['sequence']} before the target has no recorded outcome.")
    return steps


def identity_mismatch(recorded: dict[str, Any], current: dict[str, Any]) -> list[str]:
    return [field for field in IDENTITY_FIELDS if recorded.get(field) != current.get(field)]


async def _run(scope, states, operation: str, body: dict[str, Any]) -> Any:
    if operation == "goal.start":
        return await scope.start_goal(body["expr"], levels=body.get("levels", ()))
    if operation == "goal.tactic":
        return await scope.apply_tactic(states[body["stateId"]], body["tactic"], goal_index=body.get("goalId", 0))
    if operation == "goal.inspect":
        return await scope.inspect(states[body["stateId"]], body["goals"])
    if operation == "goal.certify":
        return await scope.certify(states[body["stateId"]], max_heartbeats=body.get("maxHeartbeats"))
    if operation == "goal.delete":
        for state_id in body["stateIds"]:
            if state_id in states:
                await scope.release(states.pop(state_id))
        return None
    raise ReplayUnsupported(f"Replay does not support {operation}.")


async def replay_failure(bundle: dict[str, Any], env: PantographEnv, *, allow_mismatch: bool = False) -> dict[str, Any]:
    """Re-issue the bundle's session history; report whether the failure recurred."""
    if bundle.get("schema_version") != FAILURE_SCHEMA_VERSION:
        raise ReplayUnsupported(f"Unsupported failure schema {bundle.get('schema_version')!r}.")
    if not bundle.get("replayable"):
        raise ReplayUnsupported(f"Bundle is not replayable; missing: {bundle.get('missing')}.")
    snapshot = bundle.get("snapshot") or {}
    target = snapshot.get("sequence")
    if not isinstance(target, int) or snapshot.get("exchange_stage") not in ("written", "decoded"):
        raise ReplayUnsupported("The terminal snapshot does not name a written request.")
    steps = _steps(bundle, target)
    expected = failure_signature(snapshot)
    recorded_identity = bundle.get("identity") or {}

    states: dict[int, Any] = {}
    replayed = 0
    divergence = None
    mismatch: list[str] = []
    client = await PantographClient.create(env)
    try:
        current_identity = await asyncio.to_thread(runtime_identity, client)
        mismatch = identity_mismatch(recorded_identity, current_identity)
        if mismatch and not allow_mismatch:
            return {"status": "environment-mismatch", "mismatch": mismatch,
                    "recorded": {f: recorded_identity.get(f) for f in mismatch},
                    "current": {f: current_identity.get(f) for f in mismatch}}
        async with client.state_scope() as scope:
            for prepared, done in steps:
                operation = prepared["operation"]
                if operation in SKIPPED and prepared["sequence"] != target:
                    continue
                is_target = prepared["sequence"] == target
                body = _body(prepared)
                try:
                    result = await _run(scope, states, operation, body)
                except KeyError as exc:
                    raise ReplayUnsupported(f"Recorded state {exc} was never replayed.") from exc
                except PantographTacticRejected:
                    if is_target or (done or {}).get("outcome") == "TacticSuccess":
                        divergence = {"sequence": prepared["sequence"], "expected": (done or {}).get("outcome"),
                                      "observed": "PantographTacticRejected"}
                        break
                    replayed += 1
                    continue
                except PantographError as exc:
                    if is_target:
                        break
                    recorded = (done or {}).get("error_type")
                    if done is not None and done.get("stage") == "failed" and recorded == type(exc).__name__ \
                            and client.is_usable:
                        # A recoverable failure that recurred; the session continues.
                        replayed += 1
                        continue
                    divergence = {"sequence": prepared["sequence"], "expected": recorded or (done or {}).get("outcome"),
                                  "observed": type(exc).__name__}
                    break
                replayed += 1
                if is_target:
                    divergence = {"sequence": target, "expected": expected and expected["error_type"],
                                  "observed": "success"}
                    break
                if done is not None and done.get("stage") == "failed":
                    divergence = {"sequence": prepared["sequence"], "expected": done.get("error_type"),
                                  "observed": "success"}
                    break
                allocated = (done or {}).get("allocated_state_id")
                if allocated is not None and result is not None:
                    states[allocated] = result
    except PantographError:
        pass
    finally:
        await client.aclose()
    if divergence is not None:
        return {"status": "diverged", "divergence": divergence, "replayed_requests": replayed}
    failure = client.last_failure
    observed = failure_signature(failure.to_dict()) if failure is not None and failure.sequence else None
    if observed is not None and failure.operation != (expected or {}).get("operation"):
        observed = None
    reproduced = observed is not None and observed == expected
    return {
        "status": ("reproduced" if reproduced and not mismatch
                   else "reproduced-in-different-environment" if reproduced
                   else "not-reproduced"),
        "environment_mismatch": mismatch,
        "expected": expected, "observed": observed, "replayed_requests": replayed,
        "repl_detail": failure.repl_failure.detail if failure is not None and failure.repl_failure else None,
    }
