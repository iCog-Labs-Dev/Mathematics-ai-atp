"""Strict owned-client factory and scoped canonical-goal materialization."""

from __future__ import annotations

import asyncio
from dataclasses import replace

from pantograph_async import (
    PantographClient,
    PantographEnv,
    PantographGoalRejected,
    PantographStartupError,
    PantographStateScope,
    PantographTacticRejected,
    TacticState,
)
from pantograph_async.probes import probe_model_client

from maths_ai.data_models.model_identity import with_model_identity
from maths_ai.data_models.proof_components import Goal
from maths_ai.lean_replay import (
    GoalReplaySpec,
    LeanGoalSeed,
    ReplayFailure,
    ReplayInputError,
    ReplayValidation,
    render_replay,
    validate_model_input,
    validate_replay,
)

from .failure_evidence import RequestHistory
from .pantograph_adapter import (
    execution_goal_to_goal,
    execution_goal_to_structured_locals,
    require_client_compatibility,
)


async def create_model_sexpr_client(env: PantographEnv) -> PantographClient:
    """Finish descriptor, options, and scoped model probes under one deadline."""
    require_client_compatibility()
    started = asyncio.get_running_loop().time()
    strict_env = replace(env, imports=tuple(dict.fromkeys((*env.imports, "Mathlib"))))
    client = None
    try:
        async with asyncio.timeout_at(started + strict_env.server_start_timeout_s):
            history = RequestHistory()
            client = await PantographClient.create(strict_env, observer=history)
            # Read by failure capture after the session ends; holds plain events only.
            client.request_history = history
            await probe_model_client(client)
        return client
    except BaseException as exc:
        if client is not None:
            try:
                await client.aclose()
            except BaseException as cleanup:
                if isinstance(exc, asyncio.CancelledError):
                    exc.add_note(f"Model-session cleanup also reported: {cleanup}")
                    raise exc
                raise
        if isinstance(exc, TimeoutError):
            raise PantographStartupError(
                "Model-session startup deadline expired."
            ) from exc
        raise


async def materialize_replay(
    scope: PantographStateScope, replay: GoalReplaySpec
) -> TacticState:
    """Serialize replayed locals, then release the unmaterialized start."""
    start = await scope.start_goal(replay.expression, levels=replay.universe_levels)
    try:
        tactic = (
            f"intro {' '.join(replay.local_names)}" if replay.local_names else "skip"
        )
        return await scope.apply_tactic(start, tactic)
    finally:
        if scope.is_usable:
            await scope.release(start)


async def materialize_seed(
    scope: PantographStateScope, seed: LeanGoalSeed
) -> tuple[TacticState, Goal, ReplayValidation]:
    """Replay ``seed`` and check that Lean built the goal it describes.

    Lean rejecting the input and a replay that does not match the seed are
    properties of the input, so both raise ``ReplayInputError``. Client and
    transport failures propagate unchanged. A structured seed's locals keep
    their source model identity on the returned goal; Lean's live names and
    indices are its execution identity.
    """
    replay = render_replay(seed)
    try:
        state = await materialize_replay(scope, replay)
    except (PantographGoalRejected, PantographTacticRejected) as exc:
        raise ReplayInputError(ReplayFailure.LEAN_REJECTED, str(exc)) from exc
    try:
        if len(state.goals) != 1:
            raise ReplayInputError(
                ReplayFailure.SEMANTIC_MISMATCH,
                f"Replay produced {len(state.goals)} goals instead of one.",
            )
        goal = state.goals[0]
        validation = validate_replay(
            seed, replay, goal.target.sexp, execution_goal_to_structured_locals(goal)
        )
        canonical = execution_goal_to_goal(goal)
        identities = [local.model_identity for local in seed.locals]
        if identities and all(identity is not None for identity in identities):
            canonical = with_model_identity(canonical, identities)
        validate_model_input(
            seed,
            canonical.goal_model_sexp,
            [local.type_model_sexp for local in canonical.locals or []],
        )
    except ReplayInputError:
        if scope.is_usable:
            await scope.release(state)
        raise
    return state, canonical, validation
