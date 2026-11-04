"""Batch projections use the same structured compiler as live search."""

from maths_ai.pln_inference.compiler import GoalInput, compile_goal, render
from maths_ai.hybrid_reasoner.pantograph_adapter import execution_goal_to_translator
from maths_ai.lean_replay import render_replay, seed_from_text


def essentialize_subgoal(goal_data, *, kb="kb", depth=10):
    compiled = compile_goal(GoalInput.from_projection(goal_data))
    records = [
        {
            "index": i,
            "label": f"local_hyp_{i}",
            "identity": name,
            "formula": render(formula),
        }
        for i, (name, formula) in enumerate(compiled.facts)
    ]
    return compiled.source(depth=depth, kb=kb), "", records, render(compiled.target)


async def proof_state_after_tactics(client, goal_type: str, tactics: list[str]):
    """Return owned raw-expression projections; no live handle escapes its scope."""
    replay = render_replay(seed_from_text(goal=goal_type))
    async with client.state_scope() as scope:
        state = await scope.start_goal(replay.expression, levels=replay.universe_levels)
        for tactic in tactics or ["skip"]:
            previous = state
            state = await scope.apply_tactic(state, tactic)
            await scope.release(previous)
        return [execution_goal_to_translator(goal) for goal in state.goals]
