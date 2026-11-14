"""PLN scores for the accepted search, through the shared structured compiler.

The accepted search used to send printed goal text to a text translator and
asserted the parent target as a hypothesis. PLN now translates only a goal's
raw target and locals, so a goal without them is ``unsupported`` and carries
no evidence. Such a goal, and any query without a rule proof, ranks with a
zero-evidence STV. The engine's sampled exploration fallback is never used as
evidence here: it ranks with zero evidence and keeps ``is_fallback`` set, so
the search's own DTS sampler may replace it. Unresolvable PLN dependencies
rank with zero evidence too; the backend's readiness reports them.
"""

from __future__ import annotations

from dataclasses import dataclass

from maths_ai.data_models.proof_components import STV, Goal
from maths_ai.pln_inference.compiler import GoalInput, TranslationError
from maths_ai.pln_inference.dependencies import PLNConfigurationError

NO_EVIDENCE = STV(strength=0.0, confidence=0.0)


@dataclass(frozen=True)
class GoalScore:
    stv: STV
    status: str
    is_fallback: bool


def score_goal(chainer, goal: Goal) -> GoalScore:
    try:
        structured = GoalInput.from_goal(goal)
    except TranslationError:
        return GoalScore(NO_EVIDENCE, "unsupported", False)
    try:
        result = chainer.evaluate(structured)
    except PLNConfigurationError:
        return GoalScore(NO_EVIDENCE, "pln-unavailable", False)
    if result.is_fallback or result.stv is None:
        return GoalScore(NO_EVIDENCE, result.status, result.is_fallback)
    return GoalScore(result.stv, result.status, False)
