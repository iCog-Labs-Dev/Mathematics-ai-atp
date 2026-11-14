from types import SimpleNamespace

from maths_ai.accepted_gnn.pln_scoring import NO_EVIDENCE, score_goal
from maths_ai.data_models.proof_components import STV, Goal, GoalLocal
from maths_ai.pln_inference.compiler import GoalInput


class RecordingChainer:
    def __init__(self, result):
        self.result, self.calls = result, []

    def evaluate(self, goal):
        self.calls.append(goal)
        return self.result


def _result(stv, status, is_fallback):
    return SimpleNamespace(stv=stv, status=status, is_fallback=is_fallback)


def _model_goal(**raw):
    """A live accepted-search goal: model S-expressions, raw ones only if given."""
    return Goal(
        expression="p",
        goal_model_sexp="(:fv 0)",
        goal_sexp=raw.get("target"),
        locals=[
            GoalLocal(
                user_name="p",
                internal_name="_uniq.1",
                context_index=0,
                binder_role="default",
                type_pp="Prop",
                type_model_sexp="(:sort 0)",
                type_sexp=raw.get("local_type"),
            )
        ],
        model_sexp_version=1,
    )


def _structured_goal():
    return _model_goal(target="(:fv _uniq.1)", local_type="(:sort 0)")


def test_goal_without_raw_expressions_is_unsupported_and_not_queried():
    chainer = RecordingChainer(_result(STV(strength=1.0, confidence=1.0), "ok", False))
    score = score_goal(chainer, _model_goal())
    assert (score.stv, score.status, score.is_fallback) == (NO_EVIDENCE, "unsupported", False)
    assert chainer.calls == []


def test_rule_proof_score_is_kept_and_only_the_goal_itself_is_sent():
    stv = STV(strength=0.9, confidence=0.8)
    chainer = RecordingChainer(_result(stv, "ok", False))
    goal = _structured_goal()
    score = score_goal(chainer, goal)
    assert (score.stv, score.status, score.is_fallback) == (stv, "ok", False)
    assert chainer.calls == [GoalInput.from_goal(goal)]


def test_sampled_fallback_is_never_used_as_evidence():
    chainer = RecordingChainer(_result(STV(strength=0.7, confidence=0.5), "no_stv_found", True))
    score = score_goal(chainer, _structured_goal())
    assert (score.stv, score.is_fallback) == (NO_EVIDENCE, True)


def test_missing_pln_dependencies_have_zero_evidence():
    from maths_ai.pln_inference.dependencies import PLNConfigurationError

    class Unavailable:
        def evaluate(self, goal):
            raise PLNConfigurationError("Set PETTACHAINER_ROOT")

    score = score_goal(Unavailable(), _structured_goal())
    assert (score.stv, score.status, score.is_fallback) == (NO_EVIDENCE, "pln-unavailable", False)


def test_engine_failure_has_zero_evidence():
    chainer = RecordingChainer(_result(None, "timeout", False))
    score = score_goal(chainer, _structured_goal())
    assert (score.stv, score.status, score.is_fallback) == (NO_EVIDENCE, "timeout", False)
