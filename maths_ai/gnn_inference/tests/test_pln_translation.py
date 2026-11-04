"""Semantic boundaries for shared live/batch PLN translation."""

import asyncio
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from maths_ai.pln_inference.compiler import (
    Formula,
    Term,
    TranslationError,
    GoalInput,
    LocalInput,
    Compiler,
    compile_goal,
    compile_text,
    parse_text,
    render,
    symbol,
    atom,
    TRANSLATION_VERSION,
)
from maths_ai.pln_inference.answers import rule_answers
from maths_ai.pln_inference.model import PLNInference, PLNResult

P, Q, R = [atom(Term("text-proposition", name)) for name in "PQR"]
NAMES = {name: name for name in "PQR"}


def f(kind, *args):
    return Formula(kind, args)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("P ∧ Q → P", f("imp", f("and", P, Q), P)),
        ("¬P ∧ Q", f("and", f("not", P), Q)),
        ("(P) ∧ (Q)", f("and", P, Q)),
        ("P → Q → R", f("imp", P, f("imp", Q, R))),
        ("P ∨ Q ∧ R", f("or", P, f("and", Q, R))),
        ("P ↔ Q → R", f("iff", P, f("imp", Q, R))),
        ("¬(P ∧ Q)", f("not", f("and", P, Q))),
        ("P /- nested /- comment -/ end -/ → Q", f("imp", P, Q)),
        ("P -- comment\n → Q", f("imp", P, Q)),
    ],
)
def test_text_grammar(text, expected):
    assert parse_text(text, propositions=NAMES) == expected


@pytest.mark.parametrize(
    "text",
    [
        "x = x",
        "f x",
        "∀ x : Nat, x = x",
        "P ∧",
        "(P",
        "P)",
        "P := Q",
        "P; !(bad)",
        "?m",
        "_",
        "P ↔ Q ↔ R",
        "unknown",
    ],
)
def test_text_never_silently_becomes_an_atom(text):
    with pytest.raises(TranslationError):
        parse_text(text, propositions=NAMES)


def test_text_declarations_and_injective_symbols():
    goal = compile_text("Q", ("hP : P", "hPQ : P → Q"), propositions=NAMES)
    assert goal.target == Q and [v for _, v in goal.facts] == [P, f("imp", P, Q)]
    terms = [
        Term("constant", v)
        for v in ["f x", "f_x", "Foo.same", "Bar.same", "$x", "x✝", "«quoted name»"]
    ]
    terms += [Term("natural", "2"), Term("natural", "3"), Term("local", "same")]
    assert len(set(map(symbol, terms))) == len(terms)
    for term in terms:
        assert json.loads(bytes.fromhex(symbol(term)[5:])) == term.identity()


def truth(form, values):
    if form.kind == "atom":
        return values[form.term.value]
    a = [truth(v, values) for v in form.args]
    return {
        "not": lambda: not a[0],
        "and": lambda: a[0] and a[1],
        "or": lambda: a[0] or a[1],
        "imp": lambda: not a[0] or a[1],
        "iff": lambda: a[0] == a[1],
    }[form.kind]()


def test_independent_truth_tables():
    for p, q, r in itertools.product([False, True], repeat=3):
        values = dict(zip("PQR", (p, q, r)))
        assert truth(parse_text("P ∧ Q → P", propositions=NAMES), values) == (
            not (p and q) or p
        )
        assert truth(parse_text("¬P ∧ Q", propositions=NAMES), values) == (not p and q)
        assert truth(parse_text("P ∨ Q ∧ R", propositions=NAMES), values) == (
            p or (q and r)
        )


ROWS = json.loads(
    (Path(__file__).parent / "fixtures/pln_translation/lean_states.json").read_text()
)


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r["name"])
def test_real_lean_structured_fixtures(row):
    data = GoalInput.from_projection(row["goal"])
    if row["name"] == "dependent":
        with pytest.raises(TranslationError):
            compile_goal(data)
        return
    compiled = compile_goal(data)
    assert render(compiled.target)
    if row["name"] == "implication":
        assert len(compiled.facts) == 2 and len(compiled.excluded_locals) == 2
    if row["name"] == "data_context":
        assert len(compiled.facts) == 1 and len(compiled.excluded_locals) == 2
    if row["name"] == "shadowing":
        compiler = Compiler(data)
        from maths_ai.lean_replay.raw import FVar

        ids = [v.internal_name for v in data.locals]
        assert compiler.term(FVar(ids[0])) != compiler.term(FVar(ids[1]))


@pytest.mark.parametrize(
    "output,expected",
    [
        ("(: hP (P) (STV 1 1))", False),
        ("(: (rule-proof rule h) (P) (STV 1 0.9))", True),
        ("(: (merge/revision h (rule-proof rule fact)) (P) (STV 1 0.9))", True),
        ("(: rule-proof (P) (STV 1 1))", False),
        ("(: (rule-proof bad) (Q) (STV 1 1))", False),
        ("rule-proof\n(: hP (P) (STV 1 1))", False),
        ("(: (rule-proof rule h) (P) (STV 2 1))", False),
        ("(rules (: (rule-proof bad) (P) (STV 1 1)))", False),
        ("(: (rule-proof rule h) (P) (STV nan 1))", False),
        ("(: (kb MAIN Nil) rule-proof (P) (STV 1 1))", False),
    ],
)
def test_success_requires_this_target_and_its_rule_proof(output, expected):
    assert bool(rule_answers(output, "(P)")) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "(:mv unknown)",
        "(:alien stuff)",
        "((:c Prod.fst) _ _ (:c x))",
        "(:forall h (:c True) 0)",
    ],
)
def test_unsupported_expressions_are_unscored(raw):
    result = PLNInference().evaluate(GoalInput(raw, ()))
    assert (
        result.status == "unsupported" and result.stv is None and not result.is_fallback
    )


def test_query_budget_and_unknown_hypothesis_are_explicit():
    with pytest.raises(TranslationError):
        compile_goal(GoalInput("(" * 200 + "(:c True)" + ")" * 200, ()))
    data = GoalInput("(:c True)", (LocalInput("h", "(:c Unclassified.Predicate)"),))
    with pytest.raises(TranslationError, match="locals"):
        compile_goal(data)


def test_ranker_passes_structured_goal_not_parent_and_never_learns_unsupported():
    from maths_ai.hybrid_reasoner.joint_inference import HybridReasoner
    from maths_ai.data_models.proof_components import TacticCandidate, Goal, GoalLocal
    from maths_ai.pln_inference.metta.translator.translator_modules.runner import (
        DynamicThompsonSampler,
    )

    reasoner = object.__new__(HybridReasoner)
    reasoner.pln_counts = {}
    reasoner.use_pln = True
    reasoner.pln_fallback_strategy = "thompson"
    reasoner.dts_sampler = DynamicThompsonSampler()
    captured = []

    class Scorer:
        async def evaluate_async(self, goal, **kwargs):
            captured.append(goal)
            return PLNResult(None, "unsupported", False, diagnostic={"reason": "test"})

    reasoner.petta_chainer = Scorer()
    goal = Goal(
        expression="parent text",
        goal_sexp="(:c True)",
        goal_model_sexp="(:c True)",
        locals=[],
        model_sexp_version=1,
    )
    action = TacticCandidate(tactic_name="skip", arguments=[], probability=0.7)
    ranked = asyncio.run(
        reasoner.rank_subgoals("parent text", [goal], action, gnn_probability=0.7)
    )
    assert len(captured) == 1 and captured[0].locals == ()
    assert ranked[0].stv is None and ranked[0].combined_rank == 0.7
    assert reasoner.dts_sampler.state_dict == {}


@pytest.mark.parametrize("policy", ["legacy", "puct"])
def test_unscored_and_children_remain_searchable(policy):
    from maths_ai.gnn_inference.tests.test_rl_reasoner import (
        _make_reasoner,
        _SubgoalExecutor,
        GOAL_EXPR,
        SEED,
        SAMPLE,
    )

    kwargs = {"selection_policy": policy}
    if policy == "puct":
        kwargs.update(num_simulations=16, sim_batch_size=2, puct_c=1.0)
    executor = _SubgoalExecutor({GOAL_EXPR: ["A", "B"]})
    reasoner, _, _ = _make_reasoner(executor, top_k=1, **kwargs)
    reasoner.petta_chainer = PLNInference()
    result = asyncio.run(reasoner.prove(SEED, proposal=SAMPLE))
    children = [
        node for node in result.graph.nodes.values() if node.id != result.graph.root_id
    ]
    assert {node.goal.expression for node in children} >= {"A", "B"}
    assert all(node.stv is None for node in children)
    assert result.graph.is_solved()


def test_text_context_and_malformed_declarations_are_rejected():
    with pytest.raises(TranslationError):
        parse_text("P ∧ Q", propositions={"P": "same", "Q": "same"})
    result = PLNInference().evaluate_text("P", ("h : (P",), propositions={"P": "P"})
    assert result.status == "unsupported" and result.stv is None


def test_failed_batch_process_cannot_supply_success_or_sampler_evidence(tmp_path):
    from maths_ai.pln_inference.metta.translator.translator_modules.runner import (
        parse_and_rank_logs,
        DynamicThompsonSampler,
    )

    log = tmp_path / "proof.log"
    log.write_text("(: (rule-proof rule fact) (P) (STV 1 1))")
    code = tmp_path / "exit"
    code.write_text("2")
    rows = [
        {
            "test_name": "t",
            "goal_index": 0,
            "log_path": str(log),
            "exit_code_path": str(code),
            "target_formula": "(P)",
        },
        {
            "test_name": "t",
            "goal_index": 1,
            "status": "unsupported",
            "log_path": "",
            "target_formula": None,
        },
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"translation_version": TRANSLATION_VERSION, "generated_items": rows}
        )
    )
    sampler = DynamicThompsonSampler()
    ranked = parse_and_rank_logs(
        str(manifest), fallback_strategy="thompson", thompson_sampler=sampler
    )
    assert [v["status"] for v in ranked] == ["engine_error", "unsupported"]
    assert all(not v["truth_values"] and not v["random_fallback_used"] for v in ranked)
    assert sampler.state_dict == {}


def test_generated_runner_records_exit_code(tmp_path, monkeypatch):
    import subprocess
    from maths_ai.pln_inference.metta.translator.translator_modules.runner import (
        write_runner_script,
    )

    tool = tmp_path / "petta"
    tool.write_text('#!/bin/sh\nprintf "failed process\\n"\nexit 7\n')
    tool.chmod(0o755)
    query = tmp_path / "query.metta"
    query.write_text("")
    log = tmp_path / "query.log"
    code = tmp_path / "query.exit"
    script = write_runner_script(
        [
            {
                "metta_path": str(query),
                "log_path": str(log),
                "exit_code_path": str(code),
                "test_name": "case",
                "goal_index": 0,
            }
        ],
        str(tmp_path),
    )
    monkeypatch.setenv("PETTA_BIN", str(tool))
    subprocess.run(["bash", script], check=True, capture_output=True, text=True)
    assert code.read_text().strip() == "7"
