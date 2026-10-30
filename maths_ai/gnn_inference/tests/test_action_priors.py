"""Complete-action priors: raw log-priors, aliases and live-support normalization."""

from __future__ import annotations

import asyncio
import math
import unittest

import torch

from maths_ai.data_models.proof_components import Goal, ProposedAction, TacticCandidate
from maths_ai.gnn_inference.atp_lean_gnn.actor_critic import ActionSample
from maths_ai.gnn_inference.atp_lean_gnn.inference import InferencePipeline, local_declarations
from maths_ai.gnn_inference.atp_lean_gnn.policy_actions import (
    ArgumentReference,
    CandidatePoolSnapshot,
    LocalCandidateProvider,
)
from maths_ai.gnn_inference.atp_lean_gnn.pln_rl_training import ObservationOutcome
from maths_ai.gnn_inference.tests.model_helpers import ScriptedExecutor, structured_goal
from maths_ai.gnn_inference.tests.test_rl_reasoner import (
    GOAL_EXPR,
    HYPS,
    SAMPLE,
    SEED,
    _FakeServer,
    _make_reasoner,
)
from maths_ai.hybrid_reasoner.hypergraph import EdgeStatus, ProofHypergraph


def action(name: str, probability: float, *, raw=None, tactic_probability: float = 0.5) -> ProposedAction:
    return ProposedAction(
        candidate=TacticCandidate(tactic_name=name, arguments=[], probability=tactic_probability),
        raw_action_id=raw or (name,),
        log_prior=math.log(probability),
        provider="test",
    )


def graph_with(*actions: ProposedAction) -> tuple[ProofHypergraph, list[int]]:
    graph = ProofHypergraph(Goal(expression="P", hypotheses=[]))
    child = Goal(expression="Q", hypotheses=[])
    ids = [graph.add_edge(graph.root_id, a, [(child, None)]).id for a in actions]
    return graph, ids


class ProposedActionTests(unittest.TestCase):
    def test_invalid_priors_are_contract_errors(self):
        for bad in (float("nan"), float("inf"), 0.5, -float("inf"), True, None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ProposedAction(
                    candidate=TacticCandidate(tactic_name="t", arguments=[], probability=1.0),
                    raw_action_id=("t",),
                    log_prior=bad,
                    provider="test",
                )

    def test_a_graph_insertion_needs_a_scored_proposal(self):
        graph = ProofHypergraph(Goal(expression="P", hypotheses=[]))
        with self.assertRaises(TypeError):
            graph.add_edge(graph.root_id, TacticCandidate(tactic_name="t", probability=1.0), [])


class NormalizationTests(unittest.TestCase):
    def test_priors_normalize_over_the_live_support(self):
        graph, (a, b) = graph_with(action("a", 0.32), action("b", 0.08))
        priors = graph.puct_priors(graph.root_id)
        self.assertAlmostEqual(priors[a], 0.8)
        self.assertAlmostEqual(priors[b], 0.2)

        graph.edges[a].visit_stats.N, graph.edges[a].visit_stats.W = 3, 1.5
        child = Goal(expression="R", hypotheses=[])
        c = graph.add_edge(graph.root_id, action("c", 0.10), [(child, None)]).id
        priors = graph.puct_priors(graph.root_id)
        self.assertAlmostEqual(priors[a], 0.64)
        self.assertAlmostEqual(priors[b], 0.16)
        self.assertAlmostEqual(priors[c], 0.20)
        self.assertAlmostEqual(sum(priors.values()), 1.0)
        # Adding support changes no visit statistics or raw log-priors.
        self.assertEqual((graph.edges[a].visit_stats.N, graph.edges[a].visit_stats.W), (3, 1.5))
        self.assertEqual(graph.edges[a].raw_log_priors, {("a",): math.log(0.32)})

    def test_a_dead_edge_leaves_the_support(self):
        graph, (a, b) = graph_with(action("a", 0.32), action("b", 0.08))
        graph.mark_edge_dead(b)
        self.assertEqual(graph.edges[b].status, EdgeStatus.DEAD)
        self.assertEqual(graph.puct_priors(graph.root_id), {a: 1.0})

    def test_tiny_joint_probabilities_normalize_stably(self):
        graph = ProofHypergraph(Goal(expression="P", hypotheses=[]))
        child = Goal(expression="Q", hypotheses=[])
        tiny = [
            ProposedAction(
                candidate=TacticCandidate(tactic_name=name, arguments=[], probability=0.5),
                raw_action_id=(name,),
                log_prior=log_prior,
                provider="test",
            )
            for name, log_prior in (("a", -1000.0), ("b", -1000.0 - math.log(3)))
        ]
        a, b = (graph.add_edge(graph.root_id, t, [(child, None)]).id for t in tiny)
        priors = graph.puct_priors(graph.root_id)
        self.assertAlmostEqual(priors[a], 0.75)
        self.assertAlmostEqual(priors[b], 0.25)


class AliasTests(unittest.TestCase):
    def test_a_distinct_alias_adds_its_mass_once(self):
        graph, (a, b) = graph_with(action("a", 0.2), action("b", 0.2))
        alias = action("a", 0.2, raw=("a", "other pointer tuple"))
        self.assertTrue(graph.observe_alias(a, alias))
        self.assertFalse(graph.observe_alias(a, alias))
        self.assertFalse(graph.observe_alias(a, action("a", 0.2)))
        priors = graph.puct_priors(graph.root_id)
        self.assertAlmostEqual(priors[a], 2 / 3)
        self.assertAlmostEqual(priors[b], 1 / 3)

    def test_aliases_do_not_change_ranking_confidence(self):
        graph, (a, _b) = graph_with(action("a", 0.2, tactic_probability=0.9), action("b", 0.2))
        rank = graph.root.combined_rank
        graph.observe_alias(a, action("a", 0.6, raw=("a", 2), tactic_probability=0.9))
        self.assertEqual(graph.edges[a].tactic.probability, 0.9)
        self.assertEqual(graph.root.combined_rank, rank)

    def test_a_changed_prior_for_a_known_raw_action_is_an_error(self):
        graph, (a,) = graph_with(action("a", 0.2))
        with self.assertRaises(ValueError):
            graph.observe_alias(a, action("a", 0.3))

    def test_an_alias_must_render_the_same_command(self):
        graph, (a,) = graph_with(action("a", 0.2))
        with self.assertRaises(ValueError):
            graph.observe_alias(a, action("b", 0.2))


class GenericProviderLawTests(unittest.TestCase):
    def test_the_engine_prior_is_its_greedy_complete_action_probability(self):
        reasoner, model, vocab = _make_reasoner(_CountingExecutor())
        pipeline = InferencePipeline(
            model, LocalCandidateProvider(), vocab, reasoner.tactic_vocab, torch.device("cpu")
        )
        goal = structured_goal(GOAL_EXPR, HYPS)
        asyncio.run(pipeline.prepare_goal(goal))
        model.eval()
        result = pipeline.predict_from_goal(goal, top_k=3)
        for prediction in result.top_tactic_predictions:
            self.assertAlmostEqual(
                prediction["probability"], math.exp(prediction["action_log_prior"]), places=6
            )
            tactic_id, references = prediction["raw_action_id"]
            self.assertEqual(tactic_id, prediction["tactic_id"])
            self.assertEqual(len(references), len(prediction["selected_arguments"]))
        priors = [p["action_log_prior"] for p in result.top_tactic_predictions]
        self.assertTrue(all(value <= 0.0 for value in priors))


def _stack(samples: list[ActionSample]) -> ActionSample:
    """One ActionSample whose instances are ``samples`` in order."""
    cat = torch.cat
    return ActionSample(
        tactic_action=cat([s.tactic_action for s in samples]),
        tactic_logp=cat([s.tactic_logp for s in samples]),
        tactic_entropy=cat([s.tactic_entropy for s in samples]),
        arguments=tuple(args for s in samples for args in s.arguments),
        count_logp=cat([s.count_logp for s in samples]),
        source_logp=cat([s.source_logp for s in samples]),
        identity_logp=cat([s.identity_logp for s in samples]),
        value=cat([s.value for s in samples]),
        tactic_logits=cat([s.tactic_logits for s in samples]),
    )


def _CountingExecutor():
    """Accepts every command with QED and records the Lean calls (``calls``)."""
    return ScriptedExecutor(lambda _occurrence, _command: [], server=_FakeServer())


class ActorPriorTests(unittest.TestCase):
    """The actor's priors through decoding and a real search round."""

    @staticmethod
    def _exact(index: int, arg_logp: float) -> ActionSample:
        """``exact`` with one local argument whose whole log-probability is ``arg_logp``."""
        return ActionSample(
            tactic_action=torch.tensor([2]),  # "exact"
            tactic_logp=torch.tensor([-0.5]),
            tactic_entropy=torch.tensor([1.0]),
            arguments=((ArgumentReference("local", index),),),
            count_logp=torch.tensor([arg_logp]),
            source_logp=torch.tensor([0.0]),
            identity_logp=torch.tensor([0.0]),
            value=torch.tensor([0.0]),
            tactic_logits=torch.zeros(1, 3),
        )

    def _positions(self, reasoner):
        goal = structured_goal(GOAL_EXPR, HYPS)
        dag, data = reasoner.dag_featurize(goal)
        fv = next(i for i, n in enumerate(dag.nodes) if n.label.startswith("FV"))
        hyp = next(i for i, n in enumerate(dag.nodes) if n.label == "Hyp")
        return goal, dag, data, hyp, fv

    def test_the_same_tactic_with_different_arguments_gets_different_priors(self):
        reasoner, _model, _vocab = _make_reasoner(_CountingExecutor())
        goal, dag, data, hyp, fv = self._positions(reasoner)
        declarations = local_declarations(goal)
        pool = CandidatePoolSnapshot(tuple(sorted({hyp, fv})))
        a, _ = reasoner._decode(self._exact(hyp, -0.1), dag, declarations, "fp", pool)
        b, _ = reasoner._decode(self._exact(fv, -2.0), dag, declarations, "fp", pool)
        self.assertEqual(a.candidate.tactic_name, b.candidate.tactic_name)
        self.assertAlmostEqual(a.log_prior, -0.6, places=6)
        self.assertAlmostEqual(b.log_prior, -2.5, places=6)
        # Ranking confidence is the tactic's alone.
        self.assertAlmostEqual(a.candidate.probability, math.exp(-0.5), places=6)
        self.assertEqual(a.candidate.probability, b.candidate.probability)

    def test_aliases_share_one_execution_and_keep_their_own_observations(self):
        executor = _CountingExecutor()
        reasoner, model, _vocab = _make_reasoner(executor, top_k=3)
        _goal, _dag, _data, hyp, fv = self._positions(reasoner)
        # Both the Hyp node and its FV render "exact p": two raw actions, one command.
        draws = [self._exact(hyp, -0.1), self._exact(fv, -2.0), self._exact(hyp, -0.1)]
        model.propose_actions = lambda *_args, **_kwargs: (_stack(draws), torch.tensor([0, 0, 0]))
        result = asyncio.run(reasoner.prove(SEED, proposal=SAMPLE))

        self.assertEqual(executor.calls, ["exact p"])
        (edge,) = result.graph.edges.values()
        on_hyp, on_fv = (2, (("local", hyp),)), (2, (("local", fv),))
        self.assertEqual(set(edge.raw_log_priors), {on_hyp, on_fv})
        self.assertAlmostEqual(edge.raw_log_priors[on_hyp], -0.6, places=6)
        self.assertAlmostEqual(edge.raw_log_priors[on_fv], -2.5, places=6)
        self.assertAlmostEqual(edge.log_mass, math.log(math.exp(-0.6) + math.exp(-2.5)), places=6)
        observations = {o.action.raw_action_id: o for o in result.observations}
        self.assertEqual(set(observations), {on_hyp, on_fv})
        self.assertEqual(observations[on_hyp].action.multiplicity, 2)
        self.assertEqual(observations[on_fv].action.multiplicity, 1)
        for observation in observations.values():
            self.assertEqual(observation.outcome, ObservationOutcome.ACCEPTED)
            self.assertEqual(observation.edge_id, edge.id)
        self.assertEqual(result.draws(), 3)


if __name__ == "__main__":
    unittest.main()
