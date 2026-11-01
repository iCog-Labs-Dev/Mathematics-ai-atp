"""Proposal modes, per-search generators and arity-correct argument sums."""

from __future__ import annotations

import unittest

import torch
from torch_geometric.data import Batch

from maths_ai.gnn_inference.atp_lean_gnn.policy_actions import CandidatePoolSnapshot, local_pools
from maths_ai.gnn_inference.atp_lean_gnn.graph import proof_state_to_dag
from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_premise_mask, build_vocab, dag_to_pyg
from maths_ai.gnn_inference.tests.model_helpers import actor_critic
from maths_ai.hybrid_reasoner.proposal import (
    ProposalSpec,
    derive_search_seed,
    input_identity,
    proposal_specs,
)
from maths_ai.lean_replay import seed_from_text

ID_TO_TACTIC = {0: "exact", 1: "have", 2: "trivial", 3: "apply", 4: "rw"}


def _graphs(*states: str):
    dags = [proof_state_to_dag(state) for state in states]
    vocab = build_vocab(dags)
    datas = []
    for dag in dags:
        data = dag_to_pyg(dag, vocab, add_reverse_edges=True)
        data.premise_mask = torch.tensor(build_premise_mask(dag), dtype=torch.bool)
        state_matches = (data.x == vocab.get("State", 0)).nonzero(as_tuple=False).view(-1)
        data.state_node_index = state_matches[-1:]
        datas.append(data)
    return datas, vocab


def _only(*tactic_ids: int) -> torch.Tensor:
    """A tactic mask that forces row b onto ``tactic_ids[b]``."""
    mask = torch.zeros(len(tactic_ids), len(ID_TO_TACTIC), dtype=torch.bool)
    for row, tactic_id in enumerate(tactic_ids):
        mask[row, tactic_id] = True
    return mask


STATES = ("n : Nat\nh : Even n\n⊢ Even n", "m : Nat\nk : Nat\nh : m = k\n⊢ k = m")


def _propose(model, batch, counts, **kwargs):
    with torch.no_grad():
        return model.propose_actions(batch, counts, candidate_pools=local_pools(batch), **kwargs)


def _refs(sample, instance):
    return [(ref.source, ref.identity) for ref in sample.arguments[instance]]


class ProposalModeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.datas, vocab = _graphs(*STATES)
        self.model = actor_critic(len(vocab), len(ID_TO_TACTIC))
        self.model.eval()
        self.batch = Batch.from_data_list(self.datas)

    def propose(self, counts=(1, 1), **kwargs):
        return _propose(self.model, self.batch, list(counts), **kwargs)

    def test_greedy_chooses_the_tactic_and_every_argument_by_argmax(self):
        # Greedy must not sample the argument count, source or identity.
        samples = [
            self.propose(greedy=True, generator=torch.Generator().manual_seed(seed))[0]
            for seed in range(5)
        ]
        for sample in samples[1:]:
            self.assertTrue(torch.equal(sample.tactic_action, samples[0].tactic_action))
            self.assertEqual(sample.arguments, samples[0].arguments)

    def test_greedy_takes_the_argmax_family(self):
        with torch.no_grad():
            _nodes, _state, logits, _values = self.model.encode(self.batch)
        sample, _rows = self.propose(greedy=True)
        self.assertTrue(torch.equal(sample.tactic_action, logits.argmax(dim=1)))

    def test_sampling_explores_tactic_families_and_arguments(self):
        families, arguments = set(), set()
        generator = torch.Generator().manual_seed(0)
        for _ in range(32):
            sample, _rows = self.propose((8, 8), generator=generator)
            families.update(int(t) for t in sample.tactic_action)
            arguments.update(tuple(_refs(sample, i)) for i in range(len(sample.arguments)))
        self.assertGreater(len(families), 1)
        self.assertGreater(len(arguments), 1)

    def test_a_seed_reproduces_its_draws_without_touching_the_global_rng(self):
        before = torch.get_rng_state()
        first, _ = self.propose((3, 3), generator=torch.Generator().manual_seed(7))
        second, _ = self.propose((3, 3), generator=torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        self.assertTrue(torch.equal(first.tactic_action, second.tactic_action))
        self.assertEqual(first.arguments, second.arguments)


class DecoderTests(unittest.TestCase):
    """The complete action law: tactic, argument count, then source and identity."""

    def setUp(self):
        torch.manual_seed(0)
        self.datas, self.vocab = _graphs(*STATES)
        self.model = actor_critic(len(self.vocab), len(ID_TO_TACTIC))
        self.model.eval()

    def test_an_empty_pool_proposes_no_arguments(self):
        batch = Batch.from_data_list(self.datas[:1])
        empty = (CandidatePoolSnapshot(()),)
        with torch.no_grad():
            sample, _rows = self.model.propose_actions(
                batch, [16], candidate_pools=empty, generator=torch.Generator().manual_seed(0)
            )
        self.assertTrue(all(args == () for args in sample.arguments))
        # The only legal count is zero, so it carries no probability mass away.
        self.assertTrue(torch.allclose(sample.count_logp, torch.zeros(16), atol=1e-6))
        self.assertTrue(torch.equal(sample.source_logp, torch.zeros(16)))

    def test_every_argument_is_inside_its_own_graph_pool(self):
        batch = Batch.from_data_list(self.datas)
        pools = local_pools(batch)
        sample, rows = _propose(
            self.model, batch, [16, 16], generator=torch.Generator().manual_seed(2)
        )
        for instance, row in enumerate(rows.tolist()):
            pools[row].validate(sample.arguments[instance])
            self.assertLessEqual(len(sample.arguments[instance]), self.model.max_args)

    def test_the_prior_is_the_complete_action_log_probability(self):
        batch = Batch.from_data_list(self.datas)
        sample, _rows = _propose(
            self.model, batch, [1, 1], generator=torch.Generator().manual_seed(4)
        )
        self.assertTrue(torch.allclose(
            sample.logp,
            sample.tactic_logp + sample.count_logp + sample.source_logp + sample.identity_logp,
        ))


class EncodeOnceTests(unittest.TestCase):
    """One proposal round encodes each state once and draws only what it asks for."""

    def setUp(self):
        torch.manual_seed(0)
        self.datas, vocab = _graphs(*STATES, "x : Nat\n⊢ x = x")
        self.model = actor_critic(len(vocab), len(ID_TO_TACTIC))
        self.model.eval()
        self.batch = Batch.from_data_list(self.datas)

    def propose(self, counts, **kwargs):
        return _propose(self.model, self.batch, counts, **kwargs)

    def test_one_encoder_pass_for_a_multi_draw_round(self):
        calls = []
        encode = self.model.encode_graph
        self.model.encode_graph = lambda data: calls.append(data.num_graphs) or encode(data)
        sample, rows = self.propose([4, 2, 1], generator=torch.Generator().manual_seed(0))
        self.assertEqual(calls, [3])
        self.assertEqual(rows.tolist(), [0, 0, 0, 0, 1, 1, 2])
        self.assertEqual(sample.tactic_action.shape, (7,))
        self.assertEqual(len(sample.arguments), 7)

    def test_unequal_counts_give_exactly_the_requested_instances(self):
        sample, rows = self.propose([0, 3, 1], generator=torch.Generator().manual_seed(1))
        self.assertEqual(rows.tolist(), [1, 1, 1, 2])
        self.assertEqual(sample.count_logp.shape, (4,))
        with self.assertRaises(ValueError):
            self.propose([1, 1])

    def test_each_instance_prior_equals_its_recomputed_probability(self):
        sample, rows = self.propose([3, 3, 3], generator=torch.Generator().manual_seed(3))
        pools = local_pools(self.batch)
        for instance, row in enumerate(rows.tolist()):
            with torch.no_grad():
                evaluated = self.model.evaluate_actions(
                    Batch.from_data_list([self.datas[row]]),
                    sample.tactic_action[instance : instance + 1],
                    [sample.arguments[instance]],
                    candidate_pools=(pools[row],),
                )
            # The instance's prior equals the recomputed probability of its own
            # tactic and arguments on its own state: the on-policy invariant.
            self.assertAlmostEqual(float(sample.tactic_logp[instance]), float(evaluated.tactic_logp[0]), places=5)
            self.assertAlmostEqual(float(sample.arg_logp[instance]), float(evaluated.arg_logp[0]), places=5)

    def test_a_fixed_seed_repeats_and_leaves_the_global_rng_alone(self):
        before = torch.get_rng_state()
        first, _ = self.propose([2, 2, 2], generator=torch.Generator().manual_seed(7))
        second, _ = self.propose([2, 2, 2], generator=torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        self.assertTrue(torch.equal(first.tactic_action, second.tactic_action))
        self.assertEqual(first.arguments, second.arguments)
        self.assertTrue(torch.allclose(first.arg_logp, second.arg_logp))


class SeedDerivationTests(unittest.TestCase):
    def test_seeds_depend_only_on_request_coordinates(self):
        seed = seed_from_text(goal="True")
        identity = input_identity(seed)
        self.assertEqual(identity, input_identity(seed_from_text(goal="True")))
        a = derive_search_seed("eval", 42, 0, identity)
        self.assertEqual(a, derive_search_seed("eval", 42, 0, identity))
        self.assertNotEqual(a, derive_search_seed("eval", 42, 1, identity))
        self.assertNotEqual(a, derive_search_seed("train", 42, 0, identity))
        self.assertNotEqual(a, derive_search_seed("eval", 43, 0, identity))
        self.assertLess(a, 2**63)

    def test_equal_inputs_are_told_apart_by_occurrence(self):
        seeds = [seed_from_text(goal="True"), seed_from_text(goal="False"), seed_from_text(goal="True")]
        specs = proposal_specs(seeds, mode="sample", namespace="eval", base_seed=1, pass_index=0)
        self.assertEqual(len({spec.seed for spec in specs}), 3)
        again = proposal_specs(seeds, mode="sample", namespace="eval", base_seed=1, pass_index=0)
        self.assertEqual(specs, again)
        # The first occurrence keeps its seed when the list grows.
        longer = proposal_specs([*seeds, seeds[1]], mode="sample", namespace="eval", base_seed=1, pass_index=0)
        self.assertEqual(longer[:3], specs)

    def test_specs_reject_unknown_modes_and_bad_seeds(self):
        with self.assertRaises(ValueError):
            ProposalSpec(mode="beam", seed=0)
        with self.assertRaises(ValueError):
            ProposalSpec(mode="sample", seed=-1)
        with self.assertRaises(ValueError):
            ProposalSpec(mode="sample", seed=True)


if __name__ == "__main__":
    unittest.main()
