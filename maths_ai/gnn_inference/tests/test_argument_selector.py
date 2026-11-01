from __future__ import annotations

import unittest

import torch

from maths_ai.gnn_inference.atp_lean_gnn import (
    DEMO_STATE,
    TACTIC_ARITY,
    build_premise_mask,
    build_vocab,
    get_tactic_arity,
    parse_tactic_arguments,
    proof_state_to_dag,
)

from maths_ai.gnn_inference.atp_lean_gnn.pyg import dag_to_pyg
from maths_ai.gnn_inference.tests.model_helpers import pointer


class TacticArityRegistryTests(unittest.TestCase):
    def test_known_tactics_return_correct_arity(self) -> None:
        self.assertEqual(get_tactic_arity("simp"), 0)
        self.assertEqual(get_tactic_arity("apply"), 1)
        self.assertEqual(get_tactic_arity("exact"), 1)
        self.assertEqual(get_tactic_arity("rw"), 1)
        self.assertEqual(get_tactic_arity("have"), 2)

    def test_unknown_tactic_returns_default(self) -> None:
        self.assertEqual(get_tactic_arity("totally_unknown_tactic_xyz"), 1)

    def test_all_entries_have_nonnegative_arity(self) -> None:
        for tactic, arity in TACTIC_ARITY.items():
            self.assertGreaterEqual(arity, 0, f"Tactic '{tactic}' has negative arity")


class ParseTacticArgumentsTests(unittest.TestCase):
    def test_bracket_arguments(self) -> None:
        name, args = parse_tactic_arguments("rw [foo, bar]")
        self.assertEqual(name, "rw")
        self.assertEqual(args, ["foo", "bar"])

    def test_plain_argument(self) -> None:
        name, args = parse_tactic_arguments("apply h1")
        self.assertEqual(name, "apply")
        self.assertEqual(args, ["h1"])

    def test_simp_only_with_brackets(self) -> None:
        name, args = parse_tactic_arguments("simp only [h1, h2]")
        self.assertEqual(name, "simp")
        self.assertEqual(args, ["h1", "h2"])

    def test_no_arguments(self) -> None:
        name, args = parse_tactic_arguments("simp")
        self.assertEqual(name, "simp")
        self.assertEqual(args, [])

    def test_empty_string(self) -> None:
        name, args = parse_tactic_arguments("")
        self.assertEqual(args, [])

    def test_exact_with_complex_argument(self) -> None:
        name, args = parse_tactic_arguments("exact Nat.zero_add n")
        self.assertEqual(name, "exact")
        self.assertIn("Nat.zero_add", args)


class PremiseMaskTests(unittest.TestCase):
    def test_excludes_syntax_nodes(self) -> None:
        dag = proof_state_to_dag(DEMO_STATE)
        mask = build_premise_mask(dag)

        self.assertEqual(len(mask), dag.num_nodes)

        for node, is_valid in zip(dag.nodes, mask):
            if node.label in ("App", "Arrow", "State", "Goal", "Forall"):
                self.assertFalse(
                    is_valid,
                    f"Syntax node '{node.label}' (id={node.id}) should be masked out",
                )

    def test_includes_var_and_hyp_nodes(self) -> None:
        dag = proof_state_to_dag(DEMO_STATE)
        mask = build_premise_mask(dag)

        hyp_included = any(mask[node.id] for node in dag.nodes if node.label == "Hyp")
        var_included = any(
            mask[node.id] for node in dag.nodes if node.node_type == "var"
        )
        self.assertTrue(hyp_included, "At least one 'Hyp' node should be included")
        self.assertTrue(var_included, "At least one 'var' node should be included")

    def test_at_least_one_node_is_selectable(self) -> None:
        dag = proof_state_to_dag(DEMO_STATE)
        mask = build_premise_mask(dag)
        self.assertTrue(any(mask), "Premise mask should have at least one True entry")


def test_supervised_count_loss_trains_on_zero_argument_targets():
    from torch_geometric.data import Batch, Data
    from maths_ai.gnn_inference.atp_lean_gnn.argument_selector import supervised_loss
    from maths_ai.gnn_inference.atp_lean_gnn.policy_actions import local_pools

    graph = Data(
        x=torch.tensor([1, 2]),
        node_type=torch.tensor([0, 0]),
        edge_index=torch.tensor([[0], [1]]),
        state_node_index=torch.tensor([1]),
        premise_mask=torch.tensor([True, True]),
        y=torch.tensor([1]),
        action_target_supported=torch.tensor([True]),
        arg_count=torch.tensor([0]),
        arg_lemma_ids=torch.tensor([], dtype=torch.long),
        arg_node_indices=torch.tensor([], dtype=torch.long),
    )
    batch = Batch.from_data_list([graph])
    model = pointer(4, 3)
    loss, _, _ = supervised_loss(model, batch, candidate_pools=local_pools(batch))
    loss.backward()
    assert model.decoder.arg_count_head.weight.grad.abs().sum() > 0


def test_unsupported_syntax_does_not_train_zero_argument_count():
    from torch_geometric.data import Batch, Data
    from maths_ai.gnn_inference.atp_lean_gnn.argument_selector import supervised_loss
    from maths_ai.gnn_inference.atp_lean_gnn.policy_actions import local_pools

    graph = Data(
        x=torch.tensor([1, 2]),
        node_type=torch.tensor([0, 0]),
        edge_index=torch.tensor([[0], [1]]),
        state_node_index=torch.tensor([1]),
        premise_mask=torch.tensor([True, True]),
        y=torch.tensor([1]),
        action_target_supported=torch.tensor([False]),
        arg_count=torch.tensor([0]),
        arg_lemma_ids=torch.tensor([], dtype=torch.long),
        arg_node_indices=torch.tensor([], dtype=torch.long),
    )
    batch = Batch.from_data_list([graph])
    model = pointer(4, 3)
    loss, _, _ = supervised_loss(model, batch, candidate_pools=local_pools(batch))
    loss.backward()
    assert model.decoder.arg_count_head.weight.grad is None
    assert model.tactic_classifier.weight.grad.abs().sum() > 0


def test_flat_syntax_preserves_unicode_and_rejects_fresh_names_and_modifiers():
    from maths_ai.gnn_inference.atp_lean_gnn.flat_action_targets import (
        flat_reference_syntax,
    )

    assert flat_reference_syntax("exact hα") == (["hα"], "supported")
    assert flat_reference_syntax("rw [<a>Nat.add_zero</a>, h]") == (
        ["Nat.add_zero", "h"],
        "supported",
    )
    assert flat_reference_syntax("intro x")[1] != "supported"
    assert flat_reference_syntax("simp only [h]")[1] != "supported"
