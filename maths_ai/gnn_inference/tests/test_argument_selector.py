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
from maths_ai.gnn_inference.atp_lean_gnn.argument_selector import (
    ArgumentSelector,
    TacticWithArgsClassifier,
    compute_combined_loss,
)
from maths_ai.gnn_inference.atp_lean_gnn.pyg import dag_to_pyg


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

        hyp_included = any(
            mask[node.id] for node in dag.nodes if node.label == "Hyp"
        )
        var_included = any(
            mask[node.id] for node in dag.nodes if node.node_type == "var"
        )
        self.assertTrue(hyp_included, "At least one 'Hyp' node should be included")
        self.assertTrue(var_included, "At least one 'var' node should be included")

    def test_at_least_one_node_is_selectable(self) -> None:
        dag = proof_state_to_dag(DEMO_STATE)
        mask = build_premise_mask(dag)
        self.assertTrue(any(mask), "Premise mask should have at least one True entry")


class ArgumentSelectorTests(unittest.TestCase):
    def test_output_shape_and_masking(self) -> None:
        hidden_dim = 16
        selector = ArgumentSelector(hidden_dim)

        batch_size = 2
        nodes_per_graph = [5, 7]
        total_nodes = sum(nodes_per_graph)

        state_emb = torch.randn(batch_size, hidden_dim)
        tactic_emb = torch.randn(batch_size, hidden_dim)
        node_embeddings = torch.randn(total_nodes, hidden_dim)

        # Build batch index and premise mask
        batch_index = torch.cat([
            torch.full((n,), i, dtype=torch.long) for i, n in enumerate(nodes_per_graph)
        ])
        premise_mask = torch.ones(total_nodes, dtype=torch.bool)
        # Mask out the last node in each graph
        premise_mask[4] = False   # last node of graph 0
        premise_mask[11] = False  # last node of graph 1

        scores, selected_emb = selector(
            state_emb, tactic_emb, node_embeddings, premise_mask, batch_index
        )

        max_nodes = max(nodes_per_graph)
        self.assertEqual(scores.shape, (batch_size, max_nodes))
        self.assertEqual(selected_emb.shape, (batch_size, hidden_dim))

        # Verify masked positions have -inf scores
        probs = torch.softmax(scores, dim=1)
        self.assertAlmostEqual(float(probs[0, 4].item()), 0.0, places=5)

    def test_autoregressive_step_changes_output(self) -> None:
        hidden_dim = 16
        selector = ArgumentSelector(hidden_dim)

        state_emb = torch.randn(1, hidden_dim)
        tactic_emb = torch.randn(1, hidden_dim)
        node_embeddings = torch.randn(5, hidden_dim)
        batch_index = torch.zeros(5, dtype=torch.long)
        premise_mask = torch.ones(5, dtype=torch.bool)

        scores1, sel1 = selector(
            state_emb, tactic_emb, node_embeddings, premise_mask, batch_index
        )
        scores2, sel2 = selector(
            state_emb, tactic_emb, node_embeddings, premise_mask, batch_index,
            prev_arg_emb=sel1,
        )

        # Scores should differ because the query context changed
        self.assertFalse(
            torch.allclose(scores1, scores2),
            "Autoregressive step should produce different scores",
        )


class TacticWithArgsClassifierTests(unittest.TestCase):
    def _build_tiny_batch(self):
        """Build a minimal batched PyG graph for testing."""
        from torch_geometric.data import Batch, Data

        dag1 = proof_state_to_dag("n : Nat\n⊢ Even n")
        dag2 = proof_state_to_dag("m : Nat\n⊢ Even m")

        vocab = build_vocab([dag1, dag2])
        d1 = dag_to_pyg(dag1, vocab, add_reverse_edges=True)
        d2 = dag_to_pyg(dag2, vocab, add_reverse_edges=True)

        # Add required fields
        for data, dag in [(d1, dag1), (d2, dag2)]:
            data.premise_mask = torch.tensor(build_premise_mask(dag), dtype=torch.bool)
            data.y = torch.tensor([1], dtype=torch.long)
            data.tactic_name = "apply"
            data.arg_node_indices = torch.tensor([0], dtype=torch.long)
            data.arg_count = 1

        # Find State node for state_node_index
        state_label_id = vocab.get("State", 0)
        for data in [d1, d2]:
            state_matches = (data.x == state_label_id).nonzero(as_tuple=False).view(-1)
            data.state_node_index = state_matches[-1:]

        batch = Batch.from_data_list([d1, d2])
        return batch, vocab

    def test_forward_returns_both_heads(self) -> None:
        batch, vocab = self._build_tiny_batch()

        model = TacticWithArgsClassifier(
            num_node_labels=len(vocab),
            num_tactics=5,
            hidden_dim=16,
            num_layers=2,
            dropout=0.1,
            max_args=2,
        )

        tactic_logits, arg_logits_list = model(
            batch,
            teacher_tactic_ids=batch.y.view(-1),
            tactic_names=["apply", "apply"],
        )

        self.assertEqual(tactic_logits.shape, (2, 5))
        self.assertGreater(len(arg_logits_list), 0)
        for arg_logits in arg_logits_list:
            self.assertEqual(arg_logits.shape[0], 2)

    def test_zero_arity_returns_empty_arg_list(self) -> None:
        batch, vocab = self._build_tiny_batch()

        model = TacticWithArgsClassifier(
            num_node_labels=len(vocab),
            num_tactics=5,
            hidden_dim=16,
            num_layers=2,
            dropout=0.1,
            max_args=2,
        )

        tactic_logits, arg_logits_list = model(
            batch,
            teacher_tactic_ids=batch.y.view(-1),
            tactic_names=["simp", "simp"],
        )

        self.assertEqual(tactic_logits.shape, (2, 5))
        self.assertEqual(len(arg_logits_list), 0)


class CombinedLossTests(unittest.TestCase):
    def test_masks_invalid_arg_targets(self) -> None:
        batch_size = 2
        num_tactics = 5
        num_nodes = 8

        tactic_logits = torch.randn(batch_size, num_tactics, requires_grad=True)
        arg_logits = [torch.randn(batch_size, num_nodes, requires_grad=True)]
        tactic_targets = torch.tensor([1, 2], dtype=torch.long)
        arg_targets = torch.tensor([[3], [-1]], dtype=torch.long)  # second sample unresolvable
        batch_index = torch.cat([
            torch.zeros(4, dtype=torch.long),
            torch.ones(4, dtype=torch.long),
        ])

        loss, metrics = compute_combined_loss(
            tactic_logits,
            arg_logits,
            tactic_targets,
            arg_targets,
            batch_index,
            tactic_arity_per_sample=[1, 1],
            arg_loss_weight=0.5,
            unknown_tactic_id=0,
        )

        self.assertTrue(torch.isfinite(loss), "Loss should be finite")
        self.assertGreater(float(loss.item()), 0.0, "Loss should be positive")
        self.assertIn("tactic_loss", metrics)
        self.assertIn("arg_loss", metrics)

        # Verify gradients flow
        loss.backward()
        self.assertIsNotNone(tactic_logits.grad)

    def test_zero_arity_skips_arg_loss(self) -> None:
        tactic_logits = torch.randn(2, 5, requires_grad=True)
        tactic_targets = torch.tensor([1, 2], dtype=torch.long)
        batch_index = torch.cat([
            torch.zeros(4, dtype=torch.long),
            torch.ones(4, dtype=torch.long),
        ])

        loss, metrics = compute_combined_loss(
            tactic_logits,
            [],  # no arg logits
            tactic_targets,
            torch.tensor([[-1], [-1]], dtype=torch.long),
            batch_index,
            tactic_arity_per_sample=[0, 0],
            arg_loss_weight=0.5,
            unknown_tactic_id=0,
        )

        self.assertAlmostEqual(metrics["arg_loss"], 0.0)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
