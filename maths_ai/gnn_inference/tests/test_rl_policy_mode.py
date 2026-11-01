"""Live RL must recompute actions with the layer behavior used for collection."""

from unittest import TestCase
from unittest.mock import patch

import torch
from torch_geometric.data import Batch

from maths_ai.gnn_inference.atp_lean_gnn.graph import model_goal_to_dag
from maths_ai.gnn_inference.atp_lean_gnn.model_factory import build_actor_critic_model
from maths_ai.gnn_inference.atp_lean_gnn.model_spec import ModelSpec
from maths_ai.gnn_inference.atp_lean_gnn.pln_rl_training import (
    EdgeAction,
    FailureRecord,
    compute_onpolicy_loss,
    compute_transition_loss,
    make_featurizer,
    train_step,
    train_step_onpolicy,
)
from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_vocab
from maths_ai.gnn_inference.atp_lean_gnn.search_harvest import CriticSample
from maths_ai.gnn_inference.tests.model_helpers import structured_goal


class RLPolicyModeTests(TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.goals = [
            structured_goal("p ∧ q", ["p : Prop", "q : Prop", "hp : p", "hq : q"]),
            structured_goal("n = n", ["n : Nat"]),
            structured_goal("True", []),
        ]
        vocab = build_vocab([model_goal_to_dag(goal) for goal in self.goals])
        self.featurize = make_featurizer(vocab)
        self.model = build_actor_critic_model(
            model_spec=ModelSpec.from_dict(
                {
                    "architecture": "gatv2",
                    "hidden_dim": 16,
                    "dropout": 0.4,
                    "encoder": {"num_layers": 2, "heads": 2, "readout": "state"},
                    "max_args": 2,
                }
            ),
            num_node_labels=len(vocab),
            num_tactics=3,
        )

    def test_collection_probabilities_match_repeated_batched_loss_recomputation(self):
        model = self.model
        model.eval()
        samples = []
        failures = []
        with torch.no_grad():
            for count, goal in zip((2, 1, 0), self.goals):
                # Exercise variable-length recorded argument sequences.
                model.max_args = max(count, 1)
                sample = model.act(
                    Batch.from_data_list([self.featurize(goal)]),
                    id_to_tactic=(
                        {i: "trivial" for i in range(3)} if count == 0 else None
                    ),
                )
                self.assertEqual(len(sample.arg_actions), count)
                for tensor in (sample.tactic_logp, sample.arg_logp, sample.value):
                    self.assertFalse(tensor.requires_grad)
                samples.append(sample)
                failures.append(
                    FailureRecord(
                        goal=goal,
                        action=EdgeAction(
                            tactic_id=int(sample.tactic_action[0]),
                            arg_indices=tuple(
                                int(index[0]) for index in sample.arg_actions
                            ),
                        ),
                    )
                )
        model.max_args = 2
        expected_tactic = torch.cat([sample.tactic_logp for sample in samples])
        expected_args = torch.cat([sample.arg_logp for sample in samples])
        original = model.evaluate_actions
        recomputed = []

        def capture(*args, **kwargs):
            self.assertTrue(torch.is_grad_enabled())
            self.assertTrue(all(not module.training for module in model.modules()))
            outputs = original(*args, **kwargs)
            self.assertTrue(outputs[0].requires_grad)
            self.assertTrue(outputs[1].requires_grad)
            recomputed.append(tuple(output.detach().clone() for output in outputs[:2]))
            return outputs

        with patch.object(model, "evaluate_actions", side_effect=capture):
            for _ in range(2):
                model.train()
                with torch.no_grad():
                    loss, _metrics = compute_onpolicy_loss(
                        model,
                        [],
                        [],
                        {},
                        failures,
                        self.featurize,
                    )
                    self.assertFalse(torch.is_grad_enabled())
                self.assertTrue(loss.requires_grad)
        for tactic_logp, arg_logp in recomputed:
            torch.testing.assert_close(
                tactic_logp, expected_tactic, rtol=1e-5, atol=1e-6
            )
            torch.testing.assert_close(arg_logp, expected_args, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(recomputed[0][0], recomputed[1][0], rtol=0, atol=0)
        torch.testing.assert_close(recomputed[0][1], recomputed[1][1], rtol=0, atol=0)
        loss.backward()
        pointer_gradient = model.argument_selector.query_proj.weight.grad
        self.assertIsNotNone(pointer_gradient)
        self.assertTrue(torch.isfinite(pointer_gradient).all())
        self.assertGreater(float(pointer_gradient.abs().sum()), 0)

    def test_critic_only_loss_builders_record_gradients_without_dropout(self):
        sample = CriticSample(node_id=0, goal=self.goals[0], target=0.75)
        for name in ("transition", "onpolicy"):
            with self.subTest(builder=name):
                self.model.train()
                self.model.zero_grad(set_to_none=True)
                with torch.no_grad():
                    if name == "transition":
                        result = compute_transition_loss(
                            self.model,
                            [],
                            [sample],
                            self.featurize,
                            {},
                        )
                    else:
                        result = compute_onpolicy_loss(
                            self.model,
                            [],
                            [sample],
                            {},
                            [],
                            self.featurize,
                        )
                loss, metrics = result
                self.assertTrue(
                    all(not module.training for module in self.model.modules())
                )
                self.assertEqual(metrics["actor_loss"], 0)
                self.assertTrue(loss.requires_grad)
                loss.backward()
                gradient = self.model.critic.net[-1].weight.grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(gradient.abs().sum()), 0)

    def test_empty_updates_do_not_step_or_leave_training_mode(self):
        optimizer = torch.optim.AdamW(self.model.parameters())
        before = [value.detach().clone() for value in self.model.parameters()]
        for update in (train_step, train_step_onpolicy):
            with self.subTest(update=update.__name__):
                self.model.train()
                with torch.no_grad():
                    if update is train_step:
                        metrics = update(self.model, optimizer, [], self.featurize, {})
                    else:
                        metrics = update(self.model, optimizer, [], self.featurize)
                self.assertEqual(metrics["optimizer_step"], 0)
                self.assertTrue(
                    all(not module.training for module in self.model.modules())
                )
                self.assertFalse(optimizer.state)
                for previous, current in zip(before, self.model.parameters()):
                    torch.testing.assert_close(previous, current, rtol=0, atol=0)
