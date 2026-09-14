from __future__ import annotations

import unittest
import torch
import torch.nn as nn
from torch.optim import AdamW

from maths_ai.gnn_inference.atp_lean_gnn import (
    build_premise_mask,
    build_vocab,
)
from maths_ai.gnn_inference.atp_lean_gnn.pyg import dag_to_pyg
from maths_ai.gnn_inference.atp_lean_gnn.actor_critic import (
    ActorHead,
    CriticHead,
    ActorCriticWithArgsClassifier,
    ActionSample,
    init_actor_from_supervised,
    load_from_pointer_checkpoint,
)
from maths_ai.gnn_inference.atp_lean_gnn.actor_critic_loss import (
    compute_actor_loss,
    compute_critic_loss,
    compute_entropy_bonus,
    compute_argument_rl_loss,
    compute_bc_anchor_loss,
    compute_actor_critic_combined_loss,
)
from maths_ai.gnn_inference.atp_lean_gnn.actor_critic_training import build_param_groups
from maths_ai.gnn_inference.atp_lean_gnn.checkpointing import checkpoint_payload
from maths_ai.gnn_inference.atp_lean_gnn.reward import MockRewardSource
from maths_ai.gnn_inference.atp_lean_gnn.argument_selector import TacticWithArgsClassifier
from maths_ai.gnn_inference.tests.model_helpers import actor_critic, pointer
from maths_ai.gnn_inference.atp_lean_gnn.graph import proof_state_to_dag


class ActorCriticTests(unittest.TestCase):
    def _build_tiny_batch(self):
        from torch_geometric.data import Batch
        dag1 = proof_state_to_dag("n : Nat\n⊢ Even n")
        dag2 = proof_state_to_dag("m : Nat\n⊢ Even m")

        vocab = build_vocab([dag1, dag2])
        d1 = dag_to_pyg(dag1, vocab, add_reverse_edges=True)
        d2 = dag_to_pyg(dag2, vocab, add_reverse_edges=True)

        for data, dag in [(d1, dag1), (d2, dag2)]:
            data.premise_mask = torch.tensor(build_premise_mask(dag), dtype=torch.bool)
            data.y = torch.tensor([1], dtype=torch.long)
            data.tactic_name = "apply"
            data.arg_node_indices = torch.tensor([0], dtype=torch.long)
            data.arg_count = 1

        state_label_id = vocab.get("State", 0)
        for data in [d1, d2]:
            state_matches = (data.x == state_label_id).nonzero(as_tuple=False).view(-1)
            data.state_node_index = state_matches[-1:]

        batch = Batch.from_data_list([d1, d2])
        return batch, vocab

    def test_actor_head_output_shape(self) -> None:
        hidden_dim = 16
        num_tactics = 5
        head = ActorHead(hidden_dim, num_tactics)
        state_emb = torch.randn(4, hidden_dim)
        logits = head(state_emb)
        self.assertEqual(logits.shape, (4, num_tactics))

    def test_critic_head_output_shape(self) -> None:
        hidden_dim = 16
        head = CriticHead(hidden_dim)
        state_emb = torch.randn(4, hidden_dim)
        values = head(state_emb)
        self.assertEqual(values.shape, (4, 1))

    def test_action_masking(self) -> None:
        hidden_dim = 16
        num_tactics = 5
        head = ActorHead(hidden_dim, num_tactics)
        state_emb = torch.randn(2, hidden_dim)
        mask = torch.tensor([[True, False, True, False, True],
                             [False, True, False, True, False]], dtype=torch.bool)
        logits = head(state_emb, mask=mask)
        self.assertEqual(logits.shape, (2, num_tactics))
        self.assertTrue(torch.isneginf(logits[0, 1]))
        self.assertTrue(torch.isneginf(logits[0, 3]))
        self.assertFalse(torch.isneginf(logits[0, 0]))
        self.assertTrue(torch.isneginf(logits[1, 0]))
        self.assertTrue(torch.isneginf(logits[1, 2]))
        self.assertFalse(torch.isneginf(logits[1, 1]))

    def test_full_model_forward(self) -> None:
        batch, vocab = self._build_tiny_batch()
        model = actor_critic(len(vocab), 5)
        tactic_logits, values, arg_logits_list = model(
            batch,
            tactic_names=["apply", "apply"],
        )
        self.assertEqual(tactic_logits.shape, (2, 5))
        self.assertEqual(values.shape, (2, 1))
        self.assertGreater(len(arg_logits_list), 0)
        for arg_logits in arg_logits_list:
            self.assertEqual(arg_logits.shape[0], 2)

    def test_argument_selector_unchanged(self) -> None:
        batch, vocab = self._build_tiny_batch()
        # Create pointer classifier and actor-critic classifier with same seed/weights
        torch.manual_seed(42)
        model_ptr = pointer(len(vocab), 5)
        torch.manual_seed(42)
        model_ac = actor_critic(len(vocab), 5)

        # Synchronize argument selector and embedding parameters
        model_ac.tactic_embedding.weight.data.copy_(model_ptr.tactic_embedding.weight.data)
        model_ac.argument_selector.load_state_dict(model_ptr.argument_selector.state_dict())
        model_ac.encoder.load_state_dict(model_ptr.encoder.state_dict())

        model_ptr.eval()
        model_ac.eval()

        with torch.no_grad():
            t_logits_ptr, args_ptr = model_ptr(batch, teacher_tactic_ids=batch.y.view(-1), tactic_names=["apply", "apply"])
            t_logits_ac, _, args_ac = model_ac(batch, teacher_tactic_ids=batch.y.view(-1), tactic_names=["apply", "apply"])

        self.assertEqual(len(args_ptr), len(args_ac))
        for a_ptr, a_ac in zip(args_ptr, args_ac):
            self.assertTrue(torch.allclose(a_ptr, a_ac))

    def test_combined_loss_components(self) -> None:
        tactic_logits = torch.randn(2, 5)
        value_estimates = torch.randn(2, 1)
        arg_logits_list = [torch.randn(2, 10)]
        actions = torch.tensor([1, 2], dtype=torch.long)
        returns = torch.tensor([1.0, 0.0], dtype=torch.float32)
        selected_arg_indices = [torch.tensor([3, 4], dtype=torch.long)]
        success_mask = torch.tensor([True, False], dtype=torch.bool)

        total, metrics = compute_actor_critic_combined_loss(
            tactic_logits=tactic_logits,
            value_estimates=value_estimates,
            arg_logits_list=arg_logits_list,
            actions=actions,
            returns=returns,
            selected_arg_indices=selected_arg_indices,
            success_mask=success_mask,
        )

        self.assertIn("actor_loss", metrics)
        self.assertIn("critic_loss", metrics)
        self.assertIn("entropy", metrics)
        self.assertIn("arg_loss", metrics)
        self.assertIn("total_loss", metrics)
        self.assertTrue(torch.isfinite(total))

    def test_masking_zero_sampling_prob_and_entropy(self) -> None:
        # A masked tactic must have -inf logit, exactly zero sampling probability, and
        # must be ignored by the entropy computation.
        hidden_dim = 16
        num_tactics = 5
        head = ActorHead(hidden_dim, num_tactics)
        state_emb = torch.randn(3, hidden_dim)
        mask = torch.ones(3, num_tactics, dtype=torch.bool)
        mask[:, 1] = False  # tactic 1 illegal for every example
        mask[:, 3] = False  # tactic 3 illegal for every example

        logits = head(state_emb, mask=mask)
        probs = torch.softmax(logits, dim=-1)
        self.assertTrue(torch.all(probs[:, 1] == 0.0))
        self.assertTrue(torch.all(probs[:, 3] == 0.0))
        # Legal probabilities still sum to 1.
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(3)))

        # Entropy over the masked distribution equals entropy over only the legal classes.
        ent_masked = compute_entropy_bonus(logits, mask=mask)
        legal_logits = logits[:, [0, 2, 4]]
        legal_probs = torch.softmax(legal_logits, dim=-1)
        ent_direct = -(legal_probs * torch.log(legal_probs)).sum(dim=-1).mean()
        self.assertTrue(torch.allclose(ent_masked, ent_direct, atol=1e-5))

    def test_advantage_normalization_zero_variance_is_finite(self) -> None:
        # Equal returns and equal values give a zero-variance advantage; the +1e-8
        # guard must keep the loss finite (no division by zero).
        tactic_logits = torch.randn(4, 5)
        value_estimates = torch.full((4, 1), 0.5)
        returns = torch.full((4,), 0.5)  # advantage is identically zero
        actions = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        total, metrics = compute_actor_critic_combined_loss(
            tactic_logits=tactic_logits,
            value_estimates=value_estimates,
            arg_logits_list=[],
            actions=actions,
            returns=returns,
            selected_arg_indices=[],
            success_mask=torch.zeros(4, dtype=torch.bool),
        )
        self.assertTrue(torch.isfinite(total))
        # mean_advantage reports the RAW (pre-normalization) advantage, here ~0.
        self.assertAlmostEqual(metrics["mean_advantage"], 0.0, places=5)

    def test_gradient_flow_and_advantage_detached(self) -> None:
        batch, vocab = self._build_tiny_batch()
        model = actor_critic(len(vocab), 5)

        # Use "have" (arity 2) to ensure both query_proj and query_proj_ar are used and receive gradients
        tactic_logits, values, arg_logits_list = model(batch, tactic_names=["have", "have"])
        actions = torch.tensor([1, 2], dtype=torch.long)
        returns = torch.tensor([1.0, 0.0], dtype=torch.float32)
        selected_arg_indices = [
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([0, 1], dtype=torch.long)
        ]
        success_mask = torch.tensor([True, False], dtype=torch.bool)

        # The residual's zero-initialized output layer means its inner layer receives
        # no gradient on the FIRST backward (grad flows through a zero weight). Take one
        # optimizer step to move the residual output off zero, then verify gradient
        # reaches every trainable parameter on the next backward.
        optimizer = AdamW(model.parameters(), lr=0.01)
        for _step in range(2):
            optimizer.zero_grad()
            tactic_logits, values, arg_logits_list = model(batch, tactic_names=["have", "have"])
            total, metrics = compute_actor_critic_combined_loss(
                tactic_logits=tactic_logits,
                value_estimates=values,
                arg_logits_list=arg_logits_list,
                actions=actions,
                returns=returns,
                selected_arg_indices=selected_arg_indices,
                success_mask=success_mask,
            )
            total.backward()
            optimizer.step()

        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, f"Parameter {name} has None gradient")
            self.assertFalse(torch.all(p.grad == 0), f"Parameter {name} has zero gradient")

        # Check advantage detachment:
        # Critic values should not receive gradients from the actor loss alone.
        model.zero_grad()
        tactic_logits, values, _ = model(batch)
        advantages = returns - values.squeeze(-1)
        actor_loss = compute_actor_loss(tactic_logits, actions, advantages)
        actor_loss.backward()

        for p in model.critic.parameters():
            if p.grad is not None:
                self.assertTrue(torch.all(p.grad == 0), "Critic received gradients from actor loss alone")

    def test_differential_lr_param_groups(self) -> None:
        model = actor_critic(10, 5)
        base_lr = 0.001
        arg_lr_multiplier = 0.1
        groups = build_param_groups(model, base_lr, arg_lr_multiplier)

        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["lr"], base_lr)
        self.assertEqual(groups[1]["lr"], base_lr * arg_lr_multiplier)

        g0_params = set(groups[0]["params"])
        g1_params = set(groups[1]["params"])
        self.assertEqual(len(g0_params.intersection(g1_params)), 0)
        self.assertEqual(len(g0_params) + len(g1_params), len(list(model.parameters())))

    def test_init_actor_from_supervised(self) -> None:
        model = actor_critic(10, 5)
        classifier = nn.Linear(16, 5)
        classifier.weight.data.normal_()
        classifier.bias.data.normal_()

        init_actor_from_supervised(model, classifier)

        # The inherited ``base`` layer receives the supervised classifier weights.
        self.assertTrue(torch.allclose(model.actor.base.weight, classifier.weight))
        self.assertTrue(torch.allclose(model.actor.base.bias, classifier.bias))
        # The residual branch's output layer is zero, so the branch is silent at init.
        self.assertTrue(torch.all(model.actor.residual[3].weight == 0))
        self.assertTrue(torch.all(model.actor.residual[3].bias == 0))

    def test_warmstart_behavioral_equivalence(self) -> None:
        # After warm-start, the actor must reproduce the supervised classifier exactly
        # (residual == 0), not merely share the final-layer weights.
        model = actor_critic(10, 5)
        classifier = nn.Linear(16, 5)
        classifier.weight.data.normal_()
        classifier.bias.data.normal_()
        init_actor_from_supervised(model, classifier)

        model.eval()  # disable dropout so the comparison is deterministic
        state_emb = torch.randn(8, 16)
        with torch.no_grad():
            actor_logits = model.actor(state_emb)
            classifier_logits = classifier(state_emb)
        self.assertTrue(torch.allclose(actor_logits, classifier_logits, atol=1e-6))

    def test_load_from_pointer_checkpoint(self) -> None:
        batch, vocab = self._build_tiny_batch()
        model_ptr = pointer(len(vocab), 5)
        tactic_vocab = {f"tactic_{index}": index for index in range(5)}
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "ptr_best.pt")
            torch.save(
                checkpoint_payload(
                    model_kind="tactic_with_args",
                    model_spec=model_ptr.model_spec,
                    node_vocab=vocab,
                    tactic_vocab=tactic_vocab,
                    model=model_ptr,
                ),
                ckpt_path,
            )

            model_ac = actor_critic(len(vocab), 5)

            load_from_pointer_checkpoint(
                model_ac,
                ckpt_path,
                torch.device("cpu"),
                node_vocab=vocab,
                tactic_vocab=tactic_vocab,
            )

            self.assertTrue(torch.allclose(
                model_ac.encoder.node_features.label_embedding.weight,
                model_ptr.encoder.node_features.label_embedding.weight,
            ))
            self.assertTrue(torch.allclose(model_ac.tactic_embedding.weight, model_ptr.tactic_embedding.weight))
            self.assertTrue(torch.allclose(model_ac.argument_selector.query_proj.weight, model_ptr.argument_selector.query_proj.weight))
            self.assertTrue(torch.allclose(
                model_ac.actor.base.weight,
                model_ptr.tactic_classifier.weight,
            ))

    def test_load_from_pointer_checkpoint_shape_mismatch_raises(self) -> None:
        batch, vocab = self._build_tiny_batch()
        model_ptr = pointer(len(vocab), 5)
        tactic_vocab = {f"tactic_{index}": index for index in range(5)}
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "ptr_best.pt")
            torch.save(
                checkpoint_payload(
                    model_kind="tactic_with_args",
                    model_spec=model_ptr.model_spec,
                    node_vocab=vocab,
                    tactic_vocab=tactic_vocab,
                    model=model_ptr,
                ),
                ckpt_path,
            )

            # Model built at a different hidden_dim than the checkpoint.
            model_ac = actor_critic(len(vocab), 5, hidden_dim=32)
            with self.assertRaises(ValueError):
                load_from_pointer_checkpoint(
                    model_ac,
                    ckpt_path,
                    torch.device("cpu"),
                    node_vocab=vocab,
                    tactic_vocab=tactic_vocab,
                )

    def test_mock_reward_source(self) -> None:
        reward_src = MockRewardSource()
        actions = torch.tensor([1, 2, 3])
        targets = torch.tensor([1, 4, 3])
        success = torch.tensor([True, False, True])
        rewards = reward_src.get_rewards_batch(actions, targets, success)

        self.assertTrue(torch.allclose(rewards, torch.tensor([1.0, 0.0, 1.0])))

    def test_training_step_e2e(self) -> None:
        batch, vocab = self._build_tiny_batch()
        model = actor_critic(len(vocab), 5)
        optimizer = AdamW(model.parameters(), lr=0.001)
        reward_src = MockRewardSource()

        tactic_logits, values, _ = model(batch)
        tactic_dist = torch.distributions.Categorical(logits=tactic_logits)
        actions = tactic_dist.sample()

        _, _, arg_logits_list = model(batch, teacher_tactic_ids=actions)

        selected_arg_indices = [arg_logits.argmax(dim=1) for arg_logits in arg_logits_list]
        success_mask = (actions == batch.y.view(-1))
        returns = reward_src.get_rewards_batch(actions, batch.y.view(-1), success_mask)

        optimizer.zero_grad()
        loss, metrics = compute_actor_critic_combined_loss(
            tactic_logits=tactic_logits,
            value_estimates=values,
            arg_logits_list=arg_logits_list,
            actions=actions,
            returns=returns,
            selected_arg_indices=selected_arg_indices,
            success_mask=success_mask,
        )

        loss.backward()
        optimizer.step()

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(metrics["total_loss"], -1e9)


    def test_bc_anchor_loss(self) -> None:
        # BC anchor is supervised cross-entropy toward the label; -1 labels are ignored.
        tactic_logits = torch.randn(4, 5, requires_grad=True)
        labels = torch.tensor([1, 2, -1, 0], dtype=torch.long)
        loss = compute_bc_anchor_loss(tactic_logits, labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        # Gradient flows (three labeled examples contribute).
        self.assertFalse(torch.all(tactic_logits.grad == 0))

        # All-unlabeled → zero loss, zero gradient, still finite.
        tl2 = torch.randn(3, 5, requires_grad=True)
        loss2 = compute_bc_anchor_loss(tl2, torch.tensor([-1, -1, -1]))
        self.assertEqual(float(loss2.item()), 0.0)

    def test_bc_weight_zero_reproduces_pure_rl(self) -> None:
        # bc_weight=0 must give exactly the pure-RL total (BC term contributes nothing).
        torch.manual_seed(0)
        args = dict(
            tactic_logits=torch.randn(3, 5),
            value_estimates=torch.randn(3, 1),
            arg_logits_list=[torch.randn(3, 8)],
            actions=torch.tensor([0, 1, 2]),
            returns=torch.tensor([1.0, 0.0, 1.0]),
            selected_arg_indices=[torch.tensor([0, 1, 2])],
            success_mask=torch.tensor([True, False, True]),
        )
        total_pure, _ = compute_actor_critic_combined_loss(**args)
        labels = torch.tensor([0, 1, 2], dtype=torch.long)
        total_bc0, m = compute_actor_critic_combined_loss(**args, labels=labels, bc_weight=0.0)
        self.assertAlmostEqual(float(total_pure.item()), float(total_bc0.item()), places=6)
        self.assertEqual(m["bc_loss"], 0.0)

        # A positive bc_weight changes the total and records a nonzero bc_loss.
        total_bc1, m1 = compute_actor_critic_combined_loss(**args, labels=labels, bc_weight=1.0)
        self.assertNotAlmostEqual(float(total_pure.item()), float(total_bc1.item()), places=4)
        self.assertGreater(m1["bc_loss"], 0.0)

    def test_act_sampling_shapes_and_gradients(self) -> None:
        batch, vocab = self._build_tiny_batch()
        model = actor_critic(len(vocab), 5)
        sample = model.act(batch)
        self.assertIsInstance(sample, ActionSample)
        self.assertEqual(sample.tactic_action.shape, (2,))
        self.assertEqual(sample.tactic_logp.shape, (2,))
        self.assertEqual(sample.value.shape, (2,))
        self.assertEqual(sample.arg_logp.shape, (2,))
        self.assertEqual(len(sample.arg_actions), 2)  # max_args steps

        # The pointer-as-actor policy gradient: arg_logp must flow into the argument selector.
        model.zero_grad()
        sample.arg_logp.sum().backward()
        arg_grads = [p.grad for p in model.argument_selector.parameters() if p.grad is not None]
        self.assertTrue(len(arg_grads) > 0)
        self.assertTrue(any(torch.any(g != 0) for g in arg_grads))

    def test_act_greedy_is_deterministic(self) -> None:
        batch, vocab = self._build_tiny_batch()
        model = actor_critic(len(vocab), 5)
        model.eval()
        with torch.no_grad():
            a1 = model.act(batch, greedy=True)
            a2 = model.act(batch, greedy=True)
        self.assertTrue(torch.equal(a1.tactic_action, a2.tactic_action))


if __name__ == "__main__":
    unittest.main()
