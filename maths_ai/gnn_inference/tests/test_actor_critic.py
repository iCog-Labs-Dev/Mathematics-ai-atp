from __future__ import annotations
from maths_ai.gnn_inference.atp_lean_gnn.policy_actions import local_pools
from maths_ai.gnn_inference.atp_lean_gnn.actor_critic_loss import compute_complete_policy_loss

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
    compute_bc_anchor_loss,
)
from maths_ai.gnn_inference.atp_lean_gnn.actor_critic_training import build_param_groups
from maths_ai.gnn_inference.atp_lean_gnn.checkpointing import checkpoint_payload
from maths_ai.gnn_inference.atp_lean_gnn.reward import MockRewardSource
from maths_ai.gnn_inference.atp_lean_gnn.argument_selector import TacticWithArgsClassifier
from maths_ai.gnn_inference.tests.model_helpers import actor_critic, pointer
from maths_ai.gnn_inference.atp_lean_gnn.graph import proof_state_to_dag
from maths_ai.gnn_inference.atp_lean_gnn.graph_contract import MODEL_SEXPR_GRAPH_SPEC


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

    def test_full_model_forward(self):
        batch,vocab=self._build_tiny_batch()
        model=actor_critic(len(vocab),5)
        result=model(batch,candidate_pools=local_pools(batch))
        self.assertEqual(result.tactic_logits.shape,(2,5))
        self.assertEqual(result.value.shape,(2,))
        self.assertTrue(all(len(args)<=model.max_args for args in result.arguments))

    def test_argument_selector_unchanged(self):
        batch,vocab=self._build_tiny_batch()
        ptr=pointer(len(vocab),5); model=actor_critic(len(vocab),5)
        model.encoder.load_state_dict(ptr.encoder.state_dict())
        model.decoder.load_state_dict(ptr.decoder.state_dict())
        pools=local_pools(batch)
        first=ptr.act(batch,candidate_pools=pools,greedy=True)
        second=model.evaluate_actions(batch,first.tactic_action,first.arguments,candidate_pools=pools)
        torch.testing.assert_close(first.arg_logp,second.arg_logp)

    def test_combined_loss_components(self):
        batch,vocab=self._build_tiny_batch()
        model=actor_critic(len(vocab),5)
        evaluated=model.act(batch,candidate_pools=local_pools(batch))
        loss,metrics=compute_complete_policy_loss(evaluated,torch.tensor([1.,0.]))
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue({'actor_loss','critic_loss','arg_loss','entropy'}<=metrics.keys())

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

    def test_advantage_normalization_zero_variance_is_finite(self):
        from types import SimpleNamespace
        logits=torch.randn(2,5,requires_grad=True)
        logp=logits.log_softmax(-1)[:,0]
        evaluated=SimpleNamespace(logp=logp,arg_logp=logp,value=torch.full((2,),.5),tactic_logits=logits)
        loss,metrics=compute_complete_policy_loss(evaluated,torch.full((2,),.5))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics['actor_loss'],0.)

    def test_advantage_normalization_singleton_batch_is_finite(self):
        from types import SimpleNamespace
        logits=torch.randn(1,5,requires_grad=True)
        logp=logits.log_softmax(-1)[:,0]
        evaluated=SimpleNamespace(logp=logp,arg_logp=logp,value=torch.full((1,),.5),tactic_logits=logits)
        loss,metrics=compute_complete_policy_loss(evaluated,torch.zeros(1))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics['actor_loss'],0.)

    def test_gradient_flow_and_advantage_detached(self):
        batch,vocab=self._build_tiny_batch()
        model=actor_critic(len(vocab),5)
        result=model.act(batch,candidate_pools=local_pools(batch))
        actor=-(result.logp*torch.tensor([1.,-1.])).mean()
        grads=torch.autograd.grad(actor,tuple(model.critic.parameters()),allow_unused=True,retain_graph=True)
        self.assertTrue(all(x is None for x in grads))
        loss,_=compute_complete_policy_loss(result,torch.tensor([1.,0.]))
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.decoder.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.critic.parameters()))

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
                    graph_representation=MODEL_SEXPR_GRAPH_SPEC,
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
                graph_representation=MODEL_SEXPR_GRAPH_SPEC,
            )

            self.assertTrue(torch.allclose(
                model_ac.encoder.node_features.label_embedding.weight,
                model_ptr.encoder.node_features.label_embedding.weight,
            ))
            self.assertTrue(torch.allclose(model_ac.decoder.tactic_embedding.weight, model_ptr.decoder.tactic_embedding.weight))
            self.assertTrue(torch.allclose(model_ac.decoder.argument_selector.init_proj.weight, model_ptr.decoder.argument_selector.init_proj.weight))
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
                    graph_representation=MODEL_SEXPR_GRAPH_SPEC,
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
                    graph_representation=MODEL_SEXPR_GRAPH_SPEC,
                )

    def test_mock_reward_source(self) -> None:
        reward_src = MockRewardSource()
        actions = torch.tensor([1, 2, 3])
        targets = torch.tensor([1, 4, 3])
        success = torch.tensor([True, False, True])
        rewards = reward_src.get_rewards_batch(actions, targets, success)

        self.assertTrue(torch.allclose(rewards, torch.tensor([1.0, 0.0, 1.0])))

    def test_training_step_e2e(self):
        batch,vocab=self._build_tiny_batch()
        model=actor_critic(len(vocab),5)
        optimizer=AdamW(model.parameters(),lr=.001)
        before=[p.detach().clone() for p in model.parameters()]
        result=model.act(batch,candidate_pools=local_pools(batch))
        loss,_=compute_complete_policy_loss(result,torch.tensor([1.,0.]))
        loss.backward();optimizer.step()
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,model.parameters())))


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

    def test_bc_weight_zero_reproduces_pure_rl(self):
        from types import SimpleNamespace
        logits=torch.randn(3,5,requires_grad=True)
        logp=logits.log_softmax(-1)[:,0]
        evaluated=SimpleNamespace(logp=logp,arg_logp=logp,value=torch.zeros(3),tactic_logits=logits)
        returns=torch.tensor([1.,0.,1.]);labels=torch.tensor([0,1,2])
        a,_=compute_complete_policy_loss(evaluated,returns)
        b,metrics=compute_complete_policy_loss(evaluated,returns,labels=labels,bc_weight=0.)
        torch.testing.assert_close(a,b)
        c,metrics=compute_complete_policy_loss(evaluated,returns,labels=labels,bc_weight=1.)
        self.assertGreater(metrics['bc_loss'],0.)
        self.assertGreater(float(c.detach()),float(b.detach()))

    def test_act_sampling_shapes_and_gradients(self):
        batch,vocab=self._build_tiny_batch()
        model=actor_critic(len(vocab),5)
        sample=model.act(batch,candidate_pools=local_pools(batch))
        self.assertIsInstance(sample,ActionSample)
        self.assertEqual(sample.logp.shape,(2,))
        self.assertEqual(len(sample.arguments),2)
        sample.arg_logp.sum().backward()
        self.assertTrue(model.decoder.arg_count_head.weight.grad.abs().sum()>0)

    def test_act_greedy_is_deterministic(self):
        batch,vocab=self._build_tiny_batch()
        model=actor_critic(len(vocab),5)
        pools=local_pools(batch)
        a=model.act(batch,candidate_pools=pools,greedy=True)
        b=model.act(batch,candidate_pools=pools,greedy=True)
        self.assertTrue(torch.equal(a.tactic_action,b.tactic_action))
        self.assertEqual(a.arguments,b.arguments)


if __name__ == "__main__":
    unittest.main()
