from __future__ import annotations

import unittest

import torch
from torch_geometric.data import Batch, Data

from maths_ai.gnn_inference.atp_lean_gnn.architectures import (
    amp_dtype_for_architecture,
    build_encoder,
)
from maths_ai.gnn_inference.atp_lean_gnn.checkpointing import (
    build_model_from_checkpoint,
    checkpoint_payload,
)
from maths_ai.gnn_inference.atp_lean_gnn.batching import GraphBudgetBatchSampler, GraphSize
from maths_ai.gnn_inference.atp_lean_gnn.model_factory import (
    build_actor_critic_model,
    build_pointer_model,
    build_supervised_tactic_model,
)
from maths_ai.gnn_inference.atp_lean_gnn.model_spec import ModelSpec
from maths_ai.gnn_inference.atp_lean_gnn.training_safety import require_finite_loss
from maths_ai.gnn_inference.scripts.migrate_model_checkpoint import migrate_checkpoint


def _batch() -> Batch:
    graphs = []
    for offset in (0, 1):
        data = Data(
            x=torch.tensor([0, 1, 2], dtype=torch.long),
            node_type=torch.tensor([0, 1, 2], dtype=torch.long),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
            is_bound=torch.tensor([0, 1, 1], dtype=torch.long),
            binder_depth=torch.tensor([0, 9 + offset, 100 + offset], dtype=torch.long),
            binder_kind=torch.tensor([0, 1, 2], dtype=torch.long),
            state_node_index=torch.tensor([2], dtype=torch.long),
            premise_mask=torch.tensor([True, True, True]),
            y=torch.tensor([1], dtype=torch.long),
        )
        graphs.append(data)
    return Batch.from_data_list(graphs)


def _spec(architecture: str, *, readout: str = "state") -> ModelSpec:
    encoder = {"num_layers": 2}
    if architecture == "gatv2":
        encoder.update({"heads": 4, "readout": readout})
    return ModelSpec.from_dict(
        {
            "architecture": architecture,
            "hidden_dim": 16,
            "dropout": 0.1,
            "encoder": encoder,
            "use_node_type": True,
            "max_args": 2,
        }
    )


class ArchitectureContractTests(unittest.TestCase):
    def test_each_encoder_returns_contract_shapes(self) -> None:
        batch = _batch()
        for architecture in ("graphsage", "gatv2"):
            encoder = build_encoder(
                architecture=architecture,
                encoder_config=_spec(architecture).encoder,
                num_node_labels=3,
                hidden_dim=16,
                dropout=0.1,
                use_node_type=True,
            )
            output = encoder(batch)
            self.assertEqual(output.node_embeddings.shape, (6, 16))
            self.assertEqual(output.state_embeddings.shape, (2, 16))

    def test_all_gatv2_readouts_return_attention_details(self) -> None:
        batch = _batch()
        for readout in (
            "state_mean_attention",
            "state_max_attention",
            "state_mean_max_attention",
        ):
            spec = _spec("gatv2", readout=readout)
            encoder = build_encoder(
                architecture=spec.architecture,
                encoder_config=spec.encoder,
                num_node_labels=3,
                hidden_dim=spec.hidden_dim,
                dropout=spec.dropout,
                use_node_type=spec.use_node_type,
            )
            output = encoder(batch)
            self.assertEqual(output.state_embeddings.shape, (2, 16))
            weights = output.details["attention_weights"]
            self.assertEqual(weights.shape, (6,))
            for graph_id in range(2):
                self.assertTrue(
                    torch.allclose(
                        weights[batch.batch == graph_id].sum(),
                        torch.tensor(1.0),
                        atol=1e-6,
                    )
                )

    def test_binder_depths_are_bucketed_for_both_encoders(self) -> None:
        batch = _batch()
        for architecture in ("graphsage", "gatv2"):
            spec = _spec(architecture)
            encoder = build_encoder(
                architecture=architecture,
                encoder_config=spec.encoder,
                num_node_labels=3,
                hidden_dim=16,
                dropout=0.1,
                use_node_type=True,
                max_binder_depth=10,
            )
            output = encoder(batch)
            self.assertTrue(torch.isfinite(output.node_embeddings).all())

    def test_model_spec_rejects_imprecise_architecture_names(self) -> None:
        payload = _spec("graphsage").to_dict()
        payload["architecture"] = "sage"
        with self.assertRaisesRegex(ValueError, "Unknown model architecture"):
            ModelSpec.from_dict(payload)

    def test_architecture_precision_policy(self) -> None:
        self.assertIsNone(
            amp_dtype_for_architecture("gatv2", torch.device("cpu"))
        )
        self.assertEqual(
            amp_dtype_for_architecture("graphsage", torch.device("cuda")),
            torch.float16,
        )

    def test_non_finite_loss_fails_with_architecture_and_components(self) -> None:
        with self.assertRaisesRegex(
            FloatingPointError,
            "architecture=gatv2.*precision=bfloat16.*tactic_loss=nan",
        ):
            require_finite_loss(
                torch.tensor(float("nan")),
                architecture="gatv2",
                amp_dtype=torch.bfloat16,
                epoch=3,
                batch_index=7,
                components={"tactic_loss": float("nan")},
            )


class ModelCompositionTests(unittest.TestCase):
    def test_each_model_kind_accepts_each_encoder(self) -> None:
        batch = _batch()
        for architecture in ("graphsage", "gatv2"):
            spec = _spec(architecture)
            baseline = build_supervised_tactic_model(
                model_spec=spec,
                num_node_labels=3,
                num_tactics=5,
            )
            pointer = build_pointer_model(
                model_spec=spec,
                num_node_labels=3,
                num_tactics=5,
            )
            actor_critic = build_actor_critic_model(
                model_spec=spec,
                num_node_labels=3,
                num_tactics=5,
            )
            self.assertEqual(baseline(batch).shape, (2, 5))
            self.assertEqual(pointer(batch)[0].shape, (2, 5))
            self.assertEqual(actor_critic(batch)[0].shape, (2, 5))

    def test_checkpoint_reconstructs_gatv2_without_external_architecture(self) -> None:
        spec = _spec("gatv2", readout="state_mean_max_attention")
        model = build_pointer_model(
            model_spec=spec,
            num_node_labels=3,
            num_tactics=5,
        )
        node_vocab = {"a": 0, "b": 1, "c": 2}
        tactic_vocab = {"unknown": 0, "apply": 1, "rw": 2, "simp": 3, "exact": 4}
        checkpoint = checkpoint_payload(
            model_kind="tactic_with_args",
            model_spec=spec,
            node_vocab=node_vocab,
            tactic_vocab=tactic_vocab,
            model=model,
        )
        restored, manifest, restored_spec = build_model_from_checkpoint(
            checkpoint,
            node_vocab=node_vocab,
            tactic_vocab=tactic_vocab,
            expected_model_kind="tactic_with_args",
        )
        self.assertEqual(manifest["model_kind"], "tactic_with_args")
        self.assertEqual(restored_spec, spec)
        self.assertEqual(restored.state_dict().keys(), model.state_dict().keys())

    def test_audited_version_one_layouts_migrate_strictly(self) -> None:
        node_vocab = {"a": 0, "b": 1, "c": 2}
        tactic_vocab = {"unknown": 0, "exact": 1, "rw": 2, "simp": 3, "apply": 4}
        cases = (
            ("graphsage_baseline", "supervised_tactic", _spec("graphsage")),
            ("graphsage_pointer", "tactic_with_args", _spec("graphsage")),
            (
                "ac_graphsage_actor_critic",
                "actor_critic_with_args",
                _spec("graphsage"),
            ),
            (
                "gatv2_baseline",
                "supervised_tactic",
                _spec("gatv2", readout="state_mean_max_attention"),
            ),
            ("gatv2_pointer", "tactic_with_args", _spec("gatv2")),
        )
        builders = {
            "supervised_tactic": build_supervised_tactic_model,
            "tactic_with_args": build_pointer_model,
            "actor_critic_with_args": build_actor_critic_model,
        }

        for layout, model_kind, spec in cases:
            with self.subTest(layout=layout):
                model = builders[model_kind](
                    model_spec=spec,
                    num_node_labels=len(node_vocab),
                    num_tactics=len(tactic_vocab),
                )
                legacy_state: dict[str, torch.Tensor] = {}
                for key, value in model.state_dict().items():
                    if key.startswith("encoder.node_features."):
                        suffix = key.removeprefix("encoder.node_features.")
                        prefix = "" if model_kind == "supervised_tactic" else "backbone."
                        legacy_key = prefix + suffix
                    elif key.startswith("encoder."):
                        suffix = key.removeprefix("encoder.")
                        prefix = "" if model_kind == "supervised_tactic" else "backbone."
                        legacy_key = prefix + suffix
                    elif key.startswith("tactic_classifier."):
                        suffix = key.removeprefix("tactic_classifier.")
                        prefix = "" if model_kind == "supervised_tactic" else "backbone."
                        legacy_key = prefix + "classifier." + suffix
                    else:
                        legacy_key = key
                    legacy_state[legacy_key] = value.clone()
                if model_kind == "actor_critic_with_args":
                    legacy_state["backbone.classifier.weight"] = model.actor.base.weight.clone()
                    legacy_state["backbone.classifier.bias"] = model.actor.base.bias.clone()

                legacy_config = {
                    "use_node_type": spec.use_node_type,
                    "max_args": spec.max_args,
                    "model": {
                        "hidden_dim": spec.hidden_dim,
                        "num_layers": spec.encoder["num_layers"],
                        "dropout": spec.dropout,
                        **(
                            {
                                "heads": spec.encoder["heads"],
                                "readout": spec.encoder["readout"],
                            }
                            if spec.architecture == "gatv2"
                            else {}
                        ),
                    },
                }
                migrated = migrate_checkpoint(
                    checkpoint={"model_state_dict": legacy_state, "epoch": 3},
                    legacy_config=legacy_config,
                    layout=layout,
                    node_vocab=node_vocab,
                    tactic_vocab=tactic_vocab,
                )
                restored, manifest, restored_spec = build_model_from_checkpoint(
                    migrated,
                    node_vocab=node_vocab,
                    tactic_vocab=tactic_vocab,
                    expected_model_kind=model_kind,
                )
                self.assertEqual(restored_spec, spec)
                self.assertEqual(manifest["model_kind"], model_kind)
                for key, expected in model.state_dict().items():
                    torch.testing.assert_close(restored.state_dict()[key], expected)


class GraphBudgetBatchSamplerTests(unittest.TestCase):
    def test_batches_respect_independent_node_and_edge_budgets(self) -> None:
        sizes = [
            GraphSize("a", nodes=4, edges=3),
            GraphSize("b", nodes=5, edges=7),
            GraphSize("c", nodes=3, edges=2),
        ]
        node_limited = list(
            GraphBudgetBatchSampler(
                sizes,
                max_graphs=3,
                max_nodes=8,
            )
        )
        edge_limited = list(
            GraphBudgetBatchSampler(
                sizes,
                max_graphs=3,
                max_edges=8,
            )
        )
        self.assertEqual(node_limited, [[0], [1, 2]])
        self.assertEqual(edge_limited, [[0], [1], [2]])

    def test_shuffle_is_deterministic_by_seed_and_epoch(self) -> None:
        sizes = [GraphSize(str(index), 1, 1) for index in range(8)]
        sampler = GraphBudgetBatchSampler(
            sizes,
            max_graphs=2,
            shuffle=True,
            seed=7,
        )
        epoch_zero = list(sampler)
        self.assertEqual(epoch_zero, list(sampler))
        sampler.set_epoch(1)
        self.assertNotEqual(epoch_zero, list(sampler))

    def test_oversized_graph_fails_with_identity_and_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "large.*nodes=11.*edges=4"):
            GraphBudgetBatchSampler(
                [GraphSize("large", nodes=11, edges=4)],
                max_graphs=2,
                max_nodes=10,
            )

    def test_oversized_graph_can_be_skipped_or_kept_singleton(self) -> None:
        sizes = [
            GraphSize("large", nodes=11, edges=4),
            GraphSize("small", nodes=2, edges=2),
        ]
        skipped = GraphBudgetBatchSampler(
            sizes,
            max_graphs=2,
            max_nodes=10,
            oversize_policy="skip",
        )
        self.assertEqual(list(skipped), [[1]])
        singleton = GraphBudgetBatchSampler(
            sizes,
            max_graphs=2,
            max_nodes=10,
            oversize_policy="singleton",
        )
        self.assertEqual(list(singleton), [[0], [1]])


if __name__ == "__main__":
    unittest.main()
