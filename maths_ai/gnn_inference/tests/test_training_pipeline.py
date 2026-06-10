from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

import torch

from atp_lean_gnn import (
    BaselineConfig,
    DatasetRow,
    GraphSAGEClassifierConfig,
    PreparedGraphDataset,
    TrainingLoopConfig,
    build_dataloaders,
    build_tactic_vocab,
    compute_eval_metrics_from_logits,
    encode_tactic_name,
    evaluate_baseline_run,
    label_example,
    load_prepared_metadata,
    train_baseline,
)
from atp_lean_gnn.cache import SplitReport, prepare_output_root, write_manifest, write_pyg_artifact, write_vocab
from atp_lean_gnn.graph import proof_state_to_dag
from atp_lean_gnn.pyg import build_vocab_from_labels, dag_to_pyg
from atp_lean_gnn.training import build_baseline_model


class TrainingPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prepared_root = Path("tests") / "_tmp_prepared_training"
        self.run_root = Path("tests") / "_tmp_runs"
        for path in (self.prepared_root, self.run_root):
            if path.exists():
                shutil.rmtree(path)
        self._build_fake_prepared_root()

    def tearDown(self) -> None:
        for path in (self.prepared_root, self.run_root):
            if path.exists():
                shutil.rmtree(path)

    def _split_rows(self) -> dict[str, list[DatasetRow]]:
        return {
            "train": [
                DatasetRow(
                    state="n : Nat\n|- Even n",
                    theorem="demo.train.even",
                    tactic="simp only [h1]",
                    split="train",
                    row_index=0,
                    dataset_name="fake/dataset",
                ),
                DatasetRow(
                    state="State : Nat\n|- State = State",
                    theorem="demo.train.eq",
                    tactic="rw [foo]",
                    split="train",
                    row_index=1,
                    dataset_name="fake/dataset",
                ),
            ],
            "val": [
                DatasetRow(
                    state="m : Nat\n|- Even m",
                    theorem="demo.val.known",
                    tactic="simp",
                    split="val",
                    row_index=0,
                    dataset_name="fake/dataset",
                ),
                DatasetRow(
                    state="y : Nat\n|- y = y",
                    theorem="demo.val.unknown",
                    tactic="aesop?",
                    split="val",
                    row_index=1,
                    dataset_name="fake/dataset",
                ),
            ],
            "test": [
                DatasetRow(
                    state="z : Nat\n|- z = z",
                    theorem="demo.test.known",
                    tactic="rw",
                    split="test",
                    row_index=0,
                    dataset_name="fake/dataset",
                )
            ],
        }

    def _build_fake_prepared_root(self) -> None:
        split_rows = self._split_rows()
        prepare_output_root(self.prepared_root, splits=["train", "val", "test"], force=True)

        node_labels: set[str] = set()
        train_tactic_names: list[str] = []
        dags_by_split: dict[str, list[tuple[DatasetRow, object, str]]] = {
            "train": [],
            "val": [],
            "test": [],
        }

        for split, rows in split_rows.items():
            for row in rows:
                dag = proof_state_to_dag(row.state)
                tactic_name = str(label_example(row.tactic)["tactic_name"])
                dags_by_split[split].append((row, dag, tactic_name))
                if split == "train":
                    node_labels.update(node.label for node in dag.nodes)
                    train_tactic_names.append(tactic_name)

        node_vocab = build_vocab_from_labels(node_labels)
        tactic_vocab = build_tactic_vocab(train_tactic_names)
        write_vocab(self.prepared_root, name="node_vocab.json", vocab=node_vocab)
        write_vocab(self.prepared_root, name="tactic_vocab.json", vocab=tactic_vocab)

        for split in ("train", "val", "test"):
            report = SplitReport(split=split)
            for row, dag, tactic_name in dags_by_split[split]:
                data = dag_to_pyg(dag, node_vocab)
                data.y = torch.tensor(
                    [encode_tactic_name(tactic_name, tactic_vocab)],
                    dtype=torch.long,
                )
                data.split = split
                data.row_index = row.row_index
                data.dataset_name = row.dataset_name
                data.theorem = row.theorem
                data.tactic_raw = row.tactic
                data.tactic_name = tactic_name
                write_pyg_artifact(
                    self.prepared_root,
                    split=split,
                    row_index=row.row_index,
                    data=data,
                )
                report.record_success(dag=dag, tactic_name=tactic_name)

            manifest = report.to_manifest(
                dataset_name="fake/dataset",
                output_root=self.prepared_root,
                vocab_source="train",
                sample_limit=None,
            )
            write_manifest(self.prepared_root, split=split, manifest=manifest)

    def _tiny_config(self) -> BaselineConfig:
        return BaselineConfig(
            prepared_root=self.prepared_root,
            run_root=self.run_root,
            seed=7,
            device="cpu",
            edge_mode="bidirectional",
            use_node_type=True,
            model=GraphSAGEClassifierConfig(hidden_dim=16, num_layers=2, dropout=0.1),
            training=TrainingLoopConfig(
                batch_size=2,
                epochs=1,
                learning_rate=1e-3,
                weight_decay=1e-4,
                grad_clip=1.0,
                log_every_batches=1,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
                prefetch_factor=2,
                use_amp=False,
            ),
        ).normalized()

    def test_loader_reads_manifests_and_infers_state_node_index(self) -> None:
        metadata = load_prepared_metadata(self.prepared_root)
        dataset = PreparedGraphDataset(metadata, split="train", edge_mode="forward")

        self.assertEqual(len(dataset), 2)
        sample = dataset[1]
        state_matches = (sample.x == metadata.state_label_id).nonzero(as_tuple=False).view(-1)
        source_nodes = {int(node_id) for node_id in sample.edge_index[0].tolist()}

        self.assertGreater(int(state_matches.numel()), 1)
        self.assertEqual(int(sample.state_node_index.numel()), 1)
        self.assertEqual(
            int(sample.x[sample.state_node_index.item()].item()),
            metadata.state_label_id,
        )
        self.assertNotIn(int(sample.state_node_index.item()), source_nodes)

    def test_bidirectional_transform_preserves_nodes_and_adds_reverse_edges(self) -> None:
        metadata = load_prepared_metadata(self.prepared_root)
        forward_sample = PreparedGraphDataset(metadata, split="train", edge_mode="forward")[0]
        bidirectional_sample = PreparedGraphDataset(metadata, split="train", edge_mode="bidirectional")[0]

        expected_edge_index = torch.unique(
            torch.cat([forward_sample.edge_index, forward_sample.edge_index[[1, 0], :]], dim=1),
            dim=1,
        )

        self.assertEqual(tuple(forward_sample.x.shape), tuple(bidirectional_sample.x.shape))
        self.assertEqual(
            int(bidirectional_sample.edge_index.shape[1]),
            int(expected_edge_index.shape[1]),
        )
        self.assertGreaterEqual(
            int(bidirectional_sample.edge_index.shape[1]),
            int(forward_sample.edge_index.shape[1]),
        )

    def test_model_forward_returns_batch_logits(self) -> None:
        metadata = load_prepared_metadata(self.prepared_root)
        config = self._tiny_config()
        _, loaders = build_dataloaders(metadata, config)
        batch = next(iter(loaders["train"]))
        model = build_baseline_model(metadata, config)

        logits = model(batch)

        self.assertEqual(tuple(logits.shape), (2, len(metadata.tactic_vocab)))

    def test_eval_metrics_exclude_unknown_targets(self) -> None:
        logits = torch.tensor(
            [
                [0.1, 2.0, 0.0],
                [2.0, 0.1, 0.0],
            ],
            dtype=torch.float,
        )
        targets = torch.tensor([1, 0], dtype=torch.long)

        metrics = compute_eval_metrics_from_logits(
            logits,
            targets,
            unknown_tactic_id=0,
        )

        self.assertEqual(int(metrics["known_label_count"]), 1)
        self.assertEqual(int(metrics["unknown_label_excluded_count"]), 1)
        self.assertEqual(int(metrics["top1_correct"]), 1)
        self.assertEqual(int(metrics["top5_correct"]), 1)

    def test_training_run_writes_checkpoints_and_eval_reports(self) -> None:
        summary = train_baseline(self._tiny_config())
        run_dir = Path(str(summary["run_dir"]))

        self.assertTrue((run_dir / "config.json").exists())
        self.assertTrue((run_dir / "metrics.jsonl").exists())
        self.assertTrue((run_dir / "best.pt").exists())
        self.assertTrue((run_dir / "last.pt").exists())
        self.assertTrue((run_dir / "summary.json").exists())
        self.assertTrue((run_dir / "eval_val.json").exists())
        self.assertTrue((run_dir / "eval_test.json").exists())

        metrics_lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(metrics_lines), 1)

        test_metrics = evaluate_baseline_run(run_dir, split="test")
        written_test_metrics = json.loads((run_dir / "eval_test.json").read_text(encoding="utf-8"))
        self.assertEqual(test_metrics["split"], "test")
        self.assertEqual(written_test_metrics["split"], "test")

    def test_resume_run_uses_last_checkpoint_and_appends_metrics(self) -> None:
        first_summary = train_baseline(self._tiny_config())
        run_dir = Path(str(first_summary["run_dir"]))

        resumed_config = BaselineConfig(
            prepared_root=self.prepared_root,
            run_root=self.run_root,
            seed=7,
            device="cpu",
            edge_mode="bidirectional",
            use_node_type=True,
            model=GraphSAGEClassifierConfig(hidden_dim=16, num_layers=2, dropout=0.1),
            training=TrainingLoopConfig(
                batch_size=2,
                epochs=2,
                learning_rate=1e-3,
                weight_decay=1e-4,
                grad_clip=1.0,
                log_every_batches=1,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
                prefetch_factor=2,
                use_amp=False,
            ),
        ).normalized()

        resumed_summary = train_baseline(resumed_config, resume_run_dir=run_dir)
        metrics_lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()

        self.assertEqual(Path(str(resumed_summary["run_dir"])), run_dir)
        self.assertTrue(bool(resumed_summary["resumed_from_checkpoint"]))
        self.assertEqual(int(resumed_summary["start_epoch"]), 2)
        self.assertEqual(len(metrics_lines), 2)


if __name__ == "__main__":
    unittest.main()
