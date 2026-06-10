from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

import torch

from maths_ai.gnn_inference.atp_lean_gnn import (
    BaselineConfig,
    DatasetRow,
    GraphSAGEClassifierConfig,
    TrainingLoopConfig,
    analyze_saved_run,
    build_tactic_vocab,
    compare_saved_runs,
    encode_tactic_name,
    label_example,
    render_run_comparison_markdown,
    train_baseline,
)
from maths_ai.gnn_inference.atp_lean_gnn.cache import SplitReport, prepare_output_root, write_manifest, write_pyg_artifact, write_vocab
from maths_ai.gnn_inference.atp_lean_gnn.graph import proof_state_to_dag
from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_vocab_from_labels, dag_to_pyg


class RunAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prepared_root = Path("tests") / "_tmp_analysis_prepared"
        self.run_root = Path("tests") / "_tmp_analysis_runs"
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
                    state="x : Nat\n|- x = x",
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
                    theorem="demo.val.even",
                    tactic="simp",
                    split="val",
                    row_index=0,
                    dataset_name="fake/dataset",
                ),
                DatasetRow(
                    state="y : Nat\n|- y = y",
                    theorem="demo.val.eq",
                    tactic="rw",
                    split="val",
                    row_index=1,
                    dataset_name="fake/dataset",
                ),
            ],
            "test": [
                DatasetRow(
                    state="z : Nat\n|- z = z",
                    theorem="demo.test.eq",
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
            seed=11,
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

    def test_analyze_saved_run_writes_reports_and_predictions(self) -> None:
        summary = train_baseline(self._tiny_config())
        run_dir = Path(str(summary["run_dir"]))

        analysis = analyze_saved_run(run_dir, split="val", top_k=3, min_support=1)

        self.assertTrue((run_dir / "analysis_val.json").exists())
        self.assertTrue((run_dir / "analysis_val.md").exists())
        self.assertTrue((run_dir / "predictions_val.jsonl").exists())
        self.assertEqual(analysis["split"], "val")
        self.assertEqual(int(analysis["overall"]["evaluated_count"]), 2)

        prediction_lines = (run_dir / "predictions_val.jsonl").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(prediction_lines), 2)
        first_prediction = json.loads(prediction_lines[0])
        self.assertIn("predicted_topk", first_prediction)
        self.assertIn("correct_top1", first_prediction)

    def test_compare_saved_runs_renders_markdown_table(self) -> None:
        first_summary = train_baseline(self._tiny_config())
        second_summary = train_baseline(self._tiny_config())

        comparison = compare_saved_runs(
            [
                first_summary["run_dir"],
                second_summary["run_dir"],
            ]
        )
        markdown = render_run_comparison_markdown(comparison)

        self.assertEqual(len(comparison["runs"]), 2)
        self.assertIn("Run Comparison", markdown)
        self.assertIn(Path(str(first_summary["run_dir"])).name, markdown)


if __name__ == "__main__":
    unittest.main()
