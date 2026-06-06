from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from atp_lean_gnn import (
    EMPTY_TACTIC,
    UNKNOWN_TACTIC,
    PreprocessConfig,
    build_failure_record,
    build_tactic_vocab,
    normalize_tactic,
    run_preprocessing,
)
from atp_lean_gnn.cache import build_json_payload
from atp_lean_gnn.dataset import DatasetRow, canonicalize_split_name, dataset_split_name
from atp_lean_gnn.graph import proof_state_to_dag
from atp_lean_gnn.preprocess import main as preprocess_main
from atp_lean_gnn.state import parse_state


class DatasetPreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.output_root = Path("tests") / "_tmp_prepared"
        if self.output_root.exists():
            shutil.rmtree(self.output_root)

    def tearDown(self) -> None:
        if self.output_root.exists():
            shutil.rmtree(self.output_root)

    def _fake_dataset_stream(self, dataset_name: str = "fake/dataset", *, split: str = "train"):
        streams = {
            "train": [
                {
                    "state": None,
                    "full_name": "bad.train",
                    "tactic": "simp",
                },
                {
                    "state": "n : Nat\n|- Even n",
                    "full_name": "good.train",
                    "tactic": "simp only [h1, h2]",
                },
                {
                    "state": "x : Nat\n|- x = x",
                    "full_name": "unknown.train",
                    "tactic": "linarith!",
                },
            ],
            "val": [
                {
                    "state": "y : Nat\n|- y = y",
                    "full_name": "good.val",
                    "tactic": "aesop?",
                }
            ],
            "test": [
                {
                    "state": "z : Nat\n|- z = z",
                    "full_name": "good.test",
                    "tactic": "rw [foo]",
                }
            ],
        }
        return iter(streams[split])

    def test_normalize_tactic_examples(self) -> None:
        self.assertEqual(normalize_tactic("simp only [h1, h2]"), "simp")
        self.assertEqual(normalize_tactic("rw [foo]"), "rw")
        self.assertEqual(normalize_tactic("ext x"), "ext")
        self.assertEqual(normalize_tactic("aesop?"), "aesop?")
        self.assertEqual(normalize_tactic("linarith!"), "linarith!")
        self.assertEqual(normalize_tactic("   "), EMPTY_TACTIC)

    def test_build_tactic_vocab_is_deterministic(self) -> None:
        vocab = build_tactic_vocab(["rw", "simp", "rw"])
        self.assertEqual(vocab[UNKNOWN_TACTIC], 0)
        self.assertEqual(vocab["rw"], 1)
        self.assertEqual(vocab["simp"], 2)

    def test_split_aliases_map_to_canonical_and_dataset_names(self) -> None:
        self.assertEqual(canonicalize_split_name("validation"), "val")
        self.assertEqual(canonicalize_split_name("val"), "val")
        self.assertEqual(dataset_split_name("val"), "validation")
        self.assertEqual(dataset_split_name("validation"), "validation")

    def test_failure_record_contains_required_fields(self) -> None:
        row = DatasetRow(
            state="bad state",
            theorem="demo",
            tactic="simp",
            split="train",
            row_index=7,
            dataset_name="fake/dataset",
        )
        record = build_failure_record(row, RuntimeError("boom"), phase="prepare_example")
        self.assertEqual(record["dataset"], "fake/dataset")
        self.assertEqual(record["split"], "train")
        self.assertEqual(record["row_index"], 7)
        self.assertEqual(record["error_type"], "RuntimeError")
        self.assertEqual(record["phase"], "prepare_example")
        self.assertEqual(record["failure_category"], "prepare_example:RuntimeError")
        self.assertIn("state_preview", record)

    def test_json_payload_shape(self) -> None:
        row = DatasetRow(
            state="n : Nat\n|- Even n",
            theorem="demo",
            tactic="simp",
            split="train",
            row_index=3,
            dataset_name="fake/dataset",
        )
        parsed_state = parse_state(row.state)
        dag = proof_state_to_dag(parsed_state)
        payload = build_json_payload(row, parsed_state=parsed_state, dag=dag, tactic_name="simp")

        self.assertEqual(payload["metadata"]["dataset"], "fake/dataset")
        self.assertEqual(payload["metadata"]["tactic_raw"], "simp")
        self.assertEqual(payload["metadata"]["tactic_name"], "simp")
        self.assertIn("proof_state", payload)
        self.assertIn("graph", payload)
        self.assertIn("stats", payload["graph"])
        self.assertIn("nodes", payload["graph"])
        self.assertIn("edges", payload["graph"])

    @patch("atp_lean_gnn.preprocess.iter_dataset_rows")
    def test_run_preprocessing_creates_artifacts_and_uses_train_only_vocabs(self, mock_iter_dataset_rows) -> None:
        def fake_iter_dataset_rows(*, dataset_name: str, split: str, sample_limit: int | None = None):
            rows = [
                DatasetRow(
                    state=sample["state"],
                    theorem=sample["full_name"],
                    tactic=sample["tactic"],
                    split=split,
                    row_index=index,
                    dataset_name=dataset_name,
                )
                for index, sample in enumerate(list(self._fake_dataset_stream(dataset_name, split=split)))
            ]
            if sample_limit is not None:
                rows = rows[:sample_limit]
            return iter(rows)

        mock_iter_dataset_rows.side_effect = fake_iter_dataset_rows

        config = PreprocessConfig(
            dataset_name="fake/dataset",
            splits=("train", "val", "test"),
            output_root=self.output_root,
            sample_per_split=None,
            force=True,
        )
        summary = run_preprocessing(config)

        self.assertEqual(summary["overall"]["attempted_count"], 5)
        self.assertEqual(summary["overall"]["success_count"], 4)
        self.assertEqual(summary["overall"]["failure_count"], 1)

        node_vocab = json.loads((self.output_root / "vocab" / "node_vocab.json").read_text(encoding="utf-8"))
        tactic_vocab = json.loads((self.output_root / "vocab" / "tactic_vocab.json").read_text(encoding="utf-8"))
        self.assertIn("simp", tactic_vocab)
        self.assertIn("linarith!", tactic_vocab)
        self.assertNotIn("aesop?", tactic_vocab)
        self.assertNotIn("rw", tactic_vocab)

        failure_log = (self.output_root / "failures" / "train.jsonl").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(failure_log), 1)
        failure_record = json.loads(failure_log[0])
        self.assertEqual(failure_record["row_index"], 0)
        self.assertEqual(failure_record["error_type"], "TypeError")
        self.assertEqual(failure_record["phase"], "parse_state")
        self.assertEqual(failure_record["failure_category"], "parse_state:TypeError")

        train_json = json.loads((self.output_root / "train" / "json" / "000000001.json").read_text(encoding="utf-8"))
        self.assertEqual(train_json["metadata"]["tactic_name"], "simp")

        val_data = torch.load(self.output_root / "val" / "pyg" / "000000000.pt", weights_only=False)
        test_data = torch.load(self.output_root / "test" / "pyg" / "000000000.pt", weights_only=False)
        self.assertEqual(int(val_data.y.item()), tactic_vocab[UNKNOWN_TACTIC])
        self.assertEqual(int(test_data.y.item()), tactic_vocab[UNKNOWN_TACTIC])
        self.assertEqual(val_data.split, "val")
        self.assertEqual(test_data.tactic_name, "rw")

        train_manifest = json.loads((self.output_root / "manifests" / "train.json").read_text(encoding="utf-8"))
        summary_json = json.loads((self.output_root / "reports" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(train_manifest["attempted_count"], 3)
        self.assertEqual(train_manifest["success_count"], 2)
        self.assertEqual(summary_json["splits_summary"]["train"]["success_count"], 2)

    @patch("atp_lean_gnn.preprocess.iter_dataset_rows")
    def test_cli_refuses_to_overwrite_without_force(self, mock_iter_dataset_rows) -> None:
        mock_iter_dataset_rows.return_value = iter(())
        self.output_root.mkdir(parents=True, exist_ok=True)

        exit_code = preprocess_main(
            [
                "--dataset-name",
                "fake/dataset",
                "--splits",
                "train",
                "--output-root",
                str(self.output_root),
            ]
        )
        self.assertEqual(exit_code, 1)

    @patch("atp_lean_gnn.preprocess.iter_dataset_rows")
    def test_sample_per_split_limits_processed_rows(self, mock_iter_dataset_rows) -> None:
        def fake_iter_dataset_rows(*, dataset_name: str, split: str, sample_limit: int | None = None):
            rows = [
                DatasetRow(
                    state="n : Nat\n|- Even n",
                    theorem=f"{split}.theorem.{index}",
                    tactic="simp",
                    split=split,
                    row_index=index,
                    dataset_name=dataset_name,
                )
                for index in range(3)
            ]
            if sample_limit is not None:
                rows = rows[:sample_limit]
            return iter(rows)

        mock_iter_dataset_rows.side_effect = fake_iter_dataset_rows

        summary = run_preprocessing(
            PreprocessConfig(
                dataset_name="fake/dataset",
                splits=("train", "val", "test"),
                output_root=self.output_root,
                sample_per_split=1,
                force=True,
            )
        )
        self.assertEqual(summary["overall"]["attempted_count"], 3)
        self.assertTrue((self.output_root / "train" / "json" / "000000000.json").exists())
        self.assertFalse((self.output_root / "train" / "json" / "000000001.json").exists())

    @patch("atp_lean_gnn.dataset._load_hf_split")
    def test_stream_split_translates_val_to_validation(self, mock_load_hf_split) -> None:
        mock_load_hf_split.return_value = iter(
            [
                {
                    "state": "n : Nat\n|- n = n",
                    "full_name": "demo.validation",
                    "tactic": "simp",
                }
            ]
        )

        from atp_lean_gnn.dataset import stream_split

        rows = list(stream_split(split="val", dataset_name="fake/dataset"))

        mock_load_hf_split.assert_called_once_with("val", dataset_name="fake/dataset")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].split, "val")
        self.assertEqual(rows[0].theorem, "demo.validation")


if __name__ == "__main__":
    unittest.main()
