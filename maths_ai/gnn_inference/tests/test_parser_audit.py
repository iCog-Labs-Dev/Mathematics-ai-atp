from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from maths_ai.gnn_inference.atp_lean_gnn import ParserAuditConfig, build_failure_record, run_parser_audit
from maths_ai.gnn_inference.atp_lean_gnn.audit import main as audit_main
from maths_ai.gnn_inference.atp_lean_gnn.dataset import DatasetRow
from maths_ai.gnn_inference.atp_lean_gnn.graph import proof_state_to_dag
from maths_ai.gnn_inference.atp_lean_gnn.labels import label_example
from maths_ai.gnn_inference.atp_lean_gnn.preparation import PreparationPhaseError, prepare_example


class ParserAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.output_root = Path("tests") / "_tmp_parser_audit"
        if self.output_root.exists():
            shutil.rmtree(self.output_root)

    def tearDown(self) -> None:
        if self.output_root.exists():
            shutil.rmtree(self.output_root)

    def _row(
        self,
        *,
        state: str | None,
        theorem: str,
        tactic: str,
        split: str = "train",
        row_index: int = 0,
    ) -> DatasetRow:
        return DatasetRow(
            state=state,
            theorem=theorem,
            tactic=tactic,
            split=split,
            row_index=row_index,
            dataset_name="fake/dataset",
        )

    def test_prepare_example_classifies_failures_by_phase(self) -> None:
        row = self._row(state="n : Nat\n|- n = n", theorem="demo", tactic="simp")

        with patch("maths_ai.gnn_inference.atp_lean_gnn.preparation.parse_state", side_effect=ValueError("bad state")):
            with self.assertRaises(PreparationPhaseError) as parse_ctx:
                prepare_example(row)
        self.assertEqual(parse_ctx.exception.phase, "parse_state")
        self.assertEqual(parse_ctx.exception.cause.__class__.__name__, "ValueError")

        with patch("maths_ai.gnn_inference.atp_lean_gnn.preparation.proof_state_to_dag", side_effect=RuntimeError("bad dag")):
            with self.assertRaises(PreparationPhaseError) as dag_ctx:
                prepare_example(row)
        self.assertEqual(dag_ctx.exception.phase, "proof_state_to_dag")
        self.assertEqual(dag_ctx.exception.cause.__class__.__name__, "RuntimeError")

        with patch("maths_ai.gnn_inference.atp_lean_gnn.preparation.label_example", side_effect=LookupError("bad label")):
            with self.assertRaises(PreparationPhaseError) as label_ctx:
                prepare_example(row)
        self.assertEqual(label_ctx.exception.phase, "label_example")
        self.assertEqual(label_ctx.exception.cause.__class__.__name__, "LookupError")

    def test_build_failure_record_uses_phase_and_error_type_category(self) -> None:
        row = self._row(state="n : Nat\n|- n = n", theorem="demo", tactic="simp")
        exc = PreparationPhaseError(phase="proof_state_to_dag", cause=ValueError("broken dag"))

        record = build_failure_record(row, exc)

        self.assertEqual(record["phase"], "proof_state_to_dag")
        self.assertEqual(record["error_type"], "ValueError")
        self.assertEqual(record["failure_category"], "proof_state_to_dag:ValueError")

    @patch("maths_ai.gnn_inference.atp_lean_gnn.audit.iter_dataset_rows")
    @patch("maths_ai.gnn_inference.atp_lean_gnn.preparation.label_example")
    @patch("maths_ai.gnn_inference.atp_lean_gnn.preparation.proof_state_to_dag")
    def test_run_parser_audit_writes_reports_and_categorized_failures(
        self,
        mock_proof_state_to_dag,
        mock_label_example,
        mock_iter_dataset_rows,
    ) -> None:
        real_proof_state_to_dag = proof_state_to_dag
        real_label_example = label_example

        def fake_iter_dataset_rows(*, dataset_name: str, split: str, sample_limit: int | None = None):
            rows_by_split = {
                "train": [
                    self._row(state=None, theorem="bad.parse", tactic="simp", split="train", row_index=0),
                    self._row(state="n : Nat\n|- FAIL_DAG", theorem="bad.dag", tactic="simp", split="train", row_index=1),
                    self._row(state="m : Nat\n|- m = m", theorem="bad.label", tactic="fail_label", split="train", row_index=2),
                    self._row(state="k : Nat\n|- k = k", theorem="good.train", tactic="simp", split="train", row_index=3),
                ],
                "val": [
                    self._row(state="x : Nat\n|- x = x", theorem="good.val", tactic="rw [foo]", split="val", row_index=0),
                ],
            }
            rows = rows_by_split[split]
            if sample_limit is not None:
                rows = rows[:sample_limit]
            return iter(rows)

        def fake_proof_state_to_dag(parsed_state):
            if parsed_state.goal == "FAIL_DAG":
                raise ValueError("dag failed")
            return real_proof_state_to_dag(parsed_state)

        def fake_label_example(raw_tactic: str):
            if raw_tactic == "fail_label":
                raise ValueError("label failed")
            return real_label_example(raw_tactic)

        mock_iter_dataset_rows.side_effect = fake_iter_dataset_rows
        mock_proof_state_to_dag.side_effect = fake_proof_state_to_dag
        mock_label_example.side_effect = fake_label_example

        summary = run_parser_audit(
            ParserAuditConfig(
                dataset_name="fake/dataset",
                splits=("train", "val"),
                output_root=self.output_root,
                sample_per_split=None,
                max_examples_per_category=2,
                force=True,
            )
        )

        self.assertEqual(summary["overall"]["attempted_count"], 5)
        self.assertEqual(summary["overall"]["success_count"], 2)
        self.assertEqual(summary["overall"]["failure_count"], 3)
        self.assertTrue((self.output_root / "reports" / "parser_audit.json").exists())
        self.assertTrue((self.output_root / "reports" / "parser_audit.md").exists())
        self.assertFalse((self.output_root / "train" / "json").exists())
        self.assertFalse((self.output_root / "train" / "pyg").exists())

        train_failures = (self.output_root / "failures" / "train.jsonl").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(train_failures), 3)
        failure_records = [json.loads(line) for line in train_failures]
        self.assertEqual(
            {record["failure_category"] for record in failure_records},
            {
                "parse_state:TypeError",
                "proof_state_to_dag:ValueError",
                "label_example:ValueError",
            },
        )

        manifest = json.loads((self.output_root / "manifests" / "train.json").read_text(encoding="utf-8"))
        self.assertIn("representative_failure_examples", manifest)
        self.assertIn("top_failure_phases", manifest)
        self.assertEqual(manifest["attempted_count"], 4)
        self.assertEqual(manifest["success_count"], 1)
        self.assertEqual(manifest["failure_count"], 3)

        summary_json = json.loads((self.output_root / "reports" / "parser_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(summary_json["splits_summary"]["val"]["success_count"], 1)
        self.assertEqual(summary_json["top_failure_phases"][0]["count"], 1)
        self.assertEqual(
            {item["name"] for item in summary_json["recommended_follow_up_categories"]},
            {
                "parse_state:TypeError",
                "proof_state_to_dag:ValueError",
                "label_example:ValueError",
            },
        )

        markdown = (self.output_root / "reports" / "parser_audit.md").read_text(encoding="utf-8")
        self.assertIn("Parser Coverage Audit", markdown)
        self.assertIn("Top Failure Categories", markdown)
        self.assertIn("Representative Examples", markdown)

    @patch("maths_ai.gnn_inference.atp_lean_gnn.audit.iter_dataset_rows")
    def test_audit_caps_representative_examples_per_category(self, mock_iter_dataset_rows) -> None:
        def fake_iter_dataset_rows(*, dataset_name: str, split: str, sample_limit: int | None = None):
            rows = [
                self._row(state=None, theorem=f"bad.parse.{index}", tactic="simp", split="train", row_index=index)
                for index in range(4)
            ]
            if sample_limit is not None:
                rows = rows[:sample_limit]
            return iter(rows)

        mock_iter_dataset_rows.side_effect = fake_iter_dataset_rows

        run_parser_audit(
            ParserAuditConfig(
                dataset_name="fake/dataset",
                splits=("train",),
                output_root=self.output_root,
                max_examples_per_category=2,
                force=True,
            )
        )

        manifest = json.loads((self.output_root / "manifests" / "train.json").read_text(encoding="utf-8"))
        examples = manifest["representative_failure_examples"]["parse_state:TypeError"]
        self.assertEqual(len(examples), 2)

    @patch("maths_ai.gnn_inference.atp_lean_gnn.audit.iter_dataset_rows")
    def test_audit_cli_supports_validation_alias_and_sample_limit(self, mock_iter_dataset_rows) -> None:
        def fake_iter_dataset_rows(*, dataset_name: str, split: str, sample_limit: int | None = None):
            rows_by_split = {
                "train": [
                    self._row(state="n : Nat\n|- n = n", theorem=f"train.{index}", tactic="simp", split="train", row_index=index)
                    for index in range(3)
                ],
                "val": [
                    self._row(state="m : Nat\n|- m = m", theorem=f"val.{index}", tactic="simp", split="val", row_index=index)
                    for index in range(3)
                ],
            }
            rows = rows_by_split[split]
            if sample_limit is not None:
                rows = rows[:sample_limit]
            return iter(rows)

        mock_iter_dataset_rows.side_effect = fake_iter_dataset_rows

        exit_code = audit_main(
            [
                "--dataset-name",
                "fake/dataset",
                "--splits",
                "train,validation",
                "--output-root",
                str(self.output_root),
                "--sample-per-split",
                "1",
                "--force",
            ]
        )

        self.assertEqual(exit_code, 0)
        train_manifest = json.loads((self.output_root / "manifests" / "train.json").read_text(encoding="utf-8"))
        val_manifest = json.loads((self.output_root / "manifests" / "val.json").read_text(encoding="utf-8"))
        self.assertEqual(train_manifest["attempted_count"], 1)
        self.assertEqual(val_manifest["attempted_count"], 1)


if __name__ == "__main__":
    unittest.main()
