import csv
import json
import shutil
import unittest
from pathlib import Path

from atp_lean_gnn.logger import TrainingLogger


class LoggerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.output_root = Path("tests") / "_tmp_logger"
        if self.output_root.exists():
            shutil.rmtree(self.output_root)

    def tearDown(self) -> None:
        if self.output_root.exists():
            shutil.rmtree(self.output_root)

    def test_training_logger_writes_jsonl_and_csv(self) -> None:
        logger = TrainingLogger(self.output_root)
        logger.log_epoch(1, {"train_loss": 0.5, "val_loss": 0.6})
        logger.log_epoch(2, {"train_loss": 0.4, "val_loss": 0.55})

        jsonl_path = self.output_root / "learning_curve.jsonl"
        csv_path = self.output_root / "learning_curve.csv"

        self.assertTrue(jsonl_path.exists())
        self.assertTrue(csv_path.exists())

        with jsonl_path.open("r", encoding="utf-8") as handle:
            lines = [json.loads(line) for line in handle]

        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["epoch"], 1)
        self.assertEqual(lines[1]["epoch"], 2)
        self.assertAlmostEqual(lines[0]["train_loss"], 0.5)
        self.assertAlmostEqual(lines[1]["val_loss"], 0.55)

        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)

        self.assertEqual(len(rows), 2)
        self.assertEqual(int(rows[0]["epoch"]), 1)
        self.assertAlmostEqual(float(rows[1]["val_loss"]), 0.55)


if __name__ == "__main__":
    unittest.main()
