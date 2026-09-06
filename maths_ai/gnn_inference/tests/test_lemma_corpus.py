from __future__ import annotations

import json
import unittest
import tempfile
from pathlib import Path

from maths_ai.gnn_inference.atp_lean_gnn.lemma_corpus import load_lemma_corpus, load_lemma_name_index, write_lemma_corpus, LemmaRecord


class LemmaCorpusTests(unittest.TestCase):
    def test_load_lemma_corpus_accepts_directory_path(self) -> None:
        records = [
            LemmaRecord(lemma_id=7, name="Foo.bar", statement="x = x", namespace="Foo", module=""),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_dir = Path(tmpdir)
            corpus_path = corpus_dir / "lemmas.jsonl"
            write_lemma_corpus(corpus_path, records)

            loaded = load_lemma_corpus(corpus_dir)

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].lemma_id, 7)
            self.assertEqual(loaded[0].name, "Foo.bar")

    def setUp(self) -> None:
        self.tmp_dir = Path("tests") / "_tmp_lemma_corpus"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.corpus_path = self.tmp_dir / "lemmas.jsonl"

    def tearDown(self) -> None:
        if self.tmp_dir.exists():
            for path in self.tmp_dir.iterdir():
                path.unlink()
            self.tmp_dir.rmdir()

    def test_roundtrip_and_index(self) -> None:
        records = [
            LemmaRecord(lemma_id=0, name="Foo.bar", statement="x = x", namespace="Foo", module=""),
            LemmaRecord(lemma_id=1, name="Baz.qux", statement="y = y", namespace="Baz", module=""),
        ]
        write_lemma_corpus(self.corpus_path, records)

        loaded = load_lemma_corpus(self.corpus_path)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].name, "Foo.bar")
        self.assertEqual(loaded[1].name, "Baz.qux")

        index = load_lemma_name_index(self.corpus_path)
        self.assertEqual(index["Foo.bar"], 0)
        self.assertEqual(index["Baz.qux"], 1)
