from __future__ import annotations

from types import SimpleNamespace

import pytest

from maths_ai.gnn_inference.inference_engine import GNNModelEngine


def test_validate_lemma_artifacts_rejects_ids_missing_from_corpus() -> None:
    lemma_index = SimpleNamespace(lemma_ids=[7, 11])
    lemma_corpus = {7: object()}

    with pytest.raises(ValueError, match="missing from the loaded lemma corpus"):
        GNNModelEngine._validate_lemma_artifacts(lemma_index, lemma_corpus)
