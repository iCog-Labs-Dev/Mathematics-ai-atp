from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LemmaRecord:
    lemma_id: int
    name: str
    statement: str
    namespace: str
    module: str

    def to_dict(self) -> dict[str, object]:
        return {
            "lemma_id": self.lemma_id,
            "name": self.name,
            "statement": self.statement,
            "namespace": self.namespace,
            "module": self.module,
        }


def load_lemma_corpus(path: str | Path) -> list[LemmaRecord]:
    corpus_path = Path(path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Lemma corpus not found at '{corpus_path}'.")

    records: list[LemmaRecord] = []
    with corpus_path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            try:
                lemma_id = int(payload["lemma_id"])
                name = str(payload["name"]).strip()
                statement = str(payload["statement"]).strip()
            except KeyError as exc:
                raise ValueError(
                    f"Lemma corpus entry on line {line_index} is missing {exc.args[0]!r}."
                ) from exc

            if not name:
                raise ValueError(f"Lemma corpus entry on line {line_index} has an empty name.")
            if not statement:
                raise ValueError(f"Lemma corpus entry on line {line_index} has an empty statement.")

            records.append(
                LemmaRecord(
                    lemma_id=lemma_id,
                    name=name,
                    statement=statement,
                    namespace=str(payload.get("namespace", "")),
                    module=str(payload.get("module", "")),
                )
            )

    if not records:
        raise ValueError(f"Lemma corpus '{corpus_path}' is empty.")
    return records


def load_lemma_name_index(path: str | Path) -> dict[str, int]:
    """Return a mapping from lemma name to lemma id.

    The first occurrence of a lemma name wins if duplicates exist.
    """
    name_index: dict[str, int] = {}
    for record in load_lemma_corpus(path):
        if record.name not in name_index:
            name_index[record.name] = record.lemma_id
    return name_index


def write_lemma_corpus(path: str | Path, records: list[LemmaRecord]) -> Path:
    corpus_path = Path(path)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with corpus_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return corpus_path
