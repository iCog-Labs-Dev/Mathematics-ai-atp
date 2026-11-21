"""Incremental, per-evaluation proof evidence without live execution references."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid

from maths_ai.hybrid_reasoner.proposal import input_identity
from maths_ai.hybrid_reasoner.search_results import plain, record_view



class EvaluationArchive:
    """One writer; immutable proof files and a synchronized outcome ledger.

    Writing is synchronous within the collection observer, so workers on the
    same event loop cannot interleave records. No result graphs are retained.
    """

    def __init__(self, path: Path, provenance: dict):
        self.path = Path(path).absolute()
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "proofs").mkdir()
        self.proof_count = 0
        self._seen: set[tuple[int, int]] = set()
        self.manifest = {
            "schema_version": 1,
            "evaluation_id": self.path.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "provenance": plain(provenance),
        }
        self._json(self.path / "evaluation.json", self.manifest)
        with (self.path / "attempts.jsonl").open("x"):
            pass

    @classmethod
    def unique(cls, parent: Path, prefix: str, provenance: dict):
        name = f"{prefix}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex}"
        return cls(Path(parent) / name, provenance)

    @staticmethod
    def _json(path: Path, value: dict, *, replace: bool = False):
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("x") as stream:
                json.dump(value, stream, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            if not replace and path.exists():
                raise FileExistsError(path)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def write_attempt(self, record, items, proposals, repeat: int):
        index = record.batch_index
        key = (repeat, index)
        if index < 0 or index >= len(items) or len(items) != len(proposals):
            raise ValueError("Evaluation attempt index does not match its inputs")
        if key in self._seen:
            raise ValueError(f"Duplicate evaluation attempt: {key}")
        seed = items[index].seed
        view = record_view(record, proposal=proposals[index], include_trace=False)
        proof_trace = view.pop("proof_trace", None)
        artifact = view.pop("artifact", None)
        base = {
            "schema_version": 1,
            "evaluation_id": self.manifest["evaluation_id"],
            "repeat": repeat,
            "input_index": index,
            "input_identity": input_identity(seed),
            "source": plain(seed.source_dict),
            **view,
        }
        if view.get("certified"):
            if not proof_trace or not artifact or not artifact.get("receipt") or not artifact.get("statement"):
                raise ValueError("Certified evaluation result lacks proof trace or replay evidence")
            proof_name = f"proofs/repeat-{repeat:04d}-input-{index:06d}.json"
            self._json(self.path / proof_name, {
                **base,
                "kind": "certified-evaluation-proof",
                "provenance": self.manifest["provenance"],
                "starting_state": plain(asdict(seed)),
                "proof_trace": proof_trace,
                "artifact": artifact,
            })
            base["proof_file"] = proof_name
            self.proof_count += 1
        with (self.path / "attempts.jsonl").open("a") as stream:
            stream.write(json.dumps(base, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._seen.add(key)

    def complete(self, report: dict):
        self._json(self.path / "summary.json", plain(report))
        self.manifest.update(status="completed", exported_proofs=self.proof_count, attempts=len(self._seen))
        self._json(self.path / "evaluation.json", self.manifest, replace=True)

    def fail(self, error: BaseException):
        self.manifest.update(status="failed", error_category=type(error).__name__,
                             error=str(error), exported_proofs=self.proof_count, attempts=len(self._seen))
        self._json(self.path / "evaluation.json", self.manifest, replace=True)
