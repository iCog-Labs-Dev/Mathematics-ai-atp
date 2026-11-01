"""Immutable candidate support and complete actions shared by collection/update."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math


def fingerprint(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class ArgumentReference:
    source: str
    identity: int

    def __post_init__(self):
        if (
            self.source not in {"local", "library"}
            or type(self.identity) is not int
            or self.identity < 0
        ):
            raise ValueError(
                "Argument needs a local/library source and nonnegative integer identity."
            )


@dataclass(frozen=True)
class CandidatePoolSnapshot:
    local_ids: tuple[int, ...]
    lemma_ids: tuple[int, ...] = ()
    lemma_scores: tuple[float, ...] = ()
    graph_fingerprint: str = ""
    query_fingerprint: str = ""
    artifact_fingerprint: str = ""
    policy_fingerprint: str = ""
    library: object = None
    type_signals: tuple[tuple[int, tuple[float, ...]], ...] = ()

    def __post_init__(self):
        for values in (self.local_ids, self.lemma_ids):
            if (
                type(values) is not tuple
                or len(set(values)) != len(values)
                or any(type(x) is not int or x < 0 for x in values)
            ):
                raise ValueError(
                    "Candidate IDs must be unique immutable nonnegative integers."
                )
        if type(self.lemma_scores) is not tuple or type(self.type_signals) is not tuple:
            raise ValueError("Candidate scores and features must be immutable tuples.")
        if len(self.lemma_ids) != len(self.lemma_scores) or any(
            not math.isfinite(x) for x in self.lemma_scores
        ):
            raise ValueError("Each library candidate needs a finite retrieval score.")
        if self.lemma_ids and (
            self.library is None
            or self.library.fingerprint != self.artifact_fingerprint
        ):
            raise ValueError(
                "Library candidates require their exact immutable artifact."
            )
        if len({key for key, _ in self.type_signals}) != len(self.type_signals):
            raise ValueError("Duplicate tactic-conditioned feature keys.")
        for key, signal in self.type_signals:
            if type(key) is not int or key < 0 or type(signal) is not tuple:
                raise ValueError(
                    "Tactic features require integer keys and immutable value tuples."
                )
            if len(signal) != len(self.local_ids) or any(
                not math.isfinite(x) for x in signal
            ):
                raise ValueError("Local type features do not match the pool.")

    def signal_for(self, tactic_id):
        return dict(self.type_signals).get(tactic_id, ())

    @property
    def fingerprint(self):
        return fingerprint({k: v for k, v in self.__dict__.items() if k != "library"})

    def validate(self, action):
        for ref in action:
            if ref.identity not in (
                self.local_ids if ref.source == "local" else self.lemma_ids
            ):
                raise ValueError(
                    "Stored action is outside its collection candidate support."
                )


def local_pools(data) -> tuple[CandidatePoolSnapshot, ...]:
    """Explicit local-candidate provider for local-only supervised datasets."""
    mask = data.premise_mask.bool()
    return tuple(
        CandidatePoolSnapshot(tuple((mask[start:end].nonzero().flatten()).tolist()))
        for start, end in zip(data.ptr[:-1].tolist(), data.ptr[1:].tolist())
    )


class LocalCandidateProvider:
    """Live provider whose pools hold only the goal's own local candidates.

    For runs without retrieval artifacts (smoke checks, diagnostics, local-only
    models): it never proposes a library lemma.
    """

    async def retrieve(self, goal, local_ids, graph_fingerprint, *, excluded=()):
        from .retrieval_artifacts import goal_query

        return CandidatePoolSnapshot(
            tuple(local_ids),
            graph_fingerprint=graph_fingerprint,
            query_fingerprint=fingerprint(goal_query(goal)),
        )

    async def aclose(self):
        return None


def with_type_signals(pool, goal, dag, tactic_vocab):
    """Source pilot's soft exact/rw features, tied to each actual tactic ID."""
    from dataclasses import replace
    from maths_ai.lean_replay.sexp import parse_sexp_string

    target = parse_sexp_string(goal.goal_model_sexp)
    hypotheses = {f"FV{x.context_index}": x.type_model_sexp for x in goal.locals or []}

    def contains(tree, value):
        return tree == value or (
            isinstance(tree, list) and any(contains(x, value) for x in tree)
        )

    features = []
    for name in ("exact", "rw"):
        if name not in tactic_vocab:
            continue
        values = []
        for node_id in pool.local_ids:
            text = hypotheses.get(dag.nodes[node_id].label)
            match = False
            if text is not None:
                if name == "exact":
                    match = text == goal.goal_model_sexp
                else:
                    expr = parse_sexp_string(text)
                    if (
                        isinstance(expr, list)
                        and len(expr) >= 4
                        and expr[0] == ":app"
                        and expr[1] in ([":c", "Eq"], [":c", "Iff"], [":c", "HEq"])
                    ):
                        args = [
                            x[3]
                            for x in expr[2:]
                            if isinstance(x, list)
                            and len(x) == 4
                            and x[:2] == [":arg", ":explicit"]
                        ]
                        match = len(args) >= 2 and any(
                            contains(target, x) for x in args[-2:]
                        )
            values.append(float(match))
        features.append((tactic_vocab[name], tuple(values)))
    return replace(pool, type_signals=tuple(features))


class OfflineCandidateProvider:
    """Dataset-row lookup over verified snapshots, never a live-state retriever.

    The importer supplies a table keyed by (split, row_index) and the table's
    prepared-artifact identity. A graph batch must name that same identity.
    """

    def __init__(self, rows, *, prepared_identity):
        if not prepared_identity:
            raise ValueError("Offline candidates require a prepared-artifact identity.")
        self.rows = dict(rows)
        self.prepared_identity = prepared_identity
        if any(
            not isinstance(pool, CandidatePoolSnapshot) for pool in self.rows.values()
        ):
            raise TypeError(
                "Offline candidate rows must contain immutable pool snapshots."
            )

    def __call__(self, batch):
        identities = batch.candidate_prepared_identity
        if any(value != self.prepared_identity for value in identities):
            raise ValueError("Offline candidate table and prepared graphs differ.")
        result = []
        for split, row, offset in zip(
            batch.split, batch.row_index.tolist(), batch.ptr[:-1].tolist()
        ):
            row -= offset
            try:
                result.append(self.rows[(split, row)])
            except KeyError as exc:
                raise ValueError(
                    f"Missing offline candidate row: {split}/{row}."
                ) from exc
        if len(result) != len(batch.ptr) - 1:
            raise ValueError(
                "Offline candidate row metadata does not align with graphs."
            )
        return tuple(result)
