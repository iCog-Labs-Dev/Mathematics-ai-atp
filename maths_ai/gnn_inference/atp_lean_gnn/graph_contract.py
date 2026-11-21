from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class GraphRepresentationSpec:
    """The graph representation a prepared dataset and checkpoint require."""

    representation: str
    model_sexp_version: int
    graph_schema_version: int
    hypothesis_schema: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "GraphRepresentationSpec":
        try:
            spec = cls(
                representation=str(payload["representation"]),
                model_sexp_version=int(payload["model_sexp_version"]),
                graph_schema_version=int(payload["graph_schema_version"]),
                hypothesis_schema=str(payload["hypothesis_schema"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid graph representation contract.") from exc
        spec.validate()
        return spec

    def validate(self) -> None:
        if not self.representation:
            raise ValueError("Graph representation name cannot be empty.")
        if self.model_sexp_version < 0:
            raise ValueError("model_sexp_version cannot be negative.")
        if self.graph_schema_version < 1:
            raise ValueError("graph_schema_version must be positive.")
        if not self.hypothesis_schema:
            raise ValueError("Hypothesis schema cannot be empty.")


MODEL_SEXPR_GRAPH_SPEC = GraphRepresentationSpec(
    representation="lean-model-sexp-v2",
    model_sexp_version=1,
    graph_schema_version=1,
    hypothesis_schema="fv-name-role-type-v1",
)

TEXT_GRAPH_SPEC = GraphRepresentationSpec(
    representation="lean-pretty-text-v1",
    model_sexp_version=0,
    graph_schema_version=1,
    hypothesis_schema="name-type-v1",
)


def require_graph_representation(
    actual: GraphRepresentationSpec,
    expected: GraphRepresentationSpec = MODEL_SEXPR_GRAPH_SPEC,
) -> None:
    if actual != expected:
        raise ValueError(
            "Graph representation mismatch: "
            f"actual={actual.to_dict()}, expected={expected.to_dict()}."
        )
