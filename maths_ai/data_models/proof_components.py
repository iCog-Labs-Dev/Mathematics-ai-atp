import re
from typing import List, Literal

from pydantic import BaseModel, Field, field_validator


class LocalDeclaration(BaseModel):
    """A declaration in a Lean goal's local context.

    Keeping a local definition distinct from an ordinary binder is essential:
    ``let x : Nat := 1`` cannot be faithfully reconstructed as ``x : Nat``.
    """

    name: str
    type_expression: str
    value_expression: str | None = None
    kind: Literal["variable", "let"] = "variable"

    @classmethod
    def from_text(cls, declaration: str) -> "LocalDeclaration":
        text = declaration.strip()
        match = re.fullmatch(r"let\s+([^:]+)\s*:\s*(.+?)\s*:=\s*(.+)", text, re.DOTALL)
        if match:
            return cls(
                name=match.group(1).strip(),
                type_expression=match.group(2).strip(),
                value_expression=match.group(3).strip(),
                kind="let",
            )
        if ":" not in text:
            raise ValueError(f"local declaration must contain ':': {declaration!r}")
        name, type_expression = text.split(":", 1)
        return cls(name=name.strip(), type_expression=type_expression.strip())

    def render(self) -> str:
        if self.kind == "let":
            if self.value_expression is None:
                raise ValueError("a local let declaration requires a value_expression")
            return f"let {self.name} : {self.type_expression} := {self.value_expression}"
        return f"{self.name} : {self.type_expression}"


class Goal(BaseModel):
    """A single proof goal/subgoal as exchanged between the GNN and PLN sides.

    ``expression`` is the Lean target formula (the text after ``⊢``);
    ``hypotheses`` are the local context entries available to prove it.
    """

    expression: str
    hypotheses: List[LocalDeclaration] = Field(default_factory=list)

    @field_validator("hypotheses", mode="before")
    @classmethod
    def _coerce_legacy_hypotheses(cls, values):
        if values is None:
            return []
        return [LocalDeclaration.from_text(value) if isinstance(value, str) else value for value in values]


class GoalState(BaseModel):
    """A goal positioned within a proof search branch.

    ``tactic_path`` records the tactics applied (in order) from the root
    goal down to this state, which doubles as provenance for the hypergraph
    and as the cycle-detection key (see HybridReasoner edge cases).
    """

    goal: Goal
    depth: int = 0
    tactic_path: List[str] = Field(default_factory=list)


class STV(BaseModel):
    """A PLN strength/confidence truth value, e.g. ``(STV 0.8 0.6)``."""

    strength: float
    confidence: float

    @property
    def score(self) -> float:
        """Conventional PLN ranking score: strength × confidence.

        Mirrors ``score_from_stv`` in the MeTTa translator's ranking module
        so both subsystems agree on how an STV collapses to a scalar rank.
        """
        return self.strength * self.confidence


class TacticCandidate(BaseModel):
    """A single ranked tactic prediction from the GNN engine."""

    tactic_name: str
    arguments: List[str] = Field(default_factory=list)
    probability: float


class RankedSubgoal(BaseModel):
    """A subgoal scored by the symbolic (PLN) side and combined with the
    GNN's prior probability for the tactic that produced it."""

    goal: Goal
    stv: STV
    gnn_probability: float

    @property
    def combined_rank(self) -> float:
        """score = gnn_prob × strength × confidence (see design report,
        section "Open design questions" — a simple, principled default that
        extends the existing strength×confidence convention multiplicatively
        by the policy prior; tune/replace if empirical results call for it)."""
        return self.gnn_probability * self.stv.score
