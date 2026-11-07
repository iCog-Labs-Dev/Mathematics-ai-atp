import re
from typing import Iterable, List

from pydantic import BaseModel, Field, model_validator


class Goal(BaseModel):
    """A single proof goal/subgoal as exchanged between the GNN and PLN sides.

    ``expression`` is the Lean target formula (the text after ``⊢``);
    ``hypotheses`` are the local context entries available to prove it.
    """

    expression: str
    hypotheses: List[str] = Field(default_factory=list)
    goal_model_sexp: str | None = None
    locals: List["GoalLocal"] | None = None
    model_sexp_version: int | None = None

    @model_validator(mode="after")
    def _validate_structured_state(self) -> "Goal":
        structured = self.goal_model_sexp is not None or self.locals is not None
        if not structured:
            return self
        if self.goal_model_sexp is None or self.locals is None:
            raise ValueError("Structured Goal requires goal_model_sexp and locals.")
        if self.model_sexp_version != 1:
            raise ValueError(
                f"Unsupported goal model S-expression version {self.model_sexp_version}."
            )
        indices = [local.context_index for local in self.locals]
        if indices != sorted(indices) or len(indices) != len(set(indices)):
            raise ValueError("Goal locals must have unique ordered context indices.")
        derived = [local.declaration_pp for local in self.locals]
        if self.hypotheses and self.hypotheses != derived:
            raise ValueError("Goal hypotheses disagree with structured locals.")
        self.hypotheses = derived
        return self

    @property
    def has_model_state(self) -> bool:
        return self.goal_model_sexp is not None and self.locals is not None

    def require_model_state(self) -> "Goal":
        if not self.has_model_state:
            raise ValueError(
                "The GNN requires normalized lean-model-sexp-v2 state metadata."
            )
        return self


class GoalLocal(BaseModel):
    """A local declaration and its normalized model representation."""

    user_name: str
    internal_name: str = ""
    context_index: int
    binder_role: str
    is_instance: bool = False
    is_let: bool = False
    type_pp: str
    type_model_sexp: str
    value_pp: str | None = None
    value_model_sexp: str | None = None
    model_sexp_version: int = 1

    @model_validator(mode="after")
    def _validate_local(self) -> "GoalLocal":
        if self.context_index < 0:
            raise ValueError("Goal local context_index cannot be negative.")
        if not self.binder_role or not self.type_model_sexp:
            raise ValueError("Goal local role and type model S-expression are required.")
        if self.model_sexp_version != 1:
            raise ValueError(
                f"Unsupported local model S-expression version {self.model_sexp_version}."
            )
        if self.is_let != (self.value_pp is not None and self.value_model_sexp is not None):
            raise ValueError("Local lets require both pretty and model values.")
        return self

    @property
    def declaration_pp(self) -> str:
        value = f" := {self.value_pp}" if self.is_let else ""
        return f"{self.user_name or '_'} : {self.type_pp}{value}"


Goal.model_rebuild()


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


_UNRESOLVED_LEMMA_RE = re.compile(r"^<lemma_[^>]+>$")


def has_unresolved_lemma_argument(candidate: TacticCandidate) -> bool:
    """Return whether a candidate contains an internal lemma placeholder."""
    return any(_UNRESOLVED_LEMMA_RE.fullmatch(argument) for argument in candidate.arguments)


def filter_executable_tactic_candidates(
    candidates: Iterable[TacticCandidate],
) -> list[TacticCandidate]:
    """Remove candidates that cannot be rendered as executable Lean tactics."""
    return [candidate for candidate in candidates if not has_unresolved_lemma_argument(candidate)]


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