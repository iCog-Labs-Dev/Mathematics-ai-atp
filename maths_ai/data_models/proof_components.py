from typing import List

from pydantic import BaseModel, Field


class Goal(BaseModel):
    """A single proof goal/subgoal as exchanged between the GNN and PLN sides.

    ``expression`` is the Lean target formula (the text after ``⊢``);
    ``hypotheses`` are the local context entries available to prove it.
    """

    expression: str
    hypotheses: List[str] = Field(default_factory=list)


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