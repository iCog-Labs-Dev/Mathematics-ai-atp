from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class LeanGoalSeed(BaseModel):
    """Pretty-printed Lean source that has not yet been serialized by Pantograph."""

    expression: str
    hypotheses: List[str] = Field(default_factory=list)


@dataclass(frozen=True)
class GoalReplaySpec:
    """Lean source used to reconstruct and serialize one proof state."""

    expression: str
    local_names: tuple[str, ...]


class GoalLocal(BaseModel):
    """One serialized Lean local declaration and its normalized model view."""

    user_name: str
    internal_name: str = ""
    context_index: int
    binder_role: str
    is_instance: bool = False
    is_let: bool = False
    type_pp: str
    type_model_sexp: str
    value_pp: Optional[str] = None
    value_model_sexp: Optional[str] = None
    model_sexp_version: int = 1

    @model_validator(mode="after")
    def _validate_local(self) -> "GoalLocal":
        if self.context_index < 0:
            raise ValueError("Goal local context_index cannot be negative.")
        if not self.binder_role:
            raise ValueError("Goal local binder_role cannot be empty.")
        if not self.type_model_sexp:
            raise ValueError("Goal local type_model_sexp cannot be empty.")
        if self.model_sexp_version != 1:
            raise ValueError(
                f"Unsupported local model S-expression version {self.model_sexp_version}."
            )
        if self.is_let and (self.value_pp is None or self.value_model_sexp is None):
            raise ValueError("Local let declarations require pretty and model values.")
        if not self.is_let and (
            self.value_pp is not None or self.value_model_sexp is not None
        ):
            raise ValueError("Non-let locals cannot carry a local value.")
        return self

    @property
    def declaration_pp(self) -> str:
        name = self.user_name or "_"
        value = f" := {self.value_pp}" if self.is_let and self.value_pp is not None else ""
        return f"{name} : {self.type_pp}{value}"

    def graph_record(self) -> dict[str, object]:
        return {
            "name": self.user_name or "_",
            "internal_name": self.internal_name,
            "context_index": self.context_index,
            "binder_role": self.binder_role,
            "is_instance": self.is_instance,
            "is_let": self.is_let,
            "sexp": self.type_model_sexp,
        }


class Goal(BaseModel):
    """A single proof goal/subgoal as exchanged between the GNN and PLN sides.

    ``expression`` is the Lean target formula (the text after ``⊢``);
    ``hypotheses`` are the local context entries available to prove it.
    """

    expression: str
    hypotheses: List[str] = Field(default_factory=list)
    goal_model_sexp: Optional[str] = None
    locals: Optional[List[GoalLocal]] = None
    case_tag: Optional[str] = None
    model_sexp_version: Optional[int] = None

    @model_validator(mode="after")
    def _validate_representations(self) -> "Goal":
        structured = self.goal_model_sexp is not None or self.locals is not None
        if structured and (self.goal_model_sexp is None or self.locals is None):
            raise ValueError(
                "Goal model state requires both goal_model_sexp and locals."
            )
        if structured:
            if self.model_sexp_version != 1:
                raise ValueError(
                    f"Unsupported goal model S-expression version {self.model_sexp_version}."
                )
            indices = [local.context_index for local in self.locals or []]
            if indices != sorted(indices) or len(indices) != len(set(indices)):
                raise ValueError("Goal locals must have unique ordered context indices.")
            derived = [local.declaration_pp for local in self.locals or []]
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
                "Live RL requires normalized model S-expression metadata; "
                "a text-only Goal cannot be featurized."
            )
        return self

    def graph_locals(self) -> list[dict[str, object]]:
        self.require_model_state()
        return [local.graph_record() for local in self.locals or []]

    def state_fingerprint(self) -> str:
        if self.has_model_state:
            payload: object = {
                "goal": self.goal_model_sexp,
                "locals": [
                    {
                        "user_name": local.user_name,
                        "context_index": local.context_index,
                        "binder_role": local.binder_role,
                        "is_instance": local.is_instance,
                        "is_let": local.is_let,
                        "type": local.type_model_sexp,
                        "value": local.value_model_sexp,
                    }
                    for local in self.locals or []
                ],
            }
        else:
            payload = {"expression": self.expression, "hypotheses": self.hypotheses}
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def replay_spec(self) -> GoalReplaySpec:
        """Render the structured local context back into closed Lean source."""
        self.require_model_state()
        expression = self.expression
        local_names: list[str] = []
        for local in reversed(self.locals or []):
            name = local.user_name or "_"
            local_names.append(name)
            if local.is_let:
                expression = (
                    f"let {name} : {local.type_pp} := {local.value_pp}; {expression}"
                )
                continue
            if local.is_instance or local.binder_role in {
                ":instImplicit",
                "instImplicit",
                "instance",
            }:
                binder = f"[{name} : {local.type_pp}]"
            elif local.binder_role in {":implicit", "implicit"}:
                binder = f"{{{name} : {local.type_pp}}}"
            elif local.binder_role in {
                ":strictImplicit",
                "strictImplicit",
                "strict-implicit",
            }:
                binder = f"{{{{{name} : {local.type_pp}}}}}"
            else:
                binder = f"({name} : {local.type_pp})"
            expression = f"∀ {binder}, {expression}"
        local_names.reverse()
        return GoalReplaySpec(expression=expression, local_names=tuple(local_names))


_SEED_LOCAL_RE = re.compile(r"^\s*([^:]+?)\s*:\s*(.+?)\s*$", re.DOTALL)


def seed_replay_spec(seed: LeanGoalSeed) -> GoalReplaySpec:
    """Render a text theorem seed as explicit binders for first serialization."""
    expression = seed.expression
    local_names: list[str] = []
    for hypothesis in reversed(seed.hypotheses):
        match = _SEED_LOCAL_RE.match(hypothesis)
        if match is None:
            raise ValueError(
                f"Cannot replay hypothesis {hypothesis!r}; expected 'name : type'."
            )
        name, type_pp = match.groups()
        name = name.strip()
        local_names.append(name)
        expression = f"∀ ({name} : {type_pp.strip()}), {expression}"
    local_names.reverse()
    return GoalReplaySpec(expression=expression, local_names=tuple(local_names))


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
