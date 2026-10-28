from __future__ import annotations

import hashlib
import json
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from maths_ai.lean_replay import (
    GoalReplaySpec,
    LeanGoalSeed,
    ReplayFailure,
    ReplayInputError,
    StructuredLocal,
    render_replay,
    seed_from_records,
)


class GoalLocal(BaseModel):
    """One serialized Lean local declaration and its normalized model view.

    ``user_name``, ``context_index`` and the model S-expressions are the model
    identity the graph and fingerprint read. ``internal_name`` and the
    ``execution_*`` fields are the live Lean identity (``None`` when it equals
    the model one); see ``model_identity``. ``type_sexp``/``value_sexp`` are
    Pantograph's raw expressions, kept only to replay the goal.
    """

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
    type_sexp: Optional[str] = None
    value_sexp: Optional[str] = None
    execution_name: Optional[str] = None
    execution_index: Optional[int] = None

    @property
    def execution_or_live_name(self) -> str:
        return self.execution_name if self.execution_name is not None else self.user_name

    @property
    def execution_or_live_index(self) -> int:
        return self.execution_index if self.execution_index is not None else self.context_index

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
    ``goal_sexp`` is the target's raw expression, kept for replay only.
    """

    expression: str
    hypotheses: List[str] = Field(default_factory=list)
    goal_model_sexp: Optional[str] = None
    goal_sexp: Optional[str] = None
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

    def replay_seed(self) -> LeanGoalSeed:
        """The structured seed that recreates this goal from its raw expressions.

        Replay uses the live names and indices; the model identity rides along
        so the replayed goal shows the model the same context.
        """
        self.require_model_state()
        locals_ = self.locals or []
        if self.goal_sexp is None or any(local.type_sexp is None for local in locals_):
            raise ReplayInputError(
                ReplayFailure.MISSING_CONTEXT,
                "This goal has no raw execution expressions to replay.",
            )
        records = [
            StructuredLocal(
                internal_id=local.internal_name,
                name=local.execution_or_live_name or "_",
                context_index=local.execution_or_live_index,
                binder_role=local.binder_role,
                is_instance=local.is_instance,
                is_let=local.is_let,
                type_sexp=local.type_sexp,
                value_sexp=local.value_sexp,
                model_name=local.user_name,
                model_index=local.context_index,
                model_type_sexp=local.type_model_sexp,
            )
            for local in locals_
        ]
        return seed_from_records(
            self.goal_sexp,
            records,
            source={"origin": "canonical-goal", "state": self.state_fingerprint()},
            model_target=self.goal_model_sexp,
        )

    def replay_spec(self) -> GoalReplaySpec:
        """Render the goal back into a closed Lean statement."""
        return render_replay(self.replay_seed())


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


class ArgumentRef(BaseModel):
    """What one rendered tactic argument refers to.

    ``local`` arguments name a local declaration by its live Lean ID
    (``internal_name``) in the goal the tactic was proposed for, so a replay in
    another Lean lineage can find the same declaration under its new name.
    ``global`` arguments are environment names and ``literal`` ones are passed
    through unchanged.
    """

    kind: Literal["local", "global", "literal"]
    text: str
    internal_name: Optional[str] = None

    @model_validator(mode="after")
    def _validate_ref(self) -> "ArgumentRef":
        if (self.kind == "local") != (self.internal_name is not None):
            raise ValueError("Exactly the local arguments carry an internal name.")
        return self


class TacticCandidate(BaseModel):
    """A single ranked tactic prediction from the GNN engine.

    ``argument_refs``, when present, gives one ``ArgumentRef`` per argument.
    """

    tactic_name: str
    arguments: List[str] = Field(default_factory=list)
    probability: float
    argument_refs: Optional[List[ArgumentRef]] = None

    @model_validator(mode="after")
    def _validate_refs(self) -> "TacticCandidate":
        if self.argument_refs is not None and (
            [ref.text for ref in self.argument_refs] != list(self.arguments)
        ):
            raise ValueError("Argument references must match the arguments.")
        return self


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
