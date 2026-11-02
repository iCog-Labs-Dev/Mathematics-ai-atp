"""Request and response models of the MCTS proving service."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr

ProposalMode = Literal["sample", "greedy"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProveRequest(_Strict):
    """One goal with its ordered local declarations, searched under a server profile."""

    goal: StrictStr = Field(min_length=1)
    hypotheses: list[StrictStr] = Field(default_factory=list)
    search_profile: Optional[StrictStr] = None
    proposal_mode: Optional[ProposalMode] = None
    seed: Optional[StrictInt] = Field(default=None, ge=0)
    top_k_tactics: Optional[StrictInt] = Field(default=None, ge=1)
    max_depth: Optional[StrictInt] = Field(default=None, ge=1)
    max_nodes: Optional[StrictInt] = Field(default=None, ge=1)
    num_simulations: Optional[StrictInt] = Field(default=None, ge=1)
    timeout_s: Optional[StrictFloat | StrictInt] = Field(default=None, gt=0)
    queue_timeout_s: Optional[StrictFloat | StrictInt] = Field(default=None, gt=0)


class Proposal(BaseModel):
    mode: str
    seed: int
    derivation: str


class ProveResponse(BaseModel):
    """A finished search. ``certified`` is true only for a kernel-certified root.

    ``candidate`` reports graph readiness, which is not a proof. ``proof_trace``
    keeps the replayed commands, nested obligations and discharged entries.
    """

    model_config = ConfigDict(extra="allow")

    certified: bool = False
    candidate: bool = False
    outcome: str
    end_reason: Optional[str] = None
    proof_trace: Optional[dict[str, Any]] = None
    certificate: Optional[dict[str, Any]] = None
    artifact: Optional[dict[str, Any]] = None
    search_profile: str
    search_settings: Optional[dict[str, Any]] = None
    proposal: Optional[Proposal] = None
    input_identity: Optional[str] = None
    execution: Optional[dict[str, Any]] = None
    search: Optional[dict[str, Any]] = None
    timing: dict[str, float]
    runtime: dict[str, Any]


class EvaluationRequest(_Strict):
    """An evaluation of an allowlisted dataset profile under a search profile."""

    dataset_profile: StrictStr
    search_profile: Optional[StrictStr] = None
    proposal_mode: Optional[ProposalMode] = None
    seed: Optional[StrictInt] = Field(default=None, ge=0)
    repeats: Optional[StrictInt] = Field(default=None, ge=1)
    top_k_tactics: Optional[StrictInt] = Field(default=None, ge=1)
    max_depth: Optional[StrictInt] = Field(default=None, ge=1)
    max_nodes: Optional[StrictInt] = Field(default=None, ge=1)
    num_simulations: Optional[StrictInt] = Field(default=None, ge=1)
    timeout_s: Optional[StrictFloat | StrictInt] = Field(default=None, gt=0)
    deadline_s: Optional[StrictFloat | StrictInt] = Field(default=None, gt=0)


class Job(BaseModel):
    """Persisted state of an evaluation job."""

    id: str
    state: str
    phase: Optional[str]
    created_at: str
    updated_at: str
    finished_at: Optional[str]
    settings: dict[str, Any]
    progress: dict[str, Any]
    provenance: Optional[dict[str, Any]]
    error_category: Optional[str]
    error_detail: Optional[str]
    result_partial: Optional[bool]


class Health(BaseModel):
    status: Literal["ok"] = "ok"


class Readiness(BaseModel):
    ready: bool
    runtime: str
    generation: int
    error: Optional[str]
    status: dict[str, Any]
    provenance: Optional[dict[str, Any]]
    queue: dict[str, int]


class ErrorBody(BaseModel):
    category: str
    detail: str
