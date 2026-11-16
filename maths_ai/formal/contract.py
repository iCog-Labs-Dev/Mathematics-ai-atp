"""Formal protocol version 2: the request, lifecycle, result and replay types.

These are the provider's strict consumers of the persistent-atp formal
contract (``persistent_atp_formal_service_contract_and_mcts_integration.md``).
Unknown fields are rejected everywhere, numeric fields do not accept booleans
or strings, and the input is a discriminated union on ``input.kind``: a
request carries exactly one of a source-backed or a text-based input.

``budget.max_steps`` is the only request-level tactic allowance; the removed
``search_policy.max_tactic_steps`` is rejected as an unknown field.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr

from .canonical import SHA256_PATTERN

PROTOCOL_VERSION = 2
MCTS_BACKEND = "mcts-actor-critic"
ACCEPTED_BACKEND = "accepted-gnn"
BackendName = Literal["mcts-actor-critic", "accepted-gnn"]

Sha256 = Annotated[str, Field(pattern=SHA256_PATTERN)]
RunId = Annotated[str, Field(pattern=r"^[^/]+/fr-[1-9][0-9]*$")]
LeanName = Annotated[
    str, Field(min_length=1, max_length=512, pattern=r"^[^\s.][^\s]*$")
]

# Termination reasons, separated from logical outcome and infrastructure faults.
TerminationReason = Literal[
    "certified",  # kernel-certified in search (MCTS)
    "search-closed",  # every goal closed by search, not kernel-certified (accepted)
    "step-budget",
    "node-budget",
    "simulation-budget",
    "deadline",
    "frontier-exhausted",
    "alternatives-exhausted",
    "lean-resource",
    "invalid-input",
    "environment-mismatch",
    "crash",
    "cancelled",
    "interrupted",
]
# The persistent-atp canonical dispositions (formal-search-result.schema.json).
Disposition = Literal[
    "searching",
    "proved-pending-replay",
    "budget-exhausted",
    "stagnated",
    "counterexample",
    "invalid-request",
    "environment-error",
    "internal-error",
    "cancelled",
]
RunStatus = Literal["admitting", "queued", "running", "cancelling", "finished", "interrupted"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


# Request ---------------------------------------------------------------------


class SourceInput(_Strict):
    """A declaration in a caller-supplied source artifact.

    The provider resolves the artifact by content hash, elaborates the named
    declaration's signature in its original context, and searches it with the
    declaration's outer binders introduced. The declaration's own proof is
    never used and the declaration is never offered as a premise.
    """

    kind: Literal["source"]
    artifact_hash: Sha256
    declaration: LeanName
    starting_state: Literal["binders-introduced"]


class GoalInput(_Strict):
    """A target with its ordered local context, elaborated by the provider."""

    kind: Literal["goal"]
    goal_text: StrictStr = Field(min_length=1, max_length=20_000)
    hypotheses: list[Annotated[str, Field(min_length=1, max_length=20_000)]] = Field(max_length=512)
    universe_levels: list[Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]] = Field(
        max_length=64
    )


FormalInput = Annotated[Union[SourceInput, GoalInput], Field(discriminator="kind")]


class BackendSelection(_Strict):
    name: BackendName
    profile: StrictStr = Field(min_length=1, max_length=128)
    proposal_mode: Optional[Literal["sample", "greedy"]] = None
    seed: Optional[StrictInt] = Field(default=None, ge=0, le=2**63 - 1)


class SearchPolicy(_Strict):
    policy_name: Optional[StrictStr] = Field(default=None, max_length=128)
    deny_sorry: Literal[True]


class Budget(_Strict):
    max_steps: StrictInt = Field(ge=1)
    wall_clock_ms: Optional[StrictInt] = Field(default=None, ge=1)


class VerificationBudget(_Strict):
    max_steps: StrictInt = Field(ge=1)
    wall_clock_ms: Optional[StrictInt] = Field(default=None, ge=1)


class CorpusManifest(_Strict):
    premise_corpus_artifact: Optional[Sha256] = None
    retrieval_policy: Optional[StrictStr] = None


class ModelBundle(_Strict):
    tactic_model_artifact: Optional[Sha256] = None
    argument_model_artifact: Optional[Sha256] = None
    premise_model_artifact: Optional[Sha256] = None


class FormalSearchRequest(_Strict):
    protocol_version: Literal[2]
    proof_id: StrictStr = Field(min_length=1)
    claim_id: Annotated[str, Field(pattern=r"^[^/]+/c-[1-9][0-9]*$")]
    formal_declaration_id: Annotated[str, Field(pattern=r"^[^/]+/fd-[1-9][0-9]*$")]
    run_id: RunId
    base_revision: StrictInt = Field(ge=0)
    lease_id: StrictStr
    fencing_token: StrictInt = Field(ge=1)
    input: FormalInput
    environment_id: StrictStr = Field(min_length=1)
    environment_hash: Sha256
    backend: BackendSelection
    deployment_id: StrictStr = Field(min_length=1, max_length=256)
    search_policy: SearchPolicy
    budget: Budget
    corpus_manifest: Optional[CorpusManifest] = None
    model_bundle: Optional[ModelBundle] = None


class FormalReplayRequest(_Strict):
    """Independent replay of a run's stored certificate, bound to its evidence."""

    protocol_version: Literal[2]
    run_id: RunId
    certificate_hash: Sha256
    input_artifact_hash: Sha256
    source_artifact_hash: Sha256
    environment_hash: Sha256
    budget: VerificationBudget


# Responses -------------------------------------------------------------------


class ResolvedDeployment(_Strict):
    deployment_id: str
    backend: BackendName
    profile: str
    profile_config_hash: Sha256
    environment_id: str
    environment_hash: Sha256
    model: dict[str, Any]


class AdmittedBudget(_Strict):
    requested_max_steps: int
    admitted_max_steps: int
    max_steps_reduced: StrictBool
    requested_wall_clock_ms: Optional[int]
    admitted_wall_clock_ms: int
    wall_clock_reduced: StrictBool
    consumed_steps: int
    # False once an attempt's cost was unknown and its whole reservation was charged.
    consumed_steps_exact: StrictBool
    remaining_steps: int
    verification_max_steps: Optional[int]


class Admission(_Strict):
    """The admission acknowledgement: the run's effective settings and identities."""

    protocol_version: Literal[2] = 2
    run_id: str
    status: RunStatus
    request_digest: Sha256
    input_kind: Literal["source", "goal"]
    # Null while a source input is still being admitted.
    input_artifact_hash: Optional[Sha256]
    source_artifact_hash: Optional[Sha256]
    source_origin: Optional[Literal["caller-artifact", "generated-from-text"]]
    resolved: ResolvedDeployment
    budget: AdmittedBudget
    attempts: int
    retry_from: Literal["bound-initial-input"] = "bound-initial-input"


class ReplayAttestation(_Strict):
    protocol_version: Literal[2] = 2
    status: Literal["verified", "rejected", "inconclusive"]
    reason: Optional[str] = None
    run_id: str
    certificate_hash: Sha256
    input_artifact_hash: Sha256
    source_artifact_hash: Sha256
    source_origin: Literal["caller-artifact", "generated-from-text"]
    environment_hash: str
    backend: BackendName
    verification_cost: dict[str, Any]


class Termination(_Strict):
    reason: TerminationReason
    detail: Optional[str] = None


StateRef = Annotated[str, Field(pattern=r"^[^/]+/fs-[1-9][0-9]*$")]


class FormalState(_Strict):
    state_id: StateRef
    kind: Literal["goal"]
    goal_text: str
    exact_state_hash: Sha256
    semantic_signature: Sha256
    status: Literal["open", "formally-closed"]


class TacticEdge(_Strict):
    tactic_id: Annotated[str, Field(pattern=r"^[^/]+/ta-[1-9][0-9]*$")]
    source_state_id: StateRef
    tactic_label: str
    tactic_args: Optional[list[str]] = None
    executor_result: Literal["lean-accepted"]
    subgoal_count: int = Field(ge=0)
    produced_goal_ids: list[StateRef]
    status: Literal["closed"]


class CertificateRef(_Strict):
    artifact_hash: Sha256
    status: Literal["replay-pending"]
    producer_run_id: str


class ArtifactRef(_Strict):
    sha256: Sha256
    role: Literal["certificate", "admitted-input", "source", "generated-source", "search-summary"]


class AttemptCost(_Strict):
    attempt: Optional[int] = None
    reserved_steps: Optional[int] = None
    tactic_steps: Optional[int] = None
    tactic_steps_exact: Optional[bool] = None
    wall_clock_ms: Optional[int] = None
    admission: Optional[Literal["rejected", "cancelled"]] = None
    interrupted: Optional[bool] = None


class BudgetUsed(_Strict):
    tactic_steps: int = Field(ge=0)
    tactic_steps_exact: StrictBool
    attempt: AttemptCost


class ResultBudget(_Strict):
    admitted_max_steps: int
    remaining_steps: int
    admitted_wall_clock_ms: int


class ResultBackend(_Strict):
    name: BackendName
    profile: str
    deployment_id: str
    profile_config_hash: Sha256


class CoordinatorBinding(_Strict):
    claim_id: str
    base_revision: int
    lease_id: str
    fencing_token: int


class FormalResult(_Strict):
    """A finished attempt's canonical result (``result`` of the run view)."""

    protocol_version: Literal[2]
    run_id: str
    proof_id: str
    declaration_id: str
    environment_hash: Sha256
    disposition: Disposition
    termination: Termination
    root_state_id: StateRef
    states: list[FormalState]
    tactic_edges: list[TacticEdge]
    obstructions: list[Any]
    artifacts: list[ArtifactRef]
    budget_used: BudgetUsed
    budget: ResultBudget
    input_kind: Literal["source", "goal"]
    input_artifact_hash: Optional[Sha256]
    source_artifact_hash: Optional[Sha256]
    source_origin: Optional[Literal["caller-artifact", "generated-from-text"]]
    backend: ResultBackend
    request_digest: Sha256
    coordinator_binding: CoordinatorBinding
    attempts: int
    certificate: Optional[CertificateRef] = None


class ReplaySummary(_Strict):
    replay: int
    status: Literal["verified", "rejected", "inconclusive"]


class RunView(Admission):
    """``GET /formal-search/{run_id}`` and the cancel/retry responses."""

    disposition: Optional[Disposition]
    termination: Optional[TerminationReason]
    cancel_requested: StrictBool
    result: Optional[FormalResult]
    replays: list[ReplaySummary]


class ArtifactUpload(_Strict):
    protocol_version: Literal[2] = 2
    artifact_hash: Sha256


class Capabilities(BaseModel):
    """``GET /formal-capabilities``; deployments are keyed by ``deployment_id``."""

    model_config = ConfigDict(extra="forbid")

    protocol_versions: list[Literal[2]]
    registry_hash: Sha256
    deployments: dict[str, dict[str, Any]]
    tactic_step: str
    budget_fields: list[str]
    wall_clock: str
    cancellation: str
    retry: str
    frontier_resume: Literal[False]
    replay: str
    artifact_upload: dict[str, Any]


class FieldError(_Strict):
    loc: list[Union[str, int]]
    type: str
    msg: str


class ErrorBody(_Strict):
    """Every error response: a stable ``category`` and a human-readable ``detail``.

    ``invalid-request`` carries the rejected fields as a list instead.
    """

    protocol_version: Literal[2] = 2
    category: str
    detail: Union[str, list[FieldError]]


def json_schemas() -> dict[str, dict[str, Any]]:
    """The published JSON Schemas of the provider's version-2 types."""
    return {
        "formal-search-request.v2": FormalSearchRequest.model_json_schema(),
        "formal-replay-request.v2": FormalReplayRequest.model_json_schema(),
        "formal-admission.v2": Admission.model_json_schema(),
        "formal-replay-attestation.v2": ReplayAttestation.model_json_schema(),
        "formal-error.v2": ErrorBody.model_json_schema(),
        "formal-run.v2": RunView.model_json_schema(),
        "formal-result.v2": FormalResult.model_json_schema(),
        "formal-capabilities.v2": Capabilities.model_json_schema(),
        "formal-artifact-upload.v2": ArtifactUpload.model_json_schema(),
    }
