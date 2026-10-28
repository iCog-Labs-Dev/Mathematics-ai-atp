"""Candidate recipes, argument re-binding and the portable proof artifact.

A recipe is a choice of accepted tactic attempts that may solve an obligation:
one attempt for the obligation and a recipe for each of its child slots. It is
a claim to check, not evidence. Only a sequential replay that the kernel
certifies turns a recipe into a proof.

The artifact a certified search exports is portable: the statement it proves,
each replayed step with its arguments as declaration keys, the environment it
was checked in, and the kernel receipt. A fresh session can replay it and get
a new certificate. It contains no server handles or Lean IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Protocol, Sequence

from maths_ai.data_models.proof_components import ArgumentRef, Goal, TacticCandidate

from .execution_types import DeclarationKey, ObligationKey

__all__ = [
    "ArtifactArgument",
    "ArtifactStep",
    "BindingConflict",
    "ChildRecord",
    "RecipeSource",
    "RecipeStep",
    "ReplayArtifact",
    "rebind_arguments",
]

ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class ChildRecord:
    """One child slot of an accepted attempt: its kind, how many locals the
    attempt introduced into it, and the graph node that searches it."""

    slot: int
    kind: str
    introduced: int
    node: int


@dataclass(frozen=True)
class RecipeStep:
    """One accepted attempt, as replay needs it.

    ``command`` is what ran when the attempt was explored; a replay renders the
    command again from ``arguments`` in its own context. ``dependencies`` are
    the obligations a childless attempt's proof still refers to: it closes its
    goal only once those are solved in the same execution.
    """

    attempt: int
    node: int
    tactic: str
    arguments: tuple[ArgumentRef, ...]
    command: str
    children: tuple[ChildRecord, ...]
    dependencies: tuple[ObligationKey, ...] = ()


class RecipeSource(Protocol):
    """What replay reads from the search graph."""

    def alternatives(self, node: int) -> Sequence[RecipeStep]:
        """Candidate-ready attempts of ``node``, best first."""

    def observation(self, node: int) -> Goal:
        """The model observation the node's attempts were proposed for."""

    def declarations(self, node: int) -> Mapping[str, DeclarationKey]:
        """Live local IDs of the node's own snapshot mapped to their keys."""


class BindingConflict(ValueError):
    """A recorded argument has no unambiguous declaration in the replay context."""


def _visible_name(goal: Goal, internal: str) -> str:
    locals_ = goal.locals or []
    for position, local in enumerate(locals_):
        if local.internal_name != internal:
            continue
        name = local.execution_or_live_name
        later = [other.execution_or_live_name for other in locals_[position + 1 :]]
        if not name or "✝" in name or name in later:
            raise BindingConflict(
                f"Local {name or '_'} cannot be named unambiguously in the replay context."
            )
        return name
    raise BindingConflict(f"Local {internal} is not in the replay context.")


def rebind_arguments(
    refs: Sequence[ArgumentRef],
    source: Mapping[str, DeclarationKey],
    target_goal: Goal,
    target: Mapping[str, DeclarationKey],
) -> list[ArgumentRef]:
    """The recorded arguments, re-pointed at the same declarations in ``target_goal``.

    Each local is followed from its live ID in the explored context to its
    declaration key, then to the local with that key in the replay context,
    which supplies the name to render. Global and literal arguments pass
    through. Display names are never matched against each other.
    """
    by_key = {key: name for name, key in target.items()}
    bound = []
    for ref in refs:
        if ref.kind != "local":
            bound.append(ref)
            continue
        key = source.get(ref.internal_name)
        if key is None or key not in by_key:
            raise BindingConflict(f"No declaration for argument {ref.text!r} in the replay context.")
        internal = by_key[key]
        bound.append(
            ArgumentRef(kind="local", text=_visible_name(target_goal, internal), internal_name=internal)
        )
    return bound


def candidate_for(step: RecipeStep, arguments: Sequence[ArgumentRef]) -> TacticCandidate:
    return TacticCandidate(
        tactic_name=step.tactic,
        arguments=[ref.text for ref in arguments],
        probability=1.0,
        argument_refs=list(arguments),
    )


@dataclass(frozen=True)
class ArtifactArgument:
    kind: str
    text: str
    declaration: Optional[DeclarationKey] = None

    def as_dict(self) -> dict:
        value = {"kind": self.kind, "text": self.text}
        if self.declaration is not None:
            value["declaration"] = list(_key_tuple(self.declaration))
        return value


@dataclass(frozen=True)
class ArtifactStep:
    obligation: ObligationKey
    attempt: int
    tactic: str
    arguments: tuple[ArtifactArgument, ...]
    children: tuple[tuple[int, str, int], ...]  # (slot, kind, introduced)

    def as_dict(self) -> dict:
        return {
            "obligation": [self.obligation.attempt, self.obligation.slot],
            "attempt": self.attempt,
            "tactic": self.tactic,
            "arguments": [argument.as_dict() for argument in self.arguments],
            "children": [list(child) for child in self.children],
        }


def _key_tuple(key: DeclarationKey) -> tuple:
    return (key.attempt, key.slot, key.position)


@dataclass(frozen=True)
class ReplayArtifact:
    """A certified proof as data: statement, replay steps, environment, receipt."""

    statement: dict
    steps: tuple[ArtifactStep, ...]
    environment: dict
    receipt: dict
    version: int = ARTIFACT_VERSION

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "statement": dict(self.statement),
            "steps": [step.as_dict() for step in self.steps],
            "environment": dict(self.environment),
            "receipt": dict(self.receipt),
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "ReplayArtifact":
        if payload.get("version") != ARTIFACT_VERSION:
            raise ValueError(f"Unsupported replay artifact version {payload.get('version')!r}.")
        steps = []
        for raw in payload["steps"]:
            attempt, slot = raw["obligation"]
            arguments = []
            for argument in raw["arguments"]:
                declaration = argument.get("declaration")
                arguments.append(
                    ArtifactArgument(
                        argument["kind"],
                        argument["text"],
                        DeclarationKey(*declaration) if declaration is not None else None,
                    )
                )
            steps.append(
                ArtifactStep(
                    ObligationKey(attempt, slot),
                    raw["attempt"],
                    raw["tactic"],
                    tuple(arguments),
                    tuple(tuple(child) for child in raw["children"]),
                )
            )
        return cls(
            statement=dict(payload["statement"]),
            steps=tuple(steps),
            environment=dict(payload["environment"]),
            receipt=dict(payload["receipt"]),
        )
