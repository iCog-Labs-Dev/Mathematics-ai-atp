"""Check that the goal Lean built from a replay is the goal the seed described.

A successful ``goal.start`` and ``intro`` only show that Lean accepted the input.
For structured seeds the serialized result is compared with the source raw
expressions: same constants, literals, levels, binder kinds and roles, and
local references that line up by position. Binder names are not compared, and a
``_`` projection parameter matches whatever Lean inferred. Local display names
that differ are reported, not rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from . import raw as R
from .errors import ReplayFailure, ReplayInputError
from .seed import GoalReplaySpec, LeanGoalSeed, StructuredLocal


@dataclass(frozen=True)
class ReplayValidation:
    name_drift: tuple[tuple[str, str], ...]
    source_indices: tuple[int, ...]
    replay_indices: tuple[int, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "name_drift": [list(pair) for pair in self.name_drift],
            "source_context_indices": list(self.source_indices),
            "replay_context_indices": list(self.replay_indices),
        }


def _mismatch(detail: str, where: str) -> ReplayInputError:
    return ReplayInputError(ReplayFailure.SEMANTIC_MISMATCH, detail, field=where)


def _same_level(expected: R.Level, actual: R.Level, levels: dict[str, str]) -> bool:
    if isinstance(expected, R.LevelParam):
        return isinstance(actual, R.LevelParam) and levels.get(expected.name, expected.name) == actual.name
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, R.LevelNum):
        return expected.value == actual.value
    if isinstance(expected, R.LevelOffset):
        return expected.offset == actual.offset and _same_level(expected.base, actual.base, levels)
    return (
        expected.imax == actual.imax
        and _same_level(expected.left, actual.left, levels)
        and _same_level(expected.right, actual.right, levels)
    )


def same_expression(
    expected: R.RawExpr, actual: R.RawExpr, fvars: dict[str, str], levels: dict[str, str]
) -> bool:
    stack = [(expected, actual)]
    while stack:
        left, right = stack.pop()
        if isinstance(left, R.Placeholder):
            continue
        if type(left) is not type(right):
            return False
        if isinstance(left, R.Const):
            ok = left.name == right.name
        elif isinstance(left, R.FVar):
            ok = fvars.get(left.id) == right.id
        elif isinstance(left, R.BVar):
            ok = left.index == right.index
        elif isinstance(left, (R.NatLit, R.StrLit)):
            ok = left.value == right.value
        elif isinstance(left, R.Sort):
            ok = _same_level(left.level, right.level, levels)
        elif isinstance(left, R.App):
            ok = len(left.args) == len(right.args)
        elif isinstance(left, R.Binder):
            ok = left.kind == right.kind and left.info == right.info
        else:
            ok = True
        if not ok:
            return False
        stack.extend(zip(R.children(left), R.children(right)))
    return True


def validate_model_input(
    seed: LeanGoalSeed, target_model_sexp: str, local_model_sexps: Sequence[str]
) -> None:
    """Raise ``MODEL_MISMATCH`` unless the model sees the expressions the source recorded.

    Lean can rebuild the same goal while the normalized expressions differ,
    for example where the source kept metadata that source syntax cannot
    recreate. Such a replay would show the model different input than the
    prepared data, so it is not admitted.
    """
    expected = [("target", seed.model_target, target_model_sexp)]
    expected += [
        (f"local {position} ({source.name})", source.model_type, actual)
        for position, (source, actual) in enumerate(zip(seed.locals, local_model_sexps))
    ]
    for where, recorded, actual in expected:
        if recorded is not None and recorded != actual:
            start = next(
                (i for i, (a, b) in enumerate(zip(recorded, actual)) if a != b),
                min(len(recorded), len(actual)),
            )
            raise ReplayInputError(
                ReplayFailure.MODEL_MISMATCH,
                "The replayed model S-expression differs from the source near "
                f"{recorded[max(0, start - 40):start + 40]!r}.",
                field=where,
            )


def validate_replay(
    seed: LeanGoalSeed,
    spec: GoalReplaySpec,
    target_sexp: str | None,
    replayed: Sequence[StructuredLocal],
) -> ReplayValidation:
    """Raise ``SEMANTIC_MISMATCH`` unless ``replayed`` is the seed's goal."""
    if len(replayed) != len(seed.locals):
        raise _mismatch(
            f"Replay produced {len(replayed)} locals for {len(seed.locals)} declarations.",
            "context",
        )
    drift = []
    for position, (source, local, shown) in enumerate(
        zip(seed.locals, replayed, spec.expected_names)
    ):
        where = f"local {position} ({source.name})"
        if local.is_let != source.is_let:
            raise _mismatch("A local definition was not replayed as one.", where)
        if source.role is not None and local.binder_role != source.role:
            raise _mismatch(
                f"Binder role {local.binder_role} replaced {source.role}.", where
            )
        if local.name != shown:
            drift.append((shown, local.name))

    if seed.origin == "structured":
        fvars = {
            source.internal_id: local.internal_id
            for source, local in zip(seed.locals, replayed)
        }
        levels = dict(spec.level_map)
        pairs = [(seed.target, target_sexp, "target")]
        for position, (source, local) in enumerate(zip(seed.locals, replayed)):
            where = f"local {position} ({source.name})"
            pairs.append((source.type, local.type_sexp, where))
            if source.value is not None:
                pairs.append((source.value, local.value_sexp, f"{where} value"))
        for expected, actual_sexp, where in pairs:
            if not actual_sexp:
                raise _mismatch("Lean returned no raw expression to compare.", where)
            try:
                actual = R.decode_raw(actual_sexp, where)
            except ReplayInputError as exc:
                raise _mismatch(f"Replayed expression is not closed: {exc.detail}", where) from exc
            if not same_expression(expected, actual, fvars, levels):
                raise _mismatch("Replayed expression differs from the source.", where)

    return ReplayValidation(
        tuple(drift),
        tuple(local.context_index for local in seed.locals if local.context_index is not None),
        tuple(local.context_index for local in replayed),
    )
