"""How a search proposes actions, and where its randomness comes from.

A ``ProposalSpec`` is the single description of a search's proposal policy:
``mode`` is ``"sample"`` (tactic and every argument drawn from the policy) or
``"greedy"`` (tactic and every argument chosen by argmax, step by step), and
``seed`` seeds the search's own ``torch.Generator``. No search draws from the
global Torch RNG, so evaluation never shifts the training sequence and a
search's draws do not depend on which worker ran it or in what order.

Seeds are derived, never assigned by scheduling: ``derive_search_seed`` hashes
a namespace (``train`` or ``eval``), a base seed, a pass index (training round
or evaluation repeat), the input's identity and its occurrence among equal
inputs. Draws repeat for a fixed seed on the same device; equality across
devices is not promised.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Sequence

from maths_ai.lean_replay import LeanGoalSeed, ReplayInputError, render_replay

__all__ = [
    "PROPOSAL_MODES",
    "SEED_DERIVATION_VERSION",
    "ProposalSpec",
    "derive_search_seed",
    "input_identity",
    "proposal_specs",
]

PROPOSAL_MODES = ("sample", "greedy")
SEED_DERIVATION_VERSION = "sha256-v1"


@dataclass(frozen=True)
class ProposalSpec:
    """One search's proposal mode and resolved seed (plain data, no generator)."""

    mode: str
    seed: int
    derivation: str = SEED_DERIVATION_VERSION

    def __post_init__(self) -> None:
        if self.mode not in PROPOSAL_MODES:
            raise ValueError(
                f"Unknown proposal mode {self.mode!r} (use 'sample' or 'greedy')."
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(f"A proposal seed must be a non-negative integer, got {self.seed!r}.")

    @property
    def greedy(self) -> bool:
        return self.mode == "greedy"

    def as_dict(self) -> dict[str, object]:
        return {"mode": self.mode, "seed": self.seed, "derivation": self.derivation}


def input_identity(seed: LeanGoalSeed) -> str:
    """Stable identity of one search input: its replay fingerprint.

    The fingerprint covers the rendered replay and the seed's source
    provenance. A seed that cannot be rendered never reaches a search (its
    ``prove`` raises ``ReplayInputError``), so it is identified by a digest of
    its own fields instead, which is equally stable.
    """
    try:
        return render_replay(seed).fingerprint
    except ReplayInputError:
        return "unrendered:" + hashlib.sha256(repr(seed).encode("utf-8")).hexdigest()


def derive_search_seed(
    namespace: str, base_seed: int, pass_index: int, identity: str, occurrence: int = 0
) -> int:
    """A 63-bit seed from the request's stable coordinates only."""
    body = [SEED_DERIVATION_VERSION, namespace, int(base_seed), int(pass_index), identity, int(occurrence)]
    digest = hashlib.sha256(json.dumps(body).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & (2**63 - 1)


def proposal_specs(
    seeds: Iterable[LeanGoalSeed],
    *,
    mode: str,
    namespace: str,
    base_seed: int,
    pass_index: int,
) -> list[ProposalSpec]:
    """One spec per input, in input order.

    Equal inputs are told apart by their occurrence: the n-th copy of an
    identity in ``seeds`` gets occurrence n, so the specs depend on the input
    list, not on how its items are later scheduled.
    """
    seen: dict[str, int] = {}
    specs: list[ProposalSpec] = []
    for seed in seeds:
        identity = input_identity(seed)
        occurrence = seen.get(identity, 0)
        seen[identity] = occurrence + 1
        specs.append(
            ProposalSpec(
                mode=mode,
                seed=derive_search_seed(namespace, base_seed, pass_index, identity, occurrence),
            )
        )
    return specs


def require_aligned(specs: Sequence[ProposalSpec], batch: Sequence[object]) -> None:
    if len(specs) != len(batch):
        raise ValueError(
            f"Expected one proposal spec per input: {len(specs)} specs for {len(batch)} inputs."
        )
