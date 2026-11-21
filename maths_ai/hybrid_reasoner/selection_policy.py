"""Edge-selection scoring for multi-simulation proof search.

PUCT (predictor upper confidence bound applied to trees), as used by
HTPS: balances the empirical action value Q(g,t) against the policy
prior P(g,t), with an exploration bonus that decays as the edge
accumulates visits. Virtual loss enters through both terms — it lowers
Q (pending simulations count as losses) and inflates the visit count in
the bonus denominator — so simulations selected in the same batch spread
across different branches.

This module is also the single authority for the search-mode contract:
``resolve_search_params`` validates the (selection_policy, budget)
combination and ``resolve_widening`` the progressive-widening bounds, for
both the reasoner constructor and the training config.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from maths_ai.hybrid_reasoner.hypergraph import EdgeVisitStats


def resolve_search_params(
    selection_policy: str,
    num_simulations: Optional[int],
    sim_batch_size: Optional[int],
    puct_c: Optional[float],
) -> Tuple[int, int, float]:
    """Validate the (policy, budget) combination and resolve None defaults.

    legacy: all three budget fields must be None — the legacy best-first
            loop has no simulation budget, so an explicit value would be
            silently ignored, and the run would proceed with settings the
            user believed were active. Returns ``(1, 1, 0.0)`` placeholders
            (never read by the legacy path).
    puct:   ``num_simulations`` is required (a PUCT search has no natural
            simulation count); ``sim_batch_size`` defaults to 4 and
            ``puct_c`` to 1.0.
    rp:     reserved for the deferred regularized-policy variant —
            ``NotImplementedError``.
    anything else: ``ValueError``.
    """
    if selection_policy == "legacy":
        explicit = {
            name: value
            for name, value in (
                ("num_simulations", num_simulations),
                ("sim_batch_size", sim_batch_size),
                ("puct_c", puct_c),
            )
            if value is not None
        }
        if explicit:
            raise ValueError(
                "selection_policy='legacy' runs the best-first loop, which has no "
                f"simulation budget; remove the explicit setting(s) {sorted(explicit)} "
                "or set selection_policy='puct'"
            )
        return 1, 1, 0.0
    if selection_policy == "puct":
        if num_simulations is None:
            raise ValueError(
                "selection_policy='puct' requires an explicit num_simulations "
                "(a PUCT search has no natural simulation count)"
            )
        resolved = (
            num_simulations,
            4 if sim_batch_size is None else sim_batch_size,
            1.0 if puct_c is None else puct_c,
        )
        _require_positive_int("num_simulations", resolved[0])
        _require_positive_int("sim_batch_size", resolved[1])
        if isinstance(resolved[2], bool) or not isinstance(resolved[2], (int, float)) or not (
            math.isfinite(resolved[2]) and resolved[2] >= 0
        ):
            raise ValueError(f"puct_c must be a finite non-negative number, got {resolved[2]!r}")
        return resolved
    if selection_policy == "rp":
        raise NotImplementedError(
            "selection_policy='rp' (regularized policy) is reserved but not implemented"
        )
    raise ValueError(
        f"Unknown selection_policy {selection_policy!r} (use 'legacy' or 'puct')"
    )


def _require_positive_int(name: str, value) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


@dataclass(frozen=True)
class WideningPolicy:
    """Bounded progressive widening of one node's proposal source.

    ``capacity(v) = min(max_actions, max(initial_actions, ceil(coefficient *
    (1 + v) ** exponent)))`` distinct commands may be attempted at a node with
    ``v`` completed simulation visits. Capacity counts commands attempted,
    accepted or rejected, so a rejection-only node cannot issue unbounded
    Lean requests; ``max_draws`` bounds raw policy draws, so a node whose
    draws repeat cannot either. Rounds at one node are spaced by completed
    visits (``next_round_visit``); a node keeps retrying until a bound closes
    its source.
    """

    initial_actions: int = 4
    coefficient: float = 2.0
    exponent: float = 0.5
    max_actions: int = 32
    max_draws: int = 128

    def __post_init__(self) -> None:
        _require_positive_int("widening_initial_actions", self.initial_actions)
        _require_positive_int("widening_max_actions", self.max_actions)
        _require_positive_int("widening_max_draws", self.max_draws)
        if self.max_actions < self.initial_actions:
            raise ValueError("widening_max_actions must be at least widening_initial_actions")
        if isinstance(self.coefficient, bool) or not (
            isinstance(self.coefficient, (int, float))
            and math.isfinite(self.coefficient)
            and self.coefficient > 0
        ):
            raise ValueError(f"widening_coefficient must be finite and positive, got {self.coefficient!r}")
        if isinstance(self.exponent, bool) or not (
            isinstance(self.exponent, (int, float)) and 0 < self.exponent < 1
        ):
            raise ValueError(f"widening_exponent must be in (0, 1), got {self.exponent!r}")

    def capacity(self, visits: int) -> int:
        grown = math.ceil(self.coefficient * (1 + visits) ** self.exponent)
        return min(self.max_actions, max(self.initial_actions, grown))

    def draws_for(self, *, visits: int, attempted: int, draws: int, per_round: int) -> int:
        """Raw draws one round at a node may make; 0 when it may not propose."""
        return max(0, min(per_round, self.capacity(visits) - attempted, self.max_draws - draws))

    def next_round_visit(self, *, visits: int, rounds: int) -> int:
        """Completed visits at which a node with ``rounds`` rounds may retry.

        The gap grows by one visit per round, so retries never stop while
        command capacity and draws remain, yet the node's existing edges get
        a growing share of its visits in between.
        """
        return visits + rounds

    def exhausted(self, *, attempted: int, draws: int) -> bool:
        """The hard bounds are reached: the node's source closes for good."""
        return attempted >= self.max_actions or draws >= self.max_draws

    def as_dict(self) -> dict[str, float]:
        return {
            "initial_actions": self.initial_actions,
            "coefficient": self.coefficient,
            "exponent": self.exponent,
            "max_actions": self.max_actions,
            "max_draws": self.max_draws,
        }


WIDENING_FIELDS = (
    "widening_initial_actions",
    "widening_coefficient",
    "widening_exponent",
    "widening_max_actions",
    "widening_max_draws",
)


def resolve_widening(
    selection_policy: str,
    initial_actions: Optional[int] = None,
    coefficient: Optional[float] = None,
    exponent: Optional[float] = None,
    max_actions: Optional[int] = None,
    max_draws: Optional[int] = None,
) -> Optional[WideningPolicy]:
    """The widening bounds of a search mode.

    legacy: best-first has a finite one-batch source per node, so explicit
            widening settings are rejected and the result is None.
    puct:   always widens; omitted settings take the ``WideningPolicy``
            defaults. There is no switch back to fixed expansion.
    """
    values = dict(zip(WIDENING_FIELDS, (initial_actions, coefficient, exponent, max_actions, max_draws)))
    explicit = sorted(name for name, value in values.items() if value is not None)
    if selection_policy == "legacy":
        if explicit:
            raise ValueError(
                "selection_policy='legacy' runs the best-first loop, which does not "
                f"widen; remove the explicit setting(s) {explicit} or set selection_policy='puct'"
            )
        return None
    defaults = WideningPolicy()
    return WideningPolicy(
        initial_actions=defaults.initial_actions if initial_actions is None else initial_actions,
        coefficient=defaults.coefficient if coefficient is None else coefficient,
        exponent=defaults.exponent if exponent is None else exponent,
        max_actions=defaults.max_actions if max_actions is None else max_actions,
        max_draws=defaults.max_draws if max_draws is None else max_draws,
    )


def puct_score(
    stats: EdgeVisitStats, prior: float, total_node_visits: int, c: float
) -> float:
    """PUCT score for one tactic edge.

    ``prior`` is the edge's effective prior from
    ``ProofHypergraph.puct_priors``: its observed complete-action mass,
    normalized over the parent's non-dead edges. ``total_node_visits`` is the
    sum of (N + virtual_loss) over those sibling edges; ``c`` is the
    exploration constant. Virtual loss changes Q and the visit counts, never
    the prior.
    """
    return stats.Q + c * prior * math.sqrt(total_node_visits) / (
        1 + stats.N + stats.virtual_loss
    )
