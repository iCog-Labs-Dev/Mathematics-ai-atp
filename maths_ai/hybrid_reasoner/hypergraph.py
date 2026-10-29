"""AND-OR proof hypergraph for the hybrid GNN/PLN search.

The graph is rooted at the admitted goal. Its nodes are obligation
occurrences: one Lean metavariable in one retained snapshot, observed by the
model as a canonical goal that carries the GNN prior and, with PLN, an STV.
Its hyperedges are accepted tactic attempts. An attempt can produce several
obligations that must all be proven (an AND-edge, its children kept in the
order the tactic returned them); alternative attempts on one node are
OR-alternatives.

Each node has exactly one parent edge. Nodes with identical model goals are
never merged: matching observations do not mean shared execution state or
shared proof evidence.

Edge and node statuses describe candidate readiness only. A ``READY`` root has
a candidate recipe, which still has to be replayed in one Lean execution and
certified by the kernel (``proof_replay``); ``is_solved`` is true only once a
certificate is attached.

A child whose goal has no metavariables and equals an ancestor's is a cycle
and is closed. A goal with metavariables can look identical while hidden
assignments differ, so it is never treated as a cycle; depth and node budgets
bound such branches instead.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Sequence, Set, Tuple, Union

from maths_ai.data_models.proof_components import STV, Goal, TacticCandidate
from maths_ai.data_models.proof_components import GoalState as GoalStateModel

from .execution_types import (
    DIRECT,
    ChildObligation,
    DeclarationKey,
    ExecutionOccurrence,
    ObligationKey,
    bind_arguments,
)
from .proof_recipes import ChildRecord, RecipeStep


class NodeStatus:
    OPEN = "open"  # not yet expanded; eligible for the search frontier
    EXPANDED = "expanded"  # has ≥1 outgoing edge, outcome still undetermined
    READY = "ready"  # ≥1 outgoing edge has a candidate recipe; not proof evidence
    DEAD = "dead"  # structurally closed; closure_reason determines learning evidence


class NodeClosureReason(str, Enum):
    """Why a node stopped being searchable."""

    NONE = "none"
    CANDIDATES_EXHAUSTED = "candidates_exhausted"
    NO_CANDIDATES = "no_candidates"
    DEPTH_LIMIT = "depth_limit"
    CYCLE = "cycle"
    ELABORATION_ERROR = "elaboration_error"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    RESOURCE_PRUNED = "resource_pruned"
    UNSUPPORTED_OBSERVATION = "unsupported_observation"


class UnsupportedObservation(ValueError):
    """The model cannot represent this goal, e.g. its vocabulary lacks a
    structural label the goal needs. The goal itself may be provable."""


class SearchEndReason(str, Enum):
    """Why the root search ended."""

    ROOT_CERTIFIED = "root_certified"
    ROOT_DEAD = "root_dead"
    MAX_NODES = "max_nodes"
    FRONTIER_EMPTY = "frontier_empty"
    EXTERNAL_ABORT = "external_abort"
    RESOURCE_LIMIT = "resource_limit"


class EdgeStatus:
    PENDING = "pending"  # children still open/expanded; outcome undetermined
    READY = "ready"  # no children and a complete or conditional assignment, or every child READY
    DEAD = "dead"  # structurally blocked because some required child is DEAD
    JOINT = "joint"  # no children, and the assignment depends on obligations the search does not know


def _state_key(goal: Goal) -> str:
    """Stable identity including the normalized target and ordered local context."""
    return goal.state_fingerprint()


def goal_has_holes(goal: Goal) -> bool:
    """Whether the goal's target or a local mentions an unassigned metavariable."""
    texts = [goal.goal_sexp or ""]
    for local in goal.locals or []:
        texts.extend([local.type_sexp or "", local.value_sexp or ""])
    return any("(:mv " in text for text in texts)


@dataclass
class ProofNode:
    id: int
    goal: Goal
    depth: int
    gnn_probability: float = 1.0
    stv: Optional[STV] = None
    status: str = NodeStatus.OPEN
    incoming_edge_id: Optional[int] = None
    outgoing_edge_ids: List[int] = field(default_factory=list)
    combined_rank: float = 0.0
    exhausted: bool = False
    note: Optional[str] = None
    closure_reason: NodeClosureReason = NodeClosureReason.NONE
    """Free-text annotation for terminal states the graph can't infer on its
    own — e.g. why a node was force-marked dead (cycle, depth limit, no
    candidates) — useful for inspecting/visualizing a finished search."""
    key: ObligationKey = field(default_factory=ObligationKey.root)
    declarations: Dict[str, DeclarationKey] = field(default_factory=dict)
    has_holes: bool = False
    # The retained occurrence while the search runs; cleared when it ends.
    occurrence: Optional[ExecutionOccurrence] = field(default=None, repr=False, compare=False)
    # Set when the node searches a replay continuation's actual obligation.
    continuation_of: Optional[int] = None

    def __post_init__(self) -> None:
        self.combined_rank = self.local_score
        self.has_holes = goal_has_holes(self.goal)

    @property
    def local_score(self) -> float:
        """gnn_prob × STV(strength × confidence) — this node's own
        provability heuristic in isolation, before any of *its* children
        have been explored. Exactly ``RankedSubgoal.combined_rank`` for
        non-root nodes; for the root (no incoming tactic / no STV yet) both
        factors default to neutral so it is always explored first.
        """
        stv_score = self.stv.score if self.stv is not None else 1.0
        return self.gnn_probability * stv_score

    def as_state(self, tactic_path: List[str]) -> GoalStateModel:
        return GoalStateModel(goal=self.goal, depth=self.depth, tactic_path=tactic_path)

    def summary(self) -> dict:
        return {
            "id": self.id,
            "expression": self.goal.expression,
            "depth": self.depth,
            "status": self.status,
            "gnn_probability": self.gnn_probability,
            "stv": (
                None
                if self.stv is None
                else {"strength": self.stv.strength, "confidence": self.stv.confidence}
            ),
            "combined_rank": self.combined_rank,
            "note": self.note,
            "closure_reason": self.closure_reason,
            "obligation": [self.key.attempt, self.key.slot],
            "has_holes": self.has_holes,
            "continuation_of": self.continuation_of,
        }


@dataclass
class ProofHyperedge:
    """One accepted tactic attempt. ``child_ids`` are in tactic slot order and
    ``child_slots`` gives each child's ``(slot, kind, introduced count)``."""

    id: int
    source_id: int
    tactic: TacticCandidate
    child_ids: List[int] = field(default_factory=list)
    status: str = EdgeStatus.PENDING
    command: str = ""
    child_slots: List[Tuple[int, str, int]] = field(default_factory=list)
    selected_complete: Optional[bool] = None
    # Obligations a childless attempt's assignment still refers to.
    dependencies: List[ObligationKey] = field(default_factory=list)
    closed: int = 0
    changed: int = 0

    def summary(self) -> dict:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "tactic": self.tactic.tactic_name,
            "arguments": list(self.tactic.arguments),
            "command": self.command,
            "probability": self.tactic.probability,
            "child_ids": list(self.child_ids),
            "child_slots": [list(slot) for slot in self.child_slots],
            "status": self.status,
            "dependencies": [[key.attempt, key.slot] for key in self.dependencies],
            "closed_siblings": self.closed,
            "changed_siblings": self.changed,
        }


class ProofHypergraph:
    """Root-at-goal AND-OR hypergraph, with bottom-up status/rank propagation.

    Build it exclusively through ``add_edge`` / ``mark_edge_dead`` /
    ``mark_node_exhausted`` — these keep ``status`` and ``combined_rank``
    consistent across the whole graph (objective 2: "when PLN ranks a
    subgoal, the GNN-derived score should be updated", here generalized to
    "...and that update should propagate to every ancestor").
    """

    def __init__(self, root_goal: Goal) -> None:
        self._next_node_id = 0
        self._next_edge_id = 0
        self.nodes: Dict[int, ProofNode] = {}
        self.edges: Dict[int, ProofHyperedge] = {}
        self.search_end_reason: Optional[SearchEndReason] = None
        # Proof evidence: set only by a kernel-certified replay.
        self.certificate: Optional[Any] = None
        self.witness: Tuple[Any, ...] = ()
        self.artifact: Optional[Any] = None
        # Counters for replay, retention and certification work.
        self.execution: Dict[str, Any] = {}
        # Bumped whenever an edge becomes READY, so replay runs only on change.
        self.ready_version = 0

        self.root_id = self._new_node(
            goal=root_goal, depth=0, gnn_probability=1.0, stv=None
        )

    # Construction
    def _new_node(
        self,
        *,
        goal: Goal,
        depth: int,
        gnn_probability: float,
        stv: Optional[STV],
        incoming_edge_id: Optional[int] = None,
    ) -> int:
        node_id = self._next_node_id
        self._next_node_id += 1
        node = ProofNode(
            id=node_id,
            goal=goal,
            depth=depth,
            gnn_probability=gnn_probability,
            stv=stv,
            incoming_edge_id=incoming_edge_id,
        )
        self.nodes[node_id] = node
        return node_id

    def add_edge(
        self,
        source_id: int,
        tactic: TacticCandidate,
        ranked_subgoals: Sequence[Tuple[Union[ChildObligation, Goal], Optional[STV]]],
        *,
        command: str = "",
        selected_complete: Optional[bool] = True,
        dependencies: Sequence[ObligationKey] = (),
        closed: int = 0,
        changed: int = 0,
    ) -> ProofHyperedge:
        """Record an accepted attempt from ``source_id`` with its children.

        ``ranked_subgoals`` pairs each child obligation, in tactic slot order,
        with its STV (``None`` when PLN is off). Every Lean-returned obligation
        must be present; PLN scores affect expansion order but never remove or
        reorder children. A bare ``Goal`` is taken as a direct child that
        introduced no locals.

        An attempt without children is READY when the selected obligation's
        assignment is complete. When the assignment still refers to other
        obligations, it is a conditional candidate: READY if ``dependencies``
        names them, since replay solves them in the same execution, and JOINT
        otherwise. JOINT is neither ready nor dead. Readiness is never proof;
        only the certified replay is.

        Edge case — cycles: a child whose goal has no metavariables and matches
        one of ``source_id``'s hole-free ancestors is created but force-marked
        DEAD with reason CYCLE rather than re-explored.
        """
        edge_id = self._next_edge_id
        self._next_edge_id += 1
        source = self.nodes[source_id]
        depth = source.depth + 1
        ancestry = self._ancestor_state_keys(source_id)

        child_ids: List[int] = []
        slots: List[Tuple[int, str, int]] = []
        for position, (child, stv) in enumerate(ranked_subgoals):
            if isinstance(child, Goal):
                child = ChildObligation(position, DIRECT, "", child, ())
            child_id = self._new_node(
                goal=child.goal,
                depth=depth,
                gnn_probability=tactic.probability,
                stv=stv,
                incoming_edge_id=edge_id,
            )
            node = self.nodes[child_id]
            node.key = ObligationKey(edge_id, child.slot)
            if not node.has_holes and _state_key(child.goal) in ancestry:
                node.status = NodeStatus.DEAD
                node.exhausted = True
                node.closure_reason = NodeClosureReason.CYCLE
                node.note = "cycle: identical to an ancestor goal"
            child_ids.append(child_id)
            slots.append((child.slot, child.kind, len(child.introduced)))

        edge = ProofHyperedge(
            id=edge_id,
            source_id=source_id,
            tactic=tactic,
            child_ids=child_ids,
            command=command,
            child_slots=slots,
            selected_complete=selected_complete,
            dependencies=list(dependencies),
            closed=closed,
            changed=changed,
        )
        edge.status = self._derive_edge_status(edge)
        if edge.status == EdgeStatus.READY:
            self.ready_version += 1
        self.edges[edge_id] = edge
        source.outgoing_edge_ids.append(edge_id)

        self._propagate(source_id)
        return edge

    def add_continuation(
        self, occurrence: ExecutionOccurrence, stands_for: int
    ) -> ProofNode:
        """A fresh, parentless search node for a replay continuation's obligation."""
        original = self.nodes[stands_for]
        node_id = self._new_node(
            goal=occurrence.goal,
            depth=original.depth,
            gnn_probability=original.gnn_probability,
            stv=original.stv,
        )
        node = self.nodes[node_id]
        node.key = occurrence.key
        node.declarations = dict(occurrence.declarations)
        node.occurrence = occurrence
        node.continuation_of = stands_for
        return node

    def mark_edge_dead(self, edge_id: int, *, note: Optional[str] = None) -> None:
        """The tactic failed to apply (Lean rejected it / executor error)."""
        edge = self.edges[edge_id]
        edge.status = EdgeStatus.DEAD
        if note:
            self.nodes[edge.source_id].note = note
        self._propagate(edge.source_id)

    def mark_node_exhausted(
        self,
        node_id: int,
        *,
        reason: NodeClosureReason,
        note: Optional[str] = None,
    ) -> None:
        """No more tactic candidates remain for ``node_id`` (top-k exhausted,
        depth limit reached, or the GNN returned no viable prediction).
        Lets ``_recompute_node`` decide DEAD vs. staying EXPANDED/READY.
        """
        if reason not in {
            NodeClosureReason.CANDIDATES_EXHAUSTED,
            NodeClosureReason.NO_CANDIDATES,
            NodeClosureReason.DEPTH_LIMIT,
            NodeClosureReason.CYCLE,
        }:
            raise ValueError(f"Invalid local exhaustion reason: {reason}")
        node = self.nodes[node_id]
        node.exhausted = True
        node.closure_reason = reason
        if note:
            node.note = note
        self._propagate(node_id)

    def mark_node_unelaborated(
        self, node_id: int, *, note: Optional[str] = None
    ) -> None:
        """Close a node structurally while retaining unknown learning evidence."""
        node = self.nodes[node_id]
        if node.status != NodeStatus.READY:
            node.status = NodeStatus.DEAD
        node.exhausted = True
        node.closure_reason = NodeClosureReason.ELABORATION_ERROR
        if note:
            node.note = note
        self._propagate(node_id)

    def mark_node_pruned(self, node_id: int, *, note: Optional[str] = None) -> None:
        """Close a node whose snapshot was released under retention pressure.

        Its obligation is never rebuilt from the canonical goal; the node only
        stops being searchable and its evidence stays unknown.
        """
        node = self.nodes[node_id]
        if node.status != NodeStatus.READY:
            node.status = NodeStatus.DEAD
        node.exhausted = True
        node.closure_reason = NodeClosureReason.RESOURCE_PRUNED
        node.note = note or "pruned under snapshot retention pressure"
        self._propagate(node_id)

    def mark_node_unsupported(self, node_id: int, *, note: Optional[str] = None) -> None:
        """Close a node the model cannot observe, with unknown evidence."""
        node = self.nodes[node_id]
        if node.status != NodeStatus.READY:
            node.status = NodeStatus.DEAD
        node.exhausted = True
        node.closure_reason = NodeClosureReason.UNSUPPORTED_OBSERVATION
        node.note = note
        self._propagate(node_id)

    def mark_node_infrastructure_failed(
        self, node_id: int, *, note: Optional[str] = None
    ) -> None:
        """Close a node after backend failure without creating numeric evidence."""
        node = self.nodes[node_id]
        if node.status != NodeStatus.READY:
            node.status = NodeStatus.DEAD
        node.exhausted = True
        node.closure_reason = NodeClosureReason.INFRASTRUCTURE_FAILURE
        if note:
            node.note = note
        self._propagate(node_id)

    # Queries

    @property
    def root(self) -> ProofNode:
        return self.nodes[self.root_id]

    def is_solved(self) -> bool:
        """True only when a kernel certificate of the admitted root is attached."""
        return self.certificate is not None and bool(self.certificate.is_certified)

    def has_candidate(self) -> bool:
        """The root has a candidate recipe; not evidence of a proof."""
        return self.root.status == NodeStatus.READY

    def is_exhausted(self) -> bool:
        return self.root.status == NodeStatus.DEAD

    def frontier(self) -> List[ProofNode]:
        """Open nodes ranked by ``combined_rank``, descending.

        Recomputed on demand rather than cached in a heap: ranks mutate via
        backprop (objective 2), and a stale-priority heap is its own class
        of bugs (see design report, "re-ranking thrashing"). For the node
        counts a tactic search realistically holds open at once, an O(n log n)
        re-sort per pop is the simpler-correct choice over indexed-heap
        bookkeeping.
        """
        open_nodes = [
            node for node in self.nodes.values() if node.status == NodeStatus.OPEN
        ]
        open_nodes.sort(key=lambda node: node.combined_rank, reverse=True)
        return open_nodes

    def tactic_path(self, node_id: int) -> List[str]:
        path: List[str] = []
        node = self.nodes[node_id]
        while node.incoming_edge_id is not None:
            edge = self.edges[node.incoming_edge_id]
            path.append(edge.command or " ".join(
                [edge.tactic.tactic_name, *edge.tactic.arguments]
            ).strip())
            node = self.nodes[edge.source_id]
        path.reverse()
        return path

    def proof_trace(self) -> Optional[dict]:
        """If certified, the replayed proof as a nested dict; else ``None``.

        Each entry is the command the certified replay ran for an obligation
        and one entry per child slot. A child a shared assignment discharged
        has no command of its own (``"discharged": True``).
        """
        if not self.is_solved():
            return None
        steps = {step.obligation: step for step in self.witness}

        def trace(key: ObligationKey) -> dict:
            step = steps.get(key)
            if step is None:
                return {"obligation": [key.attempt, key.slot], "discharged": True}
            node = self.nodes[step.node]
            return {
                "goal": node.goal.expression,
                "command": step.command,
                "subgoals": [
                    trace(ObligationKey(step.attempt, slot))
                    for slot, _kind, _introduced in step.step.children
                ],
            }

        return trace(ObligationKey.root())

    def witnessed_edges(self) -> Set[int]:
        """Attempts of the certified replay whose observation and command matched."""
        if not self.is_solved():
            return set()
        return {step.attempt for step in self.witness if step.observation_matched}

    def summary(self) -> dict:
        certificate = None
        if self.certificate is not None:
            certificate = {
                "status": self.certificate.status,
                "reason": self.certificate.reason,
            }
        return {
            "root_id": self.root_id,
            "solved": self.is_solved(),
            "candidate": self.has_candidate(),
            "certificate": certificate,
            "exhausted": self.is_exhausted(),
            "search_end_reason": self.search_end_reason,
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "execution": dict(self.execution),
            "nodes": [node.summary() for node in self.nodes.values()],
            "edges": [edge.summary() for edge in self.edges.values()],
        }

    # Recipe source for replay

    def alternatives(self, node_id: int) -> List[RecipeStep]:
        """READY attempts of ``node_id``, best first."""
        node = self.nodes[node_id]
        ready = [
            self.edges[eid]
            for eid in node.outgoing_edge_ids
            if self.edges[eid].status == EdgeStatus.READY
        ]
        ready.sort(key=self._edge_value, reverse=True)
        return [self._recipe_step(edge) for edge in ready]

    def observation(self, node_id: int) -> Goal:
        return self.nodes[node_id].goal

    def declarations(self, node_id: int) -> Dict[str, DeclarationKey]:
        return self.nodes[node_id].declarations

    def _recipe_step(self, edge: ProofHyperedge) -> RecipeStep:
        source = self.nodes[edge.source_id]
        refs = edge.tactic.argument_refs
        if refs is None:
            refs = bind_arguments(source.goal, edge.tactic.arguments)
        return RecipeStep(
            attempt=edge.id,
            node=edge.source_id,
            tactic=edge.tactic.tactic_name,
            arguments=tuple(refs),
            command=edge.command,
            children=tuple(
                ChildRecord(slot, kind, introduced, child_id)
                for (slot, kind, introduced), child_id in zip(edge.child_slots, edge.child_ids)
            ),
            dependencies=tuple(edge.dependencies),
        )

    def __len__(self) -> int:
        return len(self.nodes)

    # Internals: bottom-up propagation (status + combined_rank)

    def _ancestor_state_keys(self, node_id: int) -> Set[str]:
        keys: Set[str] = set()
        node = self.nodes[node_id]
        while True:
            if not node.has_holes:
                keys.add(_state_key(node.goal))
            if node.incoming_edge_id is None:
                return keys
            node = self.nodes[self.edges[node.incoming_edge_id].source_id]

    def _derive_edge_status(self, edge: ProofHyperedge) -> str:
        if not edge.child_ids:
            if edge.selected_complete is not False or edge.dependencies:
                return EdgeStatus.READY
            return EdgeStatus.JOINT
        statuses = [self.nodes[cid].status for cid in edge.child_ids]
        if any(status == NodeStatus.DEAD for status in statuses):
            return EdgeStatus.DEAD
        if all(status == NodeStatus.READY for status in statuses):
            return EdgeStatus.READY
        return EdgeStatus.PENDING

    def _edge_value(self, edge: ProofHyperedge) -> float:
        """tactic-prior × AND-aggregate(children).

        AND-aggregate = ``min`` over child ``combined_rank``s — "a
        conjunction is only as strong as its weakest still-open conjunct" —
        the conventional pessimistic aggregation for AND-nodes (mirrors
        value backup in AND-OR search / HTPS). A childless edge (the
        "no-goal" terminal) contributes just the tactic's own probability.
        """
        if not edge.child_ids:
            return edge.tactic.probability
        return edge.tactic.probability * min(
            self.nodes[cid].combined_rank for cid in edge.child_ids
        )

    def _recompute_node(self, node: ProofNode) -> bool:
        before = (node.status, node.combined_rank)

        if node.status not in (NodeStatus.READY, NodeStatus.DEAD):
            edge_statuses = [self.edges[eid].status for eid in node.outgoing_edge_ids]
            if any(status == EdgeStatus.READY for status in edge_statuses):
                node.status = NodeStatus.READY
            elif edge_statuses:
                node.status = NodeStatus.EXPANDED
                if node.exhausted and all(
                    status == EdgeStatus.DEAD for status in edge_statuses
                ):
                    node.status = NodeStatus.DEAD
            elif node.exhausted:
                # Expanded zero edges (no candidates / executor rejected all)
                # and the search loop says it won't try again ⇒ a dead end.
                node.status = NodeStatus.DEAD

        viable = [
            self.edges[eid]
            for eid in node.outgoing_edge_ids
            if self.edges[eid].status != EdgeStatus.DEAD
        ]
        best_edge_value = max((self._edge_value(edge) for edge in viable), default=None)
        node.combined_rank = (
            node.local_score
            if best_edge_value is None
            else max(node.local_score, best_edge_value)
        )

        return (node.status, node.combined_rank) != before

    def _propagate(self, start_id: int) -> None:
        """Bottom-up fixpoint walk: recompute ``start_id``, refresh the edge
        above it, and keep climbing while something keeps changing.

        BFS with a `queued` set rather than plain recursion: a change can
        legitimately re-enqueue an already-visited ancestor once its other
        children settle, and recursion depth would otherwise track proof
        depth (fine for shallow goals, riskier for deep ones).
        """
        queue: Deque[int] = deque([start_id])
        queued: Set[int] = {start_id}
        while queue:
            current_id = queue.popleft()
            queued.discard(current_id)
            node = self.nodes[current_id]

            node_changed = self._recompute_node(node)

            propagate_further = node_changed
            if node.incoming_edge_id is not None:
                edge = self.edges[node.incoming_edge_id]
                new_status = self._derive_edge_status(edge)
                if new_status != edge.status:
                    edge.status = new_status
                    propagate_further = True
                    if new_status == EdgeStatus.READY:
                        self.ready_version += 1

            if propagate_further and node.incoming_edge_id is not None:
                parent_id = self.edges[node.incoming_edge_id].source_id
                if parent_id not in queued:
                    queue.append(parent_id)
                    queued.add(parent_id)
