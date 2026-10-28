"""A scripted explorer over retained snapshots, for replay tests.

It plays the part of the search graph: each ``expand`` runs one command on a
node's own retained snapshot and records the accepted attempt with its child
nodes. Readiness follows the AND/OR rule (a childless attempt is ready when the
selected obligation is complete; an attempt with children when every child
has a ready attempt). It implements ``RecipeSource``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from maths_ai.data_models.proof_components import TacticCandidate
from maths_ai.hybrid_reasoner.execution_registry import LeaseKind, SnapshotRegistry
from maths_ai.hybrid_reasoner.execution_types import (
    ExecutionOccurrence,
    ObligationKey,
    bind_arguments,
    child_declarations,
    root_declarations,
)
from maths_ai.hybrid_reasoner.pantograph_model_sexpr import materialize_seed
from maths_ai.hybrid_reasoner.proof_recipes import ChildRecord, RecipeStep
from maths_ai.hybrid_reasoner.retained_executor import RetainedTacticExecutor
from maths_ai.hybrid_reasoner.tactic_rendering import render_tactic_command
from maths_ai.lean_replay import seed_from_text


@dataclass
class _Edge:
    step: RecipeStep
    complete: bool


@dataclass
class _Node:
    occurrence: ExecutionOccurrence
    edges: list[_Edge] = field(default_factory=list)


class Explorer:
    def __init__(self, registry: SnapshotRegistry, root: ExecutionOccurrence):
        self.registry = registry
        self.executor = RetainedTacticExecutor()
        self.nodes = {0: _Node(root)}
        self._attempts = 0

    @classmethod
    async def start(cls, scope, statement: str) -> "Explorer":
        registry = SnapshotRegistry(scope)
        state, goal, _ = await materialize_seed(scope, seed_from_text(goal=statement))
        lease = registry.adopt_root(state, "root")
        root = ExecutionOccurrence(
            0, ObligationKey.root(), state.goals[0].goal_name, goal,
            root_declarations(goal), lease,
        )
        return cls(registry, root)

    @property
    def root(self) -> ExecutionOccurrence:
        return self.nodes[0].occurrence

    def children(self, node: int, attempt: int | None = None) -> list[int]:
        edges = self.nodes[node].edges
        edge = edges[-1] if attempt is None else next(e for e in edges if e.step.attempt == attempt)
        return [child.node for child in edge.step.children]

    async def expand(self, node: int, tactic: str, *arguments: str) -> list[int] | None:
        occurrence = self.nodes[node].occurrence
        refs = bind_arguments(occurrence.goal, arguments)
        command = render_tactic_command(
            TacticCandidate(
                tactic_name=tactic, arguments=list(arguments), probability=1.0,
                argument_refs=refs,
            )
        )
        outcome = await self.executor.apply(self.registry, occurrence, command)
        if not outcome.success:
            return None
        attempt = self._attempts
        self._attempts += 1
        records = []
        for child in outcome.children:
            child_id = len(self.nodes)
            self.nodes[child_id] = _Node(
                ExecutionOccurrence(
                    child_id,
                    ObligationKey(attempt, child.slot),
                    child.goal_name,
                    child.goal,
                    child_declarations(occurrence.declarations, attempt, child),
                    self.registry.share(outcome.lease, LeaseKind.GRAPH, f"node-{child_id}"),
                )
            )
            records.append(ChildRecord(child.slot, child.kind, len(child.introduced), child_id))
        await self.registry.release(outcome.lease)
        step = RecipeStep(attempt, node, tactic, tuple(refs), command, tuple(records))
        self.nodes[node].edges.append(_Edge(step, bool(outcome.selected_complete)))
        return [record.node for record in records]

    # RecipeSource

    def _ready(self, node: int, seen=frozenset()) -> bool:
        return any(self._edge_ready(edge, seen | {node}) for edge in self.nodes[node].edges)

    def _edge_ready(self, edge: _Edge, seen) -> bool:
        if not edge.step.children:
            return edge.complete
        return all(
            child.node not in seen and self._ready(child.node, seen) for child in edge.step.children
        )

    def alternatives(self, node: int):
        return [edge.step for edge in self.nodes[node].edges if self._edge_ready(edge, {node})]

    def observation(self, node: int):
        return self.nodes[node].occurrence.goal

    def declarations(self, node: int):
        return self.nodes[node].occurrence.declarations

    def add_node(self, occurrence: ExecutionOccurrence) -> int:
        node = len(self.nodes)
        occurrence.occurrence_id = node
        self.nodes[node] = _Node(occurrence)
        return node
