from __future__ import annotations

import sys

from .graph import DAGBuilder
from .state import ProofState


def safe_console_text(text: str) -> str:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
        return text
    except UnicodeEncodeError:
        return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def console_print(text: str = "") -> None:
    print(safe_console_text(text))


def format_dag_summary(
    dag: DAGBuilder,
    parsed_state: ProofState,
    *,
    theorem: str = "",
    tactic: str = "",
) -> str:
    stats = dag.stats()
    parent_uses = dag.outgoing_counts()
    reused_nodes = dag.reused_nodes()

    lines = [
        "",
        f"  {'=' * 55}",
        f"  Theorem : {theorem}",
        f"  Tactic  : {tactic[:72]}{'...' if len(tactic) > 72 else ''}",
        f"  {'=' * 55}",
        f"  Nodes         : {stats.num_nodes}",
        f"  Edges         : {stats.num_edges}",
        f"  Roots         : {stats.num_roots}",
        f"  Leaves        : {stats.num_leaves}",
        f"  Reused nodes  : {stats.num_reused_nodes}",
        f"  Sharing ratio : {stats.sharing_ratio:.2f}  (1.0 = plain tree, >1 = shared DAG)",
    ]

    if reused_nodes:
        lines.append("")
        lines.append("  Top reused nodes (used by multiple parents):")
        for node in sorted(reused_nodes, key=lambda item: (-parent_uses[item.id], item.label))[:8]:
            uses = parent_uses[node.id]
            lines.append(f"    [{node.id:3d}]  {node.label:<30}  <- used by {uses} parents")

    lines.append("")
    lines.append(f"  Hypotheses ({len(parsed_state.hypotheses)} total):")
    for hypothesis in parsed_state.hypotheses[:5]:
        lines.append(f"    {hypothesis.name:<22} : {hypothesis.type_expr[:55]}")
    if len(parsed_state.hypotheses) > 5:
        lines.append(f"    ... and {len(parsed_state.hypotheses) - 5} more")

    goal = parsed_state.goal
    lines.append(f"\n  Goal: {goal[:80]}{'...' if len(goal) > 80 else ''}")
    return "\n".join(lines)
