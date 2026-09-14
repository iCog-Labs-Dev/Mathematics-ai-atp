from __future__ import annotations

import re
from dataclasses import dataclass


TURNSTILES = ("\u22a2", "|-")

# Lean prints a branch label above the hypothesis block of each goal a branching
# tactic produced: `case intro.zero`, `case succ`, `case h\u2082`. It names the branch;
# it is not a binder, so nothing in the goal can refer to it. Left in, it reaches
# `Goal.hypotheses` as `case intro.zero : Prop` \u2014 a hypothesis Lean never declared
# \u2014 and `goal_start` rejects the reconstructed statement.
_CASE_LABEL_RE = re.compile(r"^case\s+\S+$")


@dataclass(frozen=True)
class Hypothesis:
    name: str
    type_expr: str
    value_expr: str | None = None

    @property
    def is_local_definition(self) -> bool:
        return self.value_expr is not None

    def as_dict(self) -> dict[str, str]:
        result = {"name": self.name, "type": self.type_expr}
        if self.value_expr is not None:
            result["value"] = self.value_expr
        return result


@dataclass(frozen=True)
class ProofState:
    hypotheses: list[Hypothesis]
    goal: str

    def as_dict(self) -> dict[str, object]:
        return {
            "hypotheses": [hypothesis.as_dict() for hypothesis in self.hypotheses],
            "goal": self.goal,
        }


def _split_turnstile(state: str) -> tuple[str, str]:
    for turnstile in TURNSTILES:
        if turnstile in state:
            left, right = state.split(turnstile, maxsplit=1)
            return left.strip(), right.strip()
    return "", state.strip()


def parse_state(state: str) -> ProofState:
    """
    Split a Lean proof state into hypotheses and goal text.

    Supports both the unicode turnstile ``⊢`` and the ASCII fallback ``|-``.
    ``case <label>`` lines are dropped: they name the branch a tactic produced,
    not a binder, and carrying one into ``hypotheses`` makes the state
    unelaborable.
    """
    hyp_block, goal = _split_turnstile(state)
    hypotheses: list[Hypothesis] = []

    if hyp_block:
        for raw_line in hyp_block.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            if _CASE_LABEL_RE.match(line):
                continue

            let_match = re.fullmatch(r"let\s+([^:]+)\s*:\s*(.+?)\s*:=\s*(.+)", line)
            if let_match:
                hypotheses.append(Hypothesis(
                    let_match.group(1).strip(),
                    let_match.group(2).strip(),
                    let_match.group(3).strip(),
                ))
                continue

            if " : " in line:
                name, _, typ = line.partition(" : ")
                hypotheses.append(Hypothesis(name.strip(), typ.strip()))
                continue

            if ":" in line:
                name, _, typ = line.partition(":")
                hypotheses.append(Hypothesis(name.strip(), typ.strip()))
                continue

            hypotheses.append(Hypothesis(line, "Prop"))

    return ProofState(hypotheses=hypotheses, goal=goal)
