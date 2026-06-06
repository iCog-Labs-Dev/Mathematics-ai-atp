from __future__ import annotations

from dataclasses import dataclass


TURNSTILES = ("\u22a2", "|-")


@dataclass(frozen=True)
class Hypothesis:
    name: str
    type_expr: str

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "type": self.type_expr}


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
    """
    hyp_block, goal = _split_turnstile(state)
    hypotheses: list[Hypothesis] = []

    if hyp_block:
        for raw_line in hyp_block.splitlines():
            line = raw_line.strip()
            if not line:
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
