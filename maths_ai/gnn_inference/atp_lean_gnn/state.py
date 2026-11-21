"""Printed proof states for the GNN, parsed by the shared replay parser.

Declarations are split at lexical boundaries, so multiline types, grouped
names, local definitions and colons inside binders stay intact, and ``case``
headers are dropped only when they really are headers. A line that is not a
declaration raises ``ReplayInputError`` (a ``ValueError``) instead of becoming
an invented ``Prop`` hypothesis. ``Hypothesis.name`` keeps grouped names as
printed (``"n m"``), so graphs built from text are unchanged for valid states.
"""

from __future__ import annotations

from maths_ai.lean_replay.lexer import TURNSTILES
from maths_ai.lean_replay.text import Hypothesis, ProofState, parse_state, parse_states

__all__ = ["TURNSTILES", "Hypothesis", "ProofState", "parse_state", "parse_states"]
