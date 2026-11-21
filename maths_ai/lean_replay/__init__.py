"""Lean goal reconstruction shared by dataset seeds and canonical replay.

Lightweight on purpose: no Torch, GNN or Pantograph client imports, so data
models and loaders can depend on it without import cycles.
"""

from .errors import ReplayFailure, ReplayInputError
from .seed import (
    RENDERER_VERSION,
    GoalReplaySpec,
    LeanGoalSeed,
    SeedLocal,
    StructuredLocal,
    render_replay,
    seed_from_records,
    seed_from_text,
)
from .text import Hypothesis, ProofState, parse_state, parse_states
from .validation import ReplayValidation, validate_model_input, validate_replay

__all__ = [
    "RENDERER_VERSION",
    "GoalReplaySpec",
    "Hypothesis",
    "LeanGoalSeed",
    "ProofState",
    "ReplayFailure",
    "ReplayInputError",
    "ReplayValidation",
    "SeedLocal",
    "StructuredLocal",
    "parse_state",
    "parse_states",
    "render_replay",
    "seed_from_records",
    "seed_from_text",
    "validate_model_input",
    "validate_replay",
]
