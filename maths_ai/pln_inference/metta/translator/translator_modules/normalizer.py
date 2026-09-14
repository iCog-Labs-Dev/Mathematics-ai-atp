"""
MeTTa symbol sanitization and variable normalization.

Provides:
  - sanitize_metta_symbol(): makes Lean names safe for MeTTa atoms
  - VariableNormalizer: maps variables to v0/v1/... or preserves original names
"""

from __future__ import annotations

import re
from typing import Any


def sanitize_metta_symbol(name: Any) -> str:
    """
    Preserve Lean variable/proposition names while making them safe enough for
    simple MeTTa atoms.

    Examples:
      P        -> P
      hP       -> hP
      P.Q      -> P.Q
      ?m.123   -> m.123
      a b      -> a_b

    This intentionally avoids mapping P,Q,A,B to v0,v1,... when normalization
    is disabled.
    """
    s = str(name).strip()

    if not s:
        return "unnamed"

    s = s.replace("⟨", "").replace("⟩", "")
    s = s.replace("{", "").replace("}", "")
    s = s.replace("[", "").replace("]", "")
    s = s.replace("?", "m")

    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_'.-]", "_", s)

    if re.match(r"^[0-9-]", s):
        s = "x_" + s

    return s


class VariableNormalizer:
    """
    Variable mapper used by the translator.

    normalize=True:
        P,Q,A,B,... are mapped to v0,v1,v2,...
        This is useful for structural pattern comparison.

    normalize=False:
        the original Lean/Pantograph variable names are preserved.
        This is safer for concrete branch-aware caching because structurally
        similar states with different concrete variables will not collapse to
        the same rendered formula.
    """
    def __init__(self, *, normalize: bool = False) -> None:
        self.normalize = normalize
        self.var_map: dict[str, str] = {}
        self.counter = 0

    def get(self, var_id: Any) -> str:
        key = str(var_id)

        if not self.normalize:
            return sanitize_metta_symbol(key)

        if key not in self.var_map:
            self.var_map[key] = f"v{self.counter}"
            self.counter += 1
        return self.var_map[key]
