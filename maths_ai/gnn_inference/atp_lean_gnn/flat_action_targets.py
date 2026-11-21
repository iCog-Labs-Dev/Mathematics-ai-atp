"""Pinned flat-reference syntax supported by the source-aware decoder."""

import re
from .labels import clean_tactic_markup

_PART = r"(?:[^\W\d][\w'′]*|«[^»]+»)"
_IDENTIFIER = re.compile(rf"{_PART}(?:\.{_PART})*\Z", re.UNICODE)
_LIST_TACTICS = {
    "rw",
    "rewrite",
    "rwa",
    "erw",
    "erewrite",
    "simp",
    "simp_rw",
    "simpa",
    "simp_all",
    "dsimp",
    "norm_num",
    "rw_mod_cast",
    "field_simp",
    "linarith",
    "nlinarith",
}


def flat_reference_syntax(raw: str) -> tuple[list[str], str]:
    clean = clean_tactic_markup(raw).strip()
    parts = clean.split(maxsplit=1)
    if not parts:
        return [], "empty_tactic"
    if len(parts) == 1:
        return [], "supported"
    family, remainder = parts
    if (
        remainder.startswith("[")
        and remainder.endswith("]")
        and family in _LIST_TACTICS
    ):
        inner = remainder[1:-1].strip()
        tokens = [item.strip() for item in inner.split(",")] if inner else []
    else:
        tokens = remainder.split()
    if any(token in {"only", "at", "using", "with", "by"} for token in tokens):
        return [], "syntax_modifier_or_location"
    if not all(_IDENTIFIER.fullmatch(token) and token != "_" for token in tokens):
        return [], "compound_or_nonreference_syntax"
    # Names introduced by these tactics are not pointers into the input context.
    if family in {
        "intro",
        "intros",
        "rintro",
        "ext",
        "funext",
        "have",
        "let",
        "set",
        "obtain",
    }:
        return [], "fresh_name_or_binding_syntax"
    return tokens, "supported"
