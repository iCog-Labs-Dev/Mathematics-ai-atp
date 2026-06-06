from __future__ import annotations

import re
from collections.abc import Iterable


EMPTY_TACTIC = "<EMPTY_TACTIC>"
UNKNOWN_TACTIC = "<UNK_TACTIC>"
TACTIC_TOKEN_RE = re.compile(r"[A-Za-z0-9_.'!?]+")


def normalize_tactic(raw: str) -> str:
    text = raw.strip()
    if not text:
        return EMPTY_TACTIC

    match = TACTIC_TOKEN_RE.search(text)
    if match is None:
        return EMPTY_TACTIC
    return match.group(0)


def build_tactic_vocab(labels: Iterable[str]) -> dict[str, int]:
    vocab = {UNKNOWN_TACTIC: 0}
    for index, label in enumerate(sorted(set(labels)), start=1):
        vocab[label] = index
    return vocab


def label_example(raw_tactic: str) -> dict[str, object]:
    tactic_name = normalize_tactic(raw_tactic)
    return {
        "tactic_raw": raw_tactic,
        "tactic_name": tactic_name,
    }


def encode_tactic_name(tactic_name: str, tactic_vocab: dict[str, int]) -> int:
    return tactic_vocab.get(tactic_name, tactic_vocab[UNKNOWN_TACTIC])


# ---------------------------------------------------------------------------
# Tactic arity registry (strict static dictionary, no data-inference fallback)
# ---------------------------------------------------------------------------

TACTIC_ARITY: dict[str, int] = {
    "simp": 0,
    "ring": 0,
    "norm_num": 0,
    "omega": 0,
    "decide": 0,
    "trivial": 0,
    "contradiction": 0,
    "linarith": 0,
    "nlinarith": 0,
    "tauto": 0,
    "aesop": 0,
    "aesop?": 0,
    "apply": 1,
    "exact": 1,
    "rw": 1,
    "rewrite": 1,
    "have": 2,
    "calc": 0,
    "intro": 1,
    "intros": 0,
    "ext": 1,
    "cases": 1,
    "induction": 1,
    "constructor": 0,
    "use": 1,
    "refine": 1,
    "specialize": 1,
    "obtain": 1,
    "simp_all": 0,
    "norm_cast": 0,
    "push_cast": 0,
    "ring_nf": 0,
    "field_simp": 0,
    "positivity": 0,
    "gcongr": 0,
    "congr": 0,
    "funext": 0,
    "rfl": 0,
    "assumption": 0,
    "left": 0,
    "right": 0,
    "exfalso": 0,
    "by_contra": 0,
    "push_neg": 0,
    "contrapose": 0,
    "absurd": 1,
    "replace": 1,
    "conv": 0,
    "change": 1,
    "show": 1,
    "suffices": 1,
    "let": 2,
    "set": 1,
    "rcases": 1,
    "rintro": 0,
    "simp?": 0,
    "exact?": 0,
    "apply?": 0,
}

DEFAULT_ARITY: int = 1


def get_tactic_arity(tactic_name: str) -> int:
    """Return the expected number of pointer-selected arguments for *tactic_name*."""
    return TACTIC_ARITY.get(tactic_name, DEFAULT_ARITY)


# ---------------------------------------------------------------------------
# Best-effort tactic argument extraction
# ---------------------------------------------------------------------------

_BRACKET_OPEN = {"[", "⟨", "("}
_BRACKET_CLOSE = {"]", "⟩", ")"}
_ARG_SPLIT_RE = re.compile(r"[,\s]+")
_ARG_TOKEN_RE = re.compile(r"[A-Za-z0-9_.']+")


def parse_tactic_arguments(raw: str) -> tuple[str, list[str]]:
    """Extract the tactic family name and a list of argument tokens.

    Examples
    --------
    >>> parse_tactic_arguments("rw [foo, bar]")
    ('rw', ['foo', 'bar'])
    >>> parse_tactic_arguments("apply h1")
    ('apply', ['h1'])
    >>> parse_tactic_arguments("simp only [h1, h2]")
    ('simp', ['h1', 'h2'])
    >>> parse_tactic_arguments("simp")
    ('simp', [])
    """
    text = raw.strip()
    if not text:
        return EMPTY_TACTIC, []

    tactic_match = TACTIC_TOKEN_RE.search(text)
    if tactic_match is None:
        return EMPTY_TACTIC, []

    tactic_name = tactic_match.group(0)
    remainder = text[tactic_match.end():].strip()

    # Strip keywords that are not arguments (e.g. "only" in "simp only [...]")
    for keyword in ("only", "with", "using", "at"):
        if remainder.startswith(keyword):
            remainder = remainder[len(keyword):].strip()

    # Extract content inside brackets if present
    args: list[str] = []
    bracket_content: str | None = None
    for open_ch, close_ch in zip("[⟨(", "]⟩)"):
        start = remainder.find(open_ch)
        if start != -1:
            end = remainder.rfind(close_ch)
            if end > start:
                bracket_content = remainder[start + 1 : end]
            else:
                bracket_content = remainder[start + 1 :]
            break

    if bracket_content is not None:
        for token_match in _ARG_TOKEN_RE.finditer(bracket_content):
            args.append(token_match.group(0))
    elif remainder:
        # No brackets — split the remainder on whitespace and take identifier tokens
        for token_match in _ARG_TOKEN_RE.finditer(remainder):
            args.append(token_match.group(0))

    return tactic_name, args
