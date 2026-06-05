"""
Lean → PeTTaChainer constant mappings and type atoms.

This module contains the symbol-mapping tables used throughout the translator
to convert Lean/Pantograph names into PeTTaChainer equivalents.
"""

from __future__ import annotations


LEAN_TO_PETTA: dict[str, str] = {
    "Lean.Constant.Implies": "Implication",
    "Implies": "Implication",
    "imp": "Implication",
    "Lean.Constant.Not": "Not",
    "Not": "Not",
    "not": "Not",

    "Lean.Constant.And": "∧",
    "And": "∧",
    "Lean.Constant.Or": "∨",
    "Or": "∨",
    "Lean.Constant.Iff": "↔",
    "Iff": "↔",

    "Eq": "Equal",
    "Lean.Constant.Eq": "Equal",
    "Nat": "NaturalNumber",
    "Lean.Constant.Nat": "NaturalNumber",
    "Int": "Integer",
    "Rat": "RationalNumber",
    "Real": "RealNumber",
    "Prop": "PROP",

    "GT.gt": "GreaterThan",
    "LT.lt": "LessThan",
    "GE.ge": "GreaterEqual",
    "LE.le": "LessEqual",

    "Add.add": "Addition",
    "HAdd.hAdd": "Addition",
    "Sub.sub": "Subtraction",
    "HSub.hSub": "Subtraction",
    "Mul.mul": "Multiplication",
    "HMul.hMul": "Multiplication",
    "Div.div": "Division",
    "HDiv.hDiv": "Division",
    "Pow.pow": "Power",
    "HPow.hPow": "Power",
}

TYPE_ATOMS = {
    "PROP", "Prop", "TYPE0", "TYPE1", "TYPE2", "Type", "Sort",
    "NaturalNumber", "Integer", "RationalNumber", "RealNumber",
    "Nat", "Int", "Rat", "Real",
}


def map_const(name: str) -> str:
    if name in LEAN_TO_PETTA:
        return LEAN_TO_PETTA[name]
    if name.startswith("Lean.Constant."):
        short = name.removeprefix("Lean.Constant.")
        return LEAN_TO_PETTA.get(short, short)
    short = name.split(".")[-1]
    return LEAN_TO_PETTA.get(name, LEAN_TO_PETTA.get(short, short))
