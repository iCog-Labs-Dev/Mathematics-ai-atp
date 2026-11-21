"""Load search implementations only when their public exports are requested."""

from importlib import import_module


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _MODULES.get(name, ".hypergraph")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


_MODULES = {
    "HybridReasoner": ".joint_inference",
    "NullTacticExecutor": ".retained_executor",
    "RetainedOutcome": ".retained_executor",
    "RetainedTacticExecutor": ".retained_executor",
}


def __dir__():
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "HybridReasoner",
    "ProofHypergraph",
    "ProofNode",
    "TacticCandidate",
    "NullTacticExecutor",
    "RetainedOutcome",
    "RetainedTacticExecutor",
]
