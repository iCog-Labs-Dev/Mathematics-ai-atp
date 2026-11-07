__all__ = [
    "HybridReasoner",
    "ProofHypergraph",
    "ProofNode",
    "TacticCandidate",
    "TacticOutcome",
    "TacticExecutor",
    "NullTacticExecutor",
]


def __getattr__(name: str):
    if name == "HybridReasoner":
        from .joint_inference import HybridReasoner

        return HybridReasoner
    if name in {
        "ProofHypergraph",
        "ProofNode",
        "TacticCandidate",
        "TacticOutcome",
        "TacticExecutor",
        "NullTacticExecutor",
    }:
        from . import hypergraph

        return getattr(hypergraph, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")