"""The accepted-GNN backend: the published v2 joint policy with its own search.

This is main's benchmarked proving path (synchronous PyPantograph and the
best-first proof hypergraph), kept intact as a declared backend runtime. The
MCTS actor-critic backend lives in ``maths_ai.hybrid_reasoner``. Both share
the graph, model and data-model code in ``maths_ai.gnn_inference`` and
``maths_ai.data_models``.
"""
