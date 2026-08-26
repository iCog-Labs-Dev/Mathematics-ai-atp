# Mathematics AI ATP

This repository is an experimental proof-search and theorem-proving project that combines machine learning, symbolic reasoning, and Lean/Pantograph tooling. The codebase is organized around the `maths_ai` package, with supporting experiments and research components.

## Overview

The project explores hybrid automated theorem proving by combining:

- GNN-based tactic and argument prediction
- Premise selection and scoring
- PLN-style reasoning and ranking
- Proof hypergraph search
- Lean/Pantograph integration for tactic execution

The current implementation is research-oriented rather than a finished end-user application.

## Repository Structure

- `maths_ai/` — core library for proof search, inference, data models, and utilities
  - `core/` — shared configuration and runtime settings
  - `data/` — preprocessing and representation helpers
  - `data_models/` — proof-related data structures
  - `gnn_inference/` — GNN scoring and tactic/argument inference
  - `hybrid_reasoner/` — proof hypergraph search and tactic execution
  - `pln_inference/` — PLN-style inference components
  - `utils/` — supporting utilities
- `experiments/` — experimental projects and analysis scripts
- `tests/` — project test suites
- `.github/workflows/` — CI/CD workflows
- `Dockerfile` — reproducible Docker image definition
- `.dockerignore` — files excluded from the Docker build context

## Key Components

### Hybrid Reasoning

The `maths_ai.hybrid_reasoner` package implements the proof-search pipeline using GNN predictions, premise selection, PLN-derived ranking, and Lean/Pantograph execution.

### GNN Inference

The `maths_ai.gnn_inference` package contains the models and inference components used for tactic prediction, argument selection, premise selection, and lemma retrieval.

### PLN Inference

The `maths_ai.pln_inference` package provides symbolic reasoning components used for scoring and ranking proof states.

---

# Getting Started

## Requirements

The project targets Python 3.11+.

Core dependencies include:

- PyTorch
- PyTorch Geometric
- FAISS CPU
- Graphviz
- Pydantic
- Pantograph
- pytest
- Lean 4

## Local Development

Create and activate a Python environment, then install the project dependencies.

Using `uv`:

```bash
uv sync
uv pip install -e . --no-deps