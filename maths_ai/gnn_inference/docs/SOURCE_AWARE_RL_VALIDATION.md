# Source-aware model port: implementation validation

Validated on 2026-10-26 on `AC-branch`, starting from
`54d7cefe04aed980734798f06528ed80c71b81ac`. The architecture source is
`b8a3de7f450c2103a04f2b7a37ca85b113c45c96`.

## Scope and status

The shared model, complete-action RL objective, query-only serving, typed argument
execution, strict checkpoint conversion, and production caller migration are
implemented. The resource controls are implemented. Actual trained-artifact
rollout and hardware measurements remain pending.

| Plan stage | Status | Evidence or remaining gate |
| --- | --- | --- |
| 0: contracts | Implemented | Pinned source object, immutable typed pools, explicit reference sources, stored source numerical fixtures. |
| 1: shared decoder | Implemented | Count/source/identity logits, mixed recurrent updates, entropy gradients, and sampled/forced probability tests. |
| 2: query-only retrieval | Implemented | Local tiny-transformer artifact tests, one serving encoder, exact source pooling comparison, provenance rejection, batched requests, bounded cache/queue, shutdown tests. |
| 3: actor-critic | Implemented | Shared encoder, full-action probability, dropout-disabled recomputation, gradient checks, existing architecture/readout regression tests. |
| 4: artifacts | Implemented | Version-4 roundtrip, strict source mapping, source vocab/candidate/text-checkpoint bindings, explicit metadata/config tooling. Real selected checkpoint parity is pending. |
| 5: action/execution | Implemented | Typed records and complete pool identity, declaration-aware locals, verified library names, controlled real Lean execution. |
| 6: production consumers | Implemented | Shared supervised/inference/RL paths; complete joint policy objective; independent critic; generation/barrier/one-step regression suite. |
| 7: efficiency | Controls implemented | Shared immutable storage, query microbatching, native executor, cancellation cleanup, per-forward adapter reuse, candidate/update budgets. Actual large-artifact memory and throughput measurements are pending. |
| 8: deployment | Partially verified | Controlled real Lean gate passed. Selected trained model/index, GPU, JupyterHub, and trained multi-worker rollout gates remain pending. |

## Automated tests

Command:

```bash
.venv/bin/python -m pytest maths_ai/gnn_inference/tests -q --tb=short
```

Result: **386 passed, 10 skipped, 7 subtests passed**. The skipped tests are live or
environment-dependent checks. The default suite was run without live Lean path
variables. Native transformer/FAISS checks ran outside the execution sandbox,
where their worker threads can execute.

After the final strict-provenance change, the focused loss/reasoner/driver suite
also passed: **64 passed, 2 subtests passed**. Missing action records and empty or
changed collection graph identities are explicitly rejected.

The suite verifies these mechanisms:

- A tiny complete action space sums to probability one. Forced likelihood matches
  sampled likelihood. The expected entropy-surrogate gradient matches exact
  enumeration. Mixed source sequences deliver gradients to the decoder groups.
- Stored pinned-source fixtures compare initialization, count logits, adapter
  vectors, source/local/library logits, and GRU updates, including nonzero
  contextual residuals and optional first-position type cues.
- Collection/recomputation uses the same dropout-disabled distribution with
  nonzero configured dropout, while encoder and critic gradients remain enabled.
- Checkpoint conversion maps all trained source decoder tensors strictly, preserves
  complete probabilities on synthetic equivalent inputs, and rejects altered
  source vocabularies. Version-4 manifests roundtrip exactly.
- Query-only serving constructs one transformer tower, reproduces source masked
  mean pooling and normalization, excludes disallowed declarations, shares library
  storage, and distinguishes raw queries that share a graph.
- Active, queued, and queue-blocked requests settle during service closure. Empty
  allowed inventories remain valid empty library support. Altered index files and
  query-checkpoint bindings are rejected.
- Supervised zero-argument actions train count; unsupported syntax does not.
  Retrieval misses retain count supervision without inventing identity targets.
  Offline row lookup accounts for PyG index offsets.
- Existing graph contracts, local execution identities, failure/unknown evidence,
  update barriers, generation assertions, resource planning, and rendering tests
  remain covered by the complete inference suite.

## Controlled real Lean execution

The test used these actual local resources:

- Source root:
  `/home/nolawi/new-maths/maths_ai/sexpr_environment/sexpr_environment/mathlib4`
- Custom REPL:
  `/home/nolawi/pantograph_fork/.lake/build/bin/repl`
- Imports: `Init`, `Mathlib`.

```bash
PANTOGRAPH_TEST_SOURCE_ROOT=/home/nolawi/new-maths/maths_ai/sexpr_environment/sexpr_environment/mathlib4 \
PANTOGRAPH_TEST_REPL=/home/nolawi/pantograph_fork/.lake/build/bin/repl \
.venv/bin/python -m pytest \
  maths_ai/gnn_inference/tests/test_source_aware_live.py -q --tb=short
```

The final controlled run passed in 44.17 seconds. A controlled proposer checked
these three complete actions:

| Goal/context | Action | Result |
| --- | --- | --- |
| `p : Prop`, `h : p`, target `p` | Local `exact h` | No subgoals. |
| Target `True` | Library `exact True.intro` | No subgoals. |
| `n m : Nat`, `h : n = m`, target `n + 0 = m` | Mixed `rw [h, Nat.add_zero]` | No subgoals. |

Each action had a finite forced joint probability under the shared model. State
scopes released their allocations, and the server reported zero retained goals
after each case. This verifies execution and ownership for controlled actions;
it does not measure the accuracy of a trained policy.

The declaration-export Lean helper also executed successfully against the local
Lean toolchain with the `Init` import.

## Additional checks

- `uv pip check --python .venv/bin/python`: all 80 installed packages compatible.
- Ruff undefined-name check (`--select F821`): passed for the application modules,
  scripts, and hybrid reasoner.
- Python compilation and `git diff --check`: passed.
- RL command-line help imports successfully.

The environment added Transformers 4.57.6 and its required tokenizer support. The
existing PyTorch installation was retained. The updated lockfile records the
application dependency resolution.

## Pending gates and interpretation

No actual trained joint-source checkpoint, paired text checkpoint/index/corpus,
or selected production initialization was supplied for this run. Therefore the
following cannot yet be marked passed:

- Selected checkpoint source-to-AC parity and trained live-state retrieval coverage.
- Trained one-worker RL collection/update/resume with real artifact bindings.
- Trained multi-worker execution and GPU probability parity on deployment hardware.
- JupyterHub launch and environment compatibility.
- Large-artifact startup memory, resident corpus/index memory, queue latency,
  GPU allocation, collection throughput, and update-memory measurements.

No trained-policy success-rate or performance improvement is claimed. The
software migration does not change graph success into a kernel-certified theorem
proof. Deployment instructions are in `SOURCE_AWARE_RL.md`.
