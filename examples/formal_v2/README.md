# Formal v2 proof examples

End-to-end examples for the shared MCTS/frozen-GNN service, with independently
checkable Lean reference proofs and an explicit record of observed model results.

## Examples and evidence

| Example | Mathematical content | Reference proof | Recorded model result |
| --- | --- | --- | --- |
| `lagrange` | Six-variable Gram-determinant / sum-of-squares identity | `ring` | Both trained backends found a proof and passed fresh native replay |
| `cauchy-schwarz` | Cauchy–Schwarz for three real coordinates | Derivation from Lagrange's identity and positivity | Challenge; no model-discovered success is claimed |
| `crt` | Unique solution to three congruences below 105 | `omega`, with an explicit reference-file heartbeat allowance | MCTS exhausted its bounded search in the recorded trial |

- [`VerifiedProofs.lean`](VerifiedProofs.lean): complete Lean proofs, with axiom
  audits at the end of the file.
- [`problems.json`](problems.json): statements, ordered hypotheses and universe
  declarations submitted to the service. This file contains no proof bodies.
- [`demo.py`](demo.py): capability discovery, immutable run submission, polling,
  certificate retrieval and independent replay through the matching client.
- [`evidence/lagrange-cpu.json`](evidence/lagrange-cpu.json): a historical summary
  of an actual two-backend run. Hashes identify that run, not a reusable proof
  attestation for another deployment.

Lagrange's identity is substantial algebra, but `ring` makes it automation-friendly.
The examples demonstrate integration and native checking, not an Olympiad solve
rate or comparative superiority of one backend.

## 1. Check the reference proofs without loading a model

Use the pinned Mathlib checkout with Lean `v4.10.0-rc1` and Mathlib
`29dcec074de168ac2bf835a77ef68bbe069194c5`:

```bash
PROVIDER=/absolute/path/to/Mathematics-ai-atp
MATHLIB=/absolute/path/to/pinned/mathlib4
EXAMPLES="$PROVIDER/examples/formal_v2"

cd "$MATHLIB"
lake env lean "$EXAMPLES/VerifiedProofs.lean"
```

The command must exit successfully. The three `#print axioms` statements report
their dependencies; none may contain `sorryAx`.

The reference file is **not** sent to the models. A successful reference proof
does not imply a successful bounded neural search.

## 2. Start a provisioned service

Install the provider and its matching persistent-atp client in the Python
environment used for this example. The client pins the provider's published
schemas and checks `/formal-capabilities` before submitting work. Numeric
`protocol_version: 2` alone is not sufficient to establish compatibility.

The configuration directory must contain:

```text
service/
├── mcts-service.json          # ServiceConfig pointing to the verified MCTS deployment
├── mcts-rl.json               # Migrated checkpoint, retrieval assets and native paths
├── frozen-service.json        # Supervised frozen-GNN service configuration
├── frozen-runtime.json        # Pinned GNN bundle, source project and frozen Pantograph
└── formal-deployments.json    # Deployment/profile registry and optional default
```

Portable frozen configuration templates are in
[`docker/config`](../../docker/config). Provisioning and exact model/runtime pins
are documented in the [integration runbook](../../maths_ai/gnn_inference/docs/formal_release_integration.md)
and [MCTS deployment guide](../../maths_ai/gnn_inference/docs/mcts_hugging_face_deployment.md).
Set each template's paths and `rl_config` filename for the chosen directory.
Use short runtime `state_dir` paths: Linux Unix-domain socket paths have a limited
length; a deeply nested checkout path is unsuitable for the runtime socket.

In the first terminal:

```bash
PROVIDER=/absolute/path/to/Mathematics-ai-atp
CFG=/absolute/path/to/service
PYTHON=/absolute/path/to/installed/environment/bin/python

FORMAL_V2_PYTHON="$PYTHON" PORT=39743 \
  bash "$PROVIDER/examples/formal_v2/start.sh" "$CFG"
```

The launcher defaults to loopback port **39743**, uses the configured model
workers, and prevents an unrelated checkout on `PYTHONPATH` from shadowing the
installed provider. Runtime state and model artifacts remain local; they are not
part of these examples.

- API browser: <http://127.0.0.1:39743/docs>
- Readiness: <http://127.0.0.1:39743/ready>
- Capabilities: <http://127.0.0.1:39743/formal-capabilities>

Run one service per configured runtime state directory. If the service is already
running, use the demo against it instead of launching another owner of that state.

## 3. Submit statements and inspect fresh evidence

In a second terminal, using the same installed environment:

```bash
"$PYTHON" "$PROVIDER/examples/formal_v2/demo.py" \
  --url http://127.0.0.1:39743 --example lagrange --backend both
```

The demo submits a separate run to each model, displays its proof trace and
budget usage, then requests independent replay. Look for `status: verified` and
explicit `verification_cost.sorry_detected: false` in the receipt.

Generated output is written under `artifacts/formal-v2-examples/`, with a unique
directory per invocation:

```text
lagrange-<id>/
├── example.json
├── mcts-request.json
├── mcts-result.json
├── mcts-binding.json
├── mcts-certificate.json
├── mcts-replay.json
├── gnn-request.json
├── gnn-result.json
├── gnn-binding.json
├── gnn-certificate.json
├── gnn-replay.json
└── summary.json
```

For a bounded challenge or a single backend:

```bash
"$PYTHON" "$PROVIDER/examples/formal_v2/demo.py" --example cauchy-schwarz --backend gnn
"$PYTHON" "$PROVIDER/examples/formal_v2/demo.py" --example crt --backend both
```

Exit code `0` means every selected backend produced an independently verified
proof. Exit code `1` means a selected search/replay did not verify, or an
operation failed. A bounded failure is not a refutation. Requested limits can be
lowered by provider ceilings; the admission response reports the effective limits.

The demo uses example search identifiers. It does not create scheduler leases or
promote persistent claims. The coordinator's leased admission/gate workflow is
documented in the paired persistent-atp repository under `schemas/FORMAL_V2.md`.

## Reproducibility boundaries

- Use the matching provider/client contract revisions; schema drift fails closed.
- Original trained weights are retained. MCTS migration updates deployment
  metadata only after tensor/library/declaration checks.
- A search stays `proved-pending-replay`; independent replay is separately
  budgeted. Neither an API result nor this demo authorizes claim promotion.
- Retry restarts bound input with remaining allowance; saved-frontier resume is
  unsupported. Retry/replay POSTs are not automatically repeated after a lost
  response.
- Failed tactic attempts during search are expected. The final result and native
  replay receipt determine success.
