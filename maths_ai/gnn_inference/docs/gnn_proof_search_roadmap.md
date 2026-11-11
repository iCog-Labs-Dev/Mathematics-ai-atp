# Five-day GNN-only prover roadmap

## Goal and ownership

The GNN policy proposes every candidate action. The controller owns native Lean
snapshots, the frontier, scores, budgets, accepted paths, and failures.
DeepSeek-Prover-V2 remains a separate baseline; it supplies no proposals,
repairs, fallback proofs, or search rewards to the GNN experiment.

| Day | Deliverable | Exit check |
| --- | --- | --- |
| 1 | Native state → normalized graph → accepted GNN → rendered tactic → native successor | Current locals resolve correctly; alternative branches are independent; sibling goals survive. |
| 2 | Bounded policy-guided best-first search; structured-decoder pilot starts | Existing policy searches end-to-end while the new decoder trains. |
| 3 | Structured decoder integrated into the same policy interface | Old and new policies compared under identical search budgets. |
| 4 | Frozen checkpoint, cohort, settings, and evaluation | Verified solves and calls/trials/time recorded reproducibly. |
| 5 | Completion buffer, proof rechecks, results, and demo | Reconstructible proofs and documented artifacts. |

The first search uses accumulated action negative log-probability and explicit
depth/node/trial/time budgets. A learned value function or PUCT is not required
for this milestone. The full pending-goal state belongs to each node; finishing
one focused goal is not completing the theorem.

## Day-one implementation

- `atp_lean_gnn/live_state.py`: strict version-1 normalized model S-expressions,
  offline-compatible graph construction, runtime graph-node/local-context mapping,
  and direct pinned-Pantograph snapshots.
- `atp_lean_gnn/live_policy.py`: accepted joint bundle, matched text retrieval,
  optional accepted local cues, and rendered candidates with log probabilities.
- `scripts/run_gnn_live.py`: bounded execution smoke runner. It tries candidates
  from one parent until one executes, then continues from that child. It does
  not maintain a search frontier or backtrack to earlier proof steps; use the
  separate best-first runner below for that behavior.

Each top-ranked tactic family gets its existing greedy argument-count/source/
identity decode. Thus top-K is over tactic families, not argument-sequence beam
search. The action score includes tactic, count, and every source/identity
decision; the auxiliary stop head is not used by predicted-count decoding.

Selected local IDs are graph node IDs, not Lean context indices. The bridge
resolves actual `FV`/`Hyp` nodes through the original DAG, retaining identity
even when display names or FV labels are outside the training vocabulary.
Other selected expression nodes are explicit rendering failures, not fabricated
local or library references. Unknown tactics, ambiguous/inaccessible local names,
and incomplete decodes remain explicit rejected candidates.

### Native-state invariants

The pinned backend's automatic resumption duplicates unassigned goals for no-op
tactics. This adapter disables automatic mode and explicitly resumes the union
of child and parent goal identities, preserving order and deduplicating IDs.
Lean removes goals assigned by the action. Failed alternatives leave their
saved parent available.

`start_declaration(header)` preserves the frontend's local-context slots,
including hidden declarations. Its temporary initialization hole is reopened as
an unsolved native obligation; the initialization stub is never a model action
or a verified proof. `start(expression)` instead begins from a fresh quantified
proposition. These contexts can have different FV indices even for equivalent
sequents. Do not compact or renumber one to make them look identical.

Retrieval receives readable text reconstructed from native pretty fields,
including local let values (`live_pretty_fields_v1`). This is not guaranteed to
match the dataset's original grouped pretty-state text byte for byte. Graph
encoding uses normalized expressions, never reparsed printed Lean expressions.

### Verification and accepted-weight integration

Regression coverage includes native introduced-local lookup, sibling preservation,
no-op deduplication, failed/alternative branch isolation, local lets and instances,
and exact DAG parity between a reopened declaration and its frontend capture.
Controlled GNN/GRU fixtures exercise model → render → native execution; these
checks do not measure the accepted checkpoint's prediction quality.
The day-one focused regression run passed **64 tests**. The combined current
regression run passes **90 tests**, including **13 live-policy checks**,
**19 search checks**, and seven structured-supervision checks; six checks exercise
native Lean. All three CLI help paths load.

The accepted-checkpoint live loop now passes locally, and the accepted model's
best-first proofs pass independent compilation and axiom checks on four small
integration targets. The smoke runner still correctly labels its own zero-goal
output `closed_unverified`, with `proof_verified=false`; the separate search
runner performs independent proof verification.

The public release at `iCogLabs-Dev/Mathlib-Sexpr-GNN`, revision
`8aba62625dfeb833e4591498700f351f3bc60af9`, records the expected joint epoch 3,
local-refinement epoch 8, and retriever epoch 5. The entire local bundle download
is now complete, with all manifest members SHA-256 verified. The documented server
export location in `huggingface_joint_release.md` is
`/home/jovyan/yonatan/Mathematics-ai-atp/maths_ai/_support_files/artifacts/hf_joint_model`.
Its current server existence is not verified locally. Use that complete export
for full-Mathlib checks after deploying this code.

#### Real-weight Init-only checks

Environment: pinned Pantograph `73781c2d58456e4bf369dadd4a501e1b78a0b177`, Lean
`4.10.0-rc1`, Python `3.11.15`, Torch `2.12.1+cu130`, Transformers `5.14.1`,
FAISS `1.14.3`; CPU inference with `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`.
Runtime imports are **Init only**, with Lean imported by the independent checker.
The complete matched corpus/index/retriever is loaded; neither weights nor
retrieval candidates are replaced with fixtures. Profile: `accepted_local_cues`.

Each search used eight model calls, 24 trials, 24 nodes, depth eight, top eight
tactic families, and 60 active seconds as its upper budgets.

| Target | Verified winning actions | Calls | Trials | Active seconds |
| --- | --- | ---: | ---: | ---: |
| `(P : Prop) (h : P) : P` | `exact h` | 1 | 1 | 0.58 |
| `(P Q : Prop) (hp : P) (hq : Q) : P ∧ Q` | `simp_all` | 2 | 6 | 0.95 |
| `∀ (P : Prop), P → P` | `intro` → `simp` | 3 | 11 | 0.64 |
| `(P Q : Prop) (h : P → Q) (hp : P) : Q` | `apply h` → `exact hp` | 2 | 9 | 0.62 |

All four fresh compiler runs returned exit code zero with no forbidden axioms.
The conjunction search also executed a model-proposed `constructor` branch and
preserved its two pending goals before choosing the shorter root `simp_all` proof.
Thus these results check actual state transitions and alternative retention,
not merely a rendered tactic string. They are deliberately tiny integration
examples, not benchmark accuracy or evidence of superiority to DeepSeek.
Bundle loading and native-process startup are outside active search timing.

Reports live under `maths_ai/gnn_inference/runs/`:

| Report | SHA-256 |
| --- | --- |
| `gnn_accepted_search_identity.json` | `64ab67384b45eaf7f8c83878ca43f1a43ecc28e5a0bef23cc8cb1114fd688ded` |
| `gnn_accepted_search_conjunction.json` | `3c094cb1352a72ca251d39fc0720a0b90b22f49aa7b59006fb4122917dadc85a` |
| `gnn_accepted_search_introductions.json` | `d0a92da900a7a5a95b1bb5f0db3dca64f582fe96b0b59b3bd63581ca0dbfb2a2` |
| `gnn_accepted_search_application.json` | `98bf9c4d0497b3a9a917a73d610f5fde453b7c3ba2c6e4b6e1efa10f648aa270` |

Policy state hash: `a16d7f63ddde721c35611dcc2c604096cbe31d17490407669e2516c72d38e2ce`.
Retriever state hash: `48ac6ab921bd8a6a7c2c49b3fdc307b97293a9f3737f33484673141f456aa5fd`.
Full Mathlib environment integration and a fixed benchmark cohort remain next gates.

## Best-first prototype

- `atp_lean_gnn/gnn_search.py`: native parent/action frontier, ordered by cumulative
  action negative log-probability. Branches remain independently available after
  another candidate executes; terminal candidates are checked in frontier order.
- `atp_lean_gnn/proof_verification.py`: a fresh Lean compiler reconstructs the
  complete path against the fixed trusted target header. Each action is wrapped
  in `focus` to reproduce native stepping while preserving sibling obligations.
  Compilation must succeed and the theorem must have an explicit axiom audit
  containing only `propext`, `Classical.choice`, and/or `Quot.sound` (or no axioms).
  `sorryAx`, incomplete siblings, missing audits, and compile failures cannot count.
- `scripts/run_gnn_search.py`: one-target runner, bundle/runtime/code bindings,
  atomic JSON checkpoints, and compact progress. Exit status zero means a verified
  proof, rather than a native zero-goal state alone.

Default budgets: **32 model calls, 96 native tactic trials, 64 controller nodes
including the root, depth 16, eight tactic families per expansion, 120 active
seconds, 20 seconds per native step, 30 seconds per proof check**. Verification
also consumes the remaining active-time budget. Native process startup and bundle
loading precede active search timing. Synchronous model calls stop cooperatively:
an overrun is recorded, and no next native trial begins after the budget expires.

One expansion per node uses the existing greedy argument decoder. No proof LLM,
automation fallback, value network, or printed-goal state merging is introduced.
Duplicate candidate strings are skipped within a parent. A model-call limit stops
new expansions but permits already-queued actions to finish within other budgets.
Reports checkpoint progress; restoring an interrupted in-memory frontier is not
implemented in this prototype.

Tests verify best-first path ordering, retained alternatives, all work budgets,
duplicate/invalid-score rejection, timeout/input accounting, rejected terminal
candidates, a three-action native proof with two sibling obligations, kernel
rejection of incomplete/placeholder proofs, and the full CLI checkpoint path.
Native search decisions in the regression suite are scripted/controlled fixtures.
The separate real-weight checks above establish accepted-checkpoint integration;
neither set measures a benchmark theorem solve rate.

Training scope starts with grounded, exact-aligned structured targets and a new
decoder, freezing the GNN and matched retriever initially. V9's static 99.5%
representation coverage does not establish native validity, premise grounding,
or generation accuracy for every row. Full source-program/deferred-name handling
is an expansion target after the working prover and decoder pilot are secured.

The bounded candidate-preparation entrypoint is now implemented in
`scripts/prepare_structured_supervision.py`; see
[structured-supervision pilot](structured_supervision_pilot.md). Server-side V9
caches are required for the actual data export. Binder canonicalization,
train-only neural vocabulary/codec construction, and decoder training remain next.
