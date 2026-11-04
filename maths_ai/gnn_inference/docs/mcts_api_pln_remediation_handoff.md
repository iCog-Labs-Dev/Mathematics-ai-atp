# MCTS service remediation and PLN deployment handoff

The implementation integrates the two PLN patches from `MCTS-AC-branch`
(`119eacd`, `7cf9373`) with the API extraction at `439b361`. Evaluation stays
in the shared runtime modules; the moved contracts retain PLN translation
identity, counters and exact-resume checks. Historical source manifests are
preserved, with explicit integration and remediation records.

## Ownership and service behavior

- A Linux subreaper guardian owns each compute generation. It adopts orphaned
  Lean, source-decoder and PeTTa processes across independent sessions and
  waits for the entire tree to terminate. Linux pidfds prevent signalling a
  reused PID. Ownership startup requires supported x86-64/AArch64 Linux kernel
  facilities; failure makes the runtime unavailable.
- Proof queue expiration runs independently of the dispatcher. Total proof
  time includes queue wait. Evaluation deadlines start at durable submission.
  Cleanup holds execution ownership after a response deadline expires.
- SQLite admission atomically checks capacity and inserts jobs. Once accepted,
  a dataset job survives disconnection of its submitting HTTP client.
- Schema version 2 migrates version 1 and publishes interrupted partial
  results from committed attempts on restart. Interrupted jobs are not resumed.
  Result versions are immutable and published before database references commit.
- Admission steps consume source-row events, including offline rejections.
  A spawned decoder owns blocking source setup/downloads; one requested row
  is in flight. It can be terminated on cancellation without leaving a blocked
  thread. No CUDA model or Lean handle is sent to that decoder.
- PLN workers share one bounded subprocess owner. Slot wait counts toward
  the query/search deadline; cancellation reaps the query group before reuse.
  Output, pending work and temporary query-file lifetimes are bounded.

## Select PLN through server profiles

The supplied Docker service configuration includes `puct`, `best_first`,
`puct_pln`, and `best_first_pln`. A request selects a profile:

```bash
curl -s http://127.0.0.1:8000/prove \
  -H 'Content-Type: application/json' -d '{
    "goal": "True ∧ True",
    "search_profile": "puct_pln",
    "proposal_mode": "sample",
    "seed": 42,
    "timeout_s": 60
  }'
```

The server owns the executable/source paths and bounds. A request cannot
choose them. A service advertising PLN profiles becomes ready only after
native configuration, Python embedding and an actual rule-backed query pass.
A configuration containing only disabled profiles does not execute PLN.

Both search modes attach PLN results to child obligations. Best-first uses
the combined node ranks. PUCT retains its existing actor priors, critic
evaluation and visit-based selection; enabling PLN does not substitute STVs
for the critic. Existing continuation scheduling can consume combined ranks.

Success still requires Lean replay and final kernel certification. A genuine
PLN score requires the requested target, a structured `rule-proof`, valid STV
and zero process exit. Unsupported translation and native failures receive no
STV. Completed queries without a qualifying derivation retain their explicitly
labelled exploration behavior. No parent statement is asserted as a child fact.

## Native environment

The Docker build uses Python 3.13.13 and explicitly builds SWI-Prolog 9.3.31
with Janus against that Python. Full source revisions are pinned in
`scripts/provision_pln.sh`; the custom REPL fetch uses its complete SHA.

For JupyterHub, from the repository root:

```bash
conda env create -f environment-mcts-ac.yml
conda activate atp
bash scripts/provision_pln.sh "$CONDA_PREFIX/opt/pln" "$CONDA_PREFIX/bin/python"
source "$CONDA_PREFIX/opt/pln/activate.sh"
python -m pip check
```

For an existing environment, use `conda env update -n atp -f
environment-mcts-ac.yml` instead of creation. The YAML installs Python packages
and native build prerequisites. The explicit provisioning command builds SWI,
PeTTa and the chainer; a generic prebuilt SWI package is not used as a substitute
for the required Janus build.

The generated SWI launcher binds Python's base library and the selected
environment's site-packages. This matters for virtual environments and Conda.
Activating the generated file defines `PETTA_BIN`, `PETTA_SOURCE_ROOT`,
`PETTACHAINER_ROOT`, `SWIPL_BIN` and executable lookup paths. Provisioning runs
a real rule-proof readiness check and writes `deployment.json`.

With the existing matching model, retrieval and Lean artifacts available:

```bash
python -m maths_ai.gnn_inference.scripts.rl_train --config /absolute/trial.json
python -m maths_ai.gnn_inference.scripts.rl_train --config /absolute/eval.json --eval-only
```

Set `use_pln: true` in CLI configurations. API configurations use the named
profiles independently. Live native identities are recorded in provenance;
exact RL resume rejects a changed PLN environment. Compatible model weights
can still initialize a fresh run.

Run the full import controls separately after provisioning:

```bash
python -m maths_ai.pln_inference.diagnose \
  --petta-source-root "$PETTA_SOURCE_ROOT" \
  --output artifacts/pln_checks/new_install
```

The output directory must be new. The unanswered conjunction control remains
a recorded inference limitation; it is not required to produce a fabricated STV.

## Deployment artifacts and acceptance limits

The image includes Lean, the custom REPL, compiled Mathlib and PLN runtimes.
Mount the version-4 policy, matching prepared metadata, query encoder, full
lemma index/manifests, corpus and declaration inventory read-only. Persist
`/state` separately. Rebuilding the Lean environment can require regenerated
inventory/deployment bindings and checkpoint conversion. Do not edit hashes.

Validation evidence and the final test counts are recorded with the completed
implementation. Native tests use controlled policies rather than a trained
checkpoint benchmark. Python 3.13 native provisioning is separate from the
Torch-based HTTP tests run with the existing Python 3.11 test environment and
its own matching isolated Janus build.

The local WSL environment has no usable Docker daemon. An actual full CPU image
build/container smoke and the JupyterHub GPU/trained-artifact run remain
deployment acceptance gates. Build-file syntax or successful host tests cannot
establish those results. No code is pushed or PR merged by this handoff.
