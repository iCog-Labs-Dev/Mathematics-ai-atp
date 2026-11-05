# Compiler-aware Pantograph application integration

The application pins standalone client
`e8aa9ecc17686cfa2703c7082c02d45310ce421f` (package 0.5.0) and custom REPL
`fc740c26a1e832ecb571859af24a5c632a351bd9`. Overall protocol version 3 is
unchanged; successful transitions require validation version 3 and scope
`execution-closure-and-representation-aware-audit`.

## The incident

Periodic evaluation on JupyterHub ended in `RetainedTacticExecutor.apply()`:

```text
execution owner=declaration _private.0.Nat.pow.match_1.splitter._rarg._cstage2
path=environment audit/declaration-type/binder-type:
Missing proof dependency _obj
```

It reproduces natively with only `Init` imported, in a fresh process, with one
tactic: `unfold Nat.pow`, `simp only [Nat.pow]` or `rw [Nat.pow]`. Each needs the
equations of `Nat.pow`, which generates the match splitter on demand and
compiles it. The code generator then stores `_cstage1`/`_cstage2` entries in the
retained environment; their types use the compiler sentinel `_obj`, which the
version-2 audit read as a missing proof constant. The tactic itself is valid
Lean. Model weights, embeddings and vocabularies are not involved.

Version 3 reads those entries as compiler data when the compiler registered IR
for their owner, audits them as runtime data, and rejects any proof that refers
to one. The REPL evidence and classifier are described in the companion's
`doc/compiler_aware_validation.md`.

## Application changes

- Dependency, lockfile, installation requirements, diagnostic pins and Docker
  use the matched pair. The lockfile change is the client entry only.
- Current fake-server responses and typed fixtures use validation version 3.
  Historical captures and source manifests are unchanged; the new source-lineage
  record `tests/fixtures/pantograph/compiler_aware_validation/` follows the
  persistent-publication record.
- `maths_ai/hybrid_reasoner/command_journal.py`: an opt-in, bounded journal at
  the shared retained-execution boundary. With `MATHS_AI_COMMAND_JOURNAL=/path`
  set, every command is written before dispatch (phase `operation`, `replay` or
  `artifact`, session, scope, state, goal, target, command) and its outcome after
  (`accepted`, `rejected`, `resource` or `terminal` with the structured error
  payload). Lines are flushed immediately and the file stops at 8 MiB.
- `pantograph_diagnostics.certify_compiler_control`: the compiler-producing
  control (`intro m n`, `unfold Nat.pow`, `split <;> rfl`, then kernel
  certification). Deployment migration runs it after the existing `trivial`
  control, and the live application smoke runs it after the graph checks.
- `tests/test_compiler_aware_validation_live.py`: both searches with a
  one-command and a two-step compiler-producing proof, a multi-goal proof that
  continues after the compiler step, RL collection followed by an on-policy
  optimizer update and shared evaluation in greedy and sampled modes, and the
  journal on a successful search and on a real terminal integrity failure.

Validation errors stay terminal: search treats them as infrastructure errors,
RL collection as fatal session errors, and admission skips the row and restarts
the session. None of these paths records an actor rejection sample.

## Update JupyterHub

```bash
conda activate atp-mcts
python -m pip install --upgrade --no-deps \
  'pantograph-async-client @ git+https://github.com/123nol/Pantograph-Client@e8aa9ecc17686cfa2703c7082c02d45310ce421f'
python -m pip check
python - <<'PY'
from pantograph_async import TRANSITION_VALIDATION_VERSION
from pantograph_async.compatibility import companion_base
print("Validation version:", TRANSITION_VALIDATION_VERSION)
print("Companion:", companion_base()["source_commit"])
PY
```

Build the REPL in a new directory and keep the current one, which matches the
existing inventory:

```bash
git clone --no-checkout https://github.com/123nol/Pantograph.git \
  /home/jovyan/Pantograph-compiler-aware
git -C /home/jovyan/Pantograph-compiler-aware checkout --detach \
  fc740c26a1e832ecb571859af24a5c632a351bd9
cd /home/jovyan/Pantograph-compiler-aware
lake build repl
cd /home/jovyan/Nol/Mathematics-ai-atp
```

Rebind the active deployment without changing any model tensor. Take the paths
from the configuration that is actually in use; if the scope-aware bundle is
still active:

```bash
python -m maths_ai.gnn_inference.scripts.rebind_source_aware_environment \
  --config artifacts/source_aware_deploy_scope_aware/rl.json \
  --old-repl /home/jovyan/Pantograph-scope-aware/.lake/build/bin/repl \
  --repl /home/jovyan/Pantograph-compiler-aware/.lake/build/bin/repl \
  --output artifacts/source_aware_deploy_compiler_aware
```

The output must be a new name on persistent storage. The command, its checks,
publication layout and retry are unchanged from
[the scope-aware handoff](pantograph_scope_aware_application_integration.md);
its native control now also certifies the compiler-producing proof. The receipt
reports `transitionValidationVersion: 3`. A changed toolchain or compiled
Mathlib is refused; the REPL-only repair needs neither retraining nor a fresh
critic.

To capture a command history if a terminal error recurs:

```bash
export MATHS_AI_COMMAND_JOURNAL=/home/jovyan/Nol/Mathematics-ai-atp/runs/commands.jsonl
```

## Verification

Local, CPU only, with the pinned client, the companion executable built from
`fc740c2` and the pinned Mathlib `29dcec07`:

- Offline application/API suite: 910 passed, 56 skipped, 35 subtests passed.
  Two failures also fail on the base commit here: the `stdbuf` path test (this
  host's default `stdbuf` is the Rust one) and a 6e-8 float tolerance in
  `test_source_aware_policy`.
- Native live suite against the companion and Mathlib: 48 passed, including
  the new compiler-producing tests in both searches, the existing scope-aware
  tests, and deployment rebinding from the validation-version-2 executable.
  Four live tests fail identically on the base commit with the version-2
  pair: two need a provisioned PeTTa (`PETTACHAINER_ROOT`), two need checkpoint
  artifacts this host does not have.
- Real Uvicorn/FastAPI proving and evaluation over HTTP: passed (PLN variant
  skipped).

Not run here: the Docker image build and container smoke (no Docker daemon),
PLN-enabled native search (no provisioned PeTTa runtime locally), the
trained-checkpoint GPU trial and the JupyterHub rebind. Run the short
best-first PLN trial and a no-PLN PUCT smoke on JupyterHub with the rebound
bundle before long jobs.
