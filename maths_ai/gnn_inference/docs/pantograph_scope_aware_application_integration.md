# Scope-aware Pantograph application integration

The application pins standalone client
`053ff67dfc540391100c86e4deada06839904655` and custom REPL
`ba2c3dd8383f90b4b5327e31a645a53411f385e4`. The client package is version
0.4.0. Overall protocol version 3 is unchanged; successful transitions require
validation version 2 and scope `execution-closure-and-transition-audit`.

The REPL now checks retained executable dependencies under their owning local
contexts and separately audits auxiliary elaboration records. A valid `tauto`
can therefore succeed when its discarded elaboration work has left an inert
assignment. Real dangling executable references remain fatal integrity errors.
The application continues to use its retained snapshots, sequential replay,
and original-root kernel certification. Graph readiness alone is not success.

## Application changes

- Dependency, lockfile, installation requirements, diagnostic pins, and Docker
  use the matched client/REPL pair. Setup obtains the companion contract from
  the installed package.
- Current fake-server responses and typed fixtures use validation version 2.
  Historical captures and their source manifests remain unchanged. The new
  source-lineage record follows the transactional-resource integration.
- Native regressions cover conjunction, disjunction, the supplied Finset
  statement, and a multi-step proof with sibling obligations, using PUCT and
  best-first search. Additional tests exercise RL collection, shared evaluation,
  certification-aware API results, and fatal validation failure.
- The deployment-migration command verifies the complete old model/retrieval
  chain and preserves every learned model tensor, including the critic and
  actor residual. It does not initialize a new critic or retrain either model.

## Update JupyterHub

Run from the application checkout after pulling this change. Activate the
environment that will run the trial:

```bash
conda activate atp-mcts
python -m pip install --upgrade --no-deps \
  'pantograph-async-client @ git+https://github.com/123nol/Pantograph-Client@053ff67dfc540391100c86e4deada06839904655'
python -m pip check
python - <<'PY'
import pantograph_async
from pantograph_async.compatibility import companion_base
from pantograph_async.diagnostics import distribution_source
print("Imported client:", pantograph_async.__file__)
print("Installed source:", distribution_source())
print("Companion:", companion_base())
PY
```

Build in a separate directory to preserve the executable used by the existing
inventory. If the destination already exists, inspect it and use another new
directory rather than overwriting an installation:

```bash
git clone --no-checkout https://github.com/123nol/Pantograph.git \
  /home/jovyan/Pantograph-scope-aware
git -C /home/jovyan/Pantograph-scope-aware checkout --detach \
  ba2c3dd8383f90b4b5327e31a645a53411f385e4
cd /home/jovyan/Pantograph-scope-aware
lake build repl
cd /home/jovyan/Nol/Mathematics-ai-atp
```

The compatible compiled Mathlib project can remain at its current path. The
migration checks its toolchain, lockfile, ordered imports, every compiled
`.olean` path/content, and declaration availability.

## Migrate an existing deployment in one command

Supply the existing RL configuration that points to the current checkpoint,
prepared root, query export, index, corpus, and inventory. The artifacts may be
stored in separate directories. Relative configuration paths are interpreted
from the working directory, as they are by `rl_train`.

For an inventory generated with the transactional executable:

```bash
python -m maths_ai.gnn_inference.scripts.rebind_source_aware_environment \
  --config maths_ai/gnn_inference/configs/rl_mcts_joint_epoch5_trial.json \
  --old-repl /home/jovyan/Pantograph-transactional/.lake/build/bin/repl \
  --repl /home/jovyan/Pantograph-scope-aware/.lake/build/bin/repl \
  --output artifacts/source_aware_deploy_scope_aware
```

`--old-repl` must identify the executable used to generate that inventory. If
the inventory instead belongs to `Pantograph-companion`, supply that executable.
The command reproduces the recorded identity before changing any bindings.
It refuses a changed Mathlib, mismatched model/index/corpus, or an unverified
old identity. Merely supplying an old commit name or aggregate hash does not
replace this check.

The command performs these operations:

1. Load the old version-4 actor-critic through the ordinary strict loader and
   verify query/index/inventory/corpus bindings through the retrieval provider.
2. Establish that the imported compiled library is unchanged.
3. Generate declarations through Lean and compare them, preserving exclusions.
4. Start the new client/REPL and obtain a kernel certificate for a control goal.
5. Copy verified serving files and update the environment/index/query bindings.
6. Load all actor-critic tensors into the rebound specification, serialize the
   checkpoint, reload it, and compare every tensor's name, shape, dtype, and
   exact value.
7. Validate the staged bundle through the ordinary loaders, write a migration
   receipt, and publish a relative symlink to its complete backing directory
   without replacing an existing output.

The command first probes relative-symlink publication on the actual output
filesystem, before loading models or exporting declarations. It no longer
requires `renameat2`/`RENAME_NOREPLACE`. Space reporting gives an estimated
full-copy size and currently available bytes. Reflinks can reduce storage use,
but support and quota must be established on the actual persistent filesystem.
The tool does not require enough free space for the worst-case estimate when
reflinks may succeed; an actual space/quota failure aborts copying with the
source and destination paths.

The output name is a relative symlink to a unique sibling backing directory:

```text
artifacts/
├── .source_aware_deploy_scope_aware.bundle-<unique-id>/
└── source_aware_deploy_scope_aware -> .source_aware_deploy_scope_aware.bundle-<unique-id>
```

Both names belong to the deployment. Keep the hidden backing directory;
removing it breaks the published output. The symlink resolves to a bundle
containing:

```text
source_aware_deploy_scope_aware/
├── checkpoint.pt
├── prepared/
│   ├── vocab/node_vocab.json
│   ├── vocab/tactic_vocab.json
│   └── metadata/graph_representation.json
├── query/
├── index/
├── lemmas.jsonl
├── declarations.json
├── migration_receipt.json
└── rl.json
```

The prepared directory contains the metadata used for live model loading;
training graphs and candidate caches are not copied. The complete query and
index serving files are retained. Files use independent filesystem reflinks
where available, otherwise verified copies. Artifact files inside the bundle
do not use hard links or symlinks to external source artifacts. The root
publication symlink resolves to the persistent sibling backing directory;
normal loader containment checks still apply inside each artifact root.

The generated `rl.json` points to the new bundle and REPL, creates a separate
run root and admission cache, and preserves the original search/training
settings. Review its trial limits before starting:

```bash
python -m maths_ai.gnn_inference.scripts.rl_train \
  --config artifacts/source_aware_deploy_scope_aware/rl.json
```

Start a fresh run without `--resume`. Optimizer states, queues, curriculum,
round number, and previous best-score incumbents are not copied into the
warm-start checkpoint. The original deployment and checkpoint remain intact.

Inspect the verification receipt:

```bash
python - <<'PY'
import json
from pathlib import Path
r = json.loads(Path("artifacts/source_aware_deploy_scope_aware/migration_receipt.json").read_text())
print("Mode:", r["mode"])
print("All tensors preserved:", r["all_model_tensors_preserved"])
print("Matching model hashes:", r["model_sha256_before"] == r["model_sha256_after"])
print("Validation version:", r["protocol"]["transitionValidationVersion"])
print("New environment:", r["new_environment_identity"])
print("Publication:", r["publication"])
PY
```

## Persistent filesystem checks and publication retry

Use the persistent output path directly. Mounting the same persistent
filesystem over `/tmp` does not change its supported operations. A migration
into `/tmp` also needs full-copy space for artifacts on another filesystem
and produces temporary artifact/run paths.

To check the persistent publication operation without starting migration:

```bash
python - <<'PY'
from maths_ai.gnn_inference.atp_lean_gnn.bundle_publication import preflight_output
output = preflight_output("artifacts/source_aware_deploy_scope_aware")
print("Publication preflight passed:", output)
print("Only temporary probe entries were created and removed.")
PY
```

An existing output, including a dangling symlink, is refused. Concurrent
different publishers cannot replace one another. A handled failure or cancellation
before bundle validation removes that attempt's incomplete data. Abrupt process
termination can leave an incomplete backing directory; publication retry
refuses one without a complete valid receipt and artifact inventory. A publication failure
after validation retains the backing directory and prints its exact retry
command, for example:

```bash
python -m maths_ai.gnn_inference.scripts.publish_source_aware_bundle \
  --backing /home/jovyan/Nol/Mathematics-ai-atp/artifacts/.source_aware_deploy_scope_aware.bundle-EXAMPLE \
  --output /home/jovyan/Nol/Mathematics-ai-atp/artifacts/source_aware_deploy_scope_aware
```

Replace `EXAMPLE` with the retained directory reported by the tool. Retry
verifies the receipt, complete artifact inventory, hashes, model tensors,
configuration paths, client identity, and compiled environment before
publication. It runs a native certification control, but does not copy the
index again, regenerate embeddings, or reinitialize any model weights. An
already published output is accepted only when it names that same verified
backing directory. Later files under `runs/` and the generated admission cache
are runtime outputs, not trusted model artifacts.

Retry requires the current version-2 migration receipt containing publication
identity evidence. The older implementation deleted staging on publication
failure, so those failed attempts have no retained bundle to retry. Run a new
migration after pulling this repair. Keep `--output` at the original intended
path; retry cannot silently redirect the configuration to another location.

Space errors during byte copying still require adequate destination capacity
or quota. They are separate from publication support. The persistent output
avoids the extra full copy into `/tmp`, but does not guarantee reflink support.

For Docker, bind-mount the resolved backing directory at the artifact mount,
or mount the common parent containing both the relative symlink and its
backing directory. Verify the daemon can access the resolved source. When
archiving or uploading a standalone bundle, copy/dereference the published
root into a self-contained export; uploading only its symlink omits the data.
Generated configuration paths remain specific to the deployment location.

## Subsequent REPL iterations

Inventories generated by this change include component evidence bound to their
environment identity. Later migrations compare those components with the
current library and can omit `--old-repl`:

```bash
python -m maths_ai.gnn_inference.scripts.rebind_source_aware_environment \
  --config artifacts/source_aware_deploy_scope_aware/rl.json \
  --repl /absolute/path/to/the/next/.lake/build/bin/repl \
  --output artifacts/source_aware_deploy_next
```

This still verifies the new runtime and regenerates environment bindings. It
does not recompute lemma embeddings, export the query model again, or repeat
supervised-to-actor-critic conversion. The existing artifact schema is retained;
separating runtime and model identities is a separate future schema change.

If an older inventory has no component evidence and its executable is lost,
the narrow migration refuses to proceed. Preserve the original model and
establish a freshly verified deployment with an explicit tensor-preserving
export. Do not replace stored hashes or initialize a new critic as a repair.

## Native verification

Use explicit paths to the new executable, compiled Mathlib, the built REPL
test-fixture project, and the preserved old executable:

```bash
export PANTOGRAPH_TEST_SOURCE_ROOT=/absolute/mathlib4
export PANTOGRAPH_TEST_REPL=/absolute/Pantograph/.lake/build/bin/repl
export PANTOGRAPH_RESOURCE_SOURCE_ROOT=/absolute/Pantograph
export PANTOGRAPH_TEST_OLD_REPL=/absolute/previous/repl
python -m pytest \
  maths_ai/gnn_inference/tests/test_scope_aware_validation_live.py \
  maths_ai/gnn_inference/tests/test_deployment_migration.py \
  --run-pantograph-live -q
```

The fatal-integrity control imports `Test.ContractFixtures`. Build that fixture
with the REPL native test targets before running the control. It is not added
to production imports. CPU-native acceptance is separate from a Docker image
build and a trained-checkpoint GPU trial on JupyterHub.

For Docker, rebuild the image and mount the freshly verified artifacts with
container-specific configuration paths. A native-host bundle is not assumed
to match a separately built container binary or Mathlib tree. Environment
fingerprints must be generated or migrated against the actual container
runtime before readiness can succeed.

## Local verification, 2026-11-05

The [verification record](../tests/fixtures/pantograph/scope_aware_validation/verification.json)
records completed runs and their limits. The application/API regression sweep
passed 778 tests with 35 subtests; final migration and source-lineage checks
run separately. Native controls passed for both search modes, the supplied
Finset theorem, sibling replay, RL/evaluation acceptance, fatal integrity
failure, and genuine heartbeat recovery. A real Uvicorn process served proof
requests and an evaluation job successfully.

Native deployment migration used the preserved version-1 executable and the
new version-2 executable against the compiled Mathlib project. It generated
the real declaration inventory, obtained a kernel receipt, preserved all model
tensors, and loaded the new retrieval/checkpoint chain successfully. The models
and retrieval artifacts in that test were controlled local artifacts.

The test client was installed in an isolated temporary directory. The existing
developer environment and remote JupyterHub artifacts were not modified.
Docker image execution, Python 3.13 Conda creation, and the trained-model GPU
trial remain deployment checks.

## Persistent publication repair verification

The [publication verification record](../tests/fixtures/pantograph/persistent_publication/verification.json)
records this filesystem repair separately from the earlier validator migration.
The inference/API regression run passed 822 tests and 35 subtests. Focused
publication/retry/storage/tensor-preservation tests passed 49 checks. The native
old-to-new migration and retained retry passed against the actual compiled
Mathlib and custom REPL, including kernel certification and model loading.
These groups overlap and their counts must not be added together.

The user's JupyterHub filesystem must still run the new preflight and a short
persistent trial after pulling the repair. Neither remote symlink support nor
remote quota/capacity has been established by the local tests. No client or
REPL update is required specifically for this publication change.
