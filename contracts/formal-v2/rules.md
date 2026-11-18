# Resolution and hashing rules

Implemented in `maths_ai/formal` (`canonical.py`, `artifacts.py`,
`admission.py`, `source_elaboration.py`, `service.py`). Where this text and
the code disagree, the code is the bug: report it.

## 1. Canonical JSON and content addresses

### Deployment selection

`backend` and `deployment_id` may be omitted. The registry's `default` selects
the deployment and profile; admission resolves proposal mode/seed once and
persists that selection. Explicit selections override the default and must
match the deployment. Retry reuses the persisted selection, not a later default.
`/formal-capabilities` exposes the default and exact `schema_hashes` so the caller
can verify the full package revision before submitting work.

- A **content address** is `sha256:` followed by the lowercase hex SHA-256 of
  the exact bytes it names (pattern `^sha256:[0-9a-f]{64}$`).
- **Canonical JSON** is UTF-8 with:
  - keys sorted at every level;
  - no insignificant whitespace (`,` and `:` separators);
  - non-ASCII characters written directly, not `\u`-escaped;
  - no NaN or infinity.

  In Python: `json.dumps(value, sort_keys=True, separators=(",", ":"),
  ensure_ascii=False, allow_nan=False).encode("utf-8")`.
- **The address of a JSON object** is the content address of its canonical
  JSON bytes. Every JSON artifact the provider stores is canonical JSON, so
  fetching it and re-hashing the bytes reproduces its address.

## 2. Artifacts

### Upload and read

- **Upload:** `POST /formal-artifacts` with the raw bytes as the body (any
  content type) stores them and returns their content address (201). At most
  8 MiB; more is refused as `artifact-too-large` (413). Uploading the same
  bytes again returns the same address.
- **Read:** `GET /formal-artifacts/{address}` returns the exact bytes, with
  `X-Content-Hash`. Unknown addresses are 404 `artifact-missing`. The provider
  re-hashes on every read, and a stored file that no longer matches its
  address is refused as `artifact-corrupt`, never served.
- **No paths:** a request never names a filesystem path, import, executable
  or checkpoint. It names artifacts by address and a deployment by id.

### Source inputs (`input.kind = "source"`)

1. **Resolve.** `artifact_hash` must name an artifact already in the provider
   store; upload first. A missing artifact rejects the run as `invalid-input`
   with `input:artifact-missing`. The file must be UTF-8
   (`input:malformed-source` otherwise).
2. **Imports.** The header is the leading `import` lines; blank lines and
   comments may come between them. An `import` after the header is
   `input:malformed-source`. Every imported module must be `Init`, `Std` or
   `Lean` (or a submodule of one), or a deployment import or its submodule:
   `Mathlib` admits `Mathlib.Tactic` and so on. Anything else is
   `environment:unsupported-import`, with disposition `environment-error`.
3. **Locate the declaration.**
   - `declaration` is the full Lean name, for example `Demo.target`.
   - Names are resolved through the file's `namespace X` … `end X` blocks;
     `section` adds no prefix, and a `_root_.` prefix is honoured.
   - Only `theorem` and `lemma` are supported, with optional attributes and
     `private`, `protected`, `noncomputable`, `nonrec`, `unsafe` or `partial`
     modifiers. The name may be quoted with `«…»`.
   - The declaration must occur exactly once (`input:declaration-not-found`,
     `input:declaration-ambiguous`).
   - Its body must start with a top-level `:=`. Equation-style (`|`) and
     `where` bodies are `input:unsupported-declaration`.
4. **Elaborate** (`starting_state = "binders-introduced"`, the only supported
   convention):
   - The deployment's own REPL processes the body up to the target, with the
     target's proof replaced by `by sorry`; later declarations are dropped.
     This uses the same executable, Lake project and imports as the search
     workers, in a separate process.
   - The goal Lean reports at that `sorry` is the starting state: the
     declaration's header binders introduced in order, and the target. A `∀`
     inside the declared type stays in the target.
   - Namespaces, `open`, notation, universes, binder roles and `let` values
     are the file's own, since Lean elaborated the file itself.
   - The original proof is never elaborated, and the declaration is excluded
     from premise retrieval.
5. **Check.** The exact elaborated statement must elaborate again in the
   deployment environment. A signature that depends on a declaration of the
   file itself is `input:unsupported-context`. A file that does not elaborate
   is `input:lean-rejected`.
6. **No fallback.** Every rejection finishes the run as `invalid-request` (or
   `environment-error`), with zero steps charged. A source request never runs
   as text.

### Goal inputs (`input.kind = "goal"`)

- **Shape:**
  - `goal_text` is the target.
  - `hypotheses` are ordered local declarations, one per entry, written as
    Lean declares them: `n m : Nat`, `h : n = 2`, `x : Nat := 3`, `{α : Type u}`.
  - `universe_levels` must list exactly the universe levels the goal and
    context use; otherwise `input:universe-levels`.
  - Limits: 20 000 characters per text, 512 hypotheses, 64 levels.
- **Rendering:** the shared parser turns the input into a closed statement,
  `∀ binders, target`, with the `intro` names that recreate the state.
- **Generated source:** the provider writes and stores a Lean file for the
  task (`source_origin: "generated-from-text"`). Example `09` shows one:

  ```lean
  import Mathlib

  theorem formal_target.{u, …} :
      ∀ (n : Nat), ∀ (h : n = 2), n + 1 = 3 := by
    intro n h
    sorry
  ```
- **Elaboration:** happens when the backend admits the root. A goal Lean
  cannot elaborate ends as `invalid-request` (`invalid-input`).

## 3. Environment

- **The manifest.** Each backend reports a content manifest of the runtime it
  actually started. It holds no filesystem paths:

  ```json
  {
    "format": "maths-ai-formal-environment-v1",
    "backend": "mcts-actor-critic",
    "client": {"...": "the Pantograph client's distribution source and commit"},
    "repl_sha256": "hex SHA-256 of the REPL executable",
    "imports": ["Mathlib"],
    "lake_manifest_sha256": "hex SHA-256 of lake-manifest.json",
    "lean_toolchain_sha256": "hex SHA-256 of lean-toolchain",
    "library_olean_sha256": {"Mathlib": "hex SHA-256 of .lake/build/lib/Mathlib.olean"}
  }
  ```
- **The hash.** `environment_hash` is the address of
  `{"environment_id": <deployment environment_id>, "manifest": <manifest>}`
  (rule 1).
- **Where to get it.** `/formal-capabilities` publishes it per deployment,
  once the backend is running. Use it as an opaque value. It is not a REPL
  environment id and is not comparable to one.
- **Where it's checked.** At admission the request's `environment_id` and
  `environment_hash` must equal the deployment's (409 `environment-mismatch`
  otherwise). At replay, and before a retry, the provider checks that the
  backend still runs the run's environment.

## 4. Admitted-input manifest

- **What it is.** Admission writes one canonical JSON object and stores it.
  Its address is `input_artifact_hash`. It binds everything a run and its
  replay rely on:

  ```json
  {
    "format": "maths-ai-formal-input-v1",
    "kind": "goal | source",
    "input": "<the request's input object>",
    "seed": "<the search root: text goal/hypotheses, or raw target and locals>",
    "statement": {"expression": "...", "local_names": ["..."], "universe_levels": ["..."], "fingerprint": "..."},
    "source_artifact_hash": "<caller's file, or the generated source>",
    "source_origin": "caller-artifact | generated-from-text",
    "environment_id": "...",
    "environment_hash": "...",
    "source_imports": ["..."],
    "target_unit_start": 0
  }
  ```

  `source_imports` and `target_unit_start` are present for source inputs only.
- **`source_artifact_hash`** is the caller's file for source inputs, or the
  generated source for goal inputs.
- **Reuse.** The same file selecting a different declaration has a different
  manifest. Retry and replay load this manifest; neither accepts new text.

## 5. Request digest and identity

- **`request_digest`** is the address of the request with null-valued fields
  removed, after validation. Omitting an optional field and sending `null`
  give the same digest.
- **Repeats.** A start with an existing `run_id` and the same digest returns
  that run (200), with no new work. A different digest is 409
  `run-conflict`.
- **The result** repeats the request digest and the coordinator binding
  (`claim_id`, `base_revision`, `lease_id`, `fencing_token`). An echo is not
  authority: persistent-atp's commit gate re-checks the lease.

## 6. Certificates

- **What's stored.** A certified attempt stores its certificate as canonical
  JSON. Its address is `result.certificate.artifact_hash`. The object is the
  backend's certificate plus the bindings:

  ```json
  {
    "format": "maths-ai-mcts-certificate-v1 | maths-ai-accepted-certificate-v1",
    "...": "backend fields (below)",
    "run_id": "example/fr-1",
    "attempt": 1,
    "backend": "mcts-actor-critic",
    "input_artifact_hash": "sha256:…",
    "source_artifact_hash": "sha256:…",
    "environment_hash": "sha256:…"
  }
  ```
- **Backend fields.**
  - **MCTS:** `artifact`, the replay artifact: the closed `statement`, the
    ordered replay `steps` with their obligation keys and argument
    declarations, the client and REPL `environment`, and the kernel
    `receipt` (root type, proof term, axioms). Also `proof_trace`.
  - **Accepted:** the stored `statement` and `proof_trace`.
- **Checking a certificate offline.**
  1. `GET /formal-artifacts/{certificate_hash}`.
  2. Re-hash the bytes; they must equal the address and be canonical JSON.
  3. The bindings must equal the run's.
- **Replay.** `POST /formal-replay` names the run, `certificate_hash`,
  `input_artifact_hash`, `source_artifact_hash`, the expected
  `environment_hash` and a verification budget.
  - Any field that is not the run's own is 409 (`…-mismatch`), and a run
    without a certificate is 409 `no-certificate`.
  - The provider re-hashes the stored certificate, manifest and source, checks
    their bindings, and verifies in a fresh native session.
  - **Verdicts:**
    - `verified`;
    - `rejected`: the proof or the stored evidence does not verify;
    - `inconclusive`: a time, resource or verification-allowance limit, or a
      changed environment. This is never a verdict on the proof.
  - The attestation's identities come from the stored evidence and the
    verifying session, not the request.

## 7. Budget accounting

- **What counts as a step.** One tactic command sent to Lean, accepted or
  not, during exploration or proof-combination replay.
  `budget.max_steps` is the only request-level tactic allowance;
  `search_policy.max_tactic_steps` is an unknown field and is refused.
- **Admitted allowance.** `min(budget.max_steps, profile.max_steps)`. The
  wall clock is `min(budget.wall_clock_ms or profile default, profile
  ceiling)`, per attempt, counted from that attempt's admission including
  queueing. Both are acknowledged, with any reduction.
- **Charging.**
  - Each attempt reserves the run's whole remaining allowance and is charged
    what it issued.
  - An attempt with an unknown cost (crash, cancellation, interruption) is
    charged its whole reservation, and `consumed_steps_exact` becomes false.
  - Nothing is ever refunded.
  - A retry gets the remainder, under the profile configuration hash and
    environment fixed at admission.
- **Verification** has its own allowance: `min(replay budget,
  deployment.verification.max_steps)`.
