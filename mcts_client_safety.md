# Repair MCTS provenance and shared Pantograph setup safety

Status: proposed architecture and sequential implementation plan, 2026-10-22.
No implementation, branch switch, package update, publication, provisioning, or
training is authorized by creating this document. Review the architecture before
requesting specific implementation stages.

## 1. Outcome and applicable source trees

Address the three findings from the package-backed MCTS client review:

1. Dirty-worktree provenance loses filename characters and records missing hashes.
2. Environment setup can change `origin` before refusing a dirty checkout.
3. Setup checks whether `stdbuf` exists, not whether it is the required GNU tool.

The reviewed application implementation is contributor commit
`f1c3501b2c571ea6b2b25b2d9ccfb97c0e84a538` on
`Leul-Negash/Mathematics-ai-atp:feat/mcts-pantograph-async-client`. Its target base
is MCTS commit `89a2417e522720d2cfe3f2e6218614e4d828c8de`. The shared setup source
was ported from AC commit `72c82ce13d62f132ee1939d8e92641c100f4c418`.
The standalone client source is `123nol/Pantograph-Client` at
`664bcaa10578cb933afc618b8e8cb20a1b751106`.

The current local MCTS base does not contain `mcts_diagnostics.py` or the complete
package integration. Apply application fixes to the reviewed contributor branch
or a target containing that port, not directly to the older upstream execution
implementation. Do not silently combine this task with implementing the port.
The earlier temporary review checkout is a convenience, not an enduring source
of truth; verify the commits and working-tree state again at implementation time.

Ownership:

- MCTS provenance and Git worktree reporting: application repository.
- Safe checkout preparation: shared application setup, on both AC and MCTS.
- GNU tool-validation policy and helper-process lifecycle: standalone package.
- Setup invocation of that validation: application repository, on both branches.

Do not modify search selection, learning evidence, graph/checkpoint schemas,
optimizer scheduling, or Lean source. Preserve historical reports and user data.

## 2. Findings and important correction of scope

### Dirty status parsing

`mcts_diagnostics._target_identity()` strips the complete output of Git status
and then interprets fixed-width status fields. For the first unstaged change,
the leading status-space disappears before filename extraction. A modified
`maths_ai/hybrid_reasoner/joint_inference.py` was reproduced as
`aths_ai/hybrid_reasoner/joint_inference.py` with a null hash.

The adjacent setup helper `_changed_paths()` also parses human-readable lines
and does not correctly represent quoted filenames or rename records. Fix the
shared parsing boundary rather than only removing one `.strip()` call.

### Checkout refusal

`setup_sexpr_environment._checkout()` changes `origin` before testing whether
the destination has local changes. `_prepare_companion()` also has an existing-
patch reuse path that must validate repository identity before accepting it.

Refusal must preserve files, index, HEAD, refs and remote configuration. This
requires read-only inspection before any fetch, checkout, patch, or remote edit.
Preflight both destination repositories before toolchain installation as well.

### GNU validation already exists during actual client startup

The pinned package's `PantographClient.create()` already runs `stdbuf --version`
through `_capture_tool`, checks for GNU coreutils, and rejects an incompatible
launcher before starting the REPL. Its test suite covers non-GNU rejection.
The missing guarantee is **early provisioning validation**, not absence of all
runtime validation.

Expose/reuse that existing policy before setup changes the environment. Do not
add a separate application implementation or replace the runtime check with a
weaker existence check. The scope shift is that environment-helper validation
must also work before the Lean project and full launch specification exist.

## 3. Architectural decisions and alternatives

| Decision | Chosen approach | Alternatives and tradeoffs |
| --- | --- | --- |
| Git status | One application-owned NUL-delimited parser reused by diagnostics and setup | Removing `.strip()` alone leaves quoting/rename defects; duplicated parsers diverge |
| Parser format | Explicit porcelain v1 with `-z`, raw bytes and filesystem decoding | Porcelain v2 supplies more metadata but adds record types unnecessary for these consumers; line splitting cannot represent arbitrary filenames correctly |
| Remote mismatch | Refuse without mutation and name the expected/actual origin | Automatically rewriting origin changes user configuration; flags permitting automatic repair add an unnecessary second policy |
| Provisioning order | Inspect both destinations and validate GNU launcher before mutations | Checking each repository only when reached can mutate Mathlib/toolchains before discovering an invalid Pantograph destination |
| GNU validation | Package-owned public async preflight using existing helper-process cleanup | Application-local version checks duplicate policy; executable existence is insufficient |
| Package structure | Generalize `_capture_tool` inputs and extract its existing GNU check in place | Moving all transport/process ownership into new modules expands risk without being required for this repair |
| Buffering verification | GNU identity preflight plus actual REPL smoke | A Python `print` subprocess is not a valid substitute for verifying the native REPL's buffering; native compiler-based probes add dependencies |
| Distribution | Verified package revision, then reviewed immutable dependency-pin updates | Vendoring or import fallbacks would split execution ownership; fabricated hashes or local absolute dependency paths are not acceptable |

A version banner is a compatibility check for a trusted installed tool, not a
security attestation. Real REPL verification remains necessary. No search for an
alternate launcher or automatic coreutils installation is introduced; users must
select a suitable GNU tool on `PATH`.

## 4. Target design

### A. Shared application Git status reader

Add a small standard-library-only module, proposed as
`maths_ai/core/git_worktree.py`. It must not import search, models, or the client.

Provide one status-reading operation and one parser returning immutable entries
with these explicitly named fields:

- `index_status`: the status of the staged/index version of a path.
- `worktree_status`: the status of the on-disk version relative to the index.
- `path`: the exact repository-relative destination/current pathname.
- `original_path`: the source pathname for rename/copy records, otherwise absent.

Read `git --no-optional-locks status --porcelain=v1 -z --untracked-files=all --ignore-submodules=none`
from the verified repository root. Include submodule changes explicitly rather
than allowing user ignore settings to make a dirty destination appear clean.
Capture raw bytes; never call `.strip()` or `.splitlines()` on status output.
Decode path bytes with the filesystem encoding/surrogate-escape behavior, not
lossy replacement. NUL-delimited rename/copy records give the destination first
and the original pathname as a second record. Do not split filenames on ` -> `
or remove quotes. [Git status format](https://git-scm.com/docs/git-status#_porcelain_format_version_1)

Validate framing and status fields. Malformed or truncated output raises an
actionable error rather than silently dropping an entry. Separate status reads
from scalar Git queries such as `rev-parse HEAD`, whose trailing newline may be
removed independently. Use bounded subprocess calls with explicit arguments.

Migrate all affected application callers in the same change:

- `_target_identity()` consumes parsed entries for dirty reporting.
- Setup dirty checks consume the same entries.
- Companion changed-path comparison uses destination paths and rejects rename,
  conflict, or unexpected index changes instead of discarding their status.
- Remove the old line-oriented status parsing helpers; do not retain both modes.

### B. Accurate provenance without silently missing files

Keep the existing `dirty_sha256` report field. Its keys must be exact current
paths; hashes describe the bytes on disk, not staged bytes that were not executed.
Deleted paths remain present with a null content hash. Preserve rename source
metadata in a structured companion field so the original path is not lost.

Do not follow symlinks outside the checkout or recursively hash submodule/user
directories. Record entry kind explicitly for symlinks, submodules and deletions;
hash link-target bytes for symlinks and record submodule identity/dirty status.
Do not present a non-file null hash as if it were verified file content. An
unreadable regular file or a file disappearing unexpectedly must produce an
explicit provenance error or a report marked incomplete, not a successful null.

Preserve `worktree_clean`, package identity, revision and existing search-source
hashes. Update all current consumers/tests for additional entry metadata. Do not
rewrite historical clean reports or add a second old-format parser.

### C. Inspection before setup mutation

Separate read-only inspection from materialization in the setup script. A small
internal preparation record should identify the destination, expected origin,
required commit, status snapshot, and whether it is new, clean, or an exactly
verified companion-patch application.

Inspection must:

1. Verify an existing destination is the intended repository root, not a plain
   archive, nested directory inside another repository, or a symlink redirection.
   Use Git's repository queries rather than assuming `.git` is always a directory.
2. Require the expected origin. A missing or mismatched origin is a refusal, not
   permission to add or rewrite it. Compare using one documented policy; no broad
   URL normalization or interactive repair is needed.
3. Refuse staged, unstaged, untracked, conflicted and dirty-submodule changes for
   ordinary checkout reuse. Do not exempt a user checkout simply because its
   working directory is empty after files were deleted.
4. Permit the one existing companion case only when origin, base HEAD, allowed
   status/path set and every package-supplied patched-source digest match. Reject
   staged patch changes, renames and unrelated files. Check this identity even
   when no fetch or patch application would be needed.
5. Inspect both Mathlib and Pantograph destinations before installing toolchains,
   creating output directories, fetching caches or changing either repository.

After all preflights pass, materialize new repositories through clone and existing
clean repositories through ordinary non-forced fetch/checkout. A new clone can
check out the requested commit without using the former empty-worktree exception
for pre-existing repositories. Recheck existing state before its mutation phase;
never use `restore`, `reset --hard`, forced checkout, automatic remote changes,
or deletion to make a destination reusable.

Do not promise global rollback if a network/build operation fails after approved
materialization begins. The guarantee is that known validation refusals occur
before setup mutations. Concurrent edits/provisioning are not supported; detect
changed inspection snapshots and stop rather than force progress.

### D. One GNU policy in the standalone package

Add a documented public async helper, proposed as `require_gnu_stdbuf`, exported
from `pantograph_async`. Define it alongside the existing environment-helper
logic; a new process-management implementation is not needed.

The helper resolves a candidate from `PATH`, or validates the exact executable
already resolved for runtime startup. It runs the candidate's `--version` with
a copied environment and deterministic locale, requires successful exit and
the existing GNU coreutils identity, and returns its resolved path. Do not
modify global `PATH`/locale or silently select a different executable.

Generalize `_capture_tool` so its process ownership, stdout bounds, stderr tail,
deadline, and awaited termination/reaping do not require a complete
`PantographLaunchSpec`. Update **all** existing helper callers, including Lake
environment resolution, in the same package change.

Runtime startup invokes the shared GNU helper using the remaining total startup
budget. It must not restart that budget or block the event loop with synchronous
`subprocess.run`. Existing stdout/stderr limits, cancellation semantics and
process-group cleanup remain in force. Imports still perform no I/O.

Application setup calls this public helper before `_ensure_toolchain`, directory
creation or Git mutations. Its synchronous CLI may call `asyncio.run` once for
the short preflight when no event loop is running. Do not insert a nested event
loop into async application execution. Remove `_require_stdbuf()` and migrate
its tests/callers rather than retaining a wrapper with the old existence policy.

`build_launch_spec` and `PantographEnv.verify()` may remain static path/settings
checks. Document the distinction from executable preflight rather than implying
that a filesystem check confirms runtime compatibility.

## 5. Sequential implementation stages

| Stage | Deliverable | Exit gate |
| --- | --- | --- |
| 0 | Verified source/workspace and failing regression cases | Correct application/package trees selected; user changes preserved |
| 1 | Shared Git status parsing and corrected provenance | Exact-path hashes and adjacent setup parsing tests pass |
| 2 | Mutation-free setup inspection | Refusal tests prove files, index, refs, HEAD and remotes unchanged |
| 3 | Package-owned reusable GNU preflight | Package helper/runtime tests and installed-distribution checks pass |
| 4 | Approved dependency update and both-branch setup migration | One policy used by runtime/setup; invalid tools stop before mutation |
| 5 | Combined verification and accurate handoff | Regressions/live checks pass; unavailable provisioning/deployment gates recorded |

### Stage 0 — Freeze the applicable implementations

- Confirm whether the contributor port is being amended or has already merged.
  Select an approved workspace containing that code. Do not switch or overwrite
  the current base checkout automatically.
- Record application/package commit IDs and statuses, dependency metadata and
  existing user changes. Recheck AC's setup against the reviewed source.
- Add focused failing tests in the chosen implementation workspaces for the
  reproduced leading-space status bug, origin changes on refusal, and non-GNU
  acceptance by setup. Preserve the existing runtime non-GNU rejection test.
- Record the review baseline: 326 passing GNN tests and five passing live tests;
  broader tests had three known Thompson-sampler failures. Rerun relevant checks
  rather than treating those previous counts as new results.

Exit: sources and failing mechanisms are reproducible; no provisioning is run.

### Stage 1 — Repair shared status parsing and provenance

- Implement the byte-oriented reader/parser and migrate diagnostics and setup
  status/path consumers together.
- Add real temporary-Git tests for clean, unstaged, staged-only, mixed, untracked,
  deleted, renamed and copied paths; explicitly enable copy detection in a test
  that expects copy records. Include leading/trailing spaces, quotes, literal
  arrows, tabs/newlines and non-ASCII filenames. Unit-test malformed records.
- Confirm each regular-file hash equals an independent hash of the exact path.
  Exercise symlink/submodule classification and unreadable/disappearing files.
- Test that changing Git quoting/relative-path configuration does not alter
  reported paths, and that a rejected malformed status is not reported clean.
- Put shared parser/setup tests on both application branches; keep MCTS-specific
  report tests on the branch containing `mcts_diagnostics.py`.

Exit: the reproduced corrupted-path report is corrected, all consumers use one
parser, and existing clean report semantics remain valid.

### Stage 2 — Make setup refusals read-only

- Introduce inspection records and preflight both destinations before setup
  mutation. Remove remote rewriting and empty-existing-worktree exceptions.
- Preserve exact verified companion-patch reuse through explicit identity/status
  checks, using the installed package's allowed files and digests.
- Test dirty matching-origin and dirty mismatched-origin repositories, clean
  mismatched/missing origins, non-Git archives, nested roots, symlinks, staged
  deletions leaving an empty worktree, and dirty submodules.
- Snapshot file bytes, relevant config, HEAD, refs and index around refusals.
  Assert no clone/fetch/checkout/patch/toolchain/cache command occurs.
- Test fresh clone, matching-origin clean checkout and exact companion reuse
  positively. Companion reuse with wrong origin must fail without modification.
- Add an orchestration case where Mathlib is acceptable but Pantograph is dirty;
  assert neither destination nor toolchain/cache state is changed.

Exit: the setup repair passes on AC and MCTS. Local temporary repositories are
sufficient; no user Lean checkout or archive is used as a mutation target.

### Stage 3 — Expose and verify package GNU preflight

- Obtain write access/authorization for the package workspace before editing it.
  Implement the shared public helper and migrate runtime GNU/Lake helper calls.
- Retain and extend real-child tests for GNU success, uutils/BusyBox-like output,
  missing executable, failed exit, malformed/bannerless output, timeout, excessive
  output, cancellation during spawn/read, and repeated cancellation during cleanup.
- Assert failure leaves no helper process and never spawns Lake or the REPL.
  Keep startup's one-deadline test and verify no locale/PATH mutation or import I/O.
- Build/install the package non-editably in isolation and run its tests against
  the installed artifact. Preserve package resources and wire/schema identities.

Exit: the package has one GNU-validation policy usable before any Lean project
exists. Application integration remains pending until a real package pin exists.

### Stage 4 — Migrate application setup to the package helper

- Publish the reviewed package revision only with explicit permission. Record
  its full retrievable commit; do not invent a future hash. Without publication
  authority, finish local artifact verification and mark this handoff pending.
- Update the selected AC/MCTS dependency pins and locks to that reviewed revision,
  preserving unrelated dependency versions. Use ordinary dependency resolution;
  do not add source-path overrides, vendored functions or old/new fallback imports.
- Remove application `_require_stdbuf` and use the package preflight before any
  toolchain, filesystem, checkout, patch or cache mutation.
- Replace the hard-coded `/usr/bin/stdbuf` assertion with tests of the actual
  resolved executable and verified identity. Test GNU symlinks without assuming
  that this absolute system path identifies GNU on every machine.
- Feed a non-GNU fake to setup and assert an actionable error plus no mutations.
  Confirm GNU success proceeds to inspection/materialization in the chosen order.

Exit: both branches call the same maintained package policy; configuration errors
are detected before provisioning changes and runtime enforcement is retained.

### Stage 5 — Combined regression and handoff

- Run application Git/provenance/setup suites, full GNN/MCTS regressions and
  translator checks on the relevant branches. Run package suites independently.
- Run the five MCTS/application live checks against the existing validated REPL,
  including repeated proofs, multi-leaf PUCT, two clients and bounded updates.
  No Lean rebuild is needed for these Python repairs.
- Verify dirty and clean diagnostic reports in a disposable checkout. Write new
  evidence identifying app/package revisions and actual hashes; leave previous
  reports untouched.
- Perform fresh provisioning only with separate authorization, network/disk
  resources and a new explicit output root. Successful preflight tests alone do
  not prove a clean Lean/Mathlib build. Never use downloaded archives or the
  user's development fork as setup destinations.
- Record any unavailable fresh-build or Conda/GPU/checkpoint gate as pending.
  Do not treat the unrelated package extraction-manifest digest correction as
  resolved unless it was actually addressed and verified separately.

Exit: all three fixes are verified at their owning boundaries. Source provenance,
mutation-free refusals and early launcher validation are demonstrated separately
from provisioning/deployment success.

## 6. Completion and authorization boundaries

Completion requires exact dirty paths/hashes, no state changes on known setup
refusals, and one GNU-validation policy enforced before setup and runtime launch.
Both application branches must receive their shared fixes; package consumers must
use an explicitly reviewed pin. Do not mark a cross-repository stage complete
when only a local implementation or unpublished dependency exists.

Recommended first implementation request: **Stages 0–2** on the approved
contributor-port workspace, with the shared application changes prepared for AC.
Stage 3 requires package write access. Stage 4 requires a retrievable package
revision and explicit publication authority when a push is needed. No commit,
push, PR comment, training run or provisioning action is implicit in this plan.
