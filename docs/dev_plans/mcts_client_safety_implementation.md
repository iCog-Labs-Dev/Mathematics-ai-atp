# MCTS client safety implementation

Status: application repairs implemented on `feat/mcts-pantograph-async-client`,
2026-10-22. GNU provisioning preflight remains pending on the package owner.

The reviewed architecture is [mcts_client_safety.md](../../mcts_client_safety.md).
The user's implementation scope is the active branch only. AC migration,
package publication, dependency changes and provisioning are outside this change.
The starting application commit is
`f1c3501b2c571ea6b2b25b2d9ccfb97c0e84a538`. The installed client and unchanged
dependency pin identify `664bcaa10578cb933afc618b8e8cb20a1b751106`.
The user-provided root plan was the only pre-existing local change.

## Architecture and sequence

1. `maths_ai.core.git_worktree` reads porcelain v1 status as raw NUL-delimited
   bytes, validates records, and decodes filesystem paths without losing bytes.
   Diagnostics and setup use this same reader. Git inspection disables optional
   locks and uses ten-second subprocess deadlines.
2. Diagnostics retain `dirty_sha256` and add `dirty_entries`. File digests describe
   on-disk bytes. Metadata records status, rename/copy sources and entry kind.
   Symlink digests describe link-target bytes. Submodules record initialization,
   index and checkout revisions, and dirtiness. Directories have no recursive
   digest. Deletions retain null hashes. Failed regular-file reads raise an
   explicit incomplete-provenance error. Multiple Git records for a staged
   deletion recreated on disk retain their statuses.
3. Setup records read-only inspections of both destinations before installing
   the toolchain or materializing either checkout. An existing destination must
   be its repository root with an exactly matching origin URL. Symlinked paths,
   archives, nested directories, conflicts and ordinary dirty checkouts fail.
   Linked Git worktrees with `.git` files remain supported.
4. Companion reuse additionally requires the pinned base, exact allowed paths,
   unstaged modifications or untracked new files, and package-verified source
   digests. Staged changes, renames and symlinked sources fail. Checkout rechecks
   HEAD, status, index bytes, refs and local config before mutation, and verifies
   companion digests again. New clones and existing clean repositories use
   ordinary fetch and detached checkout. Origin is never rewritten.
5. The historical port source manifest remains unchanged. A new
   [source manifest](../../maths_ai/gnn_inference/tests/fixtures/pantograph/mcts_client_safety/source_manifest.json)
   records the current application sources and the original hashes of changed
   copies. The migration test checks that lineage and the current source hashes.

This expands the shared Git boundary to arbitrary filesystem filenames and
structured rename/copy records. Both existing callers migrated together; the
line-oriented setup parsers were removed. Setup's internal checkout and
companion functions now consume inspection records, and all callers migrated.

| Choice | Selected approach | Alternative and tradeoff |
| --- | --- | --- |
| Status parsing | Shared porcelain v1 byte parser | Removing one `strip()` leaves quoted names and rename records incorrect; porcelain v2 adds unnecessary record types. |
| Origin mismatch | Read-only refusal with expected and actual URL | Automatic repair changes user configuration before refusal. |
| Setup order | Inspect both repositories, then mutate | Inspecting one repository at a time can mutate Mathlib before rejecting Pantograph. |
| Companion reuse | Status, base, origin and digest verification | A path-only comparison accepts staged changes and checks the wrong repository identity. |
| Historical evidence | Preserve original manifest and add current source evidence | Replacing historical hashes would misstate which files were copied from AC. |
| GNU validation | Wait for the public package helper and a reviewed retrievable pin | Application-owned checks or private helper imports split ownership or depend on an unsupported API. |

## Verification

Using `/home/nolawi/new-maths/.venv/bin/python` with the pinned client installed:

| Command | Result |
| --- | --- |
| `-m pytest maths_ai/gnn_inference/tests/test_git_worktree.py maths_ai/gnn_inference/tests/test_setup_sexpr_environment.py -q` | 58 passed before the final HEAD-snapshot regression; that additional case passed in the full suite |
| `-m pytest maths_ai/gnn_inference/tests -q` | 375 passed, 6 skipped, 5 subtests passed |
| `-m pytest maths_ai/pln_inference/metta/translator/test_imports.py tests -q` | 26 passed |
| `-m pytest maths_ai/pln_inference/metta/tests -q` | 16 passed, the 3 Thompson-sampler failures already identified in the original plan |
| `git diff --check` | Passed |

The focused suite uses temporary repositories. It covers staged, unstaged,
mixed, untracked, deleted, renamed and copied paths; whitespace, quotes, arrows,
tabs, newlines, Unicode and undecodable filename bytes; malformed status;
symlinks; initialized and uninitialized submodules; failed reads; and clean and
dirty diagnostic reports. Refusal tests compare file bytes including repository
index, refs, HEAD and configuration and assert no mutation command was invoked.
Positive checks cover fresh clones, clean reuse, linked worktrees and exact
companion reuse. Orchestration rejects a dirty Pantograph destination before
creating Mathlib or installing a toolchain.

The GNN skips include five live Lean cases and the two-GPU topology check.
No validated live executable was supplied. An existing candidate under
`/home/nolawi/new-maths/maths_ai/sexpr_environment/sexpr_environment/Pantograph`
failed the installed package's companion-source digest verification, so it was
not used as live evidence. Historical live reports remain unchanged.

## Remaining dependency and limits

The pinned client validates GNU stdbuf during runtime startup but exports no
public pre-provisioning helper. `_require_stdbuf` still checks only executable
existence. The third finding is unresolved until the package exposes its shared
async validator, publishes an explicitly reviewed retrievable revision, and this
branch migrates its dependency pin and setup caller together.

Inspection prevents known invalid destinations from causing setup mutations.
Rechecks detect intervening inspected-state changes; concurrent provisioning is
unsupported. There is no global rollback after materialization starts. Fresh
Lean provisioning, package artifact verification, GPU/checkpoint deployment and
the five live application checks remain pending.
