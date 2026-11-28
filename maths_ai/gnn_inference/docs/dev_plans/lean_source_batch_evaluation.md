# Evaluate Lean source files directly

Batch evaluation should accept a pinned directory of Lean files and selected theorem names. The evaluator should derive starting states during admission and use the existing PUCT or best-first search, result reports, and proof export.

## Input contract

Add `data_source: "lean_source"` to the shared theorem-source configuration for evaluation. Require a source root, a full Git revision, and an ordered selection manifest containing relative `.lean` paths and exact theorem names. Allow multiple theorem locators per file and reject duplicate locators, ambiguous declarations, and paths that escape the source root. Resolve paths through the same configuration entry points used by CLI and evaluation job profiles. The selection manifest lists inputs and contains no converted proof states. Existing dataset and theorem-record inputs keep their current behavior and identities.

Verify selected file bytes against the pinned Git revision and use those same bytes for elaboration. Bind admission provenance to the revision, relative paths, theorem names, source digests, manifest digest, and configured Lean environment. The first release enables source mode for standalone and periodic evaluation and configured API evaluation jobs. Source-backed training remains outside scope.

## Shared admission

Extract the common source-to-seed operation from the formal service. It checks imports against the deployment, uses the existing Lean parser to isolate the selected theorem and its permitted context, elaborates the signature with binders introduced, and checks the reconstructed closed statement in the deployed environment. The formal service continues to bind source bytes and evidence through its artifact store. Batch evaluation supplies verified local bytes to the same operation.

Extend the shared admission session to elaborate source candidates under its existing deadline and pass the resulting seeds through its replay validation and reporting. Count one selected theorem per scanned row, including rejected declarations. Bound source size and native elaboration time, and await process cleanup on cancellation or timeout. Preserve the existing distinction between input rejection and operational failure. Report rejected or unsupported declarations separately from unsolved searches. Include source identity in cache and evaluation provenance so a changed selection or file cannot reuse stale admission evidence.

Preserve supported imports, namespaces, notation, binders, and local context. Exclude the original target proof and later commands. Reject statements that require unavailable local declarations or depend on placeholder answers. For PutnamBench, start with explicitly selected theorems without answer placeholders. Support for supplying answers can be reviewed separately.

## Alternatives considered

| Approach | Benefit | Cost |
| --- | --- | --- |
| Reuse formal source admission within batch evaluation | One elaboration contract and existing evaluation metrics | Requires moving the shared operation outside HTTP-specific orchestration and migrating the formal caller |
| Add a PutnamBench-only converter | Can tailor selection and reporting to the benchmark layout | Requires a separate conversion run and leaves general source input unsupported |
| Submit each theorem through the formal HTTP API | Uses the existing source endpoint directly | Adds service and artifact-store dependencies to CLI evaluation and bypasses its batch reporting |

## Verification

Check that CLI and configured API evaluation admit the same selected declarations with consistent provenance. Exercise both search modes on a small native fixture and confirm proof export. Test a file with multiple theorems, scoped notation, a target body and suffix that must stay unexecuted, a placeholder answer, a changed source file, and a timed-out elaboration. Keep existing JSONL, dataset, formal source admission, and resume tests passing. Smoke-test selected statements from PutnamBench revision `8aaccf8c40e0db1e69a3c808166d9d09b0109703` against the current Lean/Mathlib deployment and record admission results separately from solve counts.
