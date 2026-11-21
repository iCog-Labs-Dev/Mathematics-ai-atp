# Shared rendered Pantograph command identity

Implementation baseline: merge `ab3f91f` of PR #63 into MCTS-AC-branch.
The user approved implementation of the follow-up review comment.

## Architecture and alternatives

Execution identity must be the final Lean command string. Raw policy identity
remains the tactic ID and argument-index tuple used to recompute training
probabilities. Distinct raw actions may share one command and one checked outcome.

Chosen approach: move the existing renderer to a dependency-light data-model
module and make candidate identity use it. All cache, graph, alias and observation
callers migrate together. This keeps execution and identity consistent and avoids
an import cycle between data models and the reasoner.

Alternatives considered:

- Normalize only the reasoner's cache key: smaller change, but graph alias
  validation and edge uniqueness would still disagree with the executor.
- Reject every input changed by rendering: removes aliases by narrowing the
  accepted interface and changes existing tactic rendering behavior.
- Store both tuple and string execution identities: preserves old callers but
  introduces two competing identities. Raw policy IDs already preserve the
  information needed for training.

## Sequential stages

1. Extract the renderer without changing its supported tactic formatting. Empty
   tactic names and required-bracket tactics without usable arguments are
   unrenderable. Candidate/proposal command properties return the rendered string
   or `None`. Migrate all renderer imports.
2. Use strings for checked command outcomes and RL command-consistency ledgers.
   Reject unrenderable graph insertions and aliases. Settle unrenderable proposals
   separately before execution grouping, preserving raw observations and draw
   accounting without adding a checked outcome or attempted command.
3. Add regressions for shared generic/RL execution, accepted and rejected aliases
   across rounds, prior mass versus multiplicity, unrenderable actions, and
   distinct commands. Version command identity in saved search settings and test
   exact-resume sampling-schedule acceptance/rejection.
4. Refresh source provenance, run focused and offline checks, then live widening
   and bounded learning checks where the installed REPL permits them. Inspect
   local prepared-data/checkpoint availability before real driver evaluation.
   Record unavailable resources and failed acceptance runs explicitly in the
   handoff; do not replace real-checkpoint evidence with synthetic fixtures.

## Acceptance

One rendered command executes at most once per node across rounds. Its edge
contains each observed raw alias's prior once, and observations retain sample
multiplicity. Rejections are cached; unrenderable proposals never become Lean
rejections. Matching saved contracts resume, and changed/missing sampling schedules
are rejected. All shared callers use the same command identity.
