# Exact PLN translation handoff

The implementation on MCTS-AC-branch uses translation contract
`lean-pln-exact-v1-rule-proof`. The earlier uncommitted CLI sampler-name repair
is preserved. This work is not committed automatically.

## Implemented behavior

- A shared immutable formula/term representation, exact names and literals,
  collision-free atom encoding, and one renderer serve live and batch scoring.
- Live calls receive raw execution targets and ordered local declarations through
  `GoalInput`; they do not reconstruct intermediate goals from printed text.
- Only supported proof-valued local declarations become facts. Proposition/data
  declarations remain context. The parent target is not asserted automatically.
- The explicit propositional text API uses the Lean lexer, declared proposition
  identities, operator precedence, complete token consumption, and balanced
  parentheses. It is not a fallback for structured decode failures.
- Unsupported binders, unknown proposition typing, unresolved metavariables,
  placeholders, and lets produce a typed unsupported outcome with no STV.
- Genuine success requires a target-matching returned statement containing a
  structured rule-proof derivation and its attached finite STV. Direct-fact STVs,
  compiled rule output, another target, or a nonzero process exit are not success.
- Unsupported/engine-error outcomes neither update DTS evidence nor remove Lean
  obligations. Supported no-derivation queries retain labeled exploration behavior.
- Live and batch sampler keys, manifests, and search/evaluation resume contracts
  include the translation version. Batch runners record process exit codes.
- Search execution and training round metrics separate genuine, fallback, and
  unsupported counts. Raw/model graph features remain unchanged.

## Validation

The complete GNN/PLN regression invocation passed 724 tests and 35 subtests;
39 opt-in tests were skipped. The separately requested real-Lean integration
suite passed two tests. Those tests cover production scoring after real tactics,
unsupported dependent goals, allocation cleanup, and kernel-certified searches
under both best-first and PUCT with supported and unsupported AND obligations.
Native local-import checks also passed.

Validation used the repository-pinned standalone client at
955a4921b4cacaa49f4b3087f3982d7ca3629566 installed under /tmp; the project's
existing Python environment was not synchronized or replaced. Native assets were:

- PeTTa source e1490899cefc67c128d5311ff4861f9997674957; SWI-Prolog 9.3.31.
- PeTTaChainer b1678e97c4498b376b91e8cce215f5ab0398ddef.
- Companion REPL 9ecaaea242f427e534aff29314513c358b4d21ad, rebuilt with Lean 4.10.0-rc1.

Native runtime tests do not require a trained policy checkpoint: scripted proposal
providers exercise the production search, scoring, replay, and certification paths.
They do not establish the quality of learned tactic proposals on a GPU.

## Coverage and limits

`pln_validation/local_first100_coverage.json` records an ordered scan of the first
100 local `local_state_tactic_pairs.jsonl` rows. Lean admitted eight, rejected 92;
three admitted states were supported and five were explicitly unsupported because
proposition typing could not be classified. This is not a general Mathlib estimate
or proof-success rate. Dataset admission failures are separate from PLN failures.

The initial fragment covers propositional connectives, nondependent implication,
exact proposition identities, known simple data contexts, and exact uninterpreted
equality atoms. It does not implement a general Lean type checker, dependent
quantifier translation, let substitution, or arithmetic inference. General local
predicate typing will require Lean-derived metadata in the REPL/client. The fork's
unanswered conjunction query remains an inference limitation separate from parsing.

## Operational changes

Use the setup and commands in `maths_ai/pln_inference/README.md`. Existing
`PETTA_BIN` and `PETTACHAINER_ROOT` settings remain valid. Regenerate old batch
manifests; missing process exit status cannot establish a scored result.

Start a fresh RL run when changing translation semantics. Older exact-resume and
best-score contracts are rejected. Existing compatible model checkpoints can still
be used as warm starts because model graph semantics were not changed.

Use the current production APIs, not the deleted parser/normalizer helpers:

```python
from maths_ai.pln_inference.compiler import GoalInput
from maths_ai.pln_inference.model import PLNInference

result = PLNInference().evaluate(GoalInput.from_goal(canonical_goal))
if result.status == "ok":
    assert result.stv is not None and not result.is_fallback
```
