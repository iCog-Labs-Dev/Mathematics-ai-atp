# Backend capabilities

`GET /formal-capabilities` returns, for each deployment:
- its backend, admitted input kinds and environment hash;
- its profiles, with their ceilings and configuration hashes;
- its verification allowance and model identities;
- whether it is available now.

This table is what each backend supports in this provider. A deployment can
offer less than its backend supports, never more.

| | `mcts-actor-critic` | `accepted-gnn` |
| --- | --- | --- |
| Model | source-aware actor-critic checkpoint (MCTS release, e.g. `v1.0.x`) | published `iCogLabs-Dev/Mathlib-Sexpr-GNN` v2 bundle at `8aba62625dfeb833e4591498700f351f3bc60af9` |
| Runtime | `pantograph-async-client` `2b32650` with REPL `123nol/Pantograph@46a33e1` | PyPantograph `3322696` with REPL `jajos12/Pantograph@0dd5d22` |
| Lean / Mathlib | `v4.10.0-rc1` / `29dcec07` | `v4.10.0-rc1` / `29dcec07` |
| `input.kind = "goal"` | yes | yes |
| `input.kind = "source"` | yes (elaborated by the deployment's REPL; `rules.md` §2) | no: 422 `unsupported-input` |
| Universe-polymorphic goals | yes | no: `input:unsupported-context` |
| Search profiles | `puct` (simulation-based), `best_first`; node, depth and simulation limits per profile | `best_first`; node and depth limits per profile |
| `proposal_mode`, `seed` | honoured (`sample` or `greedy`) | ignored by the search (deterministic best-first) |
| Tactic-step accounting | every command through the retained-snapshot executor: exploration and combination replay together | every `goal.tactic`, including the `intro` that re-creates each node's context |
| A proved result means | the root's replayed proof was kernel-certified in search (`certified`); READY/JOINT candidates are not proofs | the best-first search closed every goal (`search-closed`); it is certified only by replay |
| Independent replay | artifact replay and kernel re-certification in a new Pantograph client, against the receipt | compiling `theorem … : <stored statement> := by intro …; <proof>` in a new Lean process under the deployment's imports, warnings as errors |
| PLN | per profile (`*_pln` profiles), with provisioned PeTTa | a deployment setting (`MATHS_AI_ACCEPTED_USE_PLN`); off ⇒ subgoals rank as `pln-disabled` and PeTTa is not required |
| Lean resource limits | heartbeats and recursion depth are recoverable; the state is kept and other candidates run | the Lean error fails that tactic |
| Cancellation | the runtime operation is cancelled, then reported terminal once the runtime is idle | the search task is cancelled |
| Frontier resume | no; retry repeats the bound initial input with the remaining allowance | no; the same |

Terminations either backend reports, and their dispositions:

| `termination.reason` | `disposition` |
| --- | --- |
| `certified` (MCTS: kernel-certified in search), `search-closed` (accepted: every goal closed, certified only by replay) | `proved-pending-replay` |
| `step-budget`, `node-budget`, `simulation-budget` (MCTS), `deadline`, `lean-resource` (MCTS) | `budget-exhausted` |
| `frontier-exhausted`, `alternatives-exhausted` | `stagnated` |
| `invalid-input` | `invalid-request` |
| `environment-mismatch` | `environment-error` |
| `crash`, `interrupted` | `internal-error` |
| `cancelled` | `cancelled` |

`counterexample` is never produced: neither backend searches for refutations.
