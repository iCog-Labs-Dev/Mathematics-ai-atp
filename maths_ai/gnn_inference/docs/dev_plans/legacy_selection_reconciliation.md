# Reconcile the HTPS search with the legacy-selection addendum

## Context

The addendum `maths_ai/gnn_inference/docs/htps_integration_addendum_legacy_selection.md`
supersedes Decision 2 and §2.2 of `htps_mcts_integration_plan.md`. It identifies two
defects in the implemented Phase 1:

1. **Unit mismatch in the mode gate.** `prove` currently selects the search mode by
   `num_simulations > 1` (joint_inference.py:417). `num_simulations` counts simulations
   — one simulation expands every unexpanded leaf of the selected partial hypertree —
   while the legacy loop runs until root resolution or the `max_nodes` node-expansion
   budget. `num_simulations = 1` therefore does not reproduce the legacy process; it is a
   different loop with a different termination condition. PUCT with one simulation also
   ranks edges by a different function than the legacy `combined_rank =
   policy_prob × PLN_score`, so no setting of the simulation count converges to legacy
   behavior.
2. **PUCT has no stored prior.** `_select_partial_hypertree` (joint_inference.py:517-519)
   reads `edge.tactic.probability` live at selection time. The addendum requires the
   prior to be stored once on `EdgeVisitStats` at expansion time (`prior_prob`) and read
   — never recomputed — by `puct_score` on every later selection.

User decisions already taken:
- `prior_prob` stores the **tactic-head probability alone** (`exp(tactic_logp)` — what
  `_decode_row` already writes into `TacticCandidate.probability` at rl_reasoner.py:219).
  No model API change.
- Misconfiguration (`num_simulations`/`sim_batch_size` set while
  `selection_policy == "legacy"`) is **rejected** with `ValueError`, not warned.

Locked constraint from the Phase 0–3 plan: every new config default reduces to current
behavior. Today's default (`num_simulations = 1`) runs the legacy loop, so the new
`selection_policy` default must be `"legacy"`.

## Changes

### 1. Re-gate `prove` on `selection_policy` — `maths_ai/hybrid_reasoner/joint_inference.py`

- `HybridReasoner.__init__` gains `selection_policy: str = "legacy"` and changes
  `num_simulations`/`sim_batch_size`/`puct_c` defaults from `1`/`4`/`1.0` to `None`
  (sentinels distinguishing "explicitly set" from "left alone"). It calls the shared
  validator (below), which returns the resolved numeric values to store on `self`.
- `prove` branches on `self.selection_policy == "puct"` → `_prove_mcts`; otherwise the
  existing best-first loop runs verbatim (frontier pop by `combined_rank`, `_expand`,
  `_propagate` — already the literal legacy path, terminating on root SOLVED/DEAD,
  empty frontier, `max_nodes`, or deadline). `num_simulations` no longer gates anything.
- Docstring of `prove` updated: the two modes are named by policy, not simulation count.
- `"rp"` is accepted by the `Literal` but raises `NotImplementedError` in the validator
  (reserved for the deferred RP variant).

### 2. Shared validator — `maths_ai/hybrid_reasoner/selection_policy.py`

New function, single authority for the mode/parameter contract, called by both the
reasoner constructor and the training config:

```python
def resolve_search_params(selection_policy, num_simulations, sim_batch_size, puct_c):
    """Validate the (policy, budget) combination and resolve None defaults.

    legacy: all three must be None -> returns (1, 1, 0.0) placeholders (never read).
    puct:   num_simulations is required (no natural default); sim_batch_size
            defaults to 4, puct_c to 1.0.
    rp:     NotImplementedError.
    anything else: ValueError.
    """
```

Rejecting rather than warning: a silently ignored search budget means the run proceeds
with settings the user believed were active.

### 3. Stored prior — `maths_ai/hybrid_reasoner/hypergraph.py` + `selection_policy.py`

- `EdgeVisitStats` gains `prior_prob: float = 0.0`.
- `ProofHypergraph.add_edge` stamps it at edge creation:
  `ProofHyperedge(..., visit_stats=EdgeVisitStats(prior_prob=tactic.probability))`.
  This is the single seam every edge creator flows through, so all callers are migrated
  at once: RL sampled edges carry `exp(tactic_logp)` from `_decode_row`; base-reasoner
  edges carry the GNN engine's probability; `PLN_fallback` pseudo-edges carry `1.0`
  (moot — those edges are created SOLVED with no children, and selection skips resolved
  nodes, so their prior is never read).
- `puct_score` signature changes to `puct_score(stats, total_node_visits, c)` and reads
  `stats.prior_prob` internally — enforcing the read-never-recomputed invariant in the
  type signature. Migrate the two callers: `_select_partial_hypertree` (drops the
  `e.tactic.probability` argument) and `test_selection_policy.py`.

### 4. Config + driver threading — `maths_ai/gnn_inference/atp_lean_gnn/rl_training_driver.py`

- `RLTrainingConfig`: `selection_policy` default `"puct"` → `"legacy"`;
  `num_simulations`/`sim_batch_size`/`puct_c` become `int | None = None` /
  `float | None = None`. Add `__post_init__` calling `resolve_search_params` (fail at
  config construction, before any Lean server spins up).
- The reasoner construction site (rl_training_driver.py:552-565) passes
  `selection_policy=cfg.selection_policy` alongside the three optional fields verbatim;
  the reasoner constructor re-runs the same validator (harmless — same rule, one
  implementation).
- `maths_ai/gnn_inference/configs/rl_actor_critic.json`: `"selection_policy": "legacy"`,
  and `num_simulations`/`sim_batch_size`/`puct_c` set to `null` (the current file says
  `"puct"` with `num_simulations: 1`, which under the new gate would silently switch the
  run from the legacy loop to a one-simulation PUCT search — exactly the confusion the
  addendum exists to prevent).

### 5. Tests

- `test_visit_stats.py`: `prior_prob` defaults to `0.0`; `add_edge` stamps
  `tactic.probability` into it.
- `test_selection_policy.py`: migrate to the new `puct_score(stats, total, c)` signature
  (priors move into `EdgeVisitStats(prior_prob=...)`); add `resolve_search_params`
  cases (legacy+explicit budget → ValueError; puct without num_simulations → ValueError;
  puct None resolution; rp → NotImplementedError).
- `test_rl_reasoner.py` `MCTSSearchTests`:
  - Multi-simulation tests add `selection_policy="puct"` to their `search_kwargs`.
  - `test_k1_legacy_loop_never_touches_visit_stats` → renamed to reflect the new gate
    (default = legacy); assertion unchanged.
  - **Corrected regression test (replaces the Phase 1 exit criterion):** a
    default-constructed reasoner and one with explicit `selection_policy="legacy"`, same
    seed and executor script, produce structurally identical graphs — same node
    expressions/statuses/creation order, same edge tactics/child sets. Both paths are
    now the same code, so this is the structural-equality check the addendum specifies,
    and it pins the default.
  - New: `selection_policy="puct"` with `num_simulations=1` accumulates visit statistics
    (N > 0 somewhere) — proving the gate is the policy, not the simulation count.
  - New: constructing a reasoner with `selection_policy="legacy"` and explicit
    `num_simulations` raises `ValueError`.
- `test_rl_training_driver.py`: config `__post_init__` rejection; JSON round-trip with
  the `None` fields (existing `test_config_json_roundtrip` covers serialization once the
  defaults change).

### 6. Documentation

- `htps_mcts_integration_plan.md`: mark Decision 2 and the §2.2 `selection_policy`
  paragraph as superseded, pointing at the addendum; correct the §2.2 description to
  three literals with legacy as default.
- `docs/dev_plans/htps_mcts_phases_0_3_integration.md`: replace the Phase 1 exit
  criterion with the structural-equality criterion.

## Alternatives considered

**Gate: keep `num_simulations > 1` and merely document the mismatch.**
Pros: no signature changes. Cons: the regression claim stays false — a user setting
`num_simulations=1` under a future default of PUCT would believe they get legacy
behavior; the addendum explicitly rejects this. Rejected.

**Prior population site: `RLHybridReasoner._link` override instead of `add_edge`.**
Pros: keeps `hypergraph.py` untouched. Cons: only RL-created edges get a prior; the base
reasoner's edges (and any future caller) stay at 0.0, splitting behavior across callers —
exactly the fallback-shaped divergence the working rules prohibit. `add_edge` is the
shared seam; one line covers everyone. Rejected.

**Prior storage: leave `puct_score(prior, ...)` taking the prior as an argument.**
Pros: no test churn. Cons: nothing stops a future caller from passing
`edge.tactic.probability` again; reading `stats.prior_prob` inside `puct_score` makes
the stored-once invariant structural. Chosen: change the signature, migrate both callers.

**Validation: sniff explicit keys in `from_json`'s payload instead of `None` sentinels.**
Pros: numeric defaults stay. Cons: only protects the JSON path — a programmatically
constructed config or a direct reasoner construction with a stray budget passes
silently. `None` sentinels validate every construction path through one function.
Rejected.

**Prior content: joint tactic × argument probability.** Would match HTPS's action prior
exactly but requires per-step argument log-probs on `ActionSample` (the current summed
`arg_logp` overcounts rows whose arity is below the batch max). User chose tactic-head
only; the joint prior can be a later, isolated change since the storage seam won't move.

## Verification

1. `uv run python -m pytest maths_ai/gnn_inference/tests/ -q` — all green except the
   pre-existing `test_premise_pool` failure.
2. `uv run python -m maths_ai.gnn_inference.scripts.rl_smoke` — exercises the default
   (legacy) path end-to-end against live Lean.
3. Grep check: no remaining reads of `edge.tactic.probability` inside selection code;
   no remaining `num_simulations > 1` gates.
