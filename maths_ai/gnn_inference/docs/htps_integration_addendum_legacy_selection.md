# Addendum: Fixing the `num_simulations` Regression Test

Supersedes Decision 2 and the Section 2.2 `selection_policy` description in
`htps_mcts_integration_plan.md`.

## Problem

The original plan claimed that setting `num_simulations = 1` would reduce the new
MCTS-style search loop to today's behavior. It doesn't, for two separate reasons:

1. **Unit mismatch.** `num_simulations` counts simulations, not node expansions. One
   simulation can expand several nodes at once — every unexpanded leaf of the selected
   partial AND-hypertree. Legacy search runs until the root resolves or a `max_nodes`
   **node-expansion** budget is exhausted, which is typically hundreds or thousands of
   expansions. `num_simulations = 1` does one small simulation and stops — not the same
   process at all.

2. **PUCT and `combined_rank` don't converge to each other at any setting.** PUCT (per
   the HTPS paper's own formula) combines only the policy prior with visit-count
   statistics — it has no PLN-style heuristic term. The legacy `combined_rank` is
   `policy_prob × PLN_score` with no visit-count term. At zero visits, PUCT's `Q` defaults
   to a constant (first-play-urgency), so it doesn't even reference PLN. Running PUCT with
   `num_simulations = 1` produces a different ranking than `combined_rank` ever did — it
   isn't a degenerate case of the same function, it's a different function.

**Consequence:** the Phase 1 exit criterion as written cannot be satisfied and should not
be used to validate correctness.

## Fix: a literal `"legacy"` selection policy

Add a third value to `selection_policy`, alongside `"puct"` and `"rp"`:

```python
selection_policy: Literal["legacy", "puct", "rp"]
```

When `selection_policy == "legacy"`:

- `select_partial_hypertree` does not use PUCT/RP or `N`/`W` at all.
- Selection reduces exactly to today's `frontier().pop()`: recompute
  `combined_rank = policy_prob × PLN_score` via the existing `_propagate` logic, pop the
  single highest-ranked OPEN node, expand only that one node.
- `sim_batch_size` is forced to `1` in this mode — no batching, no virtual loss, matching
  today's one-node-at-a-time loop.
- The loop terminates on root SOLVED/DEAD or `max_nodes` expansions, not on a simulation
  count.

This makes "legacy" its own literal code path, not something PUCT is expected to
approximate at low simulation counts.

## Corrected regression test (replaces the Phase 1 exit criterion)

> With `selection_policy = "legacy"`, running the new `run_search` loop against a fixed
> seed reproduces the exact same sequence of node expansions, edge creations, and terminal
> status as today's `HybridReasoner.prove` on the same input — a structural equality
> check, not a statistical one, since both code paths are now doing identical work.

`num_simulations` and `sim_batch_size` are only meaningful when `selection_policy` is
`"puct"` or `"rp"`. The config loader should warn (or reject) if they're set while
`selection_policy == "legacy"`.

## Related fix found during the same review: `prior_prob` was missing

`EdgeVisitStats` needs a stored prior, populated once at `expand_batch` time from the
tactic-head/pointer-head probabilities and read — never recomputed — by `puct_score` on
every later selection. Without it, PUCT has no prior to weight its exploration term
against.

```python
@dataclass
class EdgeVisitStats:
    N: int = 0
    W: float = 0.0
    virtual_loss: int = 0
    prior_prob: float = 0.0   # NEW — set once at expansion, read by puct_score thereafter
```

## Unaffected by this fix

`EdgeVisitStats`'s `N`/`W`/`virtual_loss`, `backup_simulation`, the soft critic target, and
minimal-hypertree mining are all orthogonal to selection-policy choice and need no changes
beyond the `prior_prob` addition above.
