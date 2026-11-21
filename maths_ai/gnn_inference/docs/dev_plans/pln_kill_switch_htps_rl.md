# PLN kill switch for the HTPS RL search (MCTS-AC-branch)

## Relationship to the legacy plan

`pln_kill_switch_legacy_rl.md` covers the same flag design targeting `AC-branch`
(pre-HTPS). This document applies the identical flag (`use_pln: bool = True`) to
`MCTS-AC-branch`, where the HTPS integration has restructured the call sites. Read
the legacy plan for the full design rationale, alternatives table, and background on
the three PLN-call sites. This document records only the structural divergences and
the updated change instructions.

## Structural divergences from the legacy plan

### 1. PLN call sites moved to `_execute_and_link`

On `AC-branch`, both the subgoal-ranking call and the PLN fallback block live inside
`_expand`. On `MCTS-AC-branch`, the `_expand` execution stage was factored out into
`_execute_and_link`, which is called from both:

- `_expand` (legacy best-first path, `selection_policy="legacy"`)
- `_expand_leaves` (HTPS batched path, `selection_policy="puct"`)

This means gating once in `_execute_and_link` disables PLN across both search modes.
The legacy plan's instruction to gate `_expand` is replaced by gating
`_execute_and_link`.

### 2. Line numbers throughout `joint_inference.py` are stale

`MCTS-AC-branch/joint_inference.py` is ~930 lines (vs the shorter AC-branch version).
The specific line references in the legacy plan's section 1 do not apply. Use the
function names below instead.

### 3. Test infrastructure

`MCTS-AC-branch` test files already contain `MCTSSearchTests`, `HTPSDriverTests`, and
the updated helper `_make_reasoner` (which passes `**reasoner_kwargs` to
`RLHybridReasoner`). The new `PLNDisabledTests` class slots in without changes to
that helper — the `use_pln=False` kwarg flows through `**reasoner_kwargs` automatically.

### 4. `rl_training_driver.py` construction sites

On `MCTS-AC-branch` there are two `RLHybridReasoner` construction sites:

- Main training loop (`run_rl_training`, around line 564) — already threads
  `selection_policy`, `num_simulations`, `sim_batch_size`, `puct_c`; add `use_pln`.
- `--eval-only` CLI path (`_eval` inner function, around line 792) — does not thread
  the HTPS params; add `use_pln=cfg.use_pln` here too (or it defaults to `True` on
  eval, which is acceptable if the flag's primary use is training-only ablation — but
  consistency with the training path is cleaner).

The injectable `reasoner_factory` path already receives `cfg` from the call site, so
`cfg.use_pln` is naturally available to any factory.

### 5. Config already has the HTPS fields

`rl_actor_critic.json` on `MCTS-AC-branch` already contains `selection_policy`,
`num_simulations`, `sim_batch_size`, `puct_c`. Add `"use_pln": true` in the same
search-parameters block, adjacent to `selection_policy`.

---

## Changes (MCTS-AC-branch version)

All edits happen in a new worktree off `MCTS-AC-branch`:

```bash
git worktree add .claude/worktrees/pln-kill-switch-htps \
    -b MCTS-AC-branch-pln-kill-switch MCTS-AC-branch
```

Copy this plan file into the worktree's `maths_ai/gnn_inference/docs/dev_plans/`
before committing.

### 1. `maths_ai/hybrid_reasoner/joint_inference.py`

**`__init__`**: add `use_pln: bool = True` (keyword-only, alongside the existing
`dts_sampler`/`dts_c`/`dts_random_seed` parameters). Store `self.use_pln`.

When `use_pln=False`:
- Skip `PLNInference()` — set `self.petta_chainer = None`.
- Skip the DTS construction and state-file loading block — set `self.dts_sampler = None`.
  Keep `self._dts_rng` construction unconditional (cheap, and other code may reference it).

**`rank_subgoals`**: add guard at the top:

```python
if self.petta_chainer is None:
    raise RuntimeError(
        "rank_subgoals requires PLN; the reasoner was constructed with use_pln=False"
    )
```

**`_execute_and_link`** (this is the key structural change vs. the legacy plan):

This is the single function that handles tactic execution and hypergraph linking for
BOTH `_expand` (legacy) and `_expand_leaves` (HTPS). Gate both PLN paths here.

The subgoal-ranking branch (currently: `ranked = await self.rank_subgoals(...)`,
`chosen = [(c.goal, c.stv) for c in ranked[:self.top_k_subgoals]]`) becomes:

```python
if self.use_pln:
    print(f"  [PLN Ranking] scoring {len(outcome.subgoals)} subgoal(s)...")
    ranked = await self.rank_subgoals(
        node.goal.expression, outcome.subgoals, tactic,
        gnn_probability=tactic.probability,
    )
    print(f"  [PLN Done] ranked {len(ranked)} subgoal(s)")
    for i, rs in enumerate(ranked):
        print(
            f"    subgoal {i}: {rs.goal.expression} | "
            f"stv=({rs.stv.strength:.3f}, {rs.stv.confidence:.3f}) | "
            f"combined_rank={rs.combined_rank:.4f}"
        )
    chosen = [(c.goal, c.stv) for c in ranked[: self.top_k_subgoals]]
else:
    chosen = [(g, None) for g in outcome.subgoals[: self.top_k_subgoals]]
self._link(graph, node, tactic, ranked_subgoals=chosen)
```

The PLN-fallback block (`if not any_applied: ...`, which evaluates the goal with PLN
and may close the node through a `PLN_fallback` edge) is additionally gated on
`self.use_pln`:

```python
if not any_applied:
    if self.use_pln:
        print(f"  [PLN Fallback] evaluating goal: ...")
        pln_result = await self.petta_chainer.evaluate_async(...)
        # ... existing fallback logic unchanged ...
        if stv.score >= 0.9:
            graph.add_edge(node.id, TacticCandidate("PLN_fallback", [], 1.0),
                           ranked_subgoals=[])
            self._on_expansion_complete(node)
            return
```

With `use_pln=False`, the `if not any_applied` path falls through directly to
`mark_node_exhausted` + `_on_expansion_complete` — the node is exhausted with the
existing "executor rejected every candidate tactic" note.

**`main()` CLI** (`__main__` block at the end): unchanged; it constructs with the
default `use_pln=True` and its existing `dts_sampler is not None` guards hold.

### 2. `maths_ai/hybrid_reasoner/hypergraph.py` — annotation widening only

`add_edge`: widen `ranked_subgoals: List[Tuple[Goal, STV]]` to
`List[Tuple[Goal, Optional[STV]]]` and note in the docstring that `stv=None` means
"unscored — PLN disabled". No behavior change:
- `_new_node` already accepts `stv: Optional[STV]`.
- `ProofNode.local_score` already computes
  `gnn_probability × (stv.score if stv is not None else 1.0)`.

### 3. `maths_ai/gnn_inference/atp_lean_gnn/rl_training_driver.py` — config threading

`RLTrainingConfig`: add `use_pln: bool = True` in the search-parameters block (after
`top_k_subgoals`, before or alongside `selection_policy`).

Thread it through both `RLHybridReasoner` construction sites:

```python
# main training loop site (~line 564):
reasoner = RLHybridReasoner(
    ...
    selection_policy=cfg.selection_policy,
    num_simulations=cfg.num_simulations,
    sim_batch_size=cfg.sim_batch_size,
    puct_c=cfg.puct_c,
    use_pln=cfg.use_pln,        # add this
)

# --eval-only site (~line 792):
reasoner = RLHybridReasoner(
    ...
    use_pln=cfg.use_pln,        # add this
)
```

`RLHybridReasoner.__init__` forwards it automatically via `**reasoner_kwargs`, which
is passed to `HybridReasoner.__init__` — no change to `rl_reasoner.py`.

### 4. `maths_ai/gnn_inference/configs/rl_actor_critic.json`

Add `"use_pln": true` in the search-parameters block, adjacent to `"selection_policy"`:

```json
"selection_policy": "legacy",
"use_pln": true,
"num_simulations": null,
```

### 5. `maths_ai/gnn_inference/atp_lean_gnn/pln_reward.py` — no changes

`potential()` already returns `0.0` when `node.stv is None`, so
`edge_shaped_reward` reduces to `edge_terminal_reward` when every subgoal node has
`stv=None`. State this in the commit message rather than touching the module.

### 6. Tests

New `PLNDisabledTests` class in
`maths_ai/gnn_inference/tests/test_rl_reasoner.py` (5 tests):

The existing `_make_reasoner` helper already passes `**search_kwargs` through
`**reasoner_kwargs` to `RLHybridReasoner`, so `use_pln=False` flows through without
changes to the helper.

1. **No PLN objects exist** — `_make_reasoner(use_pln=False)`: assert
   `reasoner.petta_chainer is None` and `reasoner.dts_sampler is None`. Do not assign
   a `_StubPLN` — the point is that nothing dereferences the chainer.

2. **Subgoal nodes carry `stv=None`, executor order, cap** — `_SubgoalExecutor`
   mapping the root to three subgoals with `top_k_subgoals=2`: exactly the first two
   subgoals (Lean's order) become children, each with `node.stv is None`. Covers both
   `_expand` (legacy) and `_expand_leaves` (PUCT) paths because both route through
   `_execute_and_link`.

3. **No fake QED on total rejection** — `_RejectExecutor` with `use_pln=False`: the
   graph is unsolved, no edge with `tactic_name="PLN_fallback"` exists, the root is
   exhausted with the "executor rejected every candidate tactic" note.

4. **Reward is terminal-only** — run a search with `use_pln=False`; for every
   harvested transition assert
   `edge_shaped_reward(edge, graph, cfg) == edge_terminal_reward(edge, graph, cfg)`
   (shaping term exactly zero because every `Φ` input is `None` or terminal).

5. **`rank_subgoals` guard** — calling `rank_subgoals` on a `use_pln=False` reasoner
   raises `RuntimeError`.

New tests in `maths_ai/gnn_inference/tests/test_rl_training_driver.py`:

6. **Config roundtrip** — `use_pln=False` survives `to_dict` → `from_json`.

7. **Threading** — inject a factory that asserts `cfg.use_pln is False` and returns a
   mock reasoner; run `run_rl_training` with `use_pln=False` in config and confirm the
   factory received it (avoids needing a live Lean server for this assertion).

### 7. Commit

Write the commit message to a temporary file in the project root (not committed), per
working-style rules.

---

## Verification

On the worktree:

1. `uv run python -m pytest maths_ai/gnn_inference/tests/ -q` — full suite. The
   pre-existing `test_premise_pool` failure (if any) is tolerated; confirm with a
   baseline run before the change.

2. Grep checks:

   ```bash
   # Every petta_chainer dereference must be inside a self.use_pln guard
   # or after the rank_subgoals RuntimeError guard.
   grep -n "petta_chainer\." maths_ai/hybrid_reasoner/joint_inference.py

   # Every dts_sampler dereference must be behind an existing `is not None` check.
   grep -n "dts_sampler\." maths_ai/hybrid_reasoner/joint_inference.py
   ```

3. Confirm `_execute_and_link` contains both gates (not `_expand` or `_expand_leaves`):

   ```bash
   grep -n "use_pln" maths_ai/hybrid_reasoner/joint_inference.py
   ```

   Expected: hits inside `__init__`, `rank_subgoals`, and `_execute_and_link` only.
