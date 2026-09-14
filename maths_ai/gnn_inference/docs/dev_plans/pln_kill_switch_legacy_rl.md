# PLN kill switch for the legacy RL search (AC-branch)

## Context

The legacy RL training loop (pre-HTPS, committed on `AC-branch` at 5ffd525) calls the PLN
component (a petta subprocess per query) in three places during every search:

1. **Subgoal ranking** — `HybridReasoner.rank_subgoals` (`maths_ai/hybrid_reasoner/joint_inference.py:268`)
   dispatches one `petta_chainer.evaluate_async` per subgoal and sorts by
   `combined_rank = gnn_probability × STV.score`, where the STV (strength, confidence) is
   the PLN's truth-value estimate for the subgoal and `score = strength × confidence`.
2. **The PLN fallback in `_expand`** (`joint_inference.py:477-498`) — when the executor
   rejects every candidate tactic, the node's own goal is sent to PLN; if the resulting
   `stv.score ≥ 0.9` the node is closed as SOLVED through a fabricated `PLN_fallback`
   edge added via `graph.add_edge` directly, bypassing `_link`.
3. **Reward shaping** — `edge_shaped_reward` (`maths_ai/gnn_inference/atp_lean_gnn/pln_reward.py`)
   adds a potential term `Σ_j(γ·Φ(child_j) − Φ(parent))` where the potential
   `Φ(node) = node.stv.strength` (or `.score`) comes from the PLN STV stored on each node.

The user wants a single flag that disables all PLN involvement so the actor-critic trains
on the terminal reward alone (`edge_terminal_reward`: `terminal_success − step_penalty`
on a QED edge, `terminal_failure − step_penalty` on a DEAD edge, `−step_penalty`
otherwise) without spawning any petta subprocess. The DTS (Dynamic Thompson Sampling)
bandit exists only to replace fallback STVs, so it goes down with the same switch.

Three decisions are locked (user-confirmed):
- **Worktree off `AC-branch`** — the current `MCTS-AC-branch` working tree carries
  uncommitted HTPS changes and stays untouched.
- **Single kill switch** — one constructor flag `use_pln: bool = True`; `False` disables
  the ranking calls, the `_expand` fallback (including the fake QED), the DTS sampler,
  and — through `stv=None` — the shaping term.
- **Executor order + cap** — with PLN off, subgoals keep the order Lean produced them
  in, capped at `top_k_subgoals`; frontier ordering degrades to GNN probability alone.

Two existing mechanisms make the switch almost free, and are the reason `stv=None` is
the chosen representation:

- `ProofNode.local_score` (`maths_ai/hybrid_reasoner/hypergraph.py`) already computes
  `gnn_probability × (stv.score if stv is not None else 1.0)` — frontier ranking
  degrades to GNN probability with no change.
- `potential()` in `pln_reward.py` already returns `0.0` when `node.stv is None` — the
  shaping sum vanishes and `edge_shaped_reward` reduces to `edge_terminal_reward` with
  no change to the reward module.

A side benefit: skipping the `_expand` fallback also removes the fake-QED leak, where a
`PLN_fallback` edge marks a subtree SOLVED and `backup_values` then feeds 1.0 into the
critic's `value_target` even though the actor term filters that edge out.

## Alternatives considered

| Approach | Pros | Cons | Verdict |
|---|---|---|---|
| **`use_pln=False` constructor flag; `stv=None` on all subgoal nodes** (chosen) | No petta subprocess ever spawns; shaping and frontier ranking degrade through existing `None` paths with zero reward-math changes; fake-QED leak disappears | One more constructor parameter to thread through config | **Chosen** |
| Two flags (disable shaping independently of PLN-guided search) | Finer experiments (PLN search + terminal-only reward) | Two half-modes to test; user wants full isolation from PLN | Rejected by user |
| `RewardConfig`-level flag zeroing the shaping term only | Smallest diff | PLN subprocesses still spawn per subgoal; fails the "without having to call the PLN component" requirement | Rejected |
| Placeholder `stv=STV(1.0, 1.0)` instead of `None` | Avoids widening the `add_edge` annotation | A constant Φ still shapes every edge with `γΦ − Φ ≠ 0`, and `local_score` would multiply by 1.0 anyway — `None` does both correctly for free | Rejected |
| Re-rank subgoals by a heuristic (expression length) with PLN off | Some ordering signal retained | Invents an unvalidated heuristic; user chose executor order + cap | Rejected by user |

## Changes

All edits happen in a new git worktree checked out from `AC-branch` on a new branch:

```bash
git worktree add .claude/worktrees/pln-kill-switch -b AC-branch-pln-kill-switch AC-branch
```

Copy this plan file into the worktree's `maths_ai/gnn_inference/docs/dev_plans/` so it is
committed with the change.

### 1. `maths_ai/hybrid_reasoner/joint_inference.py` — the switch itself

- `__init__` gains `use_pln: bool = True` (keyword-only, alongside the existing
  `dts_sampler`/`dts_c`/`dts_random_seed` parameters). Store `self.use_pln`.
- When `use_pln` is `False`:
  - `self.petta_chainer = None` (the `PLNInference()` construction is skipped entirely);
  - `self.dts_sampler = None` and the DTS state-file loading block
    (`joint_inference.py:181-194`) is skipped — no file I/O, no RNG needed for sampling
    (keep `self._dts_rng` construction unconditional; it is cheap and other code may
    reference it).
- `rank_subgoals` gains an explicit guard at the top:
  `if self.petta_chainer is None: raise RuntimeError("rank_subgoals requires PLN; the reasoner was constructed with use_pln=False")`
  so a future caller fails with a named error instead of an `AttributeError` on `None`.
- `_expand` (`joint_inference.py:426`):
  - The subgoal-ranking branch becomes:
    ```python
    if self.use_pln:
        ranked = await self.rank_subgoals(...)
        chosen = [(c.goal, c.stv) for c in ranked[: self.top_k_subgoals]]
    else:
        chosen = [(g, None) for g in outcome.subgoals[: self.top_k_subgoals]]
    self._link(graph, node, tactic, ranked_subgoals=chosen)
    ```
    Executor order is preserved; the `top_k_subgoals` cap still applies as the
    branching budget. Adjust the `[PLN Ranking]` prints to only fire under the flag.
  - The entire `if not any_applied:` PLN-fallback block (`joint_inference.py:476-498`)
    is additionally gated on `self.use_pln`. With the flag off, a node whose candidates
    were all rejected goes straight to `mark_node_exhausted` with the existing note.
- The CLI `main()` (`joint_inference.py:545+`) is untouched: it constructs with the
  default `use_pln=True` and its `dts_sampler is not None` guards already hold.

### 2. `maths_ai/hybrid_reasoner/hypergraph.py` — annotation widening only

- `add_edge` (`hypergraph.py:220`): widen `ranked_subgoals: List[Tuple[Goal, STV]]` to
  `List[Tuple[Goal, Optional[STV]]]` and note in the docstring that `stv=None` means
  "unscored — PLN disabled". `_new_node` already accepts `stv: Optional[STV]`, and
  `ProofNode.local_score` already handles `None`; no behavior change.

### 3. `maths_ai/gnn_inference/atp_lean_gnn/rl_training_driver.py` — config threading

- `RLTrainingConfig` gains `use_pln: bool = True` (in the search-parameters group next
  to `top_k_subgoals`).
- Both `RLHybridReasoner` construction sites pass it through:
  the training-loop site (`rl_training_driver.py:466`) and the `--eval` site
  (`rl_training_driver.py:635`).
- `RLHybridReasoner.__init__` forwards it automatically via the existing
  `**reasoner_kwargs` passthrough (`rl_reasoner.py`) — no reasoner change needed.

### 4. `maths_ai/gnn_inference/configs/rl_actor_critic.json`

- Add `"use_pln": true` to the search-parameters block (documents the knob; flipping it
  to `false` is the user's terminal-reward-only run).

### 5. `maths_ai/gnn_inference/atp_lean_gnn/pln_reward.py` — no changes

`potential()` already returns `0.0` when `node.stv is None`, so
`edge_shaped_reward = edge_terminal_reward + 0`. State this in the commit message rather
than touching the module.

### 6. Tests

New tests follow the existing fake-executor idioms in the AC-branch versions of
`maths_ai/gnn_inference/tests/test_rl_reasoner.py` (`_QEDExecutor`, `_RejectExecutor`,
`_SubgoalExecutor`, `_make_reasoner`) and `test_rl_training_driver.py` (`_write_config`).
`_make_reasoner` gains `**search_kwargs` passthrough if the AC-branch version lacks it.

In `test_rl_reasoner.py` (a new `PLNDisabledTests` class):

1. **No PLN objects exist** — construct with `use_pln=False`; assert
   `reasoner.petta_chainer is None` and `reasoner.dts_sampler is None`. Do NOT reassign
   `_StubPLN` for these tests — the point is that nothing ever dereferences the chainer.
2. **Subgoal nodes carry `stv=None`, executor order, cap** — `_SubgoalExecutor`
   mapping the root to three subgoals with `top_k_subgoals=2`: exactly the first two
   subgoals (Lean's order) become children, each with `node.stv is None`.
3. **No fake QED on total rejection** — `_RejectExecutor` with `use_pln=False`: the
   graph is unsolved, no edge named `PLN_fallback` exists, the root is exhausted with
   the "executor rejected every candidate tactic" note.
4. **Reward is terminal-only** — run a search with `use_pln=False`, then for every
   harvested transition assert
   `edge_shaped_reward(edge, graph, cfg) == edge_terminal_reward(edge, graph, cfg)`
   (shaping term exactly zero because every `Φ` input is `None` or terminal).
5. **`rank_subgoals` guard** — calling `rank_subgoals` on a `use_pln=False` reasoner
   raises `RuntimeError`.

In `test_rl_training_driver.py`:

6. **Config roundtrip** — `use_pln=False` survives `to_dict`/`from_json`.
7. **Threading** — a run with `use_pln=False` through `_write_config` +
   `run_rl_training` with the default (non-factory) path is not cheaply testable without
   Lean; instead assert at the factory seam: pass a factory that records
   `cfg.use_pln` and asserts it reached the factory as `False`.

### 7. Commit

Commit message written to a temporary file in the project root (not committed), per
working-style rules.

## Verification

On the worktree:

1. `uv run python -m pytest maths_ai/gnn_inference/tests/ -q` — full suite; only the
   pre-existing `test_premise_pool` failure is tolerated (it fails on `AC-branch`
   before this change too — confirm with a baseline run first).
2. Grep checks:
   - `grep -n "petta_chainer\." maths_ai/hybrid_reasoner/joint_inference.py` — every
     dereference is inside a `self.use_pln` guard or after the `rank_subgoals`
     RuntimeError guard.
   - `grep -n "dts_sampler\." maths_ai/hybrid_reasoner/joint_inference.py` — every
     dereference sits behind an existing `is not None` check.
3. Optional live check (needs Lean + petta absent from the environment to be
   meaningful): a small `rl_smoke`-style run with `"use_pln": false` completes without
   any petta subprocess spawning — observable because `PLNInference` is never
   constructed.
