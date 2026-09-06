# Pantograph Server Crash Recovery — Dev Plan

## Goal

Fix the RL training loop hanging on Loop 1 due to pantograph-repl process crashes going
unhandled, leaving the training driver waiting on a dead subprocess indefinitely.

---

## Root Cause

Two distinct failure modes are both printing to the terminal and contributing to the hang:

**Mode 1 — Elaboration error (alive server, bad goal):**
`ServerError` with `{'error': 'elab'}` is raised by `goal_start_async` when the goal
expression fails Lean elaboration (universe-polymorphic goals, unknown identifiers like
`rexp`). The pantograph-repl process stays alive. These are expected on malformed goals.

**Mode 2 — Server crash (dead server):**
Certain goals trigger the `getIndentAndColumn` → `String.Slice.pos!` panic in
pantograph-repl, which hard-crashes the subprocess. The Python-side `Server` object holds
a now-dead subprocess. Any subsequent `goal_start_async` call awaits a JSON response that
will never arrive — the event loop hangs there until `asyncio.wait_for` in `collect_round`
fires its timeout.

**Why the loop stalls:**
The `Server` object is created once per training run and shared across every theorem search.
After the first crash, `collect_round` catches the exception for theorem N via its bare
`except Exception` guard. But the `Server` reference on the `RLHybridReasoner` now points
to a dead process. Theorem N+1 calls `_start_state` → `goal_start_async` → waits forever
(or until the `collect_round` timeout fires). If `timeout_s` is large (or if the dead pipe
happens to accept the write without raising immediately), all subsequent theorems in the
round stall one by one.

**Second gap — `_start_state` has no error handling in `_expand`:**
`_expand` wraps `executor.apply` (tactic application) in a try/except, but calls
`_start_state` directly with no guard. A `ServerError` from `_start_state` propagates
uncaught through `_expand` → `prove`'s while loop → crashes `prove` → is caught by
`collect_round`'s bare `except Exception`. This skips the rest of the theorem but leaves
the dead server in place.

---

## Alternatives Considered

| Approach | Verdict |
|---|---|
| Redirect pantograph stderr to `/dev/null` | Hides noise; doesn't fix hang or dead server |
| Per-theorem `Server.create()` | Eliminates sharing but Lean startup (especially with Mathlib) takes 30–60 s per theorem — prohibitive |
| Blacklist panic-triggering goals | Reactive; doesn't handle novel panics; doesn't fix current hangs |
| Lower `timeout_s` in `collect_round` | Makes the driver fail-fast on individual hangs; but doesn't eliminate dead-server accumulation across theorems |
| **Wrap + restart (chosen)** | Catches both failure modes at source; restarts server transparently; skips unelab-able goals cleanly |

---

## Chosen Approach: Wrap + Restart

Three coordinated changes across two files.

### Change 1 — `_expand`: guard `_start_state` call

`_start_state` can raise `ServerError` (elab) or `ParseError` (parse). These mean the goal
cannot be elaborated — the node should be marked exhausted and the loop should continue to
the next frontier node. Currently these exceptions propagate uncaught through `_expand`,
crashing the `prove` coroutine.

```python
# _expand, before the tactic loop:
try:
    state = await self._start_state(node.goal)
except (ServerError, ParseError) as exc:
    console_print(f"  [Node {node.id} SKIP] goal elaboration failed: {exc}")
    graph.mark_node_exhausted(node.id, note=f"elaboration error: {exc}")
    return
```

### Change 2 — `_restart_server`: new method on `HybridReasoner`

```python
async def _restart_server(self) -> None:
    """Close the current pantograph subprocess and spawn a fresh one.

    Called when goal_start_async or goal_tactic_async raises a
    broken-pipe or EOF exception, which means the pantograph-repl
    process has crashed. Updates self.server and self.executor.server
    so all subsequent calls go to the new process.
    """
    try:
        await self.server.close()
    except Exception:
        pass
    self.server = await Server.create(**self._server_kwargs)
    self.executor.server = self.server
    console_print("  [Server] pantograph restarted after crash")
```

`_server_kwargs` is stored at `__init__` time (see Change 3).

### Change 3 — `HybridReasoner.__init__`: store server kwargs

```python
def __init__(self, ..., server_kwargs: dict | None = None, ...) -> None:
    self._server_kwargs = server_kwargs or {}
    ...
```

`rl_training_driver.run_rl_training` already creates `server = await Server.create()` with
no kwargs (bare Lean, no Mathlib). The driver must pass `server_kwargs={}` explicitly so
the restart uses the same environment. If Mathlib is needed, this is the single place to
update it.

### Change 4 — `_start_state`: detect dead server and restart

Wrap the two `goal_start_async` / `goal_tactic_async` calls in `_start_state`:

```python
async def _start_state(self, goal: Goal) -> GoalState:
    ...
    try:
        state = await self.server.goal_start_async(expression)
        if goal.hypotheses:
            names = ...
            state = await self.server.goal_tactic_async(state, f"intro {names}")
    except (BrokenPipeError, ConnectionResetError, EOFError):
        await self._restart_server()
        state = await self.server.goal_start_async(expression)
        if goal.hypotheses:
            names = ...
            state = await self.server.goal_tactic_async(state, f"intro {names}")
    return state
```

If the restart itself fails, the exception propagates up to `_expand`'s new guard (Change 1),
which marks the node exhausted.

### Change 5 — `rl_training_driver.run_rl_training`: pass `server_kwargs`

```python
server = await Server.create()
server_kwargs = {}          # extend with project_path/imports if Mathlib is needed
executor = PantographExecutor(server)
reasoner = RLHybridReasoner(
    ...,
    executor=executor,
    server_kwargs=server_kwargs,
    ...
)
```

`RLHybridReasoner` inherits from `HybridReasoner`, so `server_kwargs` is picked up via the
parent `__init__` with no changes needed to `rl_reasoner.py`.

---

## Files Changed

| File | Change |
|---|---|
| `maths_ai/hybrid_reasoner/joint_inference.py` | Changes 1, 2, 3, 4 |
| `maths_ai/gnn_inference/atp_lean_gnn/rl_training_driver.py` | Change 5 |

---

## What is NOT changed

- `collect_round`'s `asyncio.wait_for` + bare `except Exception` stays as a backstop.
- `PantographExecutor.apply`'s bare `except Exception` stays — tactic failures are already
  handled correctly there (returns `TacticOutcome(success=False)`).
- No changes to `rl_reasoner.py` — `RLHybridReasoner` inherits the fix.

---

## Verification

After the fix:
1. Goals with elab errors → `[Node N SKIP] goal elaboration failed` log line → next theorem
2. Server crash → `[Server] pantograph restarted after crash` → current theorem's node
   marked exhausted → training round continues
3. Loop count advances past 1 on subsequent theorems
4. Run `pytest maths_ai/gnn_inference/atp_lean_gnn/` to verify no regressions in the
   existing test suite
