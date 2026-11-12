"""A scripted runtime backend for exercising the real runtime process.

It loads nothing. A proof request's goal scripts its behavior:
``sleep:S`` waits cooperatively, ``block:S`` blocks the event loop the way a
native call does, ``spawn`` starts a child in its own session as a Lean REPL
is started, ``fail`` raises a typed failure and ``proved`` returns a certified
outcome. The startup failure is chosen by the config's checkpoint name.
"""

from __future__ import annotations

import asyncio
import time

from maths_ai.api.runtime_backend import BackendError

SPAWNED: list[int] = []


class FakeBackend:
    def __init__(self, config) -> None:
        self.config = config
        self.children: list[asyncio.subprocess.Process] = []
        self.evaluated = 0

    async def start(self):
        checkpoint = self.config.checkpoint
        if checkpoint is not None and checkpoint.name == "missing.pt":
            raise BackendError("artifacts", "checkpoint missing.pt does not exist")
        return {"checkpoint": "fake", "workers": 1}

    def status(self):
        return {"workers": 1, "usable_workers": 1, "admission_clients": 0,
                "live_clients": 1 + len(self.children), "resident_jobs": 0}

    async def prove(self, payload):
        goal = payload["goal"]
        if goal.startswith("sleep:"):
            await asyncio.sleep(float(goal.split(":")[1]))
        elif goal.startswith("block:"):
            time.sleep(float(goal.split(":")[1]))
        elif goal == "spawn":
            child = await asyncio.create_subprocess_exec(
                "sleep", "300", start_new_session=True
            )
            self.children.append(child)
            return {"spawned": child.pid}
        elif goal == "fail":
            raise BackendError("input:malformed-text", "scripted input failure")
        elif goal == "inadmissible":
            return {"outcome": "inadmissible", "exception_category": "lean-rejected"}
        elif goal in ("resource", "resource-recursion"):
            heartbeats = goal == "resource"
            return {
                "outcome": "resource_limit", "certified": False,
                "resource": {"command": "goal.tactic",
                             "kind": "heartbeats" if heartbeats else "recursion-depth",
                             "phase": "execute", "limit": 200000 if heartbeats else 512,
                             "recovery": "command-rolled-back"},
            }
        return {
            "outcome": "solved" if goal == "proved" else "unsolved",
            "certified": goal == "proved",
            "candidate": goal == "proved",
            "end_reason": "root_certified" if goal == "proved" else "num_simulations",
            "proof_trace": {"goal": goal, "command": "trivial", "subgoals": []} if goal == "proved" else None,
            "proposal": {"mode": payload["proposal_mode"], "seed": payload["seed"], "derivation": "fake"},
            "echo": payload,
        }

    async def admit(self, payload):
        self.admit_steps = getattr(self, "admit_steps", 0) + 1
        done = self.admit_steps >= 2
        repeats = payload["evaluation"]["repeats"]
        progress = {"rows_seen": 2 * self.admit_steps, "admitted": 2 * self.admit_steps,
                    "newly_admitted": 2, "done": done, "admission": {"admitted": 2 * self.admit_steps}}
        if done:
            progress.update(input_records=[{"identity": "fake-0"}, {"identity": "fake-1"}], proposal_records=[[] for _ in range(repeats)], eval_items=2, eval_subset_sha256="fake-subset",
                            proposal_seeds_sha256=[f"seeds-{r}" for r in range(repeats)],
                            admission={"admitted": 4})
        return progress

    async def evaluate(self, payload):
        self.evaluated += 1
        if payload["repeats"] == 3:  # a slow job, to cancel mid-run
            await asyncio.sleep(1.0)
        start, count = payload["start"], payload["count"]
        attempts = [
            {"index": start + i, "repeat": payload["repeat"], "outcome": "solved",
             "certified": True, "end_reason": "root_certified"}
            for i in range(count)
        ]
        stats = {"attempted": float(count), "collected": float(count), "solved": float(count),
                 "searches_failed": 0.0, "searches_cooperative_deadline": 0.0,
                 "searches_hard_timed_out": 0.0, "seeds_inadmissible": 0.0}
        return {"attempts": attempts, "stats": stats, "chunk_s": 0.0}

    async def release(self, payload):
        return {"released": True}

    async def close(self):
        for child in self.children:
            if child.returncode is None:
                child.kill()
                await child.wait()
