from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from pantograph.server import ServerError
from torch.optim import AdamW

from maths_ai.data_models.proof_components import Goal, STV
from maths_ai.hybrid_reasoner.hypergraph import TacticOutcome
from maths_ai.pln_inference.model import PLNResult

from maths_ai.gnn_inference.atp_lean_gnn.actor_critic import ActorCriticWithArgsClassifier
from maths_ai.gnn_inference.atp_lean_gnn.checkpointing import checkpoint_payload
from maths_ai.gnn_inference.atp_lean_gnn.graph import proof_state_to_dag
from maths_ai.gnn_inference.atp_lean_gnn.pln_rl_training import goal_to_state
from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_vocab
from maths_ai.gnn_inference.atp_lean_gnn.rl_reasoner import RLHybridReasoner
from maths_ai.gnn_inference.atp_lean_gnn.rl_training_driver import (
    RLTrainingConfig,
    TheoremItem,
    TheoremPool,
    bc_weight_at_round,
    build_theorem_pool,
    collect_round,
    driver_main,
    evaluate_proof_rate,
    pantograph_env,
    run_rl_training,
    save_checkpoint,
)


# ---------------------------------------------------------------------------
# Fakes (mirroring test_rl_reasoner.py — no Lean/petta)
# ---------------------------------------------------------------------------


class _FakeGoalState:
    goals: list = []


class _FakeServer:
    async def goal_start_async(self, expression):
        return _FakeGoalState()

    async def goal_tactic_async(self, state, tactic):
        return _FakeGoalState()


class _QEDExecutor:
    def __init__(self):
        self.server = _FakeServer()

    async def apply(self, server, state, tactic):
        return TacticOutcome(success=True, subgoals=[])


class _RejectExecutor:
    def __init__(self):
        self.server = _FakeServer()

    async def apply(self, server, state, tactic):
        return TacticOutcome(success=False, subgoals=[], error="rejected")


class _UnelaboratedServer(_FakeServer):
    def __init__(self):
        self.proc = object()

    async def goal_start_async(self, expression):
        raise ServerError("cannot elaborate reconstructed goal")


class _UnelaboratedExecutor:
    def __init__(self):
        self.server = _UnelaboratedServer()

    async def apply(self, server, state, tactic):
        raise AssertionError("tactics must not run when elaboration fails")


class _StubPLN:
    async def evaluate_async(self, expression, hypotheses=None, **kwargs):
        return PLNResult(stv=STV(strength=0.1, confidence=1.0), status="ok", is_fallback=False)


class _RaisingReasoner:
    """prove() raises on a chosen call index — tests per-theorem fault isolation."""

    def __init__(self, inner, raise_on: set[int]):
        self._inner = inner
        self._raise_on = raise_on
        self._calls = 0
        self.model = inner.model
        self.dag_featurize_data = inner.dag_featurize_data

    async def prove(self, goal, *, hypotheses=None, greedy=False):
        idx = self._calls
        self._calls += 1
        if idx in self._raise_on:
            raise RuntimeError("simulated Lean transport failure")
        return await self._inner.prove(goal, hypotheses=hypotheses, greedy=greedy)


class _DeadServerReasoner:
    """prove() raises on every call except a listed set of call indices.

    Reproduces the failure the log snippet recorded: `goal_start_async` raised for
    every theorem, so `collect_round` returned no results at all and there was
    nothing to take a gradient step on. Distinct from `_RejectExecutor`, whose
    searches DO complete — their rejections are failure records, which are
    training signal.
    """

    def __init__(self, inner, *, succeed_on: set[int] | None = None):
        self._inner = inner
        self._succeed_on = succeed_on or set()
        self._calls = 0
        self.model = inner.model
        self.dag_featurize_data = inner.dag_featurize_data

    async def prove(self, goal, *, hypotheses=None, greedy=False):
        idx = self._calls
        self._calls += 1
        if idx not in self._succeed_on:
            raise RuntimeError("Unknown identifier ℕ")  # what a core-only REPL says
        return await self._inner.prove(goal, hypotheses=hypotheses, greedy=greedy)


TACTIC_VOCAB = {"trivial": 0, "intro": 1, "exact": 2}
GOAL_EXPR = "p → p"
HYPS = ["p : Prop"]


def _build_node_vocab():
    goal = Goal(expression=GOAL_EXPR, hypotheses=HYPS)
    return build_vocab([proof_state_to_dag(goal_to_state(goal))])


def _make_model(node_vocab):
    from maths_ai.gnn_inference.tests.model_helpers import actor_critic
    return actor_critic(len(node_vocab), len(TACTIC_VOCAB))


def _make_reasoner(model, node_vocab, executor, *, top_k=3):
    reasoner = RLHybridReasoner(
        model,
        node_vocab,
        TACTIC_VOCAB,
        executor=executor,
        top_k_tactics=top_k,
        max_depth=3,
        max_nodes=20,
    )
    reasoner.petta_chainer = _StubPLN()
    return reasoner


def _items(n: int) -> list[TheoremItem]:
    return [
        TheoremItem(goal=Goal(expression=GOAL_EXPR, hypotheses=HYPS), tactic_label="intro", size=10 + i)
        for i in range(n)
    ]


def _write_config(tmp: Path, **overrides) -> RLTrainingConfig:
    """Config pointing at a synthetic prepared_root + warm-start checkpoint in tmp."""
    node_vocab = _build_node_vocab()
    vocab_dir = tmp / "prepared" / "vocab"
    vocab_dir.mkdir(parents=True, exist_ok=True)
    with open(vocab_dir / "node_vocab.json", "w") as f:
        json.dump(node_vocab, f)
    with open(vocab_dir / "tactic_vocab.json", "w") as f:
        json.dump(TACTIC_VOCAB, f)

    torch.manual_seed(0)
    model = _make_model(node_vocab)
    ckpt = tmp / "warmstart.pt"
    torch.save(
        checkpoint_payload(
            model_kind="actor_critic_with_args",
            model_spec=model.model_spec,
            node_vocab=node_vocab,
            tactic_vocab=TACTIC_VOCAB,
            model=model,
        ),
        ckpt,
    )

    defaults = dict(
        warmstart_checkpoint=ckpt,
        prepared_root=tmp / "prepared",
        run_root=tmp / "runs",
        device="cpu",
        num_rounds=2,
        theorems_per_round=2,
        theorem_timeout_s=30.0,
        checkpoint_every=1,
        eval_every=0,
        eval_pool_size=0,
        bc_anneal_start=0.5,
        bc_anneal_end=0.0,
        bc_anneal_rounds=10,
        top_k_tactics=2,
        max_depth=3,
        max_nodes=20,
    )
    defaults.update(overrides)
    return RLTrainingConfig(**defaults)


def _pool(n_items: int = 6, eval_size: int = 0) -> TheoremPool:
    return TheoremPool(_items(n_items), eval_pool_size=eval_size, curriculum_size=4, seed=0)


def _qed_factory(model, node_vocab, tactic_vocab, cfg):
    return _make_reasoner(model, node_vocab, _QEDExecutor(), top_k=cfg.top_k_tactics)


def _reject_factory(model, node_vocab, tactic_vocab, cfg):
    return _make_reasoner(model, node_vocab, _RejectExecutor(), top_k=cfg.top_k_tactics)


def _unelaborated_factory(model, node_vocab, tactic_vocab, cfg):
    return _make_reasoner(model, node_vocab, _UnelaboratedExecutor(), top_k=cfg.top_k_tactics)


def _dead_server_factory(succeed_on: set[int] | None = None):
    """Reasoner factory whose searches all raise, except on listed call indices.

    `theorems_per_round` prove calls make up one round, so call indices
    {2n, 2n+1} are round n when `theorems_per_round=2`.
    """

    def factory(model, node_vocab, tactic_vocab, cfg):
        inner = _make_reasoner(model, node_vocab, _QEDExecutor(), top_k=cfg.top_k_tactics)
        return _DeadServerReasoner(inner, succeed_on=succeed_on)

    return factory


class ConfigTests(unittest.TestCase):
    def test_config_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp)
            path = tmp / "cfg.json"
            with open(path, "w") as f:
                json.dump(cfg.to_dict(), f)
            loaded = RLTrainingConfig.from_json(path)
            self.assertEqual(loaded.to_dict(), cfg.to_dict())
            self.assertIsInstance(loaded.warmstart_checkpoint, Path)

    def test_config_missing_required_field_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            with open(path, "w") as f:
                json.dump({"prepared_root": "x"}, f)  # no warmstart_checkpoint
            with self.assertRaises(TypeError):
                RLTrainingConfig.from_json(path)


class PoolTests(unittest.TestCase):
    def test_file_mode_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            theorem_file = tmp / "theorems.jsonl"
            rows = [
                {"goal": "p → p", "hypotheses": ["p : Prop"], "tactic": "intro"},
                {"goal": "q ∨ p", "hypotheses": ["p : Prop", "q : Prop", "h : p ∨ q"]},
                {"goal": "x" * 500, "hypotheses": []},  # over max_state_chars → dropped
            ]
            with open(theorem_file, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            cfg = _write_config(
                tmp, data_source="file", theorem_file=theorem_file,
                max_state_chars=400, eval_pool_size=0,
            )
            pool = build_theorem_pool(cfg)
            total = len(pool.train_items) + len(pool.eval_items)
            self.assertEqual(total, 2)  # oversized row dropped
            self.assertEqual(pool.train_items[0].goal.expression, "p → p")  # size-sorted
            self.assertEqual(pool.train_items[0].tactic_label, "intro")

    def test_metavariable_rows_are_dropped_from_the_pool(self):
        """Unassigned holes (?m.4519) cannot be elaborated by goal_start, so a
        pool row carrying one would fail before its first tactic. It must count
        as dropped, never as a rollout root."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            theorem_file = tmp / "theorems.jsonl"
            rows = [
                {"goal": "p → p", "hypotheses": ["p : Prop"], "tactic": "intro"},
                {"goal": "?m.4519 = ?m.4519", "hypotheses": []},  # hole in the target
                {"goal": "p → p", "hypotheses": ["h : ?m.2235"], "tactic": "intro"},  # hole in a hyp
                {"goal": "q ∨ p", "hypotheses": ["p : Prop", "q : Prop", "h : p ∨ q"]},
            ]
            with open(theorem_file, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            cfg = _write_config(
                tmp, data_source="file", theorem_file=theorem_file,
                max_state_chars=400, eval_pool_size=0,
            )
            pool = build_theorem_pool(cfg)
            total = len(pool.train_items) + len(pool.eval_items)
            self.assertEqual(total, 2)  # both metavariable rows dropped
            self.assertEqual(pool.train_items[0].goal.expression, "p → p")  # size-sorted

    def test_curriculum_window_and_growth(self):
        pool = _pool(n_items=10)
        pool.curriculum_size = 4
        batch = pool.sample_batch(3)
        self.assertEqual(len(batch), 3)
        window = pool.train_items[:4]
        for item in batch:
            self.assertIn(item, window)
        pool.grow(2.0)
        self.assertEqual(pool.curriculum_size, 8)
        pool.grow(10.0)  # capped at the pool size
        self.assertEqual(pool.curriculum_size, len(pool.train_items))


class BCAnnealTests(unittest.TestCase):
    def test_anneal_endpoints_and_monotonicity(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _write_config(Path(tmp), bc_anneal_start=0.5, bc_anneal_end=0.1, bc_anneal_rounds=10)
        self.assertAlmostEqual(bc_weight_at_round(0, cfg), 0.5)
        self.assertAlmostEqual(bc_weight_at_round(10, cfg), 0.1)
        self.assertAlmostEqual(bc_weight_at_round(100, cfg), 0.1)
        weights = [bc_weight_at_round(i, cfg) for i in range(11)]
        self.assertTrue(all(a >= b for a, b in zip(weights, weights[1:])))


class RoundLoopTests(unittest.TestCase):
    def test_happy_path_two_rounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, num_rounds=2)
            torch.manual_seed(0)
            metrics = asyncio.run(
                run_rl_training(cfg, reasoner_factory=_qed_factory, pool=_pool())
            )
            self.assertEqual(metrics["round"], 1)
            run_dirs = list((tmp / "runs").iterdir())
            self.assertEqual(len(run_dirs), 1)
            run_dir = run_dirs[0]
            self.assertTrue((run_dir / "last.pt").exists())
            self.assertTrue((run_dir / "config.json").exists())
            rows = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertGreater(rows[0]["num_transitions"] + rows[0]["num_failures"], 0)

    def test_params_change_after_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, num_rounds=1)
            node_vocab = _build_node_vocab()
            torch.manual_seed(0)
            before = _make_model(node_vocab)
            before.load_state_dict(
                torch.load(cfg.warmstart_checkpoint, weights_only=False)["model_state_dict"]
            )
            torch.manual_seed(0)
            asyncio.run(run_rl_training(cfg, reasoner_factory=_qed_factory, pool=_pool()))
            run_dir = next((tmp / "runs").iterdir())
            after_sd = torch.load(run_dir / "last.pt", weights_only=False)["model_state_dict"]
            changed = any(
                not torch.equal(before.state_dict()[k], after_sd[k]) for k in after_sd
            )
            self.assertTrue(changed, "one training round did not update any parameters")

    def test_per_theorem_fault_isolation(self):
        node_vocab = _build_node_vocab()
        torch.manual_seed(0)
        model = _make_model(node_vocab)
        inner = _make_reasoner(model, node_vocab, _QEDExecutor(), top_k=2)
        reasoner = _RaisingReasoner(inner, raise_on={1})  # second theorem dies
        results, stats = asyncio.run(
            collect_round(reasoner, _items(3), timeout_s=10.0)
        )
        self.assertEqual(stats["attempted"], 3.0)
        self.assertEqual(stats["collected"], 2.0)
        self.assertEqual(stats["searches_failed"], 1.0)
        self.assertEqual(len(results), 2)

    def test_resume_continues_round_counter(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, num_rounds=2)
            torch.manual_seed(0)
            asyncio.run(run_rl_training(cfg, reasoner_factory=_qed_factory, pool=_pool()))
            run_dir = next((tmp / "runs").iterdir())

            cfg.num_rounds = 3
            torch.manual_seed(0)
            asyncio.run(
                run_rl_training(
                    cfg, resume_run_dir=run_dir, reasoner_factory=_qed_factory, pool=_pool()
                )
            )
            rows = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
            train_rows = [r for r in rows if "num_transitions" in r]
            self.assertEqual([r["round"] for r in train_rows], [0, 1, 2])
            state = torch.load(run_dir / "last.pt", weights_only=False)
            self.assertEqual(state["round"], 2)


class EvalTests(unittest.TestCase):
    def test_greedy_eval_deterministic(self):
        node_vocab = _build_node_vocab()
        torch.manual_seed(0)
        model = _make_model(node_vocab)
        reasoner = _make_reasoner(model, node_vocab, _QEDExecutor(), top_k=2)
        items = _items(4)
        s1 = asyncio.run(evaluate_proof_rate(reasoner, items, timeout_s=10.0))
        s2 = asyncio.run(evaluate_proof_rate(reasoner, items, timeout_s=10.0))
        self.assertEqual(s1, s2)
        self.assertEqual(s1["attempted"], 4.0)

    def test_best_checkpoint_written_on_improvement(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, num_rounds=1, eval_every=1)
            pool = TheoremPool(_items(20), eval_pool_size=2, curriculum_size=4, seed=0)
            self.assertGreater(len(pool.eval_items), 0)
            torch.manual_seed(0)
            asyncio.run(run_rl_training(cfg, reasoner_factory=_qed_factory, pool=pool))
            run_dir = next((tmp / "runs").iterdir())
            # QED executor solves everything ⇒ proof rate 1.0 > initial -1 ⇒ best.pt written.
            self.assertTrue((run_dir / "best.pt").exists())

    def test_reject_run_writes_no_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, num_rounds=1, eval_every=1)
            pool = TheoremPool(_items(20), eval_pool_size=2, curriculum_size=4, seed=0)
            torch.manual_seed(0)
            asyncio.run(run_rl_training(cfg, reasoner_factory=_reject_factory, pool=pool))
            run_dir = next((tmp / "runs").iterdir())
            # Reject executor proves nothing ⇒ proof rate 0.0 > -1 initial: best.pt IS
            # written once (first eval), but records rate 0.
            state = torch.load(run_dir / "best.pt", weights_only=False)
            self.assertEqual(state["best_proof_rate"], 0.0)


class DeadRoundTests(unittest.TestCase):
    """Issue 4: rounds that collect no transitions (every search raised — a
    broken environment) must not weaken the BC anchor, and enough of them in a
    row means the run is looping on a misconfiguration, not slowly improving."""

    def test_all_dead_rounds_keep_the_anchor_fully_weighted(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, num_rounds=2, bc_anneal_rounds=10)
            torch.manual_seed(0)
            asyncio.run(run_rl_training(cfg, reasoner_factory=_dead_server_factory(), pool=_pool()))
            run_dir = next((tmp / "runs").iterdir())
            rows = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([r["bc_weight"] for r in rows], [0.5, 0.5])
            self.assertEqual([r["anneal_rounds_done"] for r in rows], [0, 0])

    def test_unknown_only_results_do_not_advance_the_anneal(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, num_rounds=2, bc_anneal_rounds=10)
            torch.manual_seed(0)
            asyncio.run(
                run_rl_training(
                    cfg,
                    reasoner_factory=_unelaborated_factory,
                    pool=_pool(),
                )
            )
            run_dir = next((tmp / "runs").iterdir())
            rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([row["collected"] for row in rows], [2.0, 2.0])
            self.assertEqual([row["optimizer_step"] for row in rows], [0.0, 0.0])
            self.assertEqual([row["anneal_rounds_done"] for row in rows], [0, 0])

    def test_gradient_rounds_advance_the_anneal(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, num_rounds=2, bc_anneal_rounds=10)
            torch.manual_seed(0)
            asyncio.run(run_rl_training(cfg, reasoner_factory=_qed_factory, pool=_pool()))
            run_dir = next((tmp / "runs").iterdir())
            rows = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([r["anneal_rounds_done"] for r in rows], [1, 2])
            self.assertLess(rows[1]["bc_weight"], rows[0]["bc_weight"])

    def test_consecutive_dead_rounds_halt_with_a_named_error(self):
        """max_dead_rounds reached ⇒ RuntimeError naming the environment, not a
        silent forever-loop of zero-transition rounds."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, num_rounds=50, max_dead_rounds=3)
            with self.assertRaisesRegex(RuntimeError, "source-root"):
                asyncio.run(run_rl_training(cfg, reasoner_factory=_dead_server_factory(), pool=_pool()))

    def test_interleaved_gradient_round_resets_the_dead_counter(self):
        """One successful round between dead ones must restart the countdown."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, num_rounds=6, max_dead_rounds=3)
            torch.manual_seed(0)
            # Rounds 0-1 dead, round 2 gradient (counter resets), rounds 3-4 dead,
            # round 5 gradient — never three in a row, so the run completes.
            factory = _dead_server_factory(succeed_on={4, 5, 10, 11})
            asyncio.run(run_rl_training(cfg, reasoner_factory=factory, pool=_pool()))
            run_dir = next((tmp / "runs").iterdir())
            rows = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
            # Which rounds took a gradient step, not how many transitions each
            # harvested — the count depends on the sampler, the gating does not.
            self.assertEqual(
                [r["num_transitions"] > 0 for r in rows],
                [False, False, True, False, False, True],
            )
            self.assertEqual([r["anneal_rounds_done"] for r in rows], [0, 0, 1, 1, 1, 2])

    def test_round_line_reports_both_failure_counts(self):
        """Issue 3: the round line must print both `rej` (tactics refused inside
        a search that ran) and `err` (whole searches that raised or timed out).
        The old line printed only the former, rendering a broken environment as
        `fail 0` while eight theorems silently failed."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, num_rounds=1, bc_anneal_rounds=10)
            torch.manual_seed(0)
            asyncio.run(run_rl_training(cfg, reasoner_factory=_reject_factory, pool=_pool()))
            run_dir = next((tmp / "runs").iterdir())
            rows = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
            # `num_failures` counts rejections inside searches; `searches_failed`
            # counts whole searches that raised. Both must be present in metrics.
            self.assertIn("num_failures", rows[0])
            self.assertIn("searches_failed", rows[0])
            self.assertEqual(rows[0]["searches_failed"], 0.0)


class PantographEnvResolverTests(unittest.TestCase):
    """Issue 1: the config's Lean-environment fields resolve to one value the
    initial server and every post-crash restart both use."""

    def test_no_source_root_gives_core_lean_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _write_config(Path(tmp))
            env = pantograph_env(cfg)
            self.assertIsNone(env.source_root)
            self.assertEqual(env.imports, ("Init",))

    def test_source_root_adds_mathlib_to_the_import_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, source_root=tmp / "mathlib")
            env = pantograph_env(cfg)
            self.assertEqual(env.source_root, tmp / "mathlib")
            self.assertEqual(env.imports, ("Init", "Mathlib"))

    def test_explicit_imports_override_the_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(
                tmp, source_root=tmp / "proj", pantograph_imports=["Init", "MyProject"]
            )
            self.assertEqual(pantograph_env(cfg).imports, ("Init", "MyProject"))

    def test_repl_and_timeout_are_carried_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, pantograph_repl=tmp / "repl", server_timeout_s=300)
            env = pantograph_env(cfg)
            self.assertEqual(env.pantograph_repl, tmp / "repl")
            self.assertEqual(env.timeout, 300)

    def test_env_fields_survive_config_roundtrip_as_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, source_root=tmp / "mathlib", pantograph_repl=tmp / "repl")
            path = tmp / "cfg.json"
            with open(path, "w") as f:
                json.dump(cfg.to_dict(), f)
            loaded = RLTrainingConfig.from_json(path)
            self.assertIsInstance(loaded.source_root, Path)
            self.assertIsInstance(loaded.pantograph_repl, Path)
            self.assertEqual(loaded.source_root, tmp / "mathlib")


class PLNKillSwitchConfigTests(unittest.TestCase):
    """Tests for use_pln threading through training and standalone evaluation."""

    def test_use_pln_false_survives_roundtrip(self):
        """use_pln=False survives to_dict / from_json without being reset."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _write_config(Path(tmp), use_pln=False)
            path = Path(tmp) / "cfg.json"
            with open(path, "w") as f:
                json.dump(cfg.to_dict(), f)
            loaded = RLTrainingConfig.from_json(path)
            self.assertFalse(loaded.use_pln)

    def test_use_pln_false_reaches_factory(self):
        """use_pln=False is forwarded to the reasoner factory via cfg."""
        received: list[bool] = []

        def _recording_factory(model, node_vocab, tactic_vocab, cfg):
            received.append(cfg.use_pln)
            return _make_reasoner(model, node_vocab, _QEDExecutor(), top_k=cfg.top_k_tactics)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, use_pln=False, num_rounds=1)
            torch.manual_seed(0)
            asyncio.run(run_rl_training(cfg, reasoner_factory=_recording_factory, pool=_pool()))
        self.assertEqual(received, [False])

    def test_eval_only_does_not_construct_pln(self):
        """The standalone evaluation path uses the same PLN-disabled reasoner config."""
        observed: list[tuple[bool, object, object]] = []

        class _EvalEnv:
            def verify(self):
                return None

            def describe(self):
                return "test environment"

            async def create_server(self):
                return _FakeServer()

        async def _record_reasoner(reasoner, items, *, timeout_s):
            observed.append(
                (reasoner.use_pln, reasoner.petta_chainer, reasoner.dts_sampler)
            )
            return {
                "proof_rate": 0.0,
                "solved": 0.0,
                "attempted": float(len(items)),
                "searches_failed": 0.0,
            }

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, use_pln=False, eval_pool_size=1)
            config_path = tmp / "config.json"
            with open(config_path, "w") as f:
                json.dump(cfg.to_dict(), f)

            with (
                patch(
                    "maths_ai.hybrid_reasoner.joint_inference.PLNInference",
                    side_effect=AssertionError("PLNInference must not be constructed"),
                ),
                patch(
                    "maths_ai.gnn_inference.atp_lean_gnn.rl_training_driver.pantograph_env",
                    return_value=_EvalEnv(),
                ),
                patch(
                    "maths_ai.gnn_inference.atp_lean_gnn.rl_training_driver.build_theorem_pool",
                    return_value=_pool(n_items=2, eval_size=1),
                ),
                patch(
                    "maths_ai.gnn_inference.atp_lean_gnn.rl_training_driver.evaluate_proof_rate",
                    new=_record_reasoner,
                ),
            ):
                self.assertEqual(
                    driver_main(["--config", str(config_path), "--eval-only"]),
                    0,
                )

        self.assertEqual(observed, [(False, None, None)])


if __name__ == "__main__":
    unittest.main()
