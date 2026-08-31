from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.optim import AdamW

from maths_ai.data_models.proof_components import Goal, LeanGoalSeed, STV
from maths_ai.hybrid_reasoner.hypergraph import TacticOutcome
from maths_ai.pln_inference.model import PLNResult

from maths_ai.gnn_inference.atp_lean_gnn.checkpointing import checkpoint_payload
from maths_ai.gnn_inference.atp_lean_gnn.graph import model_goal_to_dag
from maths_ai.gnn_inference.atp_lean_gnn.graph_contract import MODEL_SEXPR_GRAPH_SPEC
from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_vocab
from maths_ai.gnn_inference.atp_lean_gnn.rl_reasoner import RLHybridReasoner
from maths_ai.gnn_inference.atp_lean_gnn.rl_training_driver import (
    RLTrainingConfig,
    TheoremItem,
    TheoremPool,
    bc_weight_at_round,
    build_theorem_pool,
    collect_round,
    evaluate_proof_rate,
    run_rl_training,
    save_checkpoint,
)
from maths_ai.gnn_inference.tests.model_helpers import (
    actor_critic,
    pantograph_goal,
    structured_goal,
)


# ---------------------------------------------------------------------------
# Fakes (mirroring test_rl_reasoner.py — no Lean/petta)
# ---------------------------------------------------------------------------


class _FakeGoalState:
    def __init__(self, goals=None):
        self.goals = list(goals or [])


class _FakeServer:
    async def goal_start_async(self, expression):
        return _FakeGoalState()

    async def goal_tactic_async(self, state, tactic):
        return _FakeGoalState([pantograph_goal(structured_goal(GOAL_EXPR, HYPS))])


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

    async def prove(self, goal, *, hypotheses=None, greedy=False, deadline=None):
        idx = self._calls
        self._calls += 1
        if idx in self._raise_on:
            raise RuntimeError("simulated Lean transport failure")
        return await self._inner.prove(goal, hypotheses=hypotheses, greedy=greedy, deadline=deadline)


TACTIC_VOCAB = {"trivial": 0, "intro": 1, "exact": 2, "<UNK_TACTIC>": 3}
GOAL_EXPR = "p → p"
HYPS = ["p : Prop"]


def _build_node_vocab():
    return build_vocab([model_goal_to_dag(structured_goal(GOAL_EXPR, HYPS))])


def _make_model(node_vocab):
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
        TheoremItem(
            goal=LeanGoalSeed(expression=GOAL_EXPR, hypotheses=HYPS),
            tactic_label="intro",
            size=10 + i,
        )
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
    metadata_dir = tmp / "prepared" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with open(metadata_dir / "graph_representation.json", "w") as f:
        json.dump(MODEL_SEXPR_GRAPH_SPEC.to_dict(), f)
    manifests_dir = tmp / "prepared" / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        pyg_dir = tmp / "prepared" / "pyg" / split
        pyg_dir.mkdir(parents=True, exist_ok=True)
        with open(manifests_dir / f"{split}.json", "w") as f:
            json.dump({"artifact_paths": {"pyg_dir": f"pyg/{split}"}}, f)

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
            graph_representation=MODEL_SEXPR_GRAPH_SPEC,
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

    def test_config_rejects_budget_under_legacy_policy(self):
        # __post_init__ runs resolve_search_params: an explicit simulation
        # budget under selection_policy="legacy" fails at config construction,
        # before any Lean server spins up.
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                _write_config(Path(tmp), num_simulations=8)

    def test_config_puct_requires_num_simulations(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                _write_config(Path(tmp), selection_policy="puct")
            cfg = _write_config(Path(tmp), selection_policy="puct", num_simulations=8)
            self.assertEqual(cfg.num_simulations, 8)


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


class HTPSDriverTests(unittest.TestCase):
    """Phase 2–3 driver wiring: queues, decoupled step, checkpoint round-trip."""

    def _metric_rows(self, run_dir: Path) -> list[dict]:
        rows = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
        return [r for r in rows if "num_transitions" in r]

    def test_default_off_runs_no_decoupled_step(self):
        # htps_steps_per_round defaults to 0: mining still fills the queues
        # (pure bookkeeping, no gradient), but no decoupled loss is ever logged.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, num_rounds=1)
            self.assertEqual(cfg.htps_steps_per_round, 0)
            torch.manual_seed(0)
            asyncio.run(run_rl_training(cfg, reasoner_factory=_qed_factory, pool=_pool()))
            run_dir = next((tmp / "runs").iterdir())
            for row in self._metric_rows(run_dir):
                self.assertNotIn("tactic_imitation_loss", row)
                self.assertNotIn("critic_soft_loss", row)

    def test_enabled_step_logs_losses_and_queues_grow(self):
        # QED executor ⇒ every root is SOLVED: the critic queue gets hard 1.0
        # labels and the tactic queue gets the minimal hypertree's QED edges.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, num_rounds=1, htps_steps_per_round=2, htps_batch_size=4)
            torch.manual_seed(0)
            asyncio.run(run_rl_training(cfg, reasoner_factory=_qed_factory, pool=_pool()))
            run_dir = next((tmp / "runs").iterdir())
            rows = self._metric_rows(run_dir)
            self.assertGreater(rows[-1]["tactic_queue_len"], 0)
            self.assertGreater(rows[-1]["critic_queue_len"], 0)
            self.assertGreater(rows[-1]["imitation_samples_mined"], 0)
            self.assertIn("tactic_imitation_loss", rows[-1])
            self.assertIn("critic_soft_loss", rows[-1])

    def test_checkpoint_roundtrips_optimizer_htps_and_queues(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, num_rounds=1, htps_steps_per_round=1, htps_batch_size=4)
            torch.manual_seed(0)
            asyncio.run(run_rl_training(cfg, reasoner_factory=_qed_factory, pool=_pool()))
            run_dir = next((tmp / "runs").iterdir())
            state = torch.load(run_dir / "last.pt", weights_only=False)
            # The decoupled step ran ⇒ its optimizer carries Adam moments.
            self.assertIn("optimizer_htps_state_dict", state)
            self.assertGreater(len(state["optimizer_htps_state_dict"]["state"]), 0)
            # Queues serialize canonical goals and graph fingerprints as plain dicts.
            self.assertGreater(len(state["tactic_queue"]), 0)
            tactic_row = state["tactic_queue"][0]
            self.assertIsInstance(tactic_row["goal"], dict)
            self.assertTrue(tactic_row["graph_fingerprint"])
            self.assertIsInstance(tactic_row["tactic_id"], int)
            self.assertIsInstance(tactic_row["arg_indices"], list)
            self.assertGreater(len(state["critic_queue"]), 0)
            critic_row = state["critic_queue"][0]
            self.assertIsInstance(critic_row["goal"], dict)
            self.assertTrue(critic_row["graph_fingerprint"])
            self.assertEqual(critic_row["target"], 1.0)
            self.assertEqual(critic_row["source"], "lean_status")

            # Resume restores the queues: with num_rounds == start_round the loop
            # body never runs, so the final checkpoint's queues are exactly the
            # restored ones.
            cfg.num_rounds = int(state["round"]) + 1
            torch.manual_seed(0)
            asyncio.run(run_rl_training(
                cfg, resume_run_dir=run_dir, reasoner_factory=_qed_factory, pool=_pool()
            ))
            state2 = torch.load(run_dir / "last.pt", weights_only=False)
            self.assertEqual(state2["tactic_queue"], state["tactic_queue"])
            self.assertEqual(state2["critic_queue"], state["critic_queue"])

    def test_pre_validity_checkpoint_resume_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, num_rounds=1)
            torch.manual_seed(0)
            asyncio.run(run_rl_training(cfg, reasoner_factory=_qed_factory, pool=_pool()))
            run_dir = next((tmp / "runs").iterdir())
            state = torch.load(run_dir / "last.pt", weights_only=False)
            for key in (
                "rl_target_schema_version",
                "optimizer_htps_state_dict",
                "tactic_queue",
                "critic_queue",
            ):
                state.pop(key, None)  # strip back to the pre-HTPS format
            torch.save(state, run_dir / "last.pt")

            cfg.num_rounds = 2
            torch.manual_seed(0)
            with self.assertRaisesRegex(ValueError, "canonical model-S-expression"):
                asyncio.run(run_rl_training(
                    cfg, resume_run_dir=run_dir, reasoner_factory=_qed_factory, pool=_pool()
                ))


class PLNKillSwitchDriverTests(unittest.TestCase):
    """use_pln=False: config roundtrip and threading to the reasoner factory."""

    def test_use_pln_false_survives_config_roundtrip(self):
        # use_pln=False must survive to_dict() → from_json() intact.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, use_pln=False)
            self.assertFalse(cfg.use_pln)
            path = tmp / "cfg_pln.json"
            with open(path, "w") as f:
                json.dump(cfg.to_dict(), f)
            loaded = RLTrainingConfig.from_json(path)
            self.assertFalse(loaded.use_pln)

    def test_use_pln_false_reaches_reasoner_factory(self):
        # When use_pln=False is set in config, the value must reach the factory.
        # Inject a factory that records and asserts the flag, returning a mock
        # reasoner so no live Lean server is needed.
        received: list[bool] = []

        def _recording_factory(model, node_vocab, tactic_vocab, cfg):
            received.append(cfg.use_pln)
            return _make_reasoner(model, node_vocab, _QEDExecutor(), top_k=cfg.top_k_tactics)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, use_pln=False, num_rounds=1)
            torch.manual_seed(0)
            asyncio.run(run_rl_training(cfg, reasoner_factory=_recording_factory, pool=_pool()))
        self.assertEqual(len(received), 1)
        self.assertFalse(received[0])


if __name__ == "__main__":
    unittest.main()
