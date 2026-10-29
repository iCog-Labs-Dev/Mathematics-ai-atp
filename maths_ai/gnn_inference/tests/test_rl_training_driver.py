from __future__ import annotations
from maths_ai.gnn_inference.atp_lean_gnn.retrieval_artifacts import RetrievalConfig
from maths_ai.gnn_inference.tests.model_helpers import LocalCandidateProvider

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from pantograph_async import (
    ExecutionGoal,
    ExecutionLocal,
    PantographEnv,
    PantographGoalRejected,
    PantographProtocolError,
    PantographTransportError,
    WireExpression,
)
from maths_ai.gnn_inference.tests.model_helpers import ScopedFakeClient
from torch.optim import AdamW

from maths_ai.data_models.proof_components import STV
from maths_ai.lean_replay import (
    ReplayFailure,
    ReplayInputError,
    render_replay,
    seed_from_text,
)
from maths_ai.hybrid_reasoner.retained_executor import RetainedTacticExecutor
from maths_ai.pln_inference.model import PLNResult

from maths_ai.gnn_inference.atp_lean_gnn.actor_critic import (
    ActorCriticWithArgsClassifier,
)
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
    driver_main,
    evaluate_proof_rate,
    pantograph_env,
    run_rl_training,
    save_checkpoint,
)
from maths_ai.gnn_inference.atp_lean_gnn.dataset import DatasetRow
from maths_ai.gnn_inference.tests.model_helpers import (
    ScriptedExecutor,
    pantograph_goal,
    structured_goal,
)


# ---------------------------------------------------------------------------
# Fakes (mirroring test_rl_reasoner.py — no Lean/petta)
# ---------------------------------------------------------------------------


class _FakeGoalState:
    def __init__(self):
        self.goals = [pantograph_goal(structured_goal(GOAL_EXPR, HYPS))]


class _FakeServer(ScopedFakeClient):
    def __init__(self):
        pass

    async def _start_goal(self, expression):
        return _FakeGoalState()

    async def _apply_tactic(self, state, tactic):
        return _FakeGoalState()


def _QEDExecutor():
    return ScriptedExecutor(lambda _occurrence, _command: [], server=_FakeServer())


def _RejectExecutor():
    return ScriptedExecutor(lambda _occurrence, _command: None, server=_FakeServer())


class _UnelaboratedServer(_FakeServer):
    def __init__(self):
        pass

    async def _start_goal(self, expression):
        raise PantographGoalRejected("cannot elaborate reconstructed goal")


class _UnelaboratedExecutor(RetainedTacticExecutor):
    def __init__(self):
        self.server = _UnelaboratedServer()

    async def apply(self, registry, occurrence, command, *, holder="operation"):
        raise AssertionError("tactics must not run when elaboration fails")


class _StubPLN:
    async def evaluate_async(self, expression, hypotheses=None, **kwargs):
        return PLNResult(
            stv=STV(strength=0.1, confidence=1.0), status="ok", is_fallback=False
        )


class _RaisingReasoner:
    """prove() raises on a chosen call index — tests per-theorem fault isolation."""

    def __init__(self, inner, raise_on: set[int]):
        self._inner = inner
        self._raise_on = raise_on
        self._calls = 0
        self.model = inner.model
        self.dag_featurize_data = inner.dag_featurize_data

    async def prove(self, seed, *, greedy=False):
        idx = self._calls
        self._calls += 1
        if idx in self._raise_on:
            raise RuntimeError("simulated Lean transport failure")
        return await self._inner.prove(seed, greedy=greedy)


class _DeadServerReasoner:
    """prove() raises on every call except a listed set of call indices.

    Reproduces the failure the log snippet recorded: `_start_goal` raised for
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

    async def prove(self, seed, *, greedy=False):
        idx = self._calls
        self._calls += 1
        if idx not in self._succeed_on:
            raise RuntimeError("Unknown identifier ℕ")  # what a core-only REPL says
        return await self._inner.prove(seed, greedy=greedy)


TACTIC_VOCAB = {"trivial": 0, "intro": 1, "exact": 2, "<UNK_TACTIC>": 3}
GOAL_EXPR = "p → p"
HYPS = ["p : Prop"]


def _build_node_vocab():
    goal = structured_goal(GOAL_EXPR, HYPS)
    return build_vocab([model_goal_to_dag(goal)])


def _make_model(node_vocab):
    from maths_ai.gnn_inference.tests.model_helpers import actor_critic

    return actor_critic(len(node_vocab), len(TACTIC_VOCAB))


def _make_reasoner(model, node_vocab, executor, *, top_k=3):
    reasoner = RLHybridReasoner(
        model,
        node_vocab,
        TACTIC_VOCAB,
        executor=executor,
        candidate_provider=LocalCandidateProvider(),
        top_k_tactics=top_k,
        max_depth=3,
        max_nodes=20,
    )
    reasoner.petta_chainer = _StubPLN()
    return reasoner


def _items(n: int) -> list[TheoremItem]:
    return [
        TheoremItem(
            seed=seed_from_text(goal=GOAL_EXPR, hypotheses=HYPS),
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
    for split in ("train", "val", "test"):
        pyg_dir = tmp / "prepared" / split / "pyg"
        pyg_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir = tmp / "prepared" / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        with open(manifest_dir / f"{split}.json", "w") as f:
            json.dump({"artifact_paths": {"pyg_dir": f"{split}/pyg"}}, f)

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
    return TheoremPool(
        _items(n_items), eval_pool_size=eval_size, curriculum_size=4, seed=0
    )


def _qed_factory(model, node_vocab, tactic_vocab, cfg):
    return _make_reasoner(model, node_vocab, _QEDExecutor(), top_k=cfg.top_k_tactics)


def _reject_factory(model, node_vocab, tactic_vocab, cfg):
    return _make_reasoner(model, node_vocab, _RejectExecutor(), top_k=cfg.top_k_tactics)


def _unelaborated_factory(model, node_vocab, tactic_vocab, cfg):
    return _make_reasoner(
        model, node_vocab, _UnelaboratedExecutor(), top_k=cfg.top_k_tactics
    )


def _dead_server_factory(succeed_on: set[int] | None = None):
    """Reasoner factory whose searches all raise, except on listed call indices.

    `theorems_per_round` prove calls make up one round, so call indices
    {2n, 2n+1} are round n when `theorems_per_round=2`.
    """

    def factory(model, node_vocab, tactic_vocab, cfg):
        inner = _make_reasoner(
            model, node_vocab, _QEDExecutor(), top_k=cfg.top_k_tactics
        )
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


def _wire(text: str) -> WireExpression:
    return WireExpression(pp=text, sexp=text, model_sexp="(:c Test)", model_sexp_version=1)


class _ReplayState:
    def __init__(self, goal: ExecutionGoal):
        self.goals = [goal]


class _ReplayClient(ScopedFakeClient):
    """Answers each replay with the goal ``respond(expression, tactic)`` builds.

    Expressions containing a ``rejected`` marker fail as Lean would on an
    unknown identifier.
    """

    def __init__(self, respond=None, rejected: tuple[str, ...] = ()):
        self.respond = respond or _intro_goal
        self.rejected = rejected
        self.started: list[str] = []

    async def _start_goal(self, expression):
        self.started.append(expression)
        if any(marker in expression for marker in self.rejected):
            raise PantographGoalRejected("unknown identifier 'span'")
        return expression

    async def _apply_tactic(self, state, tactic):
        return _ReplayState(self.respond(state, tactic))


def _intro_goal(expression: str, tactic: str) -> ExecutionGoal:
    """A goal whose locals are the names the intro pattern introduced."""
    names = tactic.split()[1:] if tactic.startswith("intro") else []
    return ExecutionGoal(
        goal_name="g",
        case_tag=None,
        is_conversion=False,
        fragment="tactic",
        target=_wire("(:c True)"),
        locals=tuple(
            ExecutionLocal(
                internal_name=f"_live.{index}",
                user_name=name,
                context_index=index,
                binder_role=":explicit",
                is_instance=False,
                is_let=False,
                is_inaccessible=False,
                type=_wire("(:sort 0)"),
            )
            for index, name in enumerate(names)
        ),
    )


def _factory(client):
    async def create():
        return client

    return create


def _env(tmp: Path) -> PantographEnv:
    return PantographEnv(source_root=tmp, pantograph_repl=tmp / "repl")


def _structured_row(index: int, *, let: bool = False) -> DatasetRow:
    context = [
        {
            "binder_role": ":let" if let else ":explicit",
            "context_index": 0,
            "internal_name": "_uniq.1",
            "is_instance": False,
            "is_let": let,
            "user_name": "p",
        }
    ]
    return DatasetRow(
        state="p : Prop\n⊢ p → p",
        theorem=f"demo{index}",
        tactic="intro",
        split="train",
        row_index=index,
        dataset_name="jajostrains/Mathlib-Normalized-Sexpr",
        raw_goal_sexp="(:forall h (:fv _uniq.1) (:fv _uniq.1))",
        raw_hyp_sexps=json.dumps([{"name": "p", "sexp": "(:sort 0)"}]),
        local_context=json.dumps(context),
        raw_schema_version=4,
    )


def _structured_goal(expression: str, tactic: str) -> ExecutionGoal:
    """What Lean prints after replaying ``_structured_row``."""
    return ExecutionGoal(
        goal_name="g",
        case_tag=None,
        is_conversion=False,
        fragment="tactic",
        target=_wire("(:forall h (:fv _live.1) (:fv _live.1))"),
        locals=(
            ExecutionLocal(
                internal_name="_live.1",
                user_name="p",
                context_index=0,
                binder_role=":explicit",
                is_instance=False,
                is_let=False,
                is_inaccessible=False,
                type=_wire("(:sort 0)"),
            ),
        ),
    )


def _write_theorems(tmp: Path, rows: list[dict]) -> Path:
    path = tmp / "theorems.jsonl"
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def _report(directory: Path) -> dict:
    (path,) = directory.iterdir()
    return json.loads(path.read_text())


class PoolTests(unittest.TestCase):
    @patch("maths_ai.gnn_inference.atp_lean_gnn.seed_admission.iter_dataset_rows")
    def test_dataset_rows_replay_from_their_raw_columns(self, mock_iter_dataset_rows):
        mock_iter_dataset_rows.return_value = iter(
            [_structured_row(0), _structured_row(1, let=True)]
        )
        client = _ReplayClient(_structured_goal)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(
                tmp,
                dataset_name="jajostrains/Mathlib-Normalized-Sexpr",
                max_pool_size=5,
                max_scan_rows=7,
                eval_pool_size=0,
            )
            pool = asyncio.run(
                build_theorem_pool(
                    cfg, _env(tmp), report_dir=tmp / "admission", client_factory=_factory(client)
                )
            )
            report = _report(tmp / "admission")
        self.assertEqual(len(pool.train_items), 1)
        self.assertEqual(pool.train_items[0].seed.source_dict["theorem"], "demo0")
        self.assertEqual(client.started, ["∀ (p : (Sort 0)), (∀ (h : @p), @p)"])
        mock_iter_dataset_rows.assert_called_once_with(
            dataset_name="jajostrains/Mathlib-Normalized-Sexpr",
            split="train",
            sample_limit=7,
            revision=cfg.dataset_revision,
        )
        # Schema 4 records no let values: the row is quarantined, not guessed.
        self.assertEqual(report["quarantined"], {"missing-context": 1})
        self.assertEqual((report["scanned"], report["admitted"]), (2, 1))
        self.assertEqual(
            report["coverage"]["size:<=200"], {"admitted": 1, "quarantined": 1}
        )

    def test_file_mode_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            theorem_file = _write_theorems(
                tmp,
                [
                    {"goal": "p → p", "hypotheses": ["p : Prop"], "tactic": "intro"},
                    {"goal": "q ∨ p", "hypotheses": ["p q : Prop", "h : p ∨ q"]},
                    {"goal": "x" * 500, "hypotheses": []},  # over max_state_chars
                ],
            )
            cfg = _write_config(
                tmp,
                data_source="file",
                theorem_file=theorem_file,
                max_state_chars=400,
                eval_pool_size=0,
            )
            pool = asyncio.run(build_theorem_pool(cfg, _env(tmp), client_factory=_factory(_ReplayClient())))
            total = len(pool.train_items) + len(pool.eval_items)
            self.assertEqual(total, 2)  # oversized row skipped
            self.assertEqual(pool.train_items[0].seed.target, "p → p")  # size-sorted
            self.assertEqual(pool.train_items[0].tactic_label, "intro")
            # Grouped names are separate locals in the replay.
            self.assertEqual(
                [local.name for local in pool.train_items[1].seed.locals], ["p", "q", "h"]
            )

    def test_unreconstructable_rows_are_quarantined_by_reason(self):
        """Holes, universe metavariables and Lean rejections never become roots,
        and each is reported under its own reason rather than as `dropped`."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            theorem_file = _write_theorems(
                tmp,
                [
                    {"goal": "p → p", "hypotheses": ["p : Prop"], "tactic": "intro"},
                    {"goal": "?m.4519 = ?m.4519", "hypotheses": []},
                    {"goal": "p → p", "hypotheses": ["h : ?m.2235"]},
                    {"goal": "x = x", "hypotheses": ["α : Type ?u.319125", "x : α"]},
                    {"goal": "span R s = N", "hypotheses": []},
                    {"goal": "q ∨ p", "hypotheses": ["p : Prop", "q : Prop", "h : p ∨ q"]},
                ],
            )
            cfg = _write_config(
                tmp, data_source="file", theorem_file=theorem_file, eval_pool_size=0
            )
            pool = asyncio.run(
                build_theorem_pool(
                    cfg,
                    _env(tmp),
                    report_dir=tmp / "admission",
                    client_factory=_factory(_ReplayClient(rejected=("span",))),
                )
            )
            report = _report(tmp / "admission")
        self.assertEqual(len(pool.train_items), 2)
        self.assertEqual(
            report["quarantined"],
            {"lean-rejected": 1, "unresolved-term": 2, "unresolved-universe": 1},
        )
        self.assertEqual(report["candidates"], 3)

    def test_universe_metavariables_are_not_erased(self):
        """`Type ?u.319125` has no assignment to replay; turning it into `Type`
        could change the statement, so the seed is rejected instead."""
        seed = seed_from_text(goal="x = x", hypotheses=["α : Type ?u.319125", "x : α"])
        with self.assertRaises(ReplayInputError) as error:
            render_replay(seed)
        self.assertIs(error.exception.reason, ReplayFailure.UNRESOLVED_UNIVERSE)

    def test_a_replay_that_loses_the_client_is_skipped_on_a_fresh_one(self):
        """A heavy row whose replay kills the session (the REPL printing an
        uncaught heartbeat error) is skipped and a fresh client continues."""

        class _DyingClient(_ReplayClient):
            async def _start_goal(self, expression):
                if "boom" in expression:
                    self.is_usable = False
                    raise PantographProtocolError("Unsolicited REPL stdout")
                return await super()._start_goal(expression)

        created = []

        async def factory():
            created.append(_DyingClient())
            return created[-1]

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rows = [{"goal": "boom", "hypotheses": []}, {"goal": "p → p", "hypotheses": ["p : Prop"]}]
            cfg = _write_config(
                tmp, data_source="file", theorem_file=_write_theorems(tmp, rows), eval_pool_size=0
            )
            pool = asyncio.run(
                build_theorem_pool(
                    cfg, _env(tmp), report_dir=tmp / "admission", client_factory=factory
                )
            )
            report = _report(tmp / "admission")
            self.assertEqual(len(pool.train_items), 1)
            self.assertEqual(report["skipped"], {"replay-repl-failure": 1})
            self.assertEqual(report["quarantined"], {})
            self.assertEqual(len(created), 2)

            # Consecutive heavy rows of one proof are each skipped, not fatal.
            many = tmp / "many"
            many.mkdir()
            rows = [{"goal": "boom", "hypotheses": []}] * 3 + [{"goal": "True", "hypotheses": []}]
            cfg = _write_config(
                many, data_source="file", theorem_file=_write_theorems(many, rows), eval_pool_size=0
            )
            pool = asyncio.run(build_theorem_pool(cfg, _env(many), client_factory=factory))
            self.assertEqual(len(pool.train_items), 1)

            # A client that cannot start ends admission.
            async def broken():
                raise PantographProtocolError("probe failed")

            with self.assertRaises(PantographProtocolError):
                asyncio.run(build_theorem_pool(cfg, _env(many), client_factory=broken))

    def test_oversized_replays_are_skipped_before_sending(self):
        client = _ReplayClient()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rows = [{"goal": "x = " + "x" * 300, "hypotheses": []}, {"goal": "True", "hypotheses": []}]
            cfg = _write_config(
                tmp,
                data_source="file",
                theorem_file=_write_theorems(tmp, rows),
                max_replay_chars=100,
                eval_pool_size=0,
            )
            pool = asyncio.run(
                build_theorem_pool(
                    cfg, _env(tmp), report_dir=tmp / "admission", client_factory=_factory(client)
                )
            )
            report = _report(tmp / "admission")
        self.assertEqual(len(pool.train_items), 1)
        self.assertEqual(report["skipped"], {"over-size-replay": 1})
        self.assertEqual(client.started, ["True"])

    def test_validation_cache_is_reused_only_for_the_same_identity(self):
        from maths_ai.gnn_inference.atp_lean_gnn.seed_admission import ValidationCache

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.json"
            identity = {"repl_sha256": "a", "client": {"commit": "c"}}
            cache = ValidationCache(path, identity)
            cache.add("fingerprint")
            cache.save()
            self.assertIn("fingerprint", ValidationCache(path, identity))
            changed = dict(identity, repl_sha256="b")
            self.assertNotIn("fingerprint", ValidationCache(path, changed))

    def test_cached_seeds_are_admitted_without_a_replay(self):
        client = _ReplayClient()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rows = [{"goal": "p → p", "hypotheses": ["p : Prop"]}]
            cfg = _write_config(
                tmp,
                data_source="file",
                theorem_file=_write_theorems(tmp, rows),
                seed_cache_path=tmp / "cache.json",
                eval_pool_size=0,
            )
            for _ in range(2):
                asyncio.run(
                    build_theorem_pool(
                        cfg, _env(tmp), report_dir=tmp / "admission", client_factory=_factory(client)
                    )
                )
            reports = [
                json.loads(path.read_text())
                for path in sorted((tmp / "admission").iterdir())
            ]
        # Two runs wrote two reports; the second reused the validation.
        self.assertEqual(len(reports), 2)
        self.assertEqual([r["cache_hits"] for r in reports], [0, 1])
        self.assertEqual(len(client.started), 1)

    def test_no_validated_seed_is_an_error_not_an_empty_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            theorem_file = _write_theorems(tmp, [{"goal": "span R s = N", "hypotheses": []}])
            cfg = _write_config(tmp, data_source="file", theorem_file=theorem_file)
            with self.assertRaisesRegex(RuntimeError, "No seed passed Lean validation"):
                asyncio.run(
                    build_theorem_pool(
                        cfg, _env(tmp), client_factory=_factory(_ReplayClient(rejected=("span",)))
                    )
                )

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
            cfg = _write_config(
                Path(tmp), bc_anneal_start=0.5, bc_anneal_end=0.1, bc_anneal_rounds=10
            )
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
            rows = [
                json.loads(l)
                for l in (run_dir / "metrics.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(rows), 2)
            self.assertGreater(rows[0]["num_transitions"] + rows[0]["num_failures"], 0)
            self.assertEqual([row["model_generation"] for row in rows], [1.0, 2.0])
            self.assertEqual([row["replica_generation"] for row in rows], [1.0, 2.0])

    def test_params_change_after_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, num_rounds=1)
            node_vocab = _build_node_vocab()
            torch.manual_seed(0)
            before = _make_model(node_vocab)
            before.load_state_dict(
                torch.load(cfg.warmstart_checkpoint, weights_only=False)[
                    "model_state_dict"
                ]
            )
            torch.manual_seed(0)
            asyncio.run(
                run_rl_training(cfg, reasoner_factory=_qed_factory, pool=_pool())
            )
            run_dir = next((tmp / "runs").iterdir())
            after_sd = torch.load(run_dir / "last.pt", weights_only=False)[
                "model_state_dict"
            ]
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
        results, stats = asyncio.run(collect_round(reasoner, _items(3), timeout_s=10.0))
        self.assertEqual(stats["attempted"], 3.0)
        self.assertEqual(stats["collected"], 2.0)
        self.assertEqual(stats["searches_failed"], 1.0)
        self.assertEqual(len(results), 2)

    def test_resume_continues_round_counter(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, num_rounds=2)
            torch.manual_seed(0)
            asyncio.run(
                run_rl_training(cfg, reasoner_factory=_qed_factory, pool=_pool())
            )
            run_dir = next((tmp / "runs").iterdir())

            cfg.num_rounds = 3
            torch.manual_seed(0)
            asyncio.run(
                run_rl_training(
                    cfg,
                    resume_run_dir=run_dir,
                    reasoner_factory=_qed_factory,
                    pool=_pool(),
                )
            )
            rows = [
                json.loads(l)
                for l in (run_dir / "metrics.jsonl").read_text().splitlines()
            ]
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
            asyncio.run(
                run_rl_training(cfg, reasoner_factory=_reject_factory, pool=pool)
            )
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
            asyncio.run(
                run_rl_training(
                    cfg, reasoner_factory=_dead_server_factory(), pool=_pool()
                )
            )
            run_dir = next((tmp / "runs").iterdir())
            rows = [
                json.loads(l)
                for l in (run_dir / "metrics.jsonl").read_text().splitlines()
            ]
            self.assertEqual([r["bc_weight"] for r in rows], [0.5, 0.5])
            self.assertEqual([r["anneal_rounds_done"] for r in rows], [0, 0])
            self.assertEqual([r["model_generation"] for r in rows], [0.0, 0.0])

    def test_unelaborated_roots_do_not_advance_the_anneal(self):
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
            rows = [
                json.loads(line)
                for line in (run_dir / "metrics.jsonl").read_text().splitlines()
            ]
            self.assertEqual([row["collected"] for row in rows], [0.0, 0.0])
            # A root Lean cannot elaborate is an inadmissible seed, not a failed search.
            self.assertEqual([row["searches_failed"] for row in rows], [0.0, 0.0])
            self.assertEqual([row["seeds_inadmissible"] for row in rows], [2.0, 2.0])
            self.assertEqual([row["optimizer_step"] for row in rows], [0.0, 0.0])
            self.assertEqual([row["anneal_rounds_done"] for row in rows], [0, 0])

    def test_gradient_rounds_advance_the_anneal(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, num_rounds=2, bc_anneal_rounds=10)
            torch.manual_seed(0)
            asyncio.run(
                run_rl_training(cfg, reasoner_factory=_qed_factory, pool=_pool())
            )
            run_dir = next((tmp / "runs").iterdir())
            rows = [
                json.loads(l)
                for l in (run_dir / "metrics.jsonl").read_text().splitlines()
            ]
            self.assertEqual([r["anneal_rounds_done"] for r in rows], [1, 2])
            self.assertLess(rows[1]["bc_weight"], rows[0]["bc_weight"])

    def test_consecutive_dead_rounds_halt_with_a_named_error(self):
        """max_dead_rounds reached ⇒ RuntimeError naming the environment, not a
        silent forever-loop of zero-transition rounds."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, num_rounds=50, max_dead_rounds=3)
            with self.assertRaisesRegex(RuntimeError, "source-root"):
                asyncio.run(
                    run_rl_training(
                        cfg, reasoner_factory=_dead_server_factory(), pool=_pool()
                    )
                )

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
            rows = [
                json.loads(l)
                for l in (run_dir / "metrics.jsonl").read_text().splitlines()
            ]
            # Which rounds took a gradient step, not how many transitions each
            # harvested — the count depends on the sampler, the gating does not.
            self.assertEqual(
                [r["num_transitions"] > 0 for r in rows],
                [False, False, True, False, False, True],
            )
            self.assertEqual(
                [r["anneal_rounds_done"] for r in rows], [0, 0, 1, 1, 1, 2]
            )

    def test_round_line_reports_both_failure_counts(self):
        """Issue 3: the round line must print both `rej` (tactics refused inside
        a search that ran) and `err` (whole searches that raised or timed out).
        The old line printed only the former, rendering a broken environment as
        `fail 0` while eight theorems silently failed."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, num_rounds=1, bc_anneal_rounds=10)
            torch.manual_seed(0)
            asyncio.run(
                run_rl_training(cfg, reasoner_factory=_reject_factory, pool=_pool())
            )
            run_dir = next((tmp / "runs").iterdir())
            rows = [
                json.loads(l)
                for l in (run_dir / "metrics.jsonl").read_text().splitlines()
            ]
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
            cfg = _write_config(
                tmp, source_root=tmp / "mathlib", pantograph_repl=tmp / "repl"
            )
            path = tmp / "cfg.json"
            with open(path, "w") as f:
                json.dump(cfg.to_dict(), f)
            loaded = RLTrainingConfig.from_json(path)
            self.assertIsInstance(loaded.source_root, Path)
            self.assertIsInstance(loaded.pantograph_repl, Path)
            self.assertEqual(loaded.source_root, tmp / "mathlib")


def _async_return(value):
    async def call(*_args, **_kwargs):
        return value

    return call


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
            return _make_reasoner(
                model, node_vocab, _QEDExecutor(), top_k=cfg.top_k_tactics
            )

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, use_pln=False, num_rounds=1)
            torch.manual_seed(0)
            asyncio.run(
                run_rl_training(cfg, reasoner_factory=_recording_factory, pool=_pool())
            )
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

        async def _record_reasoner(
            reasoner,
            items,
            *,
            timeout_s,
            reasoner_pool,
            model_generation,
        ):
            self.assertIsNotNone(reasoner_pool)
            self.assertEqual(model_generation, 0)
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
            cfg.retrieval = RetrievalConfig(tmp,tmp,tmp,tmp)
            config_path = tmp / "config.json"
            with open(config_path, "w") as f:
                json.dump(cfg.to_dict(), f)

            provider = LocalCandidateProvider()
            provider.library = type("Library", (), {"fingerprint": ""})()
            async def close_provider():
                pass
            provider.aclose = close_provider
            provider.verify_environment = lambda env: None
            with (
                patch("maths_ai.gnn_inference.atp_lean_gnn.rl_training_driver.LiveLemmaProvider", return_value=provider),
                patch(
                    "maths_ai.hybrid_reasoner.joint_inference.PLNInference",
                    side_effect=AssertionError("PLNInference must not be constructed"),
                ),
                patch(
                    "maths_ai.gnn_inference.atp_lean_gnn.rl_training_driver.pantograph_env",
                    return_value=_EvalEnv(),
                ),
                patch(
                    "maths_ai.gnn_inference.atp_lean_gnn.rl_training_driver.create_model_sexpr_client",
                    new=lambda env: env.create_server(),
                ),
                patch(
                    "maths_ai.gnn_inference.atp_lean_gnn.rl_training_driver.build_theorem_pool",
                    new=_async_return(_pool(n_items=2, eval_size=1)),
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
