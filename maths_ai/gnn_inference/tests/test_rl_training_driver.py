from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.optim import AdamW

from maths_ai.gnn_inference.atp_lean_gnn.policy_actions import LocalCandidateProvider
from maths_ai.data_models.proof_components import Goal, STV
from maths_ai.lean_replay import (
    ReplayFailure,
    ReplayInputError,
    render_replay,
    seed_from_text,
)
from maths_ai.hybrid_reasoner.retained_executor import RetainedTacticExecutor
from maths_ai.pln_inference.model import PLNResult

from maths_ai.gnn_inference.atp_lean_gnn.checkpointing import checkpoint_payload
from maths_ai.gnn_inference.atp_lean_gnn.graph import model_goal_to_dag
from maths_ai.gnn_inference.atp_lean_gnn.graph_contract import MODEL_SEXPR_GRAPH_SPEC
from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_vocab
from maths_ai.gnn_inference.atp_lean_gnn.rl_reasoner import RLHybridReasoner
from maths_ai.gnn_inference.atp_lean_gnn.dataset import DatasetRow
from maths_ai.gnn_inference.atp_lean_gnn.rl_training_driver import (
    EvaluationSettings,
    RLTrainingConfig,
    TheoremItem,
    TheoremPool,
    _create_live_reasoner_pool,
    apply_evaluation_overrides,
    bc_weight_at_round,
    build_theorem_pool,
    collect_round,
    evaluate_proof_rate,
    run_rl_training,
    save_checkpoint,
    write_evaluation_report,
)
from maths_ai.gnn_inference.atp_lean_gnn.rl_live_collection import LiveReasonerPool
from maths_ai.gnn_inference.atp_lean_gnn import rl_training_driver
from maths_ai.hybrid_reasoner.proposal import ProposalSpec
from maths_ai.gnn_inference.atp_lean_gnn.rl_resources import ResourcePlan
from pantograph_async import (
    TransitionProvenance,
    ExecutionGoal,
    ExecutionLocal,
    PantographConfigurationError,
    PantographEnv,
    PantographGoalRejected,
    PantographProtocolError,
    PantographRequestTimeout,
    PantographStateError,
    PantographTransportError,
    WireExpression,
)
from maths_ai.gnn_inference.tests.model_helpers import (
    ScopedFakeClient,
    ScriptedExecutor,
    actor_critic,
    pantograph_goal,
    structured_goal,
)


# ---------------------------------------------------------------------------
# Fakes (mirroring test_rl_reasoner.py — no Lean/petta)
# ---------------------------------------------------------------------------


class _FakeGoalState:
    def __init__(self, goals=None, transition=None):
        self.goals = list(goals or [])
        self.transition = transition


class _FakeServer(ScopedFakeClient):
    def __init__(self):
        self.closed = False

    async def aclose(self):
        self.closed = True
        self.is_usable = False

    async def _start_goal(self, expression):
        return _FakeGoalState()

    async def _apply_tactic(self, state, tactic):
        selected = state.goals[0].goal_name if state.goals else ""
        if tactic.startswith("intro "):
            goal = pantograph_goal(structured_goal(GOAL_EXPR, HYPS))
            return _FakeGoalState(
                [goal],
                TransitionProvenance(selected, (goal.goal_name,), ((),), (), (), (), (), ()),
            )
        # Any other tactic closes the selected goal: no children, nothing else open.
        return _FakeGoalState(
            transition=TransitionProvenance(selected, (), (), (), (), (), (), ())
        )


def _QEDExecutor():
    return ScriptedExecutor(lambda _occurrence, _command: [], server=_FakeServer())


class _InadmissibleServer(_FakeServer):
    """Lean rejects every root, as it would a statement it cannot elaborate."""

    async def _start_goal(self, expression):
        raise PantographGoalRejected("cannot elaborate reconstructed goal")


class _InadmissibleExecutor(RetainedTacticExecutor):
    def __init__(self):
        self.server = _InadmissibleServer()

    async def apply(self, registry, occurrence, command, *, holder="operation"):
        raise AssertionError("tactics must not run when the root cannot be replayed")


def _RejectExecutor():
    return ScriptedExecutor(lambda _occurrence, _command: None, server=_FakeServer())


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

    async def prove(self, seed, *, proposal, deadline=None):
        idx = self._calls
        self._calls += 1
        if idx in self._raise_on:
            raise RuntimeError("simulated Lean transport failure")
        return await self._inner.prove(seed, proposal=proposal, deadline=deadline)


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


GREEDY_EVAL = EvaluationSettings(mode="greedy", seed=0)


def _proposals(n: int) -> list[ProposalSpec]:
    return [ProposalSpec(mode="sample", seed=i) for i in range(n)]


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


def _inadmissible_factory(model, node_vocab, tactic_vocab, cfg):
    return _make_reasoner(model, node_vocab, _InadmissibleExecutor(), top_k=cfg.top_k_tactics)


class _InadmissibleAfter:
    """Searches normally for the first ``searches`` calls, then every seed is
    inadmissible -- a round with no usable root after rounds that filled replay."""

    def __init__(self, inner, searches: int):
        self._inner = inner
        self._searches = searches
        self.model = inner.model
        self.dag_featurize_data = inner.dag_featurize_data

    async def prove(self, seed, *, proposal, deadline=None):
        if self._searches <= 0:
            raise ReplayInputError(ReplayFailure.LEAN_REJECTED, "unknown identifier")
        self._searches -= 1
        return await self._inner.prove(seed, proposal=proposal, deadline=deadline)


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

    def test_live_workers_receive_complete_puct_configuration(self):
        class _Environment:
            def describe(self):
                return "test MCTS environment"

        class _RecordingReasoner:
            instances = []

            def __init__(self, **kwargs):
                self.model = kwargs["model"]
                self.kwargs = kwargs
                self._pending = {}
                self._result = None
                self.__class__.instances.append(self)

        async def create_server(_environment):
            return _FakeServer()

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(
                tmp,
                selection_policy="puct",
                num_simulations=12,
                sim_batch_size=3,
                puct_c=1.75,
                top_k_tactics=5,
                max_depth=7,
                max_nodes=33,
                use_pln=False,
                widening_max_actions=9,
                widening_max_draws=40,
            )
            node_vocab = _build_node_vocab()
            model = _make_model(node_vocab)
            plan = ResourcePlan(
                reported_cpu_count=8,
                affinity_cpu_count=8,
                quota_cpu_count=None,
                usable_cpu_count=8,
                requested_worker_count=2,
                worker_count=2,
                collection_devices=(torch.device("cpu"),),
                update_device=torch.device("cpu"),
                gpu_resources=(),
            )
            with (
                patch(
                    "maths_ai.gnn_inference.atp_lean_gnn.rl_training_driver."
                    "create_model_sexpr_client",
                    new=create_server,
                ),
                patch(
                    "maths_ai.gnn_inference.atp_lean_gnn.rl_training_driver."
                    "RLHybridReasoner",
                    new=_RecordingReasoner,
                ),
                patch(
                    "maths_ai.hybrid_reasoner.joint_inference.PantographExecutor",
                    new=lambda server: server,
                ),
            ):
                live_pool = asyncio.run(
                    _create_live_reasoner_pool(
                        model=model,
                        node_vocab=node_vocab,
                        tactic_vocab=TACTIC_VOCAB,
                        cfg=cfg,
                        env=_Environment(),
                        plan=plan,
                    )
                )
                try:
                    # A replacement worker is built by the same factory with the
                    # same model replica, so it keeps every search setting.
                    replacement = asyncio.run(
                        live_pool._replacement_worker(live_pool.workers[0])
                    )
                    self.assertIs(replacement.model, live_pool.workers[1].model)
                    live_pool.assert_generation(0)
                    _RecordingReasoner.instances.remove(replacement.reasoner)
                    settings = (
                        "selection_policy",
                        "num_simulations",
                        "sim_batch_size",
                        "puct_c",
                        "top_k_tactics",
                        "max_depth",
                        "max_nodes",
                        "use_pln",
                        "widening_max_actions",
                        "widening_max_draws",
                    )
                    self.assertEqual(
                        {key: replacement.reasoner.kwargs[key] for key in settings},
                        {
                            key: _RecordingReasoner.instances[0].kwargs[key]
                            for key in settings
                        },
                    )
                    self.assertEqual(len(_RecordingReasoner.instances), 2)
                    self.assertEqual(
                        len({id(reasoner) for reasoner in _RecordingReasoner.instances}),
                        2,
                    )
                    self.assertEqual(
                        len({id(reasoner._pending) for reasoner in _RecordingReasoner.instances}),
                        2,
                    )
                    for reasoner in _RecordingReasoner.instances:
                        self.assertEqual(reasoner.kwargs["selection_policy"], "puct")
                        self.assertEqual(reasoner.kwargs["num_simulations"], 12)
                        self.assertEqual(reasoner.kwargs["sim_batch_size"], 3)
                        self.assertEqual(reasoner.kwargs["puct_c"], 1.75)
                        self.assertEqual(reasoner.kwargs["top_k_tactics"], 5)
                        self.assertEqual(reasoner.kwargs["max_depth"], 7)
                        self.assertEqual(reasoner.kwargs["max_nodes"], 33)
                        self.assertFalse(reasoner.kwargs["use_pln"])
                        self.assertEqual(reasoner.kwargs["widening_max_actions"], 9)
                        self.assertEqual(reasoner.kwargs["widening_max_draws"], 40)
                finally:
                    asyncio.run(live_pool.close())

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
            collect_round(reasoner, _items(3), timeout_s=10.0, proposals=_proposals(3))
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
        s1 = asyncio.run(evaluate_proof_rate(reasoner, items, settings=GREEDY_EVAL, timeout_s=10.0))
        s2 = asyncio.run(evaluate_proof_rate(reasoner, items, settings=GREEDY_EVAL, timeout_s=10.0))
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
                self.assertEqual(row["htps_optimizer_steps"], 0.0)
                self.assertEqual(row["parameter_updates_this_round"], 1.0)
                self.assertEqual(row["model_generation"], 1.0)

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
            self.assertEqual(rows[-1]["onpolicy_optimizer_step"], 1.0)
            self.assertEqual(rows[-1]["htps_optimizer_steps"], 2.0)
            self.assertEqual(rows[-1]["parameter_updates_this_round"], 3.0)
            self.assertEqual(rows[-1]["model_generation"], 3.0)
            self.assertEqual(rows[-1]["replica_generation"], 3.0)

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
            self.assertIsInstance(tactic_row["arguments"], list)
            self.assertIsInstance(tactic_row["candidate_pool"], dict)
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

    def test_htps_only_update_advances_model_generation(self):
        no_onpolicy_step = {
            "num_transitions": 0.0,
            "num_failures": 0.0,
            "unknown_label_count": 0.0,
            "structural_unknown_label_count": 0.0,
            "semantic_unknown_label_count": 0.0,
            "total_label_count": 0.0,
            "onpolicy_optimizer_step": 0.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(
                tmp,
                num_rounds=1,
                htps_steps_per_round=1,
                htps_batch_size=4,
            )
            with patch(
                "maths_ai.gnn_inference.atp_lean_gnn.rl_training_driver."
                "train_step_onpolicy",
                return_value=no_onpolicy_step,
            ):
                asyncio.run(
                    run_rl_training(
                        cfg,
                        reasoner_factory=_qed_factory,
                        pool=_pool(),
                    )
                )
            run_dir = next((tmp / "runs").iterdir())
            row = self._metric_rows(run_dir)[-1]
            self.assertEqual(row["onpolicy_optimizer_step"], 0.0)
            self.assertEqual(row["htps_optimizer_steps"], 1.0)
            self.assertEqual(row["parameter_updates_this_round"], 1.0)
            self.assertEqual(row["model_generation"], 1.0)

    def test_no_update_round_keeps_model_generation_unchanged(self):
        no_onpolicy_step = {
            "num_transitions": 0.0,
            "num_failures": 0.0,
            "unknown_label_count": 0.0,
            "structural_unknown_label_count": 0.0,
            "semantic_unknown_label_count": 0.0,
            "total_label_count": 0.0,
            "onpolicy_optimizer_step": 0.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(
                tmp,
                num_rounds=1,
                htps_steps_per_round=0,
                max_dead_rounds=2,
            )
            with patch(
                "maths_ai.gnn_inference.atp_lean_gnn.rl_training_driver."
                "train_step_onpolicy",
                return_value=no_onpolicy_step,
            ):
                asyncio.run(
                    run_rl_training(
                        cfg,
                        reasoner_factory=_qed_factory,
                        pool=_pool(),
                    )
                )
            run_dir = next((tmp / "runs").iterdir())
            row = self._metric_rows(run_dir)[-1]
            self.assertEqual(row["parameter_updates_this_round"], 0.0)
            self.assertEqual(row["model_generation"], 0.0)
            self.assertEqual(row["replica_generation"], 0.0)

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
            with self.assertRaisesRegex(ValueError, "cannot be resumed exactly"):
                asyncio.run(run_rl_training(
                    cfg, resume_run_dir=run_dir, reasoner_factory=_qed_factory, pool=_pool()
                ))


class EvaluationSettingsTests(unittest.TestCase):
    """One evaluation contract reaches --eval-only, periodic evaluation and resume."""

    def test_settings_are_validated(self):
        for bad in (
            dict(mode="beam"),
            dict(seed=-1),
            dict(repeats=0),
            dict(repeats=True),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                EvaluationSettings(**bad)
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            _write_config(Path(tmp), eval_proposal_mode="argmax")

    def test_command_line_overrides_change_the_config_for_every_entry_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _write_config(Path(tmp))
            self.assertEqual(cfg.evaluation_settings(), EvaluationSettings(mode="sample", seed=42, repeats=1))
            settings = apply_evaluation_overrides(cfg, mode="greedy", seed=5, repeats=3)
            self.assertEqual(settings, EvaluationSettings(mode="greedy", seed=5, repeats=3))
            self.assertEqual(cfg.evaluation_settings(), settings)
            with self.assertRaises(ValueError):
                apply_evaluation_overrides(cfg, repeats=0)

    def test_repeats_are_reported_and_aggregated_over_searched_seeds(self):
        node_vocab = _build_node_vocab()
        torch.manual_seed(0)
        model = _make_model(node_vocab)
        reasoner = _make_reasoner(model, node_vocab, _QEDExecutor(), top_k=2)
        settings = EvaluationSettings(mode="sample", seed=3, repeats=3)
        weights = [p.detach().clone() for p in model.parameters()]
        rng = torch.get_rng_state()
        stats = asyncio.run(evaluate_proof_rate(reasoner, _items(4), settings=settings, timeout_s=10.0))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for before, after in zip(weights, model.parameters()):
            self.assertTrue(torch.equal(before, after))
        self.assertEqual([row["repeat"] for row in stats["repeats"]], [0, 1, 2])
        self.assertEqual(len({row["proposal_seeds_sha256"] for row in stats["repeats"]}), 3)
        self.assertEqual(stats["attempted"], 12.0)
        self.assertEqual(stats["searched"], 12.0)
        self.assertEqual(stats["solved"], sum(row["solved"] for row in stats["repeats"]))
        self.assertEqual(stats["proof_rate"], stats["solved"] / 12.0)
        self.assertEqual(stats["evaluation"], settings.contract())
        again = asyncio.run(evaluate_proof_rate(reasoner, _items(4), settings=settings, timeout_s=10.0))
        self.assertEqual(again, stats)

    def test_an_empty_subset_is_rejected(self):
        node_vocab = _build_node_vocab()
        reasoner = _make_reasoner(_make_model(node_vocab), node_vocab, _QEDExecutor())
        with self.assertRaisesRegex(ValueError, "empty"):
            asyncio.run(evaluate_proof_rate(reasoner, [], settings=EvaluationSettings()))

    def test_periodic_evaluation_uses_the_configured_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(
                tmp, num_rounds=1, eval_every=1, eval_proposal_mode="greedy", eval_seed=9, eval_repeats=2
            )
            pool = TheoremPool(_items(20), eval_pool_size=2, curriculum_size=4, seed=0)
            torch.manual_seed(0)
            asyncio.run(run_rl_training(cfg, reasoner_factory=_qed_factory, pool=pool))
            run_dir = next((tmp / "runs").iterdir())
            rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
            (evaluation,) = [row["eval"] for row in rows if "eval" in row]
            self.assertEqual(evaluation["evaluation"], cfg.evaluation_settings().contract())
            self.assertEqual(len(evaluation["repeats"]), 2)
            best = torch.load(run_dir / "best.pt", weights_only=False)
            self.assertEqual(best["evaluation_contract"], cfg.evaluation_settings().contract())
            self.assertEqual(best["rl_target_schema_version"], 4)

    def test_resume_rejects_a_changed_evaluation_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, num_rounds=1)
            torch.manual_seed(0)
            asyncio.run(run_rl_training(cfg, reasoner_factory=_qed_factory, pool=_pool()))
            run_dir = next((tmp / "runs").iterdir())
            cfg.num_rounds = 2
            cfg.eval_proposal_mode = "greedy"
            with self.assertRaisesRegex(ValueError, "best proof rate"):
                asyncio.run(run_rl_training(
                    cfg, resume_run_dir=run_dir, reasoner_factory=_qed_factory, pool=_pool()
                ))

    def test_resume_requires_the_same_sampling_schedule(self):
        for schedule in ("encode-once-row-major-v1", "different-schedule", None):
            with self.subTest(schedule=schedule), tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                cfg = _write_config(tmp, num_rounds=1)
                asyncio.run(run_rl_training(cfg, reasoner_factory=_qed_factory, pool=_pool()))
                run_dir = next((tmp / "runs").iterdir())
                state = torch.load(run_dir / "last.pt", weights_only=False)
                proposal = state["search_settings"]["proposal"]
                self.assertEqual(state["search_settings"]["command_identity"], "rendered-lean-command-v1")
                self.assertEqual(proposal["sampling_schedule"], "encode-once-row-major-v1")
                if schedule is None:
                    proposal.pop("sampling_schedule")
                else:
                    proposal["sampling_schedule"] = schedule
                torch.save(state, run_dir / "last.pt")
                cfg.num_rounds = 2
                if schedule == "encode-once-row-major-v1":
                    asyncio.run(run_rl_training(
                        cfg, resume_run_dir=run_dir, reasoner_factory=_qed_factory, pool=_pool()
                    ))
                    resumed = torch.load(run_dir / "last.pt", weights_only=False)
                    self.assertEqual(resumed["round"], 1)
                else:
                    with self.assertRaisesRegex(ValueError, "saved search semantics"):
                        asyncio.run(run_rl_training(
                            cfg, resume_run_dir=run_dir, reasoner_factory=_qed_factory, pool=_pool()
                        ))

    def test_resume_rejects_a_schema_3_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, num_rounds=1)
            torch.manual_seed(0)
            asyncio.run(run_rl_training(cfg, reasoner_factory=_qed_factory, pool=_pool()))
            run_dir = next((tmp / "runs").iterdir())
            state = torch.load(run_dir / "last.pt", weights_only=False)
            state["rl_target_schema_version"] = 3
            state.pop("evaluation_contract")
            torch.save(state, run_dir / "last.pt")
            cfg.num_rounds = 2
            with self.assertRaisesRegex(ValueError, "schema 3"):
                asyncio.run(run_rl_training(
                    cfg, resume_run_dir=run_dir, reasoner_factory=_qed_factory, pool=_pool()
                ))

    def test_an_evaluation_report_is_written_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.json"
            report = {"kind": "rl-evaluation-report"}
            write_evaluation_report(report, path)
            self.assertEqual(json.loads(path.read_text()), report)
            with self.assertRaises(FileExistsError):
                write_evaluation_report(report, path)


class WideningConfigTests(unittest.TestCase):
    def test_widening_settings_are_validated_and_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with self.assertRaisesRegex(ValueError, "widening"):
                _write_config(tmp, widening_max_draws=10)  # best-first does not widen
            with self.assertRaises(ValueError):
                _write_config(tmp, selection_policy="puct", num_simulations=4, widening_exponent=1.5)
            cfg = _write_config(tmp, selection_policy="puct", num_simulations=4, widening_max_actions=8)
            saved = rl_training_driver.search_settings(cfg)
            self.assertEqual(saved["widening"]["max_actions"], 8)
            self.assertEqual(saved["widening"]["initial_actions"], 4)
            self.assertIsNone(rl_training_driver.search_settings(_write_config(tmp))["widening"])


class InadmissibleRootTests(unittest.TestCase):
    """A root that cannot be replayed was never searched: it is not a failed
    proof, it adds no learning rows, and it leaves the solve-rate window alone."""

    def _rows(self, tmp: Path) -> list[dict]:
        run_dir = next((tmp / "runs").iterdir())
        rows = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
        return [row for row in rows if "num_transitions" in row]

    def test_an_inadmissible_root_is_not_a_failed_search(self):
        node_vocab = _build_node_vocab()
        reasoner = _make_reasoner(_make_model(node_vocab), node_vocab, _InadmissibleExecutor())
        results, stats = asyncio.run(collect_round(reasoner, _items(2), timeout_s=10.0, proposals=_proposals(2)))
        self.assertEqual(results, [])
        self.assertEqual(stats["seeds_inadmissible"], 2.0)
        self.assertEqual(stats["searches_failed"], 0.0)
        self.assertEqual(stats["collected"], 0.0)
        self.assertEqual(reasoner._pending, {})

    def test_evaluation_counts_only_searched_seeds(self):
        node_vocab = _build_node_vocab()
        model = _make_model(node_vocab)
        inner = _make_reasoner(model, node_vocab, _QEDExecutor())
        stats = asyncio.run(
            evaluate_proof_rate(
                _InadmissibleAfter(inner, searches=1), _items(3), settings=EvaluationSettings(), timeout_s=10.0
            )
        )
        self.assertEqual((stats["solved"], stats["seeds_inadmissible"]), (1.0, 2.0))
        self.assertEqual(stats["proof_rate"], 1.0)

    def test_rounds_without_searched_seeds_leave_the_solve_rate_window(self):
        settings = dict(
            num_rounds=2,
            curriculum_window_rounds=1,
            curriculum_solve_threshold=0.0,
            max_dead_rounds=5,
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, **settings)
            torch.manual_seed(0)
            asyncio.run(run_rl_training(cfg, reasoner_factory=_inadmissible_factory, pool=_pool()))
            rows = self._rows(tmp)
        # A zero rate appended for an unsearched round would meet the threshold
        # and widen the window; skipping the round keeps it where it was.
        self.assertEqual([row["curriculum_size"] for row in rows], [4, 4])
        self.assertEqual([row["seeds_inadmissible"] for row in rows], [2.0, 2.0])
        self.assertEqual([row["searches_failed"] for row in rows], [0.0, 0.0])
        for key in ("imitation_samples_mined", "tactic_queue_len", "critic_queue_len"):
            self.assertEqual([row[key] for row in rows], [0.0, 0.0], key)
        self.assertEqual([row["parameter_updates_this_round"] for row in rows], [0.0, 0.0])

        # The same settings do widen it once a round is actually searched.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, **settings)
            torch.manual_seed(0)
            asyncio.run(run_rl_training(cfg, reasoner_factory=_qed_factory, pool=_pool()))
            self.assertGreater(self._rows(tmp)[0]["curriculum_size"], 4)

    def test_replay_updates_continue_on_a_round_without_usable_roots(self):
        def factory(model, node_vocab, tactic_vocab, cfg):
            inner = _make_reasoner(model, node_vocab, _QEDExecutor(), top_k=cfg.top_k_tactics)
            return _InadmissibleAfter(inner, searches=cfg.theorems_per_round)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(
                tmp, num_rounds=2, htps_steps_per_round=2, htps_batch_size=4, max_dead_rounds=5
            )
            torch.manual_seed(0)
            asyncio.run(run_rl_training(cfg, reasoner_factory=factory, pool=_pool()))
            first, second = self._rows(tmp)
        self.assertEqual(second["seeds_inadmissible"], 2.0)
        self.assertEqual(second["imitation_samples_mined"], 0.0)
        self.assertEqual(second["tactic_queue_len"], first["tactic_queue_len"])
        self.assertEqual(second["critic_queue_len"], first["critic_queue_len"])
        # No on-policy step without searched roots; HTPS still learns from the
        # validated entries already in replay.
        self.assertEqual(second["onpolicy_optimizer_step"], 0.0)
        self.assertEqual(second["htps_optimizer_steps"], 2.0)
        self.assertEqual(second["model_generation"], first["model_generation"] + 2.0)


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


class _ErrorReasoner:
    """prove() raises a chosen error on its first call."""

    def __init__(self, inner, error: BaseException, *, retire_client: bool = False):
        self._inner = inner
        self._error = error
        self._retire_client = retire_client
        self.model = inner.model
        self.server = inner.server
        self.calls = 0

    async def prove(self, seed, *, proposal, deadline=None):
        self.calls += 1
        if self.calls == 1:
            if self._retire_client:
                await self.server.aclose()
            raise self._error
        return await self._inner.prove(seed, proposal=proposal, deadline=deadline)


class SingleReasonerFailureTests(unittest.TestCase):
    """The single-reasoner path cannot replace its client, so session failures end it."""

    def _reasoner(self, error, **kwargs):
        node_vocab = _build_node_vocab()
        torch.manual_seed(0)
        inner = _make_reasoner(_make_model(node_vocab), node_vocab, _QEDExecutor(), top_k=2)
        return _ErrorReasoner(inner, error, **kwargs)

    def test_session_failures_propagate_instead_of_counting_as_failed_searches(self):
        for error in (
            PantographTransportError("process exited"),
            PantographProtocolError("bad contract"),
            PantographStateError("stale state"),
            PantographConfigurationError("bad executable"),
        ):
            reasoner = self._reasoner(error)
            with self.assertRaises(type(error)):
                asyncio.run(collect_round(reasoner, _items(2), timeout_s=10.0, proposals=_proposals(2)))
            self.assertEqual(reasoner.calls, 1)

    def test_timeout_that_retired_the_only_client_ends_collection(self):
        reasoner = self._reasoner(
            PantographRequestTimeout("deadline exceeded"), retire_client=True
        )
        with self.assertRaises(PantographTransportError) as error:
            asyncio.run(collect_round(reasoner, _items(2), timeout_s=10.0, proposals=_proposals(2)))
        self.assertIn("reasoner pool", str(error.exception))
        self.assertEqual(reasoner.calls, 1)


class LivePoolDriverTests(unittest.TestCase):
    """The round loop over a real ``LiveReasonerPool`` of scoped fake clients.

    Collection finishes before any parameter update; at most one on-policy step
    and the configured HTPS steps follow; the generation advances by the actual
    optimizer-step count; replicas synchronize once to the final generation
    before the next collection.
    """

    def _run(self, **overrides):
        events: list[tuple] = []
        clients: list[_FakeServer] = []
        original_collect = LiveReasonerPool.collect_round
        original_sync = LiveReasonerPool.synchronize
        original_onpolicy = rl_training_driver.train_step_onpolicy
        original_htps = rl_training_driver.train_step_htps_style

        class _Environment:
            def verify(self):
                return None

            def describe(self):
                return "test environment"

        async def create_client(_environment):
            client = _FakeServer()
            clients.append(client)
            return client

        async def collect(pool, batch, **kwargs):
            events.append(("collect", pool.model_generation))
            return await original_collect(pool, batch, **kwargs)

        def synchronize(pool, model, generation):
            events.append(("sync", generation))
            return original_sync(pool, model, generation)

        def onpolicy(*args, **kwargs):
            metrics = original_onpolicy(*args, **kwargs)
            events.append(("onpolicy", metrics["onpolicy_optimizer_step"]))
            return metrics

        def htps(*args, **kwargs):
            metrics = original_htps(*args, **kwargs)
            events.append(("htps", metrics["htps_optimizer_step"]))
            return metrics

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(
                tmp,
                collection_workers=2,
                cpu_reserve=0,
                resource_check="warn",
                # Seed 0's sampled proposals solve a theorem in every round, so
                # each round has replay to mine.
                **{"seed": 0, **overrides},
            )
            torch.manual_seed(0)
            with (
                patch.object(rl_training_driver, "pantograph_env", lambda _cfg: _Environment()),
                patch.object(rl_training_driver, "create_model_sexpr_client", create_client),
                patch.object(LiveReasonerPool, "collect_round", collect),
                patch.object(LiveReasonerPool, "synchronize", synchronize),
                patch.object(rl_training_driver, "train_step_onpolicy", onpolicy),
                patch.object(rl_training_driver, "train_step_htps_style", htps),
            ):
                asyncio.run(run_rl_training(cfg, pool=_pool()))
            run_dir = next((tmp / "runs").iterdir())
            rows = [
                json.loads(line)
                for line in (run_dir / "metrics.jsonl").read_text().splitlines()
            ]
        return events, rows, clients

    def _assert_round_order(self, events, rows, htps_steps):
        generation = 0
        rounds = [index for index, event in enumerate(events) if event[0] == "collect"]
        self.assertEqual(len(rounds), len(rows))
        for start, end, row in zip(rounds, [*rounds[1:], len(events)], rows):
            phase = events[start:end]
            self.assertEqual(phase[0], ("collect", generation))
            onpolicy = [event for event in phase if event[0] == "onpolicy"]
            htps = [event for event in phase if event[0] == "htps"]
            syncs = [event for event in phase if event[0] == "sync"]
            self.assertLessEqual(len(onpolicy), 1)
            self.assertEqual(len(htps), htps_steps)
            steps = int(sum(event[1] for event in [*onpolicy, *htps]))
            generation += steps
            # Updates follow collection; one synchronization ends the round.
            self.assertEqual(
                [event[0] for event in phase],
                ["collect", *(["onpolicy"] * len(onpolicy)), *(["htps"] * len(htps)),
                 *(["sync"] if steps else [])],
            )
            self.assertEqual(syncs, [("sync", generation)] if steps else [])
            self.assertEqual(row["parameter_updates_this_round"], float(steps))
            self.assertEqual(row["model_generation"], float(generation))
            self.assertEqual(row["replica_generation"], float(generation))
        return generation

    def test_htps_disabled_round_order_generation_and_replica_sync(self):
        events, rows, clients = self._run(num_rounds=2)
        generation = self._assert_round_order(events, rows, htps_steps=0)
        self.assertEqual(generation, 2)
        for row in rows:
            self.assertEqual(row["onpolicy_optimizer_step"], 1.0)
            self.assertEqual(row["htps_optimizer_steps"], 0.0)
            self.assertGreater(row["imitation_samples_mined"], 0.0)
            self.assertNotIn("tactic_imitation_loss", row)
        self.assertEqual(len(clients), 2)
        self.assertTrue(all(client.closed for client in clients))

    def test_puct_rounds_widen_and_report_their_proposal_work(self):
        for htps_steps in (0, 2):
            with self.subTest(htps_steps=htps_steps):
                events, rows, _clients = self._run(
                    num_rounds=2,
                    selection_policy="puct",
                    num_simulations=6,
                    sim_batch_size=2,
                    htps_steps_per_round=htps_steps,
                    htps_batch_size=4,
                )
                self._assert_round_order(events, rows, htps_steps=htps_steps)
                for row in rows:
                    self.assertLessEqual(row["onpolicy_optimizer_step"], 1.0)
                    self.assertGreater(row["proposal_rounds"], 0.0)
                    self.assertLessEqual(row["accepted_edges"], row["distinct_commands"])
                    self.assertLessEqual(row["distinct_commands"], row["raw_draws"])
                    self.assertLessEqual(row["settled_draws"], row["raw_draws"])

    def test_htps_enabled_round_order_generation_and_replica_sync(self):
        events, rows, clients = self._run(
            num_rounds=2, htps_steps_per_round=2, htps_batch_size=4
        )
        generation = self._assert_round_order(events, rows, htps_steps=2)
        self.assertEqual(generation, 6)
        for row in rows:
            self.assertEqual(row["htps_optimizer_steps"], 2.0)
            self.assertGreater(row["tactic_queue_len"], 0.0)
            self.assertGreater(row["critic_queue_len"], 0.0)
            self.assertIn("tactic_imitation_loss", row)
        self.assertTrue(all(client.closed for client in clients))


if __name__ == "__main__":
    unittest.main()
