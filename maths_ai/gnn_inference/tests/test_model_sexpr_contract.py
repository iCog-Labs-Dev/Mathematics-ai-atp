from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import pantograph.expr as expr_mod
import torch
from torch_geometric.data import Data

from maths_ai.data_models.proof_components import Goal, GoalLocal
from maths_ai.gnn_inference.atp_lean_gnn.graph import model_goal_to_dag
from maths_ai.gnn_inference.atp_lean_gnn.inference import (
    _local_names_by_fv_label,
    _resolve_local_node_name,
)
from maths_ai.hybrid_reasoner.pantograph_model_sexpr import (
    PantographModelSExprProtocolError,
    install_model_sexpr_pypantograph_patch,
    pantograph_goal_to_goal,
)
from maths_ai.gnn_inference.scripts.migrate_prepared_graph_contract import (
    migrate_prepared_contract,
)
from maths_ai.gnn_inference.atp_lean_gnn.graph_contract import MODEL_SEXPR_GRAPH_SPEC
from maths_ai.gnn_inference.atp_lean_gnn.pln_rl_training import make_dag_featurizer
from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_vocab


def _expression(pp: str, model: str, version: int = 1) -> dict[str, object]:
    return {"pp": pp, "modelSexp": model, "modelSexpVersion": version}


def _payload() -> dict[str, object]:
    return {
        "name": "g0",
        "userName": "intro.zero",
        "fragment": "tactic",
        "vars": [
            {
                "name": "_uniq.1",
                "userName": "h",
                "contextIndex": 0,
                "binderRole": ":explicit",
                "isInstance": False,
                "isLet": False,
                "type": _expression("P", "(:c P)"),
            },
            {
                "name": "_uniq.2",
                "userName": "x",
                "contextIndex": 1,
                "binderRole": ":explicit",
                "isInstance": False,
                "isLet": True,
                "type": _expression("α", "(:c Alpha)"),
                "value": _expression("default", "(:c default)"),
            },
        ],
        "target": _expression("h = h", "(:app (:c Eq) (:fv FV0) (:fv FV0))"),
    }


class StrictCodecTests(unittest.TestCase):
    def test_payload_preserves_pretty_and_model_views_and_case_tag(self):
        install_model_sexpr_pypantograph_patch()
        parsed = expr_mod.Goal.parse(_payload(), {"g0": 0})
        goal = pantograph_goal_to_goal(parsed)

        self.assertEqual(goal.expression, "h = h")
        self.assertEqual(goal.goal_model_sexp, "(:app (:c Eq) (:fv FV0) (:fv FV0))")
        self.assertEqual(goal.case_tag, "intro.zero")
        self.assertEqual([local.user_name for local in goal.locals or []], ["h", "x"])
        self.assertEqual((goal.locals or [])[1].value_pp, "default")
        self.assertNotIn("intro.zero", goal.hypotheses)

    def test_missing_required_expression_field_raises(self):
        install_model_sexpr_pypantograph_patch()
        payload = _payload()
        payload["target"] = {"pp": "h = h", "modelSexpVersion": 1}
        with self.assertRaises(PantographModelSExprProtocolError):
            expr_mod.Goal.parse(payload, {"g0": 0})

    def test_unsupported_expression_version_raises(self):
        install_model_sexpr_pypantograph_patch()
        payload = _payload()
        payload["target"] = _expression("h = h", "(:c Bad)", version=2)
        with self.assertRaises(PantographModelSExprProtocolError):
            expr_mod.Goal.parse(payload, {"g0": 0})


class NormalizedGraphTests(unittest.TestCase):
    def test_four_child_hyp_and_shared_fv_identity(self):
        install_model_sexpr_pypantograph_patch()
        goal = pantograph_goal_to_goal(expr_mod.Goal.parse(_payload(), {"g0": 0}))
        dag = model_goal_to_dag(goal)

        hyp = next(node for node in dag.nodes if node.label == "Hyp")
        self.assertEqual(len(hyp.children), 4)
        child_labels = [dag.nodes[index].label for index in hyp.children]
        self.assertEqual(child_labels[:3], ["FV0", "h", "HypRole:explicit"])
        fv_nodes = [index for index, node in enumerate(dag.nodes) if node.label == "FV0"]
        self.assertEqual(len(fv_nodes), 1)
        fv_id = fv_nodes[0]
        eq_app = next(node for node in dag.nodes if node.label == ":app")
        self.assertEqual(eq_app.children.count(fv_id), 2)

        local_names = _local_names_by_fv_label(dag)
        self.assertEqual(_resolve_local_node_name(hyp, dag, local_names), "h")
        self.assertEqual(_resolve_local_node_name(dag.nodes[fv_id], dag, local_names), "h")

    def test_goal_root_does_not_depend_on_last_allocated_expression_node(self):
        goal = Goal(
            expression="Prop",
            goal_model_sexp="(:c Prop)",
            locals=[
                GoalLocal(
                    user_name="h",
                    context_index=0,
                    binder_role=":explicit",
                    type_pp="Prop",
                    type_model_sexp="(:c Prop)",
                )
            ],
            model_sexp_version=1,
        )
        dag = model_goal_to_dag(goal)
        goal_node = next(node for node in dag.nodes if node.label == "Goal")
        self.assertEqual(dag.nodes[goal_node.children[0]].label, "Prop")
        self.assertEqual(dag.nodes[dag.state_root_id].label, "State")

    def test_structural_oov_is_an_error_and_semantic_oov_is_counted(self):
        goal = Goal(
            expression="P",
            goal_model_sexp="(:fv FV0)",
            locals=[
                GoalLocal(
                    user_name="h",
                    context_index=0,
                    binder_role=":explicit",
                    type_pp="Prop",
                    type_model_sexp="(:c Prop)",
                )
            ],
            model_sexp_version=1,
        )
        dag = model_goal_to_dag(goal)
        vocab = build_vocab([dag])

        semantic_vocab = {key: value for key, value in vocab.items() if key != "h"}
        _dag, data = make_dag_featurizer(semantic_vocab)(goal)
        self.assertGreater(int(data.semantic_unknown_label_count), 0)
        self.assertEqual(int(data.structural_unknown_label_count), 0)

        structural_vocab = {
            key: value for key, value in vocab.items() if key != "HypRole:explicit"
        }
        with self.assertRaisesRegex(ValueError, "required structural labels"):
            make_dag_featurizer(structural_vocab)(goal)


class PreparedContractMigrationTests(unittest.TestCase):
    def test_validated_structured_graphs_receive_contract_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vocab = {
                "<UNK>": 0,
                "State": 1,
                "Goal": 2,
                "Hyp": 3,
                "FV0": 4,
                "h": 5,
                "HypRole:explicit": 6,
                "Prop": 7,
            }
            (root / "vocab").mkdir()
            (root / "vocab" / "node_vocab.json").write_text(json.dumps(vocab))
            (root / "manifests").mkdir()
            for split in ("train", "val", "test"):
                pyg_dir = root / split / "pyg"
                pyg_dir.mkdir(parents=True)
                data = Data(
                    x=torch.tensor([1, 2, 3, 4, 5, 6, 7]),
                    edge_index=torch.tensor(
                        [[2, 1, 3, 4, 5, 6], [0, 0, 2, 2, 2, 2]], dtype=torch.long
                    ),
                    state_node_index=torch.tensor([0]),
                )
                torch.save(data, pyg_dir / "0.pt")
                (root / "manifests" / f"{split}.json").write_text(
                    json.dumps({"artifact_paths": {"pyg_dir": f"{split}/pyg"}})
                )
            target = root / "target.json"
            target.write_text(json.dumps(MODEL_SEXPR_GRAPH_SPEC.to_dict()))

            output = migrate_prepared_contract(root, target, samples_per_split=1)

            self.assertEqual(
                json.loads(output.read_text()), MODEL_SEXPR_GRAPH_SPEC.to_dict()
            )


if __name__ == "__main__":
    unittest.main()
