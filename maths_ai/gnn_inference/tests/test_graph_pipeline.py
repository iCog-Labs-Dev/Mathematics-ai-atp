from __future__ import annotations

import unittest
from pathlib import Path

from maths_ai.gnn_inference.atp_lean_gnn import DEMO_STATE, build_vocab, dag_to_dict, dag_to_pyg, parse_state, proof_state_to_dag, write_dag_json


class GraphPipelineTests(unittest.TestCase):
    def test_parse_state_supports_unicode_and_ascii_turnstiles(self) -> None:
        unicode_state = parse_state("n : Nat\n\u22a2 Even n")
        ascii_state = parse_state("n : Nat\n|- Even n")

        self.assertEqual(unicode_state.goal, "Even n")
        self.assertEqual(ascii_state.goal, "Even n")
        self.assertEqual([hyp.name for hyp in unicode_state.hypotheses], ["n"])
        self.assertEqual([hyp.type_expr for hyp in ascii_state.hypotheses], ["Nat"])

    def test_reused_nodes_track_parent_uses_not_child_count(self) -> None:
        dag = proof_state_to_dag(DEMO_STATE)
        reused_by_label = {node.label for node in dag.reused_nodes()}

        self.assertIn("Even", reused_by_label)
        self.assertIn("n", reused_by_label)
        self.assertNotIn("State", reused_by_label)

    def test_json_export_includes_graph_stats(self) -> None:
        dag = proof_state_to_dag(DEMO_STATE)
        payload = dag_to_dict(dag, metadata={"source": "test"})

        self.assertEqual(payload["metadata"]["source"], "test")
        self.assertEqual(payload["stats"]["num_nodes"], dag.num_nodes)
        self.assertEqual(payload["stats"]["num_reused_nodes"], len(dag.reused_nodes()))

    def test_write_dag_json_persists_utf8_payload(self) -> None:
        dag = proof_state_to_dag(DEMO_STATE)
        output_path = Path("tests") / "_tmp_graph.json"

        try:
            written = write_dag_json(dag, output_path, metadata={"source": "test"})

            self.assertEqual(written, output_path)
            contents = output_path.read_text(encoding="utf-8")
            self.assertIn('"source": "test"', contents)
            self.assertIn('"num_nodes"', contents)
        finally:
            if output_path.exists():
                output_path.unlink()

    def test_dag_to_pyg_can_add_reverse_edges(self) -> None:
        dag = proof_state_to_dag(DEMO_STATE)
        vocab = build_vocab([dag])
        pyg = dag_to_pyg(dag, vocab, add_reverse_edges=True)

        self.assertEqual(tuple(pyg.x.shape), (dag.num_nodes,))
        self.assertEqual(tuple(pyg.node_type.shape), (dag.num_nodes,))
        self.assertEqual(tuple(pyg.edge_index.shape), (2, dag.num_edges * 2))


if __name__ == "__main__":
    unittest.main()
