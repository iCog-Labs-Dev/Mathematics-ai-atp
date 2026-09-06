import json
import networkx as nx
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import torch
from torch_geometric.data import Data

import matplotlib.pyplot as plt 
@dataclass
class GraphExpr:
    node_type: str
    name: Optional[str] = None
    idx: Optional[int] = None
    val: Optional[str] = None
    children: List['GraphExpr'] = field(default_factory=list)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'GraphExpr':
        if not isinstance(d, dict) or not d:
            return GraphExpr(node_type="unknown")

        # Handle tagged union: {"forallE": {...}, "app": {...}, "const": {...}, ...}
        for key, value in d.items():
            if key in ["forallE", "lam", "letE", "app", "var", "const", "lit", "proj", "other"]:
                node_type = key

                if node_type == "var":
                    content = value if isinstance(value, dict) else {}
                    return GraphExpr(
                        node_type="var",
                        name=content.get("name"),
                        idx=content.get("idx")
                    )
                elif node_type == "const":
                    content = value if isinstance(value, dict) else {}
                    return GraphExpr(node_type="const", name=content.get("name"))
                elif node_type == "lit":
                    content = value if isinstance(value, dict) else {}
                    return GraphExpr(node_type="lit", val=str(content.get("val", "")))
                elif node_type == "app":
                    children = []
                    content = value if isinstance(value, dict) else {}
                    if isinstance(content.get("fn"), dict):
                        children.append(GraphExpr.from_dict({"app": content["fn"]}))   # wrap back
                    if isinstance(content.get("arg"), dict):
                        children.append(GraphExpr.from_dict({"app": content["arg"]}))
                    return GraphExpr(node_type="app", children=children)

                # forallE, lam, letE, etc.
                elif isinstance(value, dict):
                    children = []
                    for k, v in value.items():
                        if isinstance(v, dict):
                            children.append(GraphExpr.from_dict({k: v}))
                    return GraphExpr(node_type=node_type, children=children)

        return GraphExpr(node_type="unknown")


def safe_attr(val):
    """Convert None to string so NetworkX accepts it."""
    if val is None:
        return ""
    return val


def build_networkx_graph(gexpr: GraphExpr, decl_name: str) -> nx.DiGraph:
    G = nx.DiGraph()
    node_counter = [0]

    def add_subtree(expr: GraphExpr, parent: Optional[int] = None) -> int:
        nid = node_counter[0]
        node_counter[0] += 1

        label = expr.node_type
        if expr.name:
            label += f"_{expr.name}"
        if expr.idx is not None:
            label += f"[{expr.idx}]"
        if expr.val:
            label += f"={expr.val}"

        G.add_node(
            nid,
            label=label,
            type=expr.node_type,
            name=safe_attr(expr.name),
            idx=safe_attr(expr.idx),
            val=safe_attr(expr.val)
        )

        if parent is not None:
            G.add_edge(parent, nid, relation="child")

        for child in expr.children:
            add_subtree(child, nid)

        return nid

    root_id = add_subtree(gexpr)
    G.graph["theorem"] = decl_name
    G.graph["root"] = root_id
    return G


def build_pyg_data(gexpr: GraphExpr, decl_name: str) -> Data:
    node_features = []
    edge_index = []
    node_types_list = []
    counter = [0]

    def traverse(expr: GraphExpr, parent_id: Optional[int] = None):
        cid = counter[0]
        counter[0] += 1

        node_type_map = {
            "var": 0, "const": 1, "app": 2, "forallE": 3, "lam": 4,
            "letE": 5, "lit": 6, "proj": 7, "unknown": 8, "other": 9
        }
        feat = [0] * 10
        feat[node_type_map.get(expr.node_type, 9)] = 1

        node_features.append(feat)
        node_types_list.append(expr.node_type)

        if parent_id is not None:
            edge_index.append([parent_id, cid])

        for child in expr.children:
            traverse(child, cid)

    traverse(gexpr)

    x = torch.tensor(node_features, dtype=torch.float)
    edge_idx = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_idx)
    data.node_types = node_types_list
    data.theorem_name = decl_name
    return data


def load_lean_graphs(json_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = {}

    for decl in data:
        if not isinstance(decl, dict):
            continue

        decl_name = decl.get("name", "unknown")
        kind = decl.get("kind", "unknown")
        type_expr_dict = decl.get("typeExpr")

        if not type_expr_dict:
            print(f"Warning: No typeExpr for {decl_name}")
            continue

        gexpr = GraphExpr.from_dict(type_expr_dict)

        nx_graph = build_networkx_graph(gexpr, decl_name)
        pyg_data = build_pyg_data(gexpr, decl_name)

        results[decl_name] = {
            "kind": kind,
            "networkx": nx_graph,
            "pyg": pyg_data,
            "gexpr": gexpr
        }

        print(f"✓ Built graph for {kind}: {decl_name} | Nodes: {nx_graph.number_of_nodes()} | Edges: {nx_graph.number_of_edges()}")

    return results



