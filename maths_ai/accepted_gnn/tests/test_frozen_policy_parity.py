"""Numerical parity with the independently published frozen GNN checkout."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_frozen_policy_matches_published_runtime():
    bundle, reference = os.environ.get("MATHS_AI_GNN_BUNDLE"), os.environ.get("MATHS_AI_FROZEN_REFERENCE")
    if not bundle or not reference:
        pytest.skip("set accepted bundle and published reference checkout for numerical parity")
    script = r'''
import json,sys,torch
if sys.argv[2] == "current":
 from maths_ai.accepted_gnn.flat_policy import LiveJointPolicy,prepare_live_goal
else:
 from maths_ai.gnn_inference.atp_lean_gnn.live_policy import LiveJointPolicy
 from maths_ai.gnn_inference.atp_lean_gnn.model_graph import prepare_live_goal
def expr(text,pp): return {"modelSexpVersion":1,"modelSexp":text,"pp":pp}
goals=[{"isConversion":False,"vars":[
 {"contextIndex":1,"userName":"P","binderRole":":explicit","type":expr("(:sort 0)","Prop")},
 {"contextIndex":7,"userName":"h_unseen","binderRole":":explicit","type":expr("(:fv FV1)","P")}],
 "target":expr("(:fv FV1)","P")},
 {"isConversion":False,"vars":[],"target":expr("(:forall x :explicit (:c Nat) (:app (:c Eq) (:arg :implicit 0 (:c Nat)) (:arg :explicit 1 0) (:arg :explicit 2 0)))","forall x : Nat, x = x")},
 {"isConversion":False,"vars":[{"contextIndex":3,"userName":"n","binderRole":":explicit","isLet":True,
 "type":expr("(:c Nat)","Nat"),"value":expr("(:lit 0)","0")}],
 "target":expr("(:app (:c Eq) (:arg :implicit 0 (:c Nat)) (:arg :explicit 1 (:fv FV3)) (:arg :explicit 2 (:lit 0)))","n = 0")}]
p=LiveJointPolicy.from_bundle(sys.argv[1],device="cpu")
rows=[]
with torch.inference_mode():
 for goal in goals:
  if sys.argv[2]=="current":
   typed,dag,batch,mapping,text,context=prepare_live_goal(goal,p.system.node_vocab)
  else:
   g=prepare_live_goal(goal,p.system.node_vocab,edge_mode="bidirectional")
   dag,batch,mapping,text=g.dag,g.batch,g.local_node_context,g.text
  nodes=p.system.policy.pointer.backbone.encode_nodes(batch)
  state=p.system.policy.pointer.backbone.readout(nodes,batch)
  pred=p.predict(goal,top_k=8)
  rows.append({"dag":[n.as_dict() for n in dag.nodes],"mapping":mapping,"text":text,
   "graph":{k:batch[k].tolist() for k in ("x","edge_index","node_type","is_bound","binder_depth","binder_kind","state_node_index","premise_mask")},
   "nodes":nodes.tolist(),"state":state.tolist(),"logits":p.system.policy.pointer.backbone.classifier(state).tolist(),
   "actions":[a.to_dict() for a in pred.candidates],"rejected":pred.rejected})
print("PARITY_JSON "+json.dumps(rows))
'''
    rows = []
    for label, workdir in (("reference", reference), ("current", Path(__file__).resolve().parents[3])):
        done = subprocess.run([sys.executable, "-B", "-c", script, bundle, label], cwd=workdir,
                              capture_output=True, text=True, timeout=180)
        assert done.returncode == 0, done.stdout + done.stderr
        rows.append(json.loads(next(line[len("PARITY_JSON "):] for line in done.stdout.splitlines() if line.startswith("PARITY_JSON "))))
    def compare(left, right):
        if isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                compare(left[key], right[key])
        elif isinstance(left, list):
            assert len(left) == len(right)
            for a, b in zip(left, right):
                compare(a, b)
        elif isinstance(left, float):
            assert right == pytest.approx(left, rel=1e-6, abs=1e-6)
        else:
            assert left == right
    compare(*rows)
