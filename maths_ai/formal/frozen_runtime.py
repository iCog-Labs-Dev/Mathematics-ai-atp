"""Frozen accepted policy in the existing supervised runtime process.

The public FormalService and ProvingService retain ownership of admission,
queues, deadlines, cancellation and evidence storage. This backend only owns
the loaded frozen policy and native sessions for one bounded operation.
"""

import asyncio
import json
from pathlib import Path
import time

from maths_ai.accepted_gnn.best_first import GnnSearchBudget, search_gnn
from maths_ai.accepted_gnn.flat_policy import LiveJointPolicy
from maths_ai.gnn_inference.atp_lean_gnn.native_session import (
    LiveLeanSession, LiveStateError, environment_binding, lean_environment,
)
from maths_ai.gnn_inference.atp_lean_gnn.proof_tasks import checked_header, masked
from maths_ai.gnn_inference.atp_lean_gnn.proof_verification import LeanProofVerifier
from maths_ai.gnn_inference.atp_lean_gnn.retrieval_artifacts import environment_components, without_repl, file_hash
from maths_ai.hybrid_reasoner.tactic_budget import TacticAllowance, TacticBudgetExhausted
from .admission import AdmissionRejected, target_unit_signature
from .source_elaboration import _local_record
from .source_units import source_units

FROZEN_BACKEND = "maths_ai.formal.frozen_runtime:FrozenRuntime"
PROFILE = "frozen-best-first"


def proof_trace(report):
    """Recover complete AND children from native focus steps on the winning path."""
    path, index = [], report["winning_node"]
    while report["nodes"][index]["parent"] is not None:
        node = report["nodes"][index]
        path.append(node)
        index = node["parent"]
    initial = report["nodes"][index]["goals"]
    if len(initial) != 1:
        raise ValueError("certificate needs exactly one root obligation")

    def occurrence(goal):
        return {"goal": goal["target"]["pp"], "subgoals": []}

    root = occurrence(initial[0])
    pending = [root]
    previous = initial
    for node in reversed(path):
        goals = node["goals"]
        siblings = previous[1:]
        if len(goals) < len(siblings) or (siblings and
            [goal["name"] for goal in goals[-len(siblings):]] != [goal["name"] for goal in siblings]):
            raise ValueError("native tactic changed unreported sibling obligations")
        count = len(goals) - len(siblings)
        children = [occurrence(goal) for goal in goals[:count]]
        current = pending.pop(0)
        current.update(command=node["action"], subgoals=children)
        pending = children + pending
        previous = goals
    if pending or previous:
        raise ValueError("winning path has unclosed obligations")
    return root


class FrozenRuntime:
    def __init__(self, config):
        self.service_config = config
        raw = json.loads(config.rl_config.read_text())
        if set(raw) != {"bundle", "source_root", "pantograph_root", "imports", "device"}:
            raise ValueError("frozen runtime needs bundle, source_root, pantograph_root, imports and device")
        base = config.rl_config.parent
        self.paths = {key: (base / raw[key]).resolve() for key in ("bundle", "source_root", "pantograph_root")}
        self.imports = tuple(raw["imports"])
        if not self.imports or any(not isinstance(name, str) or not name for name in self.imports):
            raise ValueError("frozen imports must be explicit")
        self.device = raw["device"]
        self.policy = None

    def environment(self):
        native = environment_binding(self.paths["source_root"], self.paths["pantograph_root"], imports=self.imports)
        lean, _ = lean_environment(str(self.paths["source_root"]), str(self.paths["pantograph_root"]))
        library = without_repl(environment_components(self.paths["source_root"], lean, self.imports))
        root = Path(__file__).resolve().parents[1]
        modules = [root / "accepted_gnn" / (name + ".py") for name in
                   ("flat_policy", "flat_actions", "best_first", "live_state")]
        modules += [root / "formal" / (name + ".py") for name in
                    ("frozen_runtime", "frozen_backend", "source_units", "source_elaboration", "admission")]
        modules += [root / "gnn_inference/atp_lean_gnn" / (name + ".py") for name in
                    ("native_session", "proof_tasks", "proof_verification", "model_graph", "model_factory",
                     "training", "joint_loader", "joint_source_decoder", "retrieval_artifacts")]
        return {"format": "frozen-gnn-environment-v1", "backend": "accepted-gnn", "native": native,
                "library": library, "imports": list(self.imports), "lean_sha256": file_hash(lean),
                "inference_device": self.device,
                "implementation": {str(path.relative_to(root)): file_hash(path) for path in modules}}

    async def start(self):
        self._environment = self.environment()
        self.policy = LiveJointPolicy.from_bundle(self.paths["bundle"], device=self.device)
        async with self.session() as session:
            root = await session.start_declaration("theorem FrozenRuntimePreflight (P : Prop) (h : P) : P")
            if not (await session.step(root, "exact h")).closed:
                raise RuntimeError("frozen native preflight did not close")
        return {"formal_environment": self._environment, "backend": "accepted-gnn", "profile": PROFILE,
                "bundle_sha256": file_hash(self.paths["bundle"] / "bundle.json"), "device": self.device,
                "baseline": "1315dca67529590e8729b65e27308490951afae9"}

    def status(self):
        return {"workers": int(self.policy is not None), "usable_workers": int(self.policy is not None)}

    def session(self, *, prefix=""):
        return LiveLeanSession(source_root=self.paths["source_root"], pantograph_root=self.paths["pantograph_root"],
                               imports=self.imports, prefix=prefix)

    def check_environment(self):
        if self.environment() != self._environment:
            raise RuntimeError("frozen execution environment changed after startup")

    async def admit_source(self, payload):
        self.check_environment()
        body, name = payload["body"], payload["declaration"]
        lean, env = lean_environment(str(self.paths["source_root"]), str(self.paths["pantograph_root"]))
        units = await source_units(body, declaration=name, imports=self.imports, source_root=self.paths["source_root"],
                                   lean=lean, env=env, timeout_s=payload["timeout_s"])
        encoded = body.encode("utf-8")
        start, assign = target_unit_signature(units, encoded, name)
        prefix, header = encoded[:start].decode(), encoded[start:assign].decode().strip()
        view = masked(header)
        offset = len(view) - len(view.lstrip())
        prefix += header[:offset]
        header = header[offset:]
        try:
            header, actual = checked_header(header, prefix=prefix)
        except ValueError as exc:
            raise AdmissionRejected("input:unsupported-declaration", str(exc)) from exc
        if actual != name:
            raise AdmissionRejected("input:declaration-mismatch", "Native target differs from source locator")
        async with self.session(prefix=prefix) as session:
            try:
                root = await session.start_declaration(header)
            except LiveStateError as exc:
                raise AdmissionRejected("input:lean-rejected", str(exc)) from exc
            if len(root.goals) != 1:
                raise AdmissionRejected("input:unsupported-context", "Expected exactly one target obligation")
            goal = root.goals[0]
        return {"target_sexp": goal["target"]["sexp"], "target_pp": goal["target"]["pp"],
                "locals": [_local_record(var) for var in goal["vars"]], "target_unit_start": start,
                "check": {"ok": True}, "native_target": {"header": header, "prefix": prefix, "declaration": name}}

    def target(self, manifest):
        if manifest["kind"] == "source":
            return manifest["native_target"], []
        statement = manifest["statement"]
        if statement["universe_levels"]:
            raise AdmissionRejected("input:unsupported-context", "Frozen text profile does not support universes")
        target = {"header": "theorem FormalGenerated : (" + statement["expression"] + ")",
                  "prefix": "set_option autoImplicit false\n", "declaration": "FormalGenerated"}
        actions = ["intro " + " ".join(statement["local_names"])] if statement["local_names"] else []
        return target, actions

    async def prove(self, payload):
        self.check_environment()
        manifest = payload["manifest"]
        target, actions = self.target(manifest)
        allowance = TacticAllowance(payload["max_steps"])
        deadline = time.monotonic() + payload["timeout_s"]
        verifier = LeanProofVerifier(source_root=self.paths["source_root"], imports=self.imports, prefix=target["prefix"])
        async with self.session(prefix=target["prefix"]) as session:
            try:
                root = await session.start_declaration(target["header"])
            except LiveStateError as exc:
                raise AdmissionRejected("input:lean-rejected", str(exc)) from exc
            initial = root
            for command in actions:
                allowance.reserve()
                root = await session.step(root, command)
            session.root = root
            native_step = session.step

            async def step(parent, command):
                allowance.reserve()
                return await native_step(parent, command)

            session.step = step

            class CandidateVerifier:
                async def verify(self, header, steps, *, timeout):
                    commands = actions + steps
                    if len(commands) > allowance.remaining:
                        raise TacticBudgetExhausted(allowance.limit)
                    for _ in commands:
                        allowance.reserve()
                    return await verifier.verify(header, commands, timeout=timeout)

            settings = payload["settings"]
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return {"termination": "deadline", "steps_used": allowance.issued}
            report = await search_gnn(self.policy.excluding({target["declaration"]}), session, target["header"], CandidateVerifier(),
                budget=GnnSearchBudget(model_calls=settings.get("model_calls", 32), tactic_trials=max(1, allowance.remaining),
                    nodes=settings.get("max_nodes", 64), depth=settings.get("max_depth", 16), top_k=settings.get("top_k", 8),
                    seconds=remaining, verification_seconds=remaining))
        solved = report["status"] == "solved" and report["proof_verified"]
        termination = {"solved": "certified", "trial_budget": "step-budget", "node_budget": "node-budget",
                       "time_budget": "deadline", "native_timeout": "deadline", "input_error": "invalid-input",
                       "runtime_error": "crash", "model_call_budget": "simulation-budget",
                       "depth_budget": "node-budget", "frontier_exhausted": "frontier-exhausted"}[report["status"]]
        result = {"termination": termination, "certified": solved, "steps_used": allowance.issued,
                  "details": {"profile": PROFILE, "report": report}}
        if solved:
            trace = proof_trace(report)
            if actions:
                trace = {"goal": initial.goals[0]["target"]["pp"], "command": actions[0], "subgoals": [trace]}
            result.update(trace=trace, certificate={"format": "maths-ai-accepted-certificate-v1",
                "statement": manifest["statement"], "proof_trace": trace, "native_target": target,
                "steps": actions + report["accepted_path"]})
        return result

    async def verify(self, payload):
        self.check_environment()
        certificate = payload["certificate"]
        if certificate.get("format") != "maths-ai-accepted-certificate-v1" or certificate.get("statement") != payload["statement"]:
            return {"status": "rejected", "reason": "certificate statement or format mismatch", "steps_used": 0, "environment": self._environment}
        target, commands = certificate["native_target"], certificate["steps"]
        if not commands or len(commands) > payload["max_steps"]:
            return {"status": "inconclusive", "reason": "verification allowance exhausted", "steps_used": 0, "environment": self._environment}
        verdict = await LeanProofVerifier(source_root=self.paths["source_root"], imports=self.imports,
            prefix=target["prefix"]).verify(target["header"], commands, timeout=payload["timeout_s"])
        status = {"passed": "verified", "lean_rejected": "rejected", "verification_timeout": "inconclusive"}[verdict["status"]]
        return {"status": status, "reason": None if status == "verified" else verdict["status"], "steps_used": len(commands),
                "environment": self._environment, "details": {"axioms": verdict.get("axioms"),
                    "sorry_detected": False if status == "verified" else None, "axiom_audit": verdict,
                    "session": "fresh-native-compilation"}}

    async def close(self):
        self.policy = None
