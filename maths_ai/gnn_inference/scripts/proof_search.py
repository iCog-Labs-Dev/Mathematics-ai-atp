"""Bounded proof search over native Lean snapshots and typed environment lookup."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import tempfile
import threading
import time


LEAN_HELPER = Path(__file__).resolve().parents[1] / "lean/ProofSearch.lean"


@dataclass(frozen=True)
class SearchBudget:
    model_calls: int = 8
    tactic_trials: int = 24
    nodes: int = 24
    depth: int = 8
    visits: int = 3
    seconds: float = 600
    tactic_timeout: float = 20

    def __post_init__(self):
        if (any(getattr(self, name) < 1 for name in ("model_calls", "tactic_trials", "nodes", "depth", "visits"))
                or any(not math.isfinite(value) or value <= 0 for value in (self.seconds, self.tactic_timeout))):
            raise ValueError("Search budgets must be finite and positive")


def candidate_key(code):
    """Ignore comments, keeping string contents and significant indentation."""
    out, i, depth, string = [], 0, 0, False
    while i < len(code):
        pair, char = code[i:i + 2], code[i]
        if string:
            out.append(char)
            if char == "\\" and i + 1 < len(code):
                i += 1
                out.append(code[i])
            elif char == '"':
                string = False
        elif depth:
            if pair == "/-":
                depth += 1
                i += 1
            elif pair == "-/":
                depth -= 1
                i += 1
                if not depth:
                    out.append(" ")
            elif char == "\n":
                out.append(char)
        elif pair == "/-":
            depth = 1
            i += 1
        elif pair == "--":
            end = code.find("\n", i)
            i = len(code) if end < 0 else end - 1
        else:
            out.append(char)
            string = char == '"'
        i += 1
    return "\n".join(line.rstrip() for line in "".join(out).splitlines() if line.strip()).strip()


def parse_proposal(raw, branching):
    from maths_ai.gnn_inference.scripts import run_minif2f_dev as dev

    if raw.count("```") % 2:
        raise ValueError("Unclosed proposal block")
    blocks = re.findall(r"```([^\n`]*)\n(.*?)```", raw, re.DOTALL)
    text = blocks[-1][1].strip() if blocks else raw.strip()
    if text.startswith("{") and (not blocks or blocks[-1][0].strip().lower() == "json"
                                  or re.match(r'\{\s*"(?:lookup|tactics)"', text)):
        payload = json.loads(text)
        if set(payload) == {"lookup"} and isinstance(payload["lookup"], str):
            query = payload["lookup"].strip()
            if not query or len(query) > 128:
                raise ValueError("Invalid lookup query")
            return {"lookup": query}
        if set(payload) != {"tactics"} or not isinstance(payload["tactics"], list):
            raise ValueError("Expected lookup or tactics JSON")
        tactics = payload["tactics"][:branching]
    else:
        # Prover checkpoints often keep emitting a complete theorem despite a
        # next-step prompt. Its header is NEVER installed in the native state.
        if blocks and re.fullmatch(r"lean4?(?:\s+tactic(?:\s+block(?:\s+\d+)?)?)?",
                                   blocks[-1][0].strip(), re.IGNORECASE):
            text = blocks[-1][1].strip()
            code = dev._proof_code("```lean4\n" + text + "\n```")
        else:
            code = dev._proof_code(raw)
        if re.match(r"\s*(?:theorem|lemma|example)\b", code):
            code = dev.control._extract_body(code)
        elif code.startswith("by\n") or code.startswith("by "):
            code = dev.control._extract_body(code)
        tactics = [code]
    if not tactics or not all(isinstance(tactic, str) and tactic.strip() for tactic in tactics):
        raise ValueError("Expected nonempty tactic blocks")
    for tactic in tactics:
        dev._proof_code("```lean4\n" + tactic + "\n```")
        if re.search(r"^\s*(?:theorem|lemma|import|namespace|axiom)\b", dev._masked(tactic), re.MULTILINE):
            raise ValueError("Return tactics for the current goal, not a new declaration")
    return {"tactics": [tactic.strip() for tactic in tactics]}


def proof_body(steps):
    return "\n".join("focus\n" + "\n".join("  " + line for line in step.splitlines()) for step in steps)


def proof_text(task, steps):
    return ("```lean4\n" + task["header"] + " := by\n"
            + "\n".join("  " + line for line in proof_body(steps).splitlines()) + "\n```")


class NativeSession:
    """One native elaboration; state IDs are never reused across processes."""

    def __init__(self, task, context, *, heartbeats, rec_depth, timeout):
        from maths_ai.gnn_inference.scripts import run_minif2f_dev as dev

        self.proc = None
        self.temporary = tempfile.TemporaryDirectory(prefix="atp_native_search_")
        self.lines = queue.Queue()
        self.log = []
        self.spec = "Spec_" + task["id"]
        project, toolchain, lean_path = context
        helper = LEAN_HELPER.read_text().removeprefix("import Lean\n")
        header = re.sub(r"^theorem\s+" + re.escape(task["id"]), "theorem SearchRoot_" + task["id"],
                        task["header"].lstrip(), count=1)
        source = ("import Lean\n" + dev._spec(task, heartbeats, rec_depth) + "\n" + helper
                  + dev._options(heartbeats, rec_depth) + header + " := by\n  atp_native_search\n")
        path = Path(self.temporary.name) / "Session.lean"
        path.write_text(source, encoding="utf-8")
        try:
            self.proc = subprocess.Popen(
                ["elan", "run", toolchain, "lean", str(path)], cwd=project,
                env={**os.environ, "LEAN_PATH": lean_path}, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                bufsize=1, start_new_session=True,
            )
            self.reader = threading.Thread(target=self._read, daemon=True)
            self.reader.start()
            self.root = self._response(timeout)
            if self.root.get("status") != "ok":
                raise RuntimeError(f"Native initialization failed: {self.root}")
        except BaseException:
            self.close()
            raise

    def _read(self):
        for line in self.proc.stdout:
            if line.startswith("ATP_SEARCH "):
                self.lines.put(line[len("ATP_SEARCH "):])
            else:
                self.log.append(line.rstrip())
                del self.log[:-30]
        self.lines.put(None)

    def _response(self, timeout):
        try:
            line = self.lines.get(timeout=max(0.001, timeout))
        except queue.Empty:
            self.close()
            raise TimeoutError("Native Lean request exceeded its wall-time budget") from None
        if line is None:
            raise RuntimeError("Native Lean exited: " + "\n".join(self.log))
        return json.loads(line)

    def request(self, command, state, timeout, **payload):
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError("Native session is not running")
        self.proc.stdin.write(json.dumps({"command": command, "state": state, "spec": self.spec, **payload}, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        return self._response(timeout)

    def close(self):
        if self.proc is not None:
            if self.proc.poll() is None:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.proc.wait()
            self.reader.join(timeout=2)
            for stream in (self.proc.stdin, self.proc.stdout):
                if stream:
                    stream.close()
            self.proc = None
        self.temporary.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def lookup_queries(goals):
    text = "\n".join(goals)
    queries = []
    if "↑" in text and ("ℕ" in text or "Nat" in text):
        queries.append("cast lt")
    if "choose" in text:
        queries.extend(["sum choose", "mul choose"])
    elif "∑" in text or "Finset.sum" in text:
        queries.extend(["sum lt", "sum le"])
    else:
        types = dict(re.findall(r"^([A-Za-z_][A-Za-z_0-9]*) : (Nat|Int|Real|ℕ|ℤ|ℝ)\b", text, re.MULTILINE))
        aliases = {"ℕ": "Nat", "ℤ": "Int", "ℝ": "Real"}
        for receiver, method in re.findall(r"\b([A-Za-z_][A-Za-z_0-9]*)\.([A-Za-z_][A-Za-z_0-9]*)", text):
            namespace = types.get(receiver, receiver if receiver in {"Nat", "Int", "Real", "Finset"} else "")
            namespace = aliases.get(namespace, namespace)
            queries.append(namespace + "." + method if namespace else method)
    return list(dict.fromkeys(queries))[:2]


def messages(task, node, branching):
    feedback = node["feedback"][-3:]
    content = (
        "Return the next Lean tactic for the FIRST current goal. "
        "Start with the Lean code block; omit the mathematical essay and theorem declaration. "
        "Intermediate steps may leave goals; sibling goals are preserved. "
        "Do not use sorry/admit, add assumptions, or repeat previously rejected candidates.\n"
        "Variables/hypotheses printed above ⊢ are ALREADY introduced; do not introduce them again. "
        "Use intro only when the current target itself is a forall or implication. "
        "Use explicit casts when combining Nat bounds with Real inequalities.\n"
        "Example response format:\n```lean4\nassumption\n```\n"
        "Return one such Lean block, JSON {\"tactics\": [\"block1\", \"block2\"]} "
        f"with at most {branching} independent alternatives, or JSON "
        "{\"lookup\": \"sum choose\"} to obtain real declaration names AND types. "
        "Lookup matches space-separated name fragments in the current imported environment; "
        "use it instead of guessing an unfamiliar lemma's signature.\n\n"
        "Original import/open context:\n" + task["prefix"]
        + "\n\nVerified tactic prefix:\n" + (proof_body(node["steps"]) or "(none)")
        + "\n\nALL current goals:\n" + "\n\n".join(node["goals"])
        + "\n\nTyped lookup results:\n" + json.dumps(node["lookups"], ensure_ascii=False)
        + "\n\nRecent rejected candidates / feedback:\n" + json.dumps(feedback, ensure_ascii=False)
    )
    return [{"role": "user", "content": content}]


def search(task, record, generate, validate, save, session_factory, *, budget, branching, seed):
    """Checkpoint graph and raw calls; reconstruct native prefixes after resume/timeouts."""
    if record.get("status") == "passed":
        return record
    started = time.monotonic()
    elapsed = record.get("elapsed_seconds", 0)
    record.setdefault("calls", [])
    record.setdefault("nodes", [])
    record.setdefault("trials", [])
    record.setdefault("duplicates", 0)
    record.setdefault("lookup_cache", {})
    for node in record["nodes"]:
        node.setdefault("expansions", 0)
    record["budget"] = asdict(budget)
    for call in record["calls"]:
        if "raw_text" not in call:
            call.update(status="interrupted", processed=True)
    session, native_ids = None, {}

    def remaining():
        return budget.seconds - elapsed - (time.monotonic() - started)

    def checkpoint():
        record["elapsed_seconds"] = elapsed + time.monotonic() - started
        save()

    def close():
        nonlocal session
        if session is not None:
            session.close()
        session = None
        native_ids.clear()

    def restore(index):
        nonlocal session
        if remaining() <= 0:
            raise TimeoutError("Theorem search budget exhausted")
        if session is None:
            session = session_factory(remaining())
            native_ids[0] = session.root["state"]
        if index not in native_ids:
            state = session.root["state"]
            for step in record["nodes"][index]["steps"]:
                result = session.request("step", state, min(budget.tactic_timeout, remaining()), tactic=step)
                if result["status"] != "ok":
                    raise RuntimeError("Previously accepted prefix did not replay: " + str(result))
                state = result["state"]
            native_ids[index] = state
        return native_ids[index]

    try:
        if remaining() <= 0:
            record.update(status="budget_exhausted", stop_reason="wall_time")
            return record
        if not record["nodes"]:
            session = session_factory(remaining())
            native_ids[0] = session.root["state"]
            record["nodes"].append({"steps": [], "goals": session.root["goals"], "visits": 0, "expansions": 0,
                                    "tried": [], "lookups": [], "feedback": []})
            checkpoint()
        while remaining() > 0:
            pending = next((call for call in record["calls"] if "raw_text" in call and not call.get("processed")), None)
            if pending is not None:
                call = pending
                index, node = call["node"], record["nodes"][call["node"]]
            else:
                if len(record["calls"]) >= budget.model_calls or len(record["trials"]) >= budget.tactic_trials:
                    record["stop_reason"] = "model_call_limit" if len(record["calls"]) >= budget.model_calls else "tactic_trial_limit"
                    break
                eligible = [(i, node) for i, node in enumerate(record["nodes"])
                            if node["expansions"] < budget.visits and len(node["steps"]) < budget.depth
                            and not node.get("blocked")]
                if not eligible:
                    record["stop_reason"] = "frontier_exhausted"
                    break
                index, node = min(eligible, key=lambda item: (item[1]["expansions"], len(item[1]["goals"]),
                                                             -item[1].get("progress", len(item[1]["steps"])), item[0]))
                call = {"node": index, "seed": seed + len(record["calls"]), "status": "started"}
                record["calls"].append(call)
                node["visits"] += 1
                checkpoint()
            state = restore(index)
            if not node.get("auto_lookup_done"):
                for query in lookup_queries(node["goals"]):
                    if query not in record["lookup_cache"]:
                        try:
                            state = restore(index)
                            result = session.request("lookup", state, min(budget.tactic_timeout, remaining()), query=query)
                        except (TimeoutError, RuntimeError) as error:
                            close()
                            result = {"status": "lookup_error", "error": str(error)}
                        record["lookup_cache"][query] = {"query": query, **result, "automatic": True}
                    if not any(hit.get("query") == query for hit in node["lookups"]):
                        node["lookups"].append(record["lookup_cache"][query])
                node["auto_lookup_done"] = True
                checkpoint()
                if remaining() <= 0:
                    record["stop_reason"] = "wall_time"
                    break
                state = restore(index)
            try:
                if "raw_text" not in call:
                    call.update(generate(messages(task, node, branching), call["seed"], remaining()))
                    checkpoint()  # Raw model output survives interruption before execution.
                proposal = parse_proposal(call["raw_text"], branching)
            except TimeoutError as error:
                call.update(status="time_budget_exhausted", processed=True, error=str(error))
                checkpoint()
                break
            except ValueError as error:
                call.update(proposal_error=str(error), processed=True)
                node["feedback"].append({"error": str(error)})
                checkpoint()
                continue
            call["proposal"] = proposal
            if "lookup" in proposal:
                result = session.request("lookup", state, min(budget.tactic_timeout, remaining()), query=proposal["lookup"])
                call["lookup_result"] = result
                node["lookups"].append(result)
                call["processed"] = True
                checkpoint()
                continue
            if not call.get("expanded"):
                node["expansions"] += 1
                call["expanded"] = True
            # Every cumulative prefix uses the original parser tree boundaries.
            # Each success is saved before the next prefix can fail or time out.
            if not call.get("prepared_tactics"):
                groups = []
                for tactic in proposal["tactics"]:
                    before = len(groups)
                    split = session.request("split", state, min(budget.tactic_timeout, remaining()), tactic=tactic)
                    call.setdefault("splits", []).append(split)
                    if split.get("status") == "ok":
                        for key in ("inner_prefixes", "adapted_prefixes", "prefixes"):
                            if split.get(key) and split[key] not in groups:
                                groups.append(split[key][:32])
                    else:
                        groups.append([tactic])
                    if len(groups) == before:
                        groups.append([tactic])
                groups = groups[:branching]
                call["prepared_tactics"] = [tactic for group in groups for tactic in group]
                call["prepared_groups"] = [i for i, group in enumerate(groups) for _ in group]
                call["prefix_lengths"] = [i + 1 for group in groups for i in range(len(group))]
                checkpoint()
            for alternative, tactic in enumerate(call["prepared_tactics"]):
                if alternative in call.get("done_tactics", []):
                    continue
                group = call["prepared_groups"][alternative]
                if group in call.get("failed_groups", []):
                    call.setdefault("done_tactics", []).append(alternative)
                    continue
                trial = next((trial for trial in record["trials"]
                              if trial["seed"] == call["seed"] and trial["alternative"] == alternative), None)
                if remaining() <= 0 or (trial is None and len(record["trials"]) >= budget.tactic_trials):
                    break
                key = candidate_key(tactic)
                if trial is None and (key in node["tried"] or key == "skip"):
                    record["duplicates"] += 1
                    node["feedback"].append({"tactic": tactic, "error": "Repeated candidate; choose another branch"})
                    call.setdefault("done_tactics", []).append(alternative)
                    checkpoint()
                    continue
                if trial is None:
                    node["tried"].append(key)
                    trial = {"node": index, "seed": call["seed"], "alternative": alternative,
                             "tactic": tactic, "status": "started"}
                    record["trials"].append(trial)
                    checkpoint()
                try:
                    state = restore(index)
                    result = session.request("step", state, min(budget.tactic_timeout, remaining()), tactic=tactic)
                except (TimeoutError, RuntimeError) as error:
                    close()
                    result = {"status": "native_error", "error": str(error)}
                trial.update(result)
                if result["status"] != "ok":
                    call.setdefault("failed_groups", []).append(group)
                    node["feedback"].append({"tactic": tactic, "error": result.get("error", str(result))})
                else:
                    steps = node["steps"] + [tactic]
                    if not result["goals"]:
                        raw = proof_text(task, steps)
                        trial["proof"] = raw
                        checkpoint()
                        trial["final_lean"] = validate(task, raw, max(0.001, remaining()))
                        if trial["final_lean"]["status"] == "passed":
                            record.update(status="passed", proof=raw, lean=trial["final_lean"])
                            checkpoint()
                            return record
                        node["feedback"].append({"tactic": tactic, "error": trial["final_lean"]})
                    elif len(record["nodes"]) < budget.nodes:
                        child = len(record["nodes"])
                        native_ids[child] = result["state"]
                        record["nodes"].append({"steps": steps, "goals": result["goals"], "visits": 0, "expansions": 0,
                                                "progress": node.get("progress", 0) + call["prefix_lengths"][alternative],
                                                "tried": [], "lookups": [], "feedback": []})
                call.setdefault("done_tactics", []).append(alternative)
                checkpoint()
            call["processed"] = True
            checkpoint()
        record["status"] = "budget_exhausted"
        record.setdefault("stop_reason", "wall_time")
        checkpoint()
        return record
    finally:
        close()
        checkpoint()
