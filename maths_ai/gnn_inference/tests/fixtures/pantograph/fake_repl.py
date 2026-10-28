#!/usr/bin/env python3
"""Real diagnostic child for owned transport tests; never used by application code."""

import json
from copy import deepcopy
import os
from pathlib import Path
import signal
import sys
import time


def event(kind, **values):
    with open(os.environ["PANTOGRAPH_FAKE_REPORT"], "a", encoding="utf-8") as stream:
        stream.write(json.dumps({"kind": kind, "pid": os.getpid(), **values}) + "\n")


def send(value, split=False):
    raw = json.dumps(value, separators=(",", ":")).encode() + b"\n"
    if split:
        for offset in range(0, len(raw), 3):
            sys.stdout.buffer.write(raw[offset : offset + 3])
            sys.stdout.buffer.flush()
            time.sleep(0.001)
    else:
        sys.stdout.buffer.write(raw)
        sys.stdout.buffer.flush()


def main():
    event(
        "spawn",
        executable=Path(sys.argv[0]).name,
        cwd=os.getcwd(),
        args=sys.argv[1:],
        lean_path=os.environ.get("LEAN_PATH"),
    )
    if Path(sys.argv[0]).name == "stdbuf":
        print("diagnostic non-GNU launcher", flush=True)
        return 0
    if Path(sys.argv[0]).name == "lake":
        mode = os.environ.get("PANTOGRAPH_FAKE_RESOLVER", "normal")
        if mode in ("delay", "ignore_term"):
            if mode == "ignore_term":
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
            time.sleep(5)
        if mode == "failure":
            print("resolver diagnostic", file=sys.stderr, flush=True)
            return 7
        if mode == "stderr_flood":
            sys.stderr.write("x" * 200000 + "HELPER-TAIL")
            sys.stderr.flush()
        if mode == "empty":
            return 0
        if mode == "invalid_utf8":
            sys.stdout.buffer.write(b"\xff\n")
            return 0
        if mode == "oversized":
            print("x" * 100000, flush=True)
            return 0
        print("/fake/lean/path", flush=True)
        return 0

    scenario = os.environ.get("PANTOGRAPH_FAKE_SCENARIO", "normal")
    if scenario == "ignore_term":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if scenario == "ready_delay":
        time.sleep(0.4)
    if scenario == "ready_eof":
        return 0
    if scenario == "stderr_flood":
        sys.stderr.write("x" * 200000 + "TAIL-MARKER")
        sys.stderr.flush()
    print("wrong" if scenario == "bad_ready" else "ready.", flush=True)
    if scenario == "unsolicited":
        send({"unexpected": True})
    options = {
        "printJsonPretty": False,
        "printExprPretty": True,
        "printExprAST": True,
        "printExprModelAST": True,
        "noRepeat": False,
        "automaticMode": True,
        "printDependentMVars": False,
        "printAuxDecls": False,
        "printImplementationDetailHyps": False,
    }
    states = set()
    # Per live state: its root and listed goals, for provenance and inspection.
    roots = {}
    listed = {}
    next_id = 0
    for line in sys.stdin:
        command, raw = line.split(" ", 1)
        args = json.loads(raw)
        event("request", command=command, args=args)
        if command == "protocol.describe":
            result = {
                "protocol": "maths-ai-pantograph",
                "protocolVersion": 2,
                "modelSexpVersion": 1,
                "transitionValidationVersion": 1,
                "transitionProvenanceVersion": 1,
                "inspectionVersion": 1,
                "certificationVersion": 1,
                "messageVersion": 1,
                "goalSelection": "explicit-focused-v1",
                "tacticErrorsFormat": "string-array-v1",
            }
            if scenario == "bad_descriptor":
                result["protocolVersion"] = 99
            if scenario == "unextended":
                result = {
                    "error": "command",
                    "desc": "Unknown command protocol.describe",
                }
            if scenario == "startup_total_delay":
                time.sleep(0.15)
        elif command == "options.set":
            options.update(args)
            result = (
                {}
                if scenario != "options_error"
                else {"error": "command", "desc": "options failed"}
            )
            if scenario == "startup_total_delay":
                time.sleep(0.15)
        elif command == "options.print":
            result = dict(options)
            if scenario == "options_mismatch":
                result["noRepeat"] = True
        elif command == "stat":
            if scenario == "reply_delay":
                time.sleep(0.25)
            if scenario == "malformed":
                sys.stdout.buffer.write(b"{bad}\n")
                sys.stdout.buffer.flush()
                continue
            if scenario == "invalid_utf8":
                sys.stdout.buffer.write(b"\xff\n")
                sys.stdout.buffer.flush()
                continue
            if scenario == "oversized":
                print("x" * 100000, flush=True)
                continue
            if scenario == "partial":
                print('{"nGoals":0}', end="", flush=True)
                return 0
            if scenario == "reply_eof":
                return 0
            result = {"nGoals": False if scenario == "invalid_schema" else len(states)}
        elif command == "goal.start":
            # "universe:u,v" elaborates only when exactly those levels are declared.
            needed = (
                args["expr"].split(":", 1)[1].split(",")
                if args["expr"].startswith("universe:")
                else []
            )
            if args["expr"] == "reject":
                result = {"error": "elab", "desc": "bad expression"}
            elif sorted(args.get("levels", [])) != sorted(needed):
                result = {"error": "elab", "desc": "unknown universe level"}
            else:
                if scenario == "reuse_released_zero" and not states:
                    next_id = 0
                states.add(next_id)
                roots[next_id] = "g" + str(next_id)
                listed[next_id] = [{"name": roots[next_id]}]
                result = {"stateId": next_id, "root": roots[next_id]}
                if scenario != "duplicate_allocation":
                    next_id += 1
        elif command == "goal.delete":
            if scenario == "delete_delay":
                time.sleep(0.25)
            if scenario == "delete_error":
                send({"error": "command", "desc": "deletion unavailable"})
                continue
            states.difference_update(args["stateIds"])
            for state_id in args["stateIds"]:
                roots.pop(state_id, None)
                listed.pop(state_id, None)
            result = {}
        elif command == "goal.tactic":
            if args["stateId"] not in states:
                result = {"error": "invalid", "desc": "unknown state"}
            else:
                fixtures = json.loads(
                    Path(os.environ["PANTOGRAPH_FAKE_CONTRACT"]).read_text()
                )
                tactic = args["tactic"]
                if tactic in (
                    "parse_failure",
                    "tactic_failure",
                    "logged_error_failure",
                ):
                    result = fixtures[tactic]
                elif tactic == "validation_error":
                    result = {"error": "validation", "desc": "inspection unavailable"}
                else:
                    key = {
                        "skip": "identity_intro",
                        "split": "conjunction_split",
                        "trivial": "valid_truth",
                        "sorry": "admission",
                        "missing_validation": "identity_intro",
                        "conv": "identity_intro",
                    }.get(tactic, tactic)
                    result = deepcopy(fixtures["successes"][key])
                    result["nextStateId"] = next_id
                    selected = listed[args["stateId"]][args["goalId"]]["name"]
                    if scenario == "wrong_selected":
                        selected = "unrelated"
                    result["transition"]["selected"] = selected
                    states.add(next_id)
                    roots[next_id] = roots[args["stateId"]]
                    listed[next_id] = result["goals"]
                    next_id += 1
                    if tactic == "missing_validation":
                        result.pop("validation")
                    if tactic == "conv":
                        result["goals"][0]["fragment"] = "conv"
                        result["goals"][0]["isConversion"] = True
                    if scenario == "tactic_delay":
                        time.sleep(0.25)
        elif command in ("goal.inspect", "goal.certify"):
            if args["stateId"] not in states:
                result = {"error": "index", "desc": "unknown state"}
            elif command == "goal.inspect":
                goals = {goal["name"]: goal for goal in listed[args["stateId"]]}
                names = list(args["goals"])
                if scenario == "inspect_mismatch":
                    names.reverse()
                result = {
                    "obligations": [
                        {
                            "name": name,
                            "status": "active",
                            "goal": goals[name],
                            "holes": [],
                            "universeHoles": False,
                        }
                        if "vars" in goals.get(name, {})
                        else {
                            "name": name,
                            "status": "complete",
                            "holes": [],
                            "universeHoles": False,
                        }
                        for name in names
                    ]
                }
            else:
                fixtures = json.loads(
                    Path(os.environ["PANTOGRAPH_FAKE_CONTRACT"]).read_text()
                )
                if listed[args["stateId"]]:
                    result = deepcopy(fixtures["incomplete"])
                else:
                    result = deepcopy(fixtures["certified"])
                    result["receipt"]["root"] = (
                        "unrelated"
                        if scenario == "wrong_root"
                        else roots[args["stateId"]]
                    )
        else:
            result = {"error": "command", "desc": "Unsupported diagnostic command"}
        send(result, scenario == "split")
        if scenario == "broken_pipe" and command == "options.print":
            os.close(0)
            time.sleep(5)
        if scenario == "unsolicited_after_reply" and command == "stat":
            send({"unexpected": True})
        if scenario == "unsolicited_idle" and command == "options.print":
            time.sleep(0.05)
            send({"unexpected": True})
        if scenario == "no_read" and command == "options.print":
            time.sleep(5)
    if scenario == "ignore_term":
        time.sleep(5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
