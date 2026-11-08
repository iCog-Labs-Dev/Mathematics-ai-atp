"""Native header extraction and conversion ownership regression controls."""

import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess

import pytest
from pantograph_async import PantographEnv

from maths_ai.gnn_inference.atp_lean_gnn import minif2f_conversion as conversion
from maths_ai.gnn_inference.atp_lean_gnn.goal_file_records import decode_goal_record, execution_goal_record
from maths_ai.hybrid_reasoner.pantograph_model_sexpr import create_model_sexpr_client, materialize_seed
from maths_ai.lean_replay import render_replay


@pytest.fixture
def native_env(request):
    root, repl = os.environ.get("PANTOGRAPH_TEST_SOURCE_ROOT"), os.environ.get("PANTOGRAPH_TEST_REPL")
    if not root or not repl:
        if request.config.getoption("--run-pantograph-live"):
            pytest.fail("Native conversion needs PANTOGRAPH_TEST_SOURCE_ROOT and PANTOGRAPH_TEST_REPL")
        pytest.skip("Explicit native Lean paths not configured")
    return PantographEnv(source_root=Path(root), pantograph_repl=Path(repl),
                         imports=("Init", "Mathlib"), server_start_timeout_s=300)


SOURCE = '''import MiniF2F.Minif2fImport
open Nat
/- A comment containing := and : does not delimit declarations. -/
theorem binders.{u} {α : Type u} [Inhabited α] (x : α)
    (h : True) : ∀ n : Nat, n = n := by
  run_tac throwError "proof bodies must never execute"
theorem «quoted:name» (n m : Nat) : n + m = m + n := by sorry
theorem missing : unknownBenchmarkConstant := by sorry
'''


@pytest.mark.pantograph_live
def test_native_header_roles_roundtrip_and_proof_nonexecution(native_env):
    async def run():
        header, rows = await conversion.run_helper(SOURCE, "import Mathlib\n", native_env)
        assert header["context_commands"] == ["open Nat"]
        assert len(rows) == 3
        first, quoted, missing = rows
        assert first["status"] == quoted["status"] == "ok"
        assert missing["status"] == "incompatible"
        p = first["problem"]
        assert p["levels"] == ["u"]
        assert [x["binder_role"] for x in p["locals"]] == [
            ":implicit", ":instance-implicit", ":explicit", ":explicit"]
        changed = SOURCE.replace('run_tac throwError "proof bodies must never execute"', "sorry")
        _, again = await conversion.run_helper(changed, "import Mathlib\n", native_env)
        assert again[0]["problem"]["expression"] == p["expression"]
        client = await create_model_sexpr_client(native_env)
        try:
            async with client.state_scope() as scope:
                original = await scope.start_goal(p["expression"], levels=p["levels"])
                state = await scope.apply_tactic(original, "intro α inst x h")
                assert len(state.goals[0].locals) == 4
                assert "∀" in state.goals[0].target.pp
                record = execution_goal_record(state.goals[0], source={"problem_id":"control"})
                seed = decode_goal_record(record, path=Path("control.jsonl"), line=1)
                replay = render_replay(seed)
                await materialize_seed(scope, seed)
                _, valid = await conversion.run_helper(SOURCE, "import Mathlib\n", native_env,
                                                       checks={"binders": replay.expression})
                assert valid[0]["problem"]["checked"]
                _, invalid = await conversion.run_helper(SOURCE, "import Mathlib\n", native_env,
                                                         checks={"binders": "False"})
                assert invalid[0]["status"] == "incompatible"
                # The initial state is executable, independent of the supplied proof.
                result = await scope.apply_tactic(state, "intro n; rfl")
                assert not result.goals
                certificate = await scope.certify(result)
                assert certificate is not None
            assert (await client.stat()).n_goals == 0
        finally:
            await client.aclose()
    asyncio.run(run())


@pytest.mark.pantograph_live
@pytest.mark.parametrize("source,wrapper", [
    ("import MiniF2F.Minif2fImport\nnamespace Unsafe\n", "import Mathlib\n"),
    ("import MiniF2F.Test\n", "import Mathlib\n"),
    ("import MiniF2F.Minif2fImport\n", "import Mathlib\naxiom injected : False\n"),
    ("import MiniF2F.Minif2fImport\ntheorem malformed ( : True := by sorry", "import Mathlib\n"),
])
def test_native_rejects_unsupported_context_and_contamination(native_env, source, wrapper):
    with pytest.raises(conversion.ConversionInfrastructureError):
        asyncio.run(conversion.run_helper(source, wrapper, native_env))


def test_source_bytes_must_match_git_revision(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for name in conversion.SOURCE_FILES:
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text("original")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Test", "-c", "user.email=test@example.org",
                    "commit", "-qm", "fixture"], check=True)
    revision = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"]).decode().strip()
    assert len(conversion.verify_source(tmp_path, revision)) == len(conversion.SOURCE_FILES)
    (tmp_path / "MiniF2F/Valid.lean").write_text("changed")
    with pytest.raises(ValueError, match="Consumed source"):
        conversion.verify_source(tmp_path, revision)


def test_helper_timeout_and_output_bounds_reap_processes(tmp_path, monkeypatch):
    # Exercise the real subprocess reader/cleanup without requiring Lean.
    lake = tmp_path / "lake"
    lake.write_text("#!/bin/sh\nprintf '%s' \"$$\" > \"$HELPER_PID\"\nexec sleep 30\n")
    lake.chmod(0o755)
    pidfile = tmp_path / "pid"
    monkeypatch.setenv("PATH", str(tmp_path) + ":" + os.environ["PATH"])
    monkeypatch.setenv("HELPER_PID", str(pidfile))
    env = PantographEnv(source_root=tmp_path, pantograph_repl=tmp_path / "unused")
    with pytest.raises(TimeoutError):
        asyncio.run(conversion.run_helper("", "", env, timeout_s=0.1))
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)
    lake.write_text("#!/bin/sh\nyes excessive\n")
    with pytest.raises(conversion.ConversionInfrastructureError, match="byte limit"):
        asyncio.run(conversion.run_helper("", "", env, max_bytes=100))
