"""Native admission controls. Set FORMAL_NATIVE_PROJECT and FORMAL_NATIVE_REPL."""
import asyncio
import os
from pathlib import Path

import pytest

from maths_ai.formal.admission import source_seed
from maths_ai.formal.source_elaboration import ReplSession, elaborate_source


@pytest.mark.parametrize('case', ['bad-proof', 'bad-suffix', 'effects', 'namespace'])
def test_native_signature_isolation(case, tmp_path):
    project = os.environ.get('FORMAL_NATIVE_PROJECT')
    repl = os.environ.get('FORMAL_NATIVE_REPL')
    if not project or not repl:
        pytest.skip('Set native Lean project and REPL paths to run admission controls')
    marker = tmp_path / 'executed'
    bodies = {
        'bad-proof': 'theorem target : True := by exact (0 : Nat)\n',
        'bad-suffix': 'theorem target : True := by trivial\n#check deliberately_missing_suffix\n',
        'effects': f'theorem target : True := by\n  run_tac liftIO (IO.FS.writeFile "{marker}" "proof")\n  trivial\nrun_cmd liftIO (IO.FS.writeFile "{marker}" "suffix")\n',
        'namespace': 'namespace N\nuniverse u\ntheorem target (α : Type u) (x : α) : x = x := by rfl\nend N\n',
    }
    declaration = 'N.target' if case == 'namespace' else 'target'
    async def run():
        return await elaborate_source(
            lambda: ReplSession(Path(repl), Path(project), ['Init'], timeout_s=30),
            bodies[case], declaration, lambda g: source_seed(g, declaration)[1])
    result = asyncio.run(run())
    assert result['check']['ok'], result
    assert not marker.exists()
