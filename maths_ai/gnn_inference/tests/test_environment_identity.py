"""The environment identity binds a REPL by its source, and still by its bytes."""

import subprocess

import pytest

from maths_ai.gnn_inference.atp_lean_gnn.deployment_migration import verify_previous_environment
from maths_ai.gnn_inference.atp_lean_gnn.retrieval_artifacts import (
    COMPONENTS_V1, COMPONENTS_V2, RetrievalError, environment_component_variants,
    environment_components, environment_fingerprints, fingerprint_environment_components,
    repl_source,
)

IMPORTS = ("Init", "Mathlib")


def git(root, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@example.org",
                    *args], check=True, capture_output=True)


def checkout(tmp_path, name="pantograph", binary=b"build one"):
    root = tmp_path / name
    root.mkdir()
    (root / "lean-toolchain").write_text("leanprover/lean4:v4.10.0-rc1\n")
    (root / "lakefile.lean").write_text("import Lake\n")
    (root / ".gitignore").write_text("/.lake\n")
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "commit", "-qm", "source")
    repl = root / ".lake/build/bin/repl"
    repl.parent.mkdir(parents=True)
    repl.write_bytes(binary)
    return root, repl


@pytest.fixture
def library(tmp_path):
    root = tmp_path / "mathlib"
    (root / ".lake/build/lib").mkdir(parents=True)
    (root / "lean-toolchain").write_text("leanprover/lean4:v4.10.0-rc1\n")
    (root / "lake-manifest.json").write_text("{}")
    (root / ".lake/build/lib/Fixture.olean").write_bytes(b"compiled library")
    return root


def test_a_clean_checkout_identifies_its_repl_by_commit(tmp_path):
    root, repl = checkout(tmp_path)
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    assert repl_source(repl)["commit"] == commit
    # Build products and other untracked files do not change the source.
    (root / "notes.txt").write_text("untracked")
    assert repl_source(repl)["commit"] == commit
    (root / "lakefile.lean").write_text("import Lake -- edited\n")
    assert repl_source(repl) is None


def test_a_repl_outside_a_git_checkout_has_no_source_identity(tmp_path):
    plain = tmp_path / "plain/.lake/build/bin/repl"
    plain.parent.mkdir(parents=True)
    plain.write_bytes(b"binary")
    (tmp_path / "plain/lean-toolchain").write_text("leanprover/lean4:v4.10.0-rc1\n")
    assert repl_source(plain) is None
    loose = tmp_path / "repl"
    loose.write_bytes(b"binary")
    assert repl_source(loose) is None


def test_a_new_commit_is_a_new_environment(tmp_path, library):
    root, repl = checkout(tmp_path)
    v2, v1 = environment_component_variants(library, repl, IMPORTS)
    assert (v2["schema"], v1["schema"]) == (COMPONENTS_V2, COMPONENTS_V1)
    (root / "README.md").write_text("change")
    git(root, "add", ".")
    git(root, "commit", "-qm", "next")
    after = environment_components(library, repl, IMPORTS)
    assert after["repl_source"]["commit"] != v2["repl_source"]["commit"]
    assert fingerprint_environment_components(after) != fingerprint_environment_components(v2)


def test_identical_sources_agree_across_checkouts(tmp_path, library):
    first_root, first = checkout(tmp_path, "first", b"one")
    second_root = tmp_path / "second"
    git(tmp_path, "clone", "-q", str(first_root), str(second_root))
    second = second_root / ".lake/build/bin/repl"
    second.parent.mkdir(parents=True)
    second.write_bytes(b"other bytes")
    assert environment_fingerprints(library, first, IMPORTS)[0] == environment_fingerprints(library, second, IMPORTS)[0]
    assert environment_fingerprints(library, first, IMPORTS)[1] != environment_fingerprints(library, second, IMPORTS)[1]


def test_a_byte_bound_inventory_still_verifies_on_its_own_executable(tmp_path, library):
    _, repl = checkout(tmp_path)
    v2, v1 = environment_component_variants(library, repl, IMPORTS)
    accepted = environment_fingerprints(library, repl, IMPORTS)
    assert fingerprint_environment_components(v1) in accepted
    assert fingerprint_environment_components(v2) in accepted


def test_a_byte_bound_inventory_migrates_to_a_source_bound_one(tmp_path, library):
    _, old = checkout(tmp_path, "old", b"published executable")
    _, new = checkout(tmp_path, "new", b"container build")
    recorded = environment_component_variants(library, old, IMPORTS)[1]
    inventory = {"environment_identity": fingerprint_environment_components(recorded),
                 "environment_components": recorded}
    current = environment_components(library, new, IMPORTS)
    assert current["schema"] == COMPONENTS_V2
    assert verify_previous_environment(inventory, current, old_repl=old) == recorded
    with pytest.raises(RetrievalError, match="Old executable differs"):
        verify_previous_environment(inventory, current, old_repl=new)
    (library / ".lake/build/lib/Fixture.olean").write_bytes(b"rebuilt library")
    changed = environment_components(library, new, IMPORTS)
    with pytest.raises(RetrievalError, match="not a REPL-only migration"):
        verify_previous_environment(inventory, changed)


def test_malformed_source_evidence_is_rejected(tmp_path, library):
    _, repl = checkout(tmp_path)
    components = environment_components(library, repl, IMPORTS)
    for source in ({"commit": "abc", "toolchain_sha256": "0" * 64},
                   {"commit": "a" * 40},
                   {"commit": "a" * 40, "toolchain_sha256": "0" * 64, "extra": 1}):
        with pytest.raises(RetrievalError):
            fingerprint_environment_components({**components, "repl_source": source})
    with pytest.raises(RetrievalError, match="Incomplete"):
        fingerprint_environment_components({**components, "repl_sha256": "0" * 64})
