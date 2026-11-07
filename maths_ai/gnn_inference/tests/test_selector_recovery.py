"""Cached-only selector coverage keeps immutable denominators and delta selection."""

import json

import pytest

from maths_ai.gnn_inference.atp_lean_gnn import semantic_templates
from maths_ai.gnn_inference.scripts import audit_scoped_recovery as scoped
from maths_ai.gnn_inference.scripts import audit_selector_recovery as selectors
from maths_ai.gnn_inference.tests.test_scoped_recovery import frozen_v5
from maths_ai.gnn_inference.tests.test_template_coverage import _snapshot


@pytest.fixture
def frozen_v6(tmp_path, frozen_v5):
    previous, cache = frozen_v5
    output = tmp_path / "v6"
    assert scoped.main(["--previous-recovery-dir", str(previous), "--output-dir", str(output)]) == 0
    return output, cache


def test_selector_audit_uses_no_semantic_cache_and_counts_only_new_rows(tmp_path, frozen_v6, monkeypatch, capsys):
    previous, _ = frozen_v6
    output = tmp_path / "v7"
    before = _snapshot(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("syntax-only V7 must not access V2 semantic traces")

    monkeypatch.setattr(semantic_templates, "load_semantic_trace", forbidden)
    assert selectors.main(["--previous-recovery-dir", str(previous), "--output-dir", str(output)]) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["complete"] and report["manifest"]["semantic_cache_used"] is False
    for split in ("train", "val"):
        summary = report["splits"][split]["summary"]
        assert summary["selected_steps"] == 7 and summary["quarantined_steps"] == 1
        assert summary["baseline_v6_renderable_steps"] == 5 and summary["template_v7_renderable_steps"] == 6
        assert summary["v7_rescues_vs_v6"] == summary["exact_aligned_v7_rescues"] == 1
        assert summary["v7_losses_vs_v6"] == summary["exact_aligned_v7_losses"] == 0
        assert summary["method_selector_steps"] == summary["method_selector_nodes"] == 2
        assert summary["method_selector_rescues"] == 1
    assert json.loads((output / "new_val_pilot_rows.json").read_text()) == [5]
    after = _snapshot(tmp_path)
    assert all(after[name] == digest for name, digest in before.items())
    assert all(name in before or name.startswith("v7/") for name in after)
    assert "old_support" in capsys.readouterr().out


def test_selector_audit_rejects_changed_cache_and_reports_incomplete(tmp_path, frozen_v6):
    previous, cache = frozen_v6
    path = cache / "val/action_trace_v3/000000005.json"
    path.write_bytes(path.read_bytes() + b" ")
    output = tmp_path / "changed"
    with pytest.raises(ValueError, match="cache changed"):
        selectors.main(["--previous-recovery-dir", str(previous), "--output-dir", str(output)])
    assert not json.loads((output / "report.json").read_text())["complete"]


def test_selector_audit_rejects_tampered_previous_rows(tmp_path, frozen_v6):
    previous, _ = frozen_v6
    path = previous / "rows.jsonl"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="immutable V6"):
        selectors.main(["--previous-recovery-dir", str(previous), "--output-dir", str(tmp_path / "changed")])
