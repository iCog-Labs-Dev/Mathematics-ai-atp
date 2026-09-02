"""Tests for DynamicThompsonSampler and parse_and_rank_logs (thompson path)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from runner import DynamicThompsonSampler, ThompsonSampler, parse_and_rank_logs


def _write_manifest(tmp_dir: str, items: list[dict]) -> str:
    path = os.path.join(tmp_dir, "manifest.json")
    with open(path, "w") as f:
        json.dump({"generated_items": items}, f)
    return path


def _write_log(tmp_dir: str, name: str, content: str) -> str:
    path = os.path.join(tmp_dir, f"{name}.log")
    with open(path, "w") as f:
        f.write(content)
    return path


class TestDynamicThompsonSamplerCore(unittest.TestCase):
    def test_initial_prior_is_uniform(self):
        ts = DynamicThompsonSampler()
        self.assertEqual(ts.alpha("new_key"), 1.0)
        self.assertEqual(ts.beta("new_key"), 1.0)

    def test_record_observation_high_score_increases_alpha(self):
        ts = DynamicThompsonSampler()
        ts.record_observation("k", 0.9)
        self.assertGreater(ts.alpha("k"), ts.beta("k"))

    def test_record_observation_low_score_increases_beta(self):
        ts = DynamicThompsonSampler()
        ts.record_observation("k", 0.1)
        self.assertGreater(ts.beta("k"), ts.alpha("k"))

    def test_record_failure_increases_beta(self):
        ts = DynamicThompsonSampler()
        ts.record_failure("k", penalty=1.0)
        self.assertEqual(ts.alpha("k"), 1.0)
        self.assertEqual(ts.beta("k"), 2.0)

    def test_dts_discount_caps_total_at_C(self):
        ts = DynamicThompsonSampler(C=10.0)
        for _ in range(50):
            ts.record_observation("k", 0.8)
        total = ts.alpha("k") + ts.beta("k")
        self.assertAlmostEqual(total, 10.0, places=5)

    def test_score_clamped_outside_0_1(self):
        ts = DynamicThompsonSampler()
        ts.record_observation("k", 5.0)   # clamped to 1.0
        self.assertEqual(ts.alpha("k"), 2.0)
        ts2 = DynamicThompsonSampler()
        ts2.record_observation("k2", -3.0)  # clamped to 0.0
        self.assertEqual(ts2.beta("k2"), 2.0)

    def test_sample_returns_value_in_0_1(self):
        import random
        ts = DynamicThompsonSampler()
        rng = random.Random(42)
        for _ in range(100):
            v = ts.sample("k", rng)
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_subgoal_key_format(self):
        item = {"test_name": "modus_ponens", "goal_index": 2}
        self.assertEqual(ThompsonSampler.subgoal_key(item), "modus_ponens_goal_2")

    def test_state_dict_roundtrip(self):
        ts = DynamicThompsonSampler()
        ts.record_observation("k", 0.7)
        state = ts.state_dict
        ts2 = DynamicThompsonSampler(state=state)
        self.assertAlmostEqual(ts2.alpha("k"), ts.alpha("k"))
        self.assertAlmostEqual(ts2.beta("k"), ts.beta("k"))

    def test_save_and_load(self):
        ts = DynamicThompsonSampler()
        ts.record_observation("k", 0.6)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "state.json")
            ts.save_to(path)
            ts2 = DynamicThompsonSampler.load_from(path)
        self.assertAlmostEqual(ts2.alpha("k"), ts.alpha("k"))


class TestParseAndRankLogsThompson(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp)

    def _item(self, name: str, goal: int, log_content: str) -> dict:
        log = _write_log(self.tmp, f"{name}_{goal}", log_content)
        return {
            "test_name": name,
            "goal_index": goal,
            "metta_path": f"/fake/{name}_{goal}.metta",
            "log_path": log,
        }

    def test_stv_found_updates_sampler_alpha(self):
        """When STV is present, record_observation should push alpha up."""
        item = self._item("t", 0, "(STV 0.9 0.9)")
        manifest = _write_manifest(self.tmp, [item])
        ts = DynamicThompsonSampler()
        ranked = parse_and_rank_logs(manifest, fallback_strategy="thompson", thompson_sampler=ts)
        self.assertEqual(ranked[0]["status"], "ok")
        self.assertGreater(ts.alpha("t_goal_0"), ts.beta("t_goal_0"))

    def test_no_stv_updates_sampler_beta(self):
        """When no STV, record_failure should push beta up."""
        item = self._item("t", 0, "no proof here")
        manifest = _write_manifest(self.tmp, [item])
        ts = DynamicThompsonSampler()
        ranked = parse_and_rank_logs(manifest, fallback_strategy="thompson", thompson_sampler=ts)
        self.assertGreater(ts.beta("t_goal_0"), ts.alpha("t_goal_0"))

    def test_thompson_fallback_scores_assigned_when_no_stv_globally(self):
        """All logs have no STV → fallback scoring uses thompson posterior."""
        items = [self._item("t", i, "nothing") for i in range(3)]
        manifest = _write_manifest(self.tmp, items)
        ranked = parse_and_rank_logs(manifest, fallback_strategy="thompson", random_seed=42)
        for r in ranked:
            self.assertEqual(r["status"], "thompson_fallback_no_stv_global")
            self.assertGreater(r["score"], 0.0)

    def test_thompson_not_triggered_with_default_strategy(self):
        """Default strategy='random' means ts stays None — no thompson fields."""
        item = self._item("t", 0, "no stv")
        manifest = _write_manifest(self.tmp, [item])
        ranked = parse_and_rank_logs(manifest)
        self.assertNotIn("thompson_alpha", ranked[0])

    def test_mixed_stv_and_no_stv_no_fallback_applied(self):
        """If at least one subgoal has an STV, fallback block is skipped entirely."""
        items = [
            self._item("t", 0, "(STV 0.8 0.9)"),
            self._item("t", 1, "nothing"),
        ]
        manifest = _write_manifest(self.tmp, items)
        ranked = parse_and_rank_logs(manifest, fallback_strategy="thompson")
        statuses = {r["goal_index"]: r["status"] for r in ranked}
        self.assertEqual(statuses[0], "ok")
        # goal 1 gets no_stv_found but NOT thompson_fallback_no_stv_global
        self.assertNotEqual(statuses[1], "thompson_fallback_no_stv_global")

    def test_ranking_sorted_by_score_descending(self):
        items = [
            self._item("t", 0, "(STV 0.3 0.5)"),
            self._item("t", 1, "(STV 0.9 0.9)"),
            self._item("t", 2, "(STV 0.1 0.2)"),
        ]
        manifest = _write_manifest(self.tmp, items)
        ranked = parse_and_rank_logs(manifest, fallback_strategy="thompson")
        scores = [r["score"] for r in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_state_persisted_to_file(self):
        item = self._item("t", 0, "(STV 0.7 0.8)")
        manifest = _write_manifest(self.tmp, [item])
        state_path = os.path.join(self.tmp, "ts_state.json")
        parse_and_rank_logs(
            manifest,
            fallback_strategy="thompson",
            thompson_state_output=state_path,
        )
        self.assertTrue(os.path.exists(state_path))
        with open(state_path) as f:
            state = json.load(f)
        self.assertIn("t_goal_0", state)

    def test_state_reloaded_carries_over_across_calls(self):
        """Sampler passed between calls retains learning."""
        ts = DynamicThompsonSampler()
        for i in range(5):
            item = self._item("t", i, "(STV 0.9 0.9)")
            manifest = _write_manifest(self.tmp, [item])
            parse_and_rank_logs(manifest, fallback_strategy="thompson", thompson_sampler=ts)

        # After 5 successes alpha should be well above beta for those keys
        for i in range(5):
            self.assertGreater(ts.alpha(f"t_goal_{i}"), ts.beta(f"t_goal_{i}"))

    def test_missing_log_skipped_in_fallback(self):
        items = [
            {"test_name": "t", "goal_index": 0, "metta_path": "/fake.metta", "log_path": "/nonexistent.log"},
            self._item("t", 1, "nothing"),
        ]
        manifest = _write_manifest(self.tmp, items)
        ranked = parse_and_rank_logs(manifest, fallback_strategy="thompson", random_seed=0)
        status_map = {r["goal_index"]: r["status"] for r in ranked}
        self.assertEqual(status_map[0], "missing_log")
        self.assertEqual(status_map[1], "thompson_fallback_no_stv_global")


if __name__ == "__main__":
    unittest.main()
