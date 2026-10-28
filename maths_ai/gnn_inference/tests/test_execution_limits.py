"""Retention and memory limits."""

from __future__ import annotations

import os
import unittest

from maths_ai.hybrid_reasoner.execution_limits import (
    ResidentMemoryMonitor,
    RetentionLimits,
    resident_mb,
)


class LimitTests(unittest.TestCase):
    def test_limits_are_positive_and_validated(self):
        for bad in (
            {"max_retained_snapshots": 0},
            {"max_retained_snapshots": True},
            {"max_replay_tactic_steps": 1.5},
            {"max_replay_continuations": -1},
            {"max_repl_rss_growth_mb": 0},
            {"max_collection_rss_growth_mb": float("inf")},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                RetentionLimits(**bad)
        with self.assertRaises(ValueError):
            RetentionLimits.from_config({"max_snapshots": 4})
        limits = RetentionLimits.from_config({"max_replay_combinations": 3})
        self.assertEqual(limits.replay_limits().max_combinations, 3)
        self.assertEqual(RetentionLimits.from_config(limits.as_dict()), limits)

    def test_resident_memory_is_sampled_against_a_baseline(self):
        self.assertGreater(resident_mb(), 0)
        self.assertEqual(resident_mb(os.getpid()) is not None, True)
        self.assertIsNone(resident_mb(2**30))
        monitor = ResidentMemoryMonitor(1e9)
        growth = monitor.growth_mb()
        self.assertIsNotNone(growth)
        self.assertFalse(monitor.exceeded())
        blocks = [bytearray(32 * 1024 * 1024) for _ in range(2)]
        strict = ResidentMemoryMonitor(1e-6)
        blocks.append(bytearray(16 * 1024 * 1024))
        for block in blocks:
            block[::4096] = b"x" * len(block[::4096])
        self.assertTrue(strict.exceeded())
        unavailable = ResidentMemoryMonitor(1.0, pid=2**30)
        self.assertFalse(unavailable.exceeded())


if __name__ == "__main__":
    unittest.main()
