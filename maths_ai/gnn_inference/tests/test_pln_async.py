from __future__ import annotations

import asyncio
import time
import unittest

from maths_ai.pln_inference.model import PLNInference, PLNResult
from maths_ai.data_models.proof_components import STV


class PLNAsyncTests(unittest.TestCase):
    def test_evaluate_async_matches_sync(self) -> None:
        pln = PLNInference(max_concurrency=4)

        # Stub the blocking evaluate with a deterministic fake result.
        def fake_evaluate(expression, hypotheses=None):
            return PLNResult(stv=STV(strength=0.7, confidence=0.5), status="ok", is_fallback=False)

        pln.evaluate = fake_evaluate  # type: ignore[assignment]

        async def _run():
            return await pln.evaluate_async("x = x", ["h : Nat"])

        result = asyncio.run(_run())
        self.assertEqual(result.stv.strength, 0.7)
        self.assertEqual(result.stv.confidence, 0.5)

    def test_evaluate_async_overlaps_blocking_calls(self) -> None:
        # N concurrent blocking calls (each sleeps D) must finish in ~D, not ~N*D,
        # proving the blocking subprocess.run is off the event-loop thread.
        pln = PLNInference(max_concurrency=8)
        delay = 0.2
        n = 6

        def slow_evaluate(expression, hypotheses=None):
            time.sleep(delay)  # stand-in for the blocking subprocess.run
            return PLNResult(stv=STV(strength=1.0, confidence=1.0), status="ok", is_fallback=False)

        pln.evaluate = slow_evaluate  # type: ignore[assignment]

        async def _run():
            return await asyncio.gather(
                *(pln.evaluate_async(f"g{i}") for i in range(n))
            )

        start = time.perf_counter()
        results = asyncio.run(_run())
        elapsed = time.perf_counter() - start

        self.assertEqual(len(results), n)
        # Serialized would be n*delay = 1.2s; overlapped should be well under half that.
        self.assertLess(elapsed, n * delay * 0.5)


if __name__ == "__main__":
    unittest.main()
