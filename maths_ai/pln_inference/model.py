"""Native PeTTa scoring with exact shared compilation and rule-proof validation."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import random
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from maths_ai.data_models.proof_components import STV

from .dependencies import PLNDependencies, resolve_petta

from .compiler import (
    GoalInput,
    CompiledGoal,
    TranslationError,
    compile_goal,
    compile_text,
    render,
)
from .answers import rule_answers
from .metta.translator.translator_modules.runner import (
    sample_fallback_score,
    score_from_stv,
)

_ERROR_MARKERS = ("error", "exception", "failed")


@dataclass(frozen=True)
class PLNResult:
    """Only ``ok`` is a target-matching rule derivation with an attached STV.

    Unsupported translations and engine failures have no STV. A completed query
    with no derivation may carry an explicitly marked exploration fallback.
    """

    stv: STV | None
    status: str
    is_fallback: bool
    raw_output: str = ""
    diagnostic: dict | None = None


class PLNInference:
    """
    Evaluate a Lean subgoal's provability via PeTTaChainer (MeTTa/petta).
    """

    def __init__(
        self,
        *,
        petta_bin: Optional[str] = None,
        axioms_path: Optional[Path] = None,
        chainer_root: Optional[Path] = None,
        depth: int = 10,
        timeout: float = 60.0,
        fallback_low: float = 0.0,
        fallback_high: float = 1.0,
        random_seed: Optional[int] = None,
        max_concurrency: int = 8,
    ) -> None:
        self.petta_bin = petta_bin
        self.chainer_root = chainer_root
        self.axioms_path = axioms_path
        self.depth = depth
        self.timeout = timeout
        self.fallback_low = fallback_low
        self.fallback_high = fallback_high
        self._rng = random.Random(random_seed)
        # The bounded executor is created per active wave of evaluations and shut down
        # when the final queued evaluation completes.
        self._max_concurrency = max_concurrency
        self._executor: ThreadPoolExecutor | None = None
        self._active_evaluations = 0

    # Public API

    async def evaluate_async(
        self,
        goal: GoalInput,
        *,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> PLNResult:
        """Non-blocking ``evaluate``: run the blocking subprocess in a bounded thread pool.

        ``evaluate`` calls ``subprocess.run`` (up to ``timeout`` seconds), which does not
        yield and would freeze the event loop — stalling every other concurrent proof
        search. ``subprocess.run`` releases the GIL while waiting on the ``petta`` process,
        so a thread executor gives genuine concurrency here. The pool is bounded by
        ``max_concurrency`` so many searches cannot spawn unbounded ``petta`` processes.
        """
        running_loop = asyncio.get_running_loop()
        if loop is not None and loop is not running_loop:
            raise ValueError(
                "evaluate_async received an event loop that is not running here."
            )
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self._max_concurrency)
        future = self._executor.submit(self.evaluate, goal)
        self._active_evaluations += 1
        try:
            # Polling keeps the event loop responsive and avoids depending on executor
            # callback wakeups, which are unavailable in some sandboxed runtimes.
            while not future.done():
                await asyncio.sleep(0.005)
            return future.result()
        except asyncio.CancelledError:
            future.cancel()
            raise
        finally:
            self._active_evaluations -= 1
            if self._active_evaluations == 0 and self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
                self._executor = None

    def evaluate(self, goal: GoalInput) -> PLNResult:
        """Score a structured goal; unsupported translation has no STV."""
        try:
            compiled = compile_goal(goal)
        except TranslationError as exc:
            return PLNResult(None, "unsupported", False, diagnostic=exc.as_dict())
        return self._evaluate_compiled(compiled)

    def evaluate_text(
        self, target: str, hypotheses: tuple[str, ...], *, propositions: dict[str, str]
    ) -> PLNResult:
        """Explicit propositional text API; never a structured-decoding fallback."""
        try:
            compiled = compile_text(target, hypotheses, propositions=propositions)
        except TranslationError as exc:
            return PLNResult(None, "unsupported", False, diagnostic=exc.as_dict())
        return self._evaluate_compiled(compiled)

    def _evaluate_compiled(self, compiled: CompiledGoal) -> PLNResult:
        source = self._render_query_source(compiled)
        log_text, launch_status = self._run_petta(source)
        if launch_status is not None:
            return PLNResult(None, launch_status, False, log_text)
        scores = rule_answers(log_text, render(compiled.target))
        if scores:
            strength, confidence = max(scores, key=lambda pair: score_from_stv(*pair))
            return PLNResult(
                STV(strength=strength, confidence=confidence),
                "ok",
                False,
                log_text,
                {"symbols": compiled.symbols()},
            )
        status = (
            "log_error_no_stv"
            if any(marker in log_text.lower() for marker in _ERROR_MARKERS)
            else "no_stv_found"
        )
        if status == "log_error_no_stv":
            return PLNResult(None, status, False, log_text)
        return self._fallback(status=status, raw_output=log_text)

    def _render_query_source(self, compiled: CompiledGoal) -> str:
        dependencies = PLNDependencies.resolve(self.chainer_root, self.axioms_path)
        return dependencies.imports() + "\n\n" + compiled.source(depth=self.depth)

    def _run_petta(self, source: str) -> Tuple[str, Optional[str]]:
        """Run ``petta`` on rendered MeTTa ``source``.

        Returns ``(captured_output, launch_status)``; ``launch_status`` is
        ``None`` on a zero exit (regardless of whether a proof was found).
        Failed launches and nonzero exits retain diagnostics. Configuration
        errors raise before starting a subprocess.
        """
        executable = resolve_petta(self.petta_bin)
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".metta", delete=False, encoding="utf-8"
        )
        try:
            handle.write(source)
            handle.close()

            try:
                completed = subprocess.run(
                    [executable, handle.name],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired as exc:

                def decode(value):
                    return (
                        value.decode(errors="replace")
                        if isinstance(value, bytes)
                        else (value or "")
                    )

                return decode(exc.stdout) + decode(exc.stderr), "timeout"
            except OSError as exc:
                return str(exc), "petta_unavailable"

            output = completed.stdout + completed.stderr
            if completed.returncode:
                return (
                    f"PeTTa exited with code {completed.returncode}.\n{output}",
                    "engine_error",
                )
            return output, None
        finally:
            try:
                os.unlink(handle.name)
            except OSError:
                pass

    def _fallback(self, *, status: str, raw_output: str = "") -> PLNResult:
        score = sample_fallback_score(
            self._rng, low=self.fallback_low, high=self.fallback_high
        )
        return PLNResult(
            stv=STV(strength=score, confidence=1.0),
            status=status,
            is_fallback=True,
            raw_output=raw_output,
        )
