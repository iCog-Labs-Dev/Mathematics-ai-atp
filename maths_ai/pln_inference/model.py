"""Native PeTTa scoring with exact shared compilation and rule-proof validation."""

from __future__ import annotations

import asyncio
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from maths_ai.data_models.proof_components import STV

from .dependencies import PLNDependencies, resolve_petta

from .processes import QueryProcesses
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
        max_concurrency: int = 2,
        pending_limit: int = 64,
        output_limit: int = 1048576,
        temp_dir: str | None = None,
    ) -> None:
        self.petta_bin = petta_bin
        self.chainer_root = chainer_root
        self.axioms_path = axioms_path
        self.depth = depth
        self.timeout = timeout
        self.fallback_low = fallback_low
        self.fallback_high = fallback_high
        self._rng = random.Random(random_seed)
        self.processes = QueryProcesses(max_concurrency, pending_limit, output_limit, temp_dir)

    async def evaluate_async(self, goal: GoalInput, *, deadline=None) -> PLNResult:
        try:
            compiled = compile_goal(goal)
        except TranslationError as exc:
            return PLNResult(None, "unsupported", False, diagnostic=exc.as_dict())
        source = self._render_query_source(compiled)
        output, status = await self.processes.run(
            resolve_petta(self.petta_bin), source, timeout=self.timeout, deadline=deadline,
        )
        return self._answer(compiled, output, status)

    async def aclose(self):
        await self.processes.aclose()

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
        output, status = self._run_petta(source)
        return self._answer(compiled, output, status)

    def _answer(self, compiled, log_text, launch_status):
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
        """Synchronous entry point over the same owned async process runner."""
        return asyncio.run(self.processes.run(resolve_petta(self.petta_bin), source, timeout=self.timeout))

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
