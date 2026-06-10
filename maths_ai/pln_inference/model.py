"""Symbolic (PLN) subgoal evaluation via PeTTaChainer/MeTTa.

Wraps the formula-rendering and STV-parsing utilities already built for the
batch ``translator.py`` pipeline (see ``metta/translator/translator_modules``)
into a single-query interface suitable for the hybrid reasoner's recursive
search loop, which needs to score one subgoal at a time rather than generate
a directory of ``.metta``/``.log`` files.
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from maths_ai.data_models.proof_components import STV

from .metta.translator.translator_modules.normalizer import VariableNormalizer
from .metta.translator.translator_modules.parser import parse_simple_pp
from .metta.translator.translator_modules.renderer import build_query, render_formula
from .metta.translator.translator_modules.runner import (
    extract_stv_scores,
    sample_fallback_score,
    score_from_stv,
)

_AXIOMS_DIR = Path(__file__).parent / "metta" / "axioms"
_DEFAULT_IMPORT_HEADER = _AXIOMS_DIR / "import-pettachainer.metta"
_DEFAULT_AXIOMS_FILE = _AXIOMS_DIR / "metamath_axioms.metta"

_ERROR_MARKERS = ("error", "exception", "failed")


@dataclass(frozen=True)
class PLNResult:
    """Outcome of scoring one subgoal against the symbolic engine.

    ``status`` mirrors the vocabulary the MeTTa translator's ranking mode
    already established (``ok``, ``no_stv_found``, ``log_error_no_stv``,
    ``missing_log``) plus a few that only arise on the single-query path
    (``petta_unavailable``, ``timeout``, ``render_error``). ``is_fallback``
    tells the caller whether ``stv`` is a real PLN result or a random
    exploration score — see the "Important limitation" note in the
    translator readme: a fallback score is *not* evidence of provability.
    """

    stv: STV
    status: str
    is_fallback: bool
    raw_output: str = ""


class PLNInference:
    """
        Evaluate a Lean subgoal's provability via PeTTaChainer (MeTTa/petta).
    """

    def __init__(
        self,
        *,
        petta_bin: Optional[str] = None,
        axioms_path: Optional[Path] = None,
        import_header_path: Optional[Path] = None,
        depth: int = 10,
        timeout: float = 60.0,
        fallback_low: float = 0.0,
        fallback_high: float = 1.0,
        random_seed: Optional[int] = None,
        normalize_variables: bool = False,
    ) -> None:
        self.petta_bin = petta_bin or os.environ.get("PETTA_BIN") or shutil.which("petta")
        self.axioms_path = Path(axioms_path) if axioms_path else _DEFAULT_AXIOMS_FILE
        self.import_header_path = (
            Path(import_header_path) if import_header_path else _DEFAULT_IMPORT_HEADER
        )
        self.depth = depth
        self.timeout = timeout
        self.fallback_low = fallback_low
        self.fallback_high = fallback_high
        self._rng = random.Random(random_seed)
        self._normalizer = VariableNormalizer(normalize=normalize_variables)

     
    # Public API
     
    def evaluate(self, expression: str, hypotheses: Optional[Sequence[str]] = None) -> PLNResult:
        """Score ``expression``'s provability given ``hypotheses``.

        ``expression`` is the Lean target formula (the text after ``⊢``);
        ``hypotheses`` are local-context formula strings, asserted into the
        knowledge base as ``(STV 1.0 1.0)`` facts before the query — exactly
        as the batch translator does for local hypotheses.
        """
        try:
            formula = parse_simple_pp(expression, self._normalizer)
        except Exception as exc:  # malformed/unparseable expression string
            return self._fallback(status="render_error", raw_output=repr(exc))

        if self.petta_bin is None:
            return self._fallback(status="petta_unavailable")

        source = self._render_query_source(formula, hypotheses or [])
        log_text, launch_status = self._run_petta(source)
        if launch_status is not None:
            return self._fallback(status=launch_status)

        stvs = extract_stv_scores(log_text)
        if stvs:
            strength, confidence = max(stvs, key=lambda pair: score_from_stv(*pair))
            return PLNResult(
                stv=STV(strength=strength, confidence=confidence),
                status="ok",
                is_fallback=False,
                raw_output=log_text,
            )

        lowered = log_text.lower()
        status = (
            "log_error_no_stv"
            if any(marker in lowered for marker in _ERROR_MARKERS)
            else "no_stv_found"
        )
        return self._fallback(status=status, raw_output=log_text)

     
    # Internals
     
    def _render_query_source(self, formula: object, hypotheses: Sequence[str]) -> str:
        lines: List[str] = []

        if self.import_header_path.exists():
            lines.append(self.import_header_path.read_text(encoding="utf-8").strip())
        if self.axioms_path.exists():
            lines.append(f"!(import! &self {self.axioms_path})")

        for index, raw_hypothesis in enumerate(hypotheses):
            try:
                hyp_formula = parse_simple_pp(raw_hypothesis, self._normalizer)
            except Exception:
                continue  # skip hypotheses we cannot render rather than abort the query
            lines.append(
                f"!(compileadd kb (: local-hyp-{index} {render_formula(hyp_formula)} (STV 1.0 1.0)))"
            )

        lines.append(build_query(formula, depth=self.depth))
        return "\n\n".join(lines) + "\n"

    def _run_petta(self, source: str) -> Tuple[str, Optional[str]]:
        """Run ``petta`` on rendered MeTTa ``source``.

        Returns ``(captured_output, launch_status)``; ``launch_status`` is
        ``None`` on a successful run (regardless of whether a proof was
        found) or one of ``timeout`` / ``petta_unavailable`` if the process
        itself could not be completed.
        """
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".metta", delete=False, encoding="utf-8"
        )
        try:
            handle.write(source)
            handle.close()

            try:
                completed = subprocess.run(
                    [self.petta_bin, handle.name],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired:
                return "", "timeout"
            except OSError:
                return "", "petta_unavailable"

            return completed.stdout + completed.stderr, None
        finally:
            try:
                os.unlink(handle.name)
            except OSError:
                pass

    def _fallback(self, *, status: str, raw_output: str = "") -> PLNResult:
        score = sample_fallback_score(self._rng, low=self.fallback_low, high=self.fallback_high)
        return PLNResult(
            stv=STV(strength=score, confidence=1.0),
            status=status,
            is_fallback=True,
            raw_output=raw_output,
        )
