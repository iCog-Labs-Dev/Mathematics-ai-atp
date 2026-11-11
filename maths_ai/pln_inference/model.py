"""Symbolic (PLN) subgoal evaluation via PeTTaChainer/MeTTa.

Wraps the formula-rendering and STV-parsing utilities already built for the
batch ``translator.py`` pipeline (see ``metta/translator/translator_modules``)
into a single-query interface suitable for the hybrid reasoner's recursive
search loop, which needs to score one subgoal at a time rather than generate
a directory of ``.metta``/``.log`` files.

How queries are executed
========================

``petta`` (the PeTTa MeTTa interpreter) resolves ``import!`` paths relative to
the directory of the *top-level* ``.metta`` file, not relative to the importing
module. PeTTaChainer's modules import their siblings by bare name
(``!(import! &self compile)`` etc.), so a query must run from a directory that
contains (or links) the PeTTaChainer ``.metta`` modules. ``PLNInference``
therefore prepares a *workspace*: a directory that links every entry of the
vendored PeTTaChainer sources (``PETTA_CHAINER_DIR``, default the Docker
layout ``/opt/pettachainer/pettachainer/metta``; not part of the image if not
present). Query files are generated inside that workspace
(``PETTA_WORKSPACE``) so the relative chainer import in
``metta/axioms/import-pettachainer.metta`` resolves.

Random fallback scores
======================

When PeTTa or PeTTaChainer cannot produce an STV, ``evaluate()`` returns
``is_fallback=True`` with a *deterministic* zero-evidence value
(``STV 0.0 0.0``). Sampling random exploration scores that mimic evidence is
disabled by default; it can be re-enabled explicitly with
``fallback_enabled=True`` (or the ``MATHS_AI_PLN_FALLBACK`` environment
variable) for experiments that deliberately substitute DTS exploration noise.

What counts as a "real" result
==============================

A non-fallback result requires an STV for the queried formula in the output
PeTTa emitted *after* the query started executing (see
:func:`_query_output`). STVs printed earlier — compiled axioms,
``compileadd`` hypothesis facts — are knowledge-base state, not query
results, and are never reused. Consequently only formulas the chainer can
actually derive from the knowledge base return real STVs (e.g. ``P`` given
hypothesis ``P``). Formulas with no derivation path — bare ``True``, ``P``
without evidence, contradictions — return the zero-evidence fallback with
``status="no_stv_found"``; they are *not* errors.
"""

from __future__ import annotations

import os
import random
import re
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
    extract_stv_scores_for_formula,
    sample_fallback_score,
    score_from_stv,
)

_AXIOMS_DIR = Path(__file__).parent / "metta" / "axioms"
_DEFAULT_IMPORT_HEADER = _AXIOMS_DIR / "import-pettachainer.metta"
_DEFAULT_AXIOMS_FILE = _AXIOMS_DIR / "metamath_axioms.metta"

# Vendored PeTTaChainer sources (Docker layout). Imported directly instead of
# via `git-import!` at query time, which would require network access.
_DEFAULT_CHAINER_DIR = "/opt/pettachainer/pettachainer/metta"

# A successful petta run (exit code 0) with no query STV is the *normal*
# no-proof outcome, not an error. Only line-anchored runtime failures in the
# query's own output count as errors: bare words like "error"/"failed" also
# appear in imported-source and compiler chatter, which must not flip the
# classification (see `_query_output`).
_ERROR_LINE_PATTERN = re.compile(
    r"^\s*(?:ERROR:|Exception|Traceback \(most recent call last\))",
    re.MULTILINE,
)

# The interpreter compiles ``!(query ...)`` runnables into a Prolog
# ``findall(A, query(...), ...)`` goal and prints it; its presence separates
# "the query ran through the PeTTaChainer language" from an unevaluated echo.
_QUERY_GOAL_PATTERN = re.compile(r"findall\(\s*[A-Z_0-9]+\s*,\s*query\(")

# The interpreter echoes the ``!(query ...)`` runnable before evaluating it.
_QUERY_RUNNABLE_PATTERN = re.compile(r"!\(\s*query\s")

# Startup probe: assert P as a known fact, then query P. An arbitrary
# proposition without hypotheses is generally not provable, so the readiness
# probe queries a fact it just asserted — the minimal query that must yield
# a real STV on a healthy PeTTa + PeTTaChainer install.
_PROBE_FORMULA = "P"


def _query_output(log_text: str) -> str:
    """Return the portion of ``log_text`` emitted after the query starts.

    PeTTa echoes the ``!(query ...)`` runnable and/or prints its compiled
    Prolog ``findall(A, query(...), ...)`` goal when the query begins.
    Everything before the *last* such marker is query setup — imports,
    compiled axioms, ``compileadd`` hypothesis facts — and may legitimately
    contain the queried formula with a perfect STV. Only post-marker output
    may be trusted as a query result. When no marker appears the query never
    ran through the chainer and there is no query output at all.
    """
    cut = -1
    for pattern in (_QUERY_RUNNABLE_PATTERN, _QUERY_GOAL_PATTERN):
        for match in pattern.finditer(log_text):
            cut = max(cut, match.start())
    return log_text[cut:] if cut >= 0 else ""


@dataclass(frozen=True)
class PLNResult:
    """Outcome of scoring one subgoal against the symbolic engine.

    ``status`` mirrors the vocabulary the MeTTa translator's ranking mode
    already established (``ok``, ``no_stv_found``, ``log_error_no_stv``,
    ``missing_log``) plus a few that only arise on the single-query path
    (``petta_unavailable``, ``timeout``, ``render_error``). ``is_fallback``
    tells the caller whether ``stv`` is a real PLN result or a zero-evidence
    placeholder — see the "Important limitation" note in the translator
    readme: a fallback score is *not* evidence of provability.
    """

    stv: STV
    status: str
    is_fallback: bool
    raw_output: str = ""

    @property
    def score(self) -> float:
        """Conventional PLN ranking score: strength × confidence."""
        return self.stv.score


def _env_flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


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
        chainer_dir: Optional[Path] = None,
        workspace_dir: Optional[Path] = None,
        depth: int = 10,
        timeout: float = 60.0,
        fallback_low: float = 0.0,
        fallback_high: float = 1.0,
        random_seed: Optional[int] = None,
        fallback_enabled: Optional[bool] = None,
        normalize_variables: bool = False,
    ) -> None:
        self.petta_bin = petta_bin or os.environ.get("PETTA_BIN") or shutil.which("petta")
        self.axioms_path = Path(axioms_path) if axioms_path else _DEFAULT_AXIOMS_FILE
        self.import_header_path = (
            Path(import_header_path) if import_header_path else _DEFAULT_IMPORT_HEADER
        )
        self.chainer_dir = Path(
            chainer_dir or os.environ.get("PETTA_CHAINER_DIR") or _DEFAULT_CHAINER_DIR
        )
        raw_workspace = workspace_dir or os.environ.get("PETTA_WORKSPACE")
        self.workspace_dir = (
            Path(raw_workspace)
            if raw_workspace
            else Path(tempfile.gettempdir()) / "petta_workspace"
        )
        self.depth = depth
        self.timeout = timeout
        self.fallback_low = fallback_low
        self.fallback_high = fallback_high
        self.fallback_enabled = (
            _env_flag("MATHS_AI_PLN_FALLBACK") if fallback_enabled is None else fallback_enabled
        )
        self._rng = random.Random(random_seed)
        self._normalizer = VariableNormalizer(normalize=normalize_variables)
        self._initialized = False
        self._initialization_error: Optional[str] = None
        # (<chainer module name> -> path in the workspace) — empty until
        # :meth:`_prepare_workspace` successfully links the vendored modules.
        self._workspace_entries: dict = {}
        self._prepare_workspace()

      
    # Startup / introspection
      
    @property
    def is_available(self) -> bool:
        """Whether the PeTTa executable passed its startup probe."""
        return self._initialized and self._initialization_error is None

    @property
    def initialization_error(self) -> Optional[str]:
        return self._initialization_error

    def initialize(self) -> None:
        """Probe the full /pln/evaluate pipeline: PeTTa + PeTTaChainer.

        Runs a minimal *known-fact* query through the exact code path
        ``evaluate()`` uses (chainer import, axioms import, hypothesis
        assertion, query executable): P is asserted as a fact and then
        queried. Querying an arbitrary proposition with no hypotheses is
        generally unprovable and would fail a healthy install; querying a
        just-asserted fact must yield a real STV. The probe requires that
        petta completed the run, that the PeTTaChainer library actually
        loaded (the chainer modules silently no-op a failed ``import!`` in
        some PeTTa versions), and that a real STV reaches the *query*
        output. Any failure raises, so callers can mark the component
        unavailable instead of serving meaningless scores.
        """
        if self.petta_bin is None:
            self._initialization_error = "PeTTa executable was not found on PATH"
            raise RuntimeError(self._initialization_error)

        probe_formula = parse_simple_pp(_PROBE_FORMULA, self._normalizer)
        source = self._render_query_source(probe_formula, [_PROBE_FORMULA])
        probe_text, launch_status = self._run_petta(source)
        if launch_status is not None:
            self._initialization_error = f"PeTTa startup probe failed: {launch_status}"
            raise RuntimeError(self._initialization_error)
        if not _QUERY_GOAL_PATTERN.search(probe_text):
            self._initialization_error = (
                "PeTTa startup probe failed: the query was not evaluated — "
                "PeTTaChainer did not load (check PETTA_CHAINER_DIR / "
                f"PETTA_WORKSPACE, chainer_dir={self.chainer_dir})"
            )
            raise RuntimeError(self._initialization_error)
        if not extract_stv_scores_for_formula(
            _query_output(probe_text), render_formula(probe_formula)
        ):
            self._initialization_error = (
                "PeTTa startup probe failed: no STV in PeTTa output "
                f"(queried {_PROBE_FORMULA} given hypothesis {_PROBE_FORMULA})"
            )
            raise RuntimeError(self._initialization_error)

        self._initialized = True
        self._initialization_error = None

    def _prepare_workspace(self) -> None:
        """Link the vendored PeTTaChainer modules into ``workspace_dir``.

        ``petta`` resolves imports relative to the directory of the top-level
        file, so every PeTTaChainer entry must appear inside the directory the
        generated query file lives in. Symlinks are preferred (cheap, follow
        the vendor tree); copying is the fallback on filesystems without
        them. Failures are recorded, not raised — :meth:`initialize` turns
        them into the degraded startup state.
        """
        self._workspace_entries = {}
        if not self.chainer_dir.is_dir():
            return
        try:
            self.workspace_dir.mkdir(parents=True, exist_ok=True)
            for entry in sorted(self.chainer_dir.iterdir()):
                link = self.workspace_dir / entry.name
                if link.is_symlink():
                    pass  # already prepared (idempotent across processes/requests)
                elif link.exists():
                    if entry.is_dir() and not link.is_dir() or not entry.is_dir() and link.is_dir():
                        raise OSError(f"workspace entry {link} collides with {entry}")
                else:
                    try:
                        link.symlink_to(entry)
                    except OSError:
                        (
                            shutil.copytree(entry, link)
                            if entry.is_dir()
                            else shutil.copy2(entry, link)
                        )
                self._workspace_entries[entry.name] = link
        except OSError:
            self._workspace_entries = {}

      
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
            if launch_status == "petta_unavailable":
                self._initialized = False
                self._initialization_error = "PeTTa process could not be executed"
            return self._fallback(status=launch_status)

        # Only trust output emitted after the query began: pre-query records
        # (compiled axioms, compileadd hypothesis facts) may contain the very
        # same formula with a perfect STV and must not score the query.
        query_output = _query_output(log_text)
        stvs = extract_stv_scores_for_formula(query_output, render_formula(formula))
        if stvs:
            strength, confidence = max(stvs, key=lambda pair: score_from_stv(*pair))
            return PLNResult(
                stv=STV(strength=strength, confidence=confidence),
                status="ok",
                is_fallback=False,
                raw_output=log_text,
            )

        # A successful run (exit 0) with no query STV is the normal no-proof
        # outcome. Classify as an error only when the query's own output
        # contains a line-anchored runtime failure — imported-source and
        # compiler chatter before the query may freely contain words like
        # "error" or "failed".
        status = (
            "log_error_no_stv"
            if _ERROR_LINE_PATTERN.search(query_output)
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
        # The query file must live in the PeTTaChainer workspace so relative
        # module imports (see import-pettachainer.metta) resolve. When the
        # workspace is unavailable there is no viable chainer layout; the run
        # happens in the plain temp dir and reports its failure.
        run_dir = self.workspace_dir if self._workspace_entries else None
        handle = tempfile.NamedTemporaryFile(
            "w",
            prefix="pln-query-",
            suffix=".metta",
            delete=False,
            encoding="utf-8",
            dir=run_dir,
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

            output = completed.stdout + completed.stderr
            if completed.returncode != 0:
                return output, "petta_unavailable"
            return output, None
        finally:
            try:
                os.unlink(handle.name)
            except OSError:
                pass

    def _fallback(self, *, status: str, raw_output: str = "") -> PLNResult:
        """Build the no-evidence placeholder result.

        Deterministic zero-evidence STV unless random exploration sampling
        was explicitly enabled (``fallback_enabled``); identical requests
        therefore produce identical responses.
        """
        score = (
            sample_fallback_score(self._rng, low=self.fallback_low, high=self.fallback_high)
            if self.fallback_enabled
            else 0.0
        )
        strength, confidence = (score, self.fallback_high) if self.fallback_enabled else (0.0, 0.0)
        return PLNResult(
            stv=STV(strength=strength, confidence=confidence),
            status=status,
            is_fallback=True,
            raw_output=raw_output,
        )
