# Running PLN with local dependencies

PLN-enabled search needs an installed PeTTa runtime and a separate PeTTaChainer
checkout. Application startup does not clone either repository. The same local
import builder serves live training/evaluation and batch MeTTa generation.

## JupyterHub installation

For the service branch, use its Python 3.13 native provisioner from the
repository root:

```bash
conda activate atp
bash scripts/provision_pln.sh "$CONDA_PREFIX/opt/pln" "$CONDA_PREFIX/bin/python"
source "$CONDA_PREFIX/opt/pln/activate.sh"
```

It builds a matching SWI-Prolog/Janus and pinned local source checkouts.
For an already provisioned environment, check the runtime prerequisites:

```bash
conda activate atp
swipl --version
swipl -q -g "use_module(library(janus)),writeln(janus_ok),halt"
```

The tested runtime uses SWI-Prolog 9.3.31. PeTTa's documentation requires 9.3.x
or later and Python interoperability. Install a suitable SWI-Prolog build with
Janus before continuing if either check fails. Installing a Python package alone
does not provide that Prolog runtime.

For a new installation, clone both projects and select the tested revisions:

```bash
mkdir -p /home/jovyan/pln
cd /home/jovyan/pln
git clone https://github.com/trueagi-io/PeTTa.git
git -C PeTTa checkout e1490899cefc67c128d5311ff4861f9997674957
git clone https://github.com/123nol/petta_chainer.git
git -C petta_chainer checkout b1678e97c4498b376b91e8cce215f5ab0398ddef

mkdir -p /home/jovyan/bin
cat > /home/jovyan/bin/petta <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
query_file="$(realpath "$1")"
cd /home/jovyan/pln/PeTTa
exec sh ./run.sh "$query_file" -s
EOF
chmod +x /home/jovyan/bin/petta

export PETTA_BIN=/home/jovyan/bin/petta
export PETTA_SOURCE_ROOT=/home/jovyan/pln/PeTTa
export PETTACHAINER_ROOT=/home/jovyan/pln/petta_chainer
```

Keep the PeTTa checkout available: the launcher uses its shell script and Prolog
source. Optional MORK/FAISS extensions are not needed by the import diagnostic.
The application's lemma-retrieval FAISS index is a separate component.

Verify the interpreter independently:

```bash
printf '!(+ 1 2)\n' > /tmp/petta-arithmetic.metta
"$PETTA_BIN" /tmp/petta-arithmetic.metta
```

Expect `3`. This does not yet verify chainer or axiom loading.

## Verify imports and preserve evidence

From the application repository root, with its Python dependencies installed:

```bash
python -m maths_ai.pln_inference.diagnose \
  --petta-source-root /home/jovyan/pln/PeTTa \
  --output artifacts/pln_checks/first_install
```

The output directory must be new. It retains all query sources, separate stdout
and stderr logs, subprocess exit codes, file hashes, Git revisions when available,
and `summary.json`. A nonzero exit means a required check failed. Git revision
metadata does not attest that every file in a checkout is unmodified; select clean
checkouts when reproducing an experiment.

The diagnostic checks:

- A supplied implication derives its conclusion through the core chainer.
- An imported control axiom derives a uniquely named conclusion. Without that
  import the same premise does not derive the conclusion.
- The full fork collection produces an identity proof naming `pm2.18` and
  `pm2.18i`. Without the collection those imported derivations are absent.
  The core chainer can still prove identity through a contextual proof.
- A conjunction query is an observation, not a required successful derivation.
  The tested configuration returns no answer for this case.

Answer inspection is restricted to returned statements with the requested target
and a finite strength/confidence pair. Echoed facts and compiled rules do not
count as answers. A successful score also requires an actual `rule-proof` subtree in that returned
statement. A direct fact or a standalone proof-marker string is insufficient.

The supported module paths contain only ASCII letters, digits, `/`, `_`, `-`,
and `.`. Paths with spaces or MeTTa syntax characters are rejected. Imports use
absolute module names without the final `.metta`; PeTTa appends that extension.

## Live search and batch generation

Live search reads `PETTACHAINER_ROOT` and `PETTA_BIN`. It defaults to the fork's
`pettachainer/metta/metamath/axioms/metamath_axioms.metta`. Configuration is
validated when preparing a query. Missing files/executable raise
`PLNConfigurationError` rather than becoming random scores. A configured process
that fails returns `engine_error`, `timeout`, or `petta_unavailable` with retained
diagnostics. Only supported completed queries without a qualifying rule derivation may receive
a labeled exploration fallback. Engine errors and unsupported translation have no STV.

The Python API supports explicit `chainer_root`, `axioms_path`, and `petta_bin`
arguments. There is no arbitrary import-header parameter. An explicit axiom file
is an override of the fork default, not a second execution implementation.

After successful verification, set `"use_pln": true` in the normal RL config and
launch from the same activated terminal. With `use_pln` false, dependency
resolution and PeTTa execution are not required.

Batch generation uses the same environment configuration, or explicit paths:

```bash
python -m maths_ai.pln_inference.metta.translator.translator_modules.cli \
  --input /absolute/input.json --output /absolute/new_query_directory \
  --chainer-root /home/jovyan/pln/petta_chainer \
  --source-root /absolute/mathlib4 --pantograph-repl /absolute/repl
```

`--axioms-path` optionally overrides the fork default. Generated shell runners
honor `PETTA_BIN`. Regenerate archived examples with the current translator.
The old hardcoded `metta/scripts/run_tests.py` runner is replaced by the native
diagnostic above.

## Known limits and the Git-import alternative

Live scoring now compiles `GoalInput` from raw execution expressions and unique
local IDs. Batch generation uses the same typed compiler. The old permissive
parser, name sanitization, numeric buckets, and binder-erasing normalization
have been removed. The parent target is no longer asserted as a child hypothesis.

Supported initial forms are propositional conjunction, disjunction, equivalence,
negation, nondependent implication, exact proposition identities, and exact
uninterpreted equality atoms retaining their type and operands. Equality atoms
are not an arithmetic solver. `P : Prop` introduces a symbol, `h : P` supplies a
fact, and known data declarations supply terms rather than facts.

Unknown proposition typing, dependent binders, unresolved metavariables, ambiguous
projection placeholders, and lets return `unsupported` with a typed diagnostic.
No structured decode error falls back to printed text. `evaluate_text` is a
separate explicit propositional grammar with declared symbols, named hypotheses,
Lean precedence, and no arbitrary application/equality/quantifier text.

```python
from maths_ai.pln_inference.compiler import GoalInput
from maths_ai.pln_inference.model import PLNInference

# canonical_goal is an actual structured Goal returned by the application adapter.
result = PLNInference().evaluate(GoalInput.from_goal(canonical_goal))
# For an explicit text-only propositional query:
result = PLNInference().evaluate_text(
    "Q", ("hP : P", "hPQ : P → Q"), propositions={"P": "P", "Q": "Q"}
)
```

Success requires a target-matching returned statement with a structured
`rule-proof` (possibly inside a merged proof) and its attached finite strength and
confidence. A direct-fact STV is intentionally not success. An unsupported goal
retains the policy prior, receives no STV or PLN potential, and does not train the
Thompson sampler. It remains a Lean search obligation. Genuine scores, fallback
scores, and unsupported counts are recorded separately in search execution data
and round metrics.

Translation version `lean-pln-exact-v1-rule-proof` is included in sampler keys,
batch manifests, and search/evaluation contracts. Regenerate old batch manifests
and start a fresh RL run when changing translation semantics. Model warm starts
remain supported; model graph features and weights have not been changed.

The first 100 local JSONL rows yielded 92 Lean admission rejections and eight
admitted states. Three admitted states were supported and five had unknown sort
classification. This ordered local sample is not a Mathlib-wide coverage estimate.
See `maths_ai/gnn_inference/docs/pln_validation/local_first100_coverage.json`.
General predicate classification needs additional Lean-derived metadata; the
current client does not expose a per-local `isProp` field. Kernel certification,
not a PLN score, still determines proof success.

To repeat the audit on your dataset:

```bash
python -m maths_ai.pln_inference.coverage \
  --input /absolute/states.jsonl --source-root /absolute/mathlib4 \
  --repl /absolute/repl --output /absolute/new_coverage_directory \
  --max-scan-rows 100 --time-budget 120
```


Native tests verified that `git-import!` can download the available fork and
register its library root after explicit Prolog bootstrap. The ordinary header
still failed in `use-module!` lookup. The importer reuses existing directories
without pulling updates and has no revision argument. This implementation uses
local imports throughout and performs no network requests during inference.

## Service ownership

PLN-enabled CLI startup and API readiness validate the resolved native source
identity, the Python version embedded by Janus, and a genuine rule-backed
query. All live Lean workers share one bounded PLN process owner. Waiting for
a query slot counts toward the search deadline; cancellation reaps the query
process group before releasing capacity. Query output is bounded.

API clients select a server-owned PLN profile. They cannot choose source or
executable paths. PUCT keeps its actor/critic visit-based selection; PLN scores
are recorded on obligations and used by existing rank/continuation paths.
