# Running PLN with local dependencies

PLN-enabled search needs an installed PeTTa runtime and a separate PeTTaChainer
checkout. Application startup does not clone either repository. The same local
import builder serves live training/evaluation and batch MeTTa generation.

## JupyterHub installation

In the Conda environment used for RL, check the runtime prerequisites:

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
count as answers. The diagnostic does not use the live extractor's `rule-proof`
filter as its only oracle.

The supported module paths contain only ASCII letters, digits, `/`, `_`, `-`,
and `.`. Paths with spaces or MeTTa syntax characters are rejected. Imports use
absolute module names without the final `.metta`; PeTTa appends that extension.

## Live search and batch generation

Live search reads `PETTACHAINER_ROOT` and `PETTA_BIN`. It defaults to the fork's
`pettachainer/metta/metamath/axioms/metamath_axioms.metta`. Configuration is
validated when preparing a query. Missing files/executable raise
`PLNConfigurationError` rather than becoming random scores. A configured process
that fails returns `engine_error`, `timeout`, or `petta_unavailable` with retained
diagnostics. Completed queries with no recognized answer remain fallback results.

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

Import verification does not validate general Lean-to-MeTTa translation. Named
hypotheses, operator precedence, parentheses, opaque arithmetic/quantifier
translation, and direct-fact STV extraction have separately identified defects.
Search also supplies the parent goal as a heuristic assumption. These are not
repaired here. Genuine PLN scores are heuristics; Lean replay and kernel
certification determine proof success.

Native tests verified that `git-import!` can download the available fork and
register its library root after explicit Prolog bootstrap. The ordinary header
still failed in `use-module!` lookup. The importer reuses existing directories
without pulling updates and has no revision argument. This implementation uses
local imports throughout and performs no network requests during inference.
