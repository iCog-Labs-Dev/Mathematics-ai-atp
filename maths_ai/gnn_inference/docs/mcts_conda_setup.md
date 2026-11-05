# Restore a Conda environment for MCTS actor–critic search

`requirements-mcts-ac.txt` at the repository root contains the Python
dependencies exported from this branch's `uv.lock`. It pins transitive
dependencies and the standalone custom Pantograph client. Use it instead of
the older, incomplete root `requirements.txt` for this branch.

The export was made on `MCTS-AC-branch` at `7cf9373`. It supports the CLI search,
evaluation and RL training paths, including source-aware retrieval. It does
not include the unmerged FastAPI service from PR #72.

`environment-mcts-ac.yml` adds Python 3.13, GNU coreutils,
Git, and build prerequisites for Janus. It installs the Python requirements
through pip after Conda finishes. The explicit native provisioner below builds
SWI-Prolog/Janus, PeTTa and PeTTaChainer. It targets Linux JupyterHub.

## Install in JupyterHub

Run these commands in a JupyterLab terminal from the repository root:

```bash
cd /home/jovyan/Nol/Mathematics-ai-atp
conda env create -f environment-mcts-ac.yml
conda activate atp
python --version
python -m pip check
```

If `atp` already exists, update it instead of creating another environment:

```bash
conda deactivate
conda env update -n atp -f environment-mcts-ac.yml
conda activate atp
python --version
python -m pip check
```

Run either command from the repository root, where the environment file and
its referenced requirements file live. Installation requires access to
conda-forge, the Python package index, and GitHub. The Linux Torch installation
includes large NVIDIA runtime packages. The pinned Python dependencies were
resolved for CPython 3.13 on Linux x86-64; this is not an end-to-end GPU or
Conda installation test.

This export pins `torch==2.12.0` and its CUDA 13 dependencies on Linux. The
host NVIDIA driver must support that runtime. Installing Python packages
does not update the host driver. If CUDA initialization fails, retain the
diagnostics below and check the driver/runtime pairing with the JupyterHub
administrator before changing package versions. The existing lockfile was
exported without selecting a different Torch build.

## Check the installation

```bash
command -v python
command -v stdbuf
stdbuf --version
nvidia-smi

python - <<'PY'
import sys
from importlib.metadata import distribution, version

import datasets
import faiss
import numpy
import pantograph_async
import pydantic
import torch
import torch_geometric
import transformers

print("Python:", sys.executable)
for name in (
    "torch", "torch-geometric", "datasets", "faiss-cpu",
    "transformers", "pydantic", "pantograph-async-client",
):
    print(f"{name}: {version(name)}")
print("Client source:", distribution("pantograph-async-client").read_text("direct_url.json"))
print("Torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU tensor:", (torch.ones(1, device="cuda:0") + 1).cpu())
PY

python -m maths_ai.gnn_inference.scripts.rl_train --help
```

`stdbuf --version` must identify GNU coreutils. The custom client source
metadata should identify commit
`e8aa9ecc17686cfa2703c7082c02d45310ce421f`. For a GPU run,
`torch.cuda.is_available()` must return `True` and the GPU tensor operation
must succeed. `--help` verifies CLI imports, not model or Lean readiness.

If you run Python inside notebooks as well as the terminal, optionally
register this environment as a kernel:

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name atp --display-name "Python (atp)"
```

Select that kernel in JupyterLab and restart it after installing packages.
The search CLI itself does not require `ipykernel`.

## What this file does not install

The following existing runtime files must still be available:

- Matching Lean/Elan/Lake toolchain, the custom Pantograph REPL executable,
  and the configured Mathlib source root with its compiled dependencies.
- Version-4 source-aware actor–critic checkpoint, matching prepared metadata,
  exported query encoder, lemma index and manifests, corpus, and declaration
  inventory referenced by your trial configuration.
- For `use_pln: true`, the separately provisioned PeTTa runtime and configured
  local PeTTaChainer fork, installed with the native provisioning command
  below. Janus must pass its checks before enabling PLN. Follow the PLN setup instructions
  in `maths_ai/pln_inference/README.md` and preserve `PETTA_BIN` and
  `PETTACHAINER_ROOT`.

Restoring the Conda packages does not require rebuilding an intact custom
REPL or regenerating intact model/retrieval artifacts. Check that the config
still points to their actual locations.

Once those checks pass, run your existing trial configuration:

```bash
python -m maths_ai.gnn_inference.scripts.rl_train --config /absolute/path/to/trial.json
```

For evaluation without optimizer updates:

```bash
python -m maths_ai.gnn_inference.scripts.rl_train --config /absolute/path/to/trial.json --eval-only
```

The evaluation configuration must have a nonempty evaluation subset.

## Verify SWI-Prolog and Python embedding for PLN

After activating `atp`, from the repository root:

```bash
bash scripts/provision_pln.sh "$CONDA_PREFIX/opt/pln" "$CONDA_PREFIX/bin/python"
source "$CONDA_PREFIX/opt/pln/activate.sh"
swipl --version
swipl -q -g "use_module(library(janus)),py_call(json:loads('[]'),[]),py_call(sys:version_info,V),writeln(V),halt"
```

The provisioner builds pinned SWI-Prolog 9.3.31 and Janus with Python 3.13,
checks out the pinned PeTTa and forked PeTTaChainer sources, creates launchers,
and requires a real rule-backed PLN answer. The SWI launcher selects the
correct Python base library and environment site-packages. It records the
source/native identities in `deployment.json`.

Run this as an explicit setup operation; search never downloads these
repositories. Source the generated activation file in each terminal used for
PLN-enabled search. Installing `janus-swi` through pip alone does not install
Prolog's required embedding library.

See [the remediation handoff](mcts_api_pln_remediation_handoff.md) for profile
selection, import controls, and host-versus-container acceptance limits.

## Refresh the exported file when dependencies change

From the repository root, using an installed `uv`:

```bash
uv export --frozen --no-dev --no-emit-project --no-hashes \
  --format requirements.txt --output-file requirements-mcts-ac.txt
```

This exports existing locked versions without updating `uv.lock`. Regenerate
it after dependency changes are merged rather than maintaining a separate
list by hand.
