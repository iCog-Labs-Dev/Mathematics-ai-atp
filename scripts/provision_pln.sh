#!/usr/bin/env bash
# Explicit build/provision step; never called during proof search.
set -euo pipefail
prefix="${1:?Usage: provision_pln.sh INSTALL_PREFIX PYTHON_EXECUTABLE}"
python_bin="${2:?Specify the Python 3.13 executable}"
prefix="$(realpath -m "$prefix")"
python_bin="$("$python_bin" -c 'import sys; print(sys.executable)')"
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 13), sys.version'
mkdir -p "$prefix/src" "$prefix/bin"

checkout() {
    local repo="$1" revision="$2" destination="$3"
    if [ ! -d "$destination/.git" ]; then
        git init -q "$destination"
        git -C "$destination" remote add origin "$repo"
    fi
    if ! git -C "$destination" rev-parse --verify HEAD >/dev/null 2>&1; then
        git -C "$destination" fetch --depth 1 origin "$revision"
        git -C "$destination" checkout --detach FETCH_HEAD
    fi
    test "$(git -C "$destination" rev-parse HEAD)" = "$revision"
    test -z "$(git -C "$destination" status --porcelain --untracked-files=no)"
}

checkout https://github.com/SWI-Prolog/swipl-devel.git \
    c617497d18bebe8d44d68329ef5f467b74572b9b "$prefix/src/swipl"
git -C "$prefix/src/swipl" submodule update --init --recursive --depth 1
checkout https://github.com/trueagi-io/PeTTa.git \
    e1490899cefc67c128d5311ff4861f9997674957 "$prefix/src/PeTTa"
checkout https://github.com/123nol/petta_chainer.git \
    b1678e97c4498b376b91e8cce215f5ab0398ddef "$prefix/src/petta_chainer"

python_libdir="$("$python_bin" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
cmake -S "$prefix/src/swipl" -B "$prefix/build/swipl" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix/swipl" \
    -DCMAKE_INSTALL_RPATH="$python_libdir" \
    -DINSTALL_DOCUMENTATION=OFF \
    -DSWIPL_PACKAGES_X=OFF -DSWIPL_PACKAGES_JAVA=OFF \
    '-DCPYTHON_VERSION=3.13;EXACT' -DPython_EXECUTABLE="$python_bin"
cmake --build "$prefix/build/swipl" --parallel "${PLN_BUILD_JOBS:-2}"
cmake --install "$prefix/build/swipl"

python_home="$("$python_bin" -c 'import sys; print(sys.base_prefix)')"
python_site="$("$python_bin" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
cat > "$prefix/bin/swipl" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export PYTHONHOME="$python_home"
export PYTHONPATH="$python_site\${PYTHONPATH:+:\$PYTHONPATH}"
exec "$prefix/swipl/bin/swipl" "\$@"
EOF
chmod 0755 "$prefix/bin/swipl"
cat > "$prefix/bin/petta" <<EOF
#!/usr/bin/env bash
set -euo pipefail
query_file="\$(realpath "\$1")"
export PATH="$prefix/bin:$prefix/swipl/bin:\$PATH"
cd "$prefix/src/PeTTa"
exec sh ./run.sh "\$query_file" -s
EOF
chmod 0755 "$prefix/bin/petta"
cat > "$prefix/activate.sh" <<EOF
export PATH="$prefix/bin:$prefix/swipl/bin:\$PATH"
export SWIPL_BIN="$prefix/bin/swipl"
export PETTA_BIN="$prefix/bin/petta"
export PETTA_SOURCE_ROOT="$prefix/src/PeTTa"
export PETTACHAINER_ROOT="$prefix/src/petta_chainer"
EOF
# shellcheck source=/dev/null
source "$prefix/activate.sh"
"$python_bin" - <<'PY'
from maths_ai.pln_inference.runtime import PLNRuntimeSpec
import asyncio
import json
from pathlib import Path
import os

async def verify():
    scorer, identity = await PLNRuntimeSpec.from_environment().open()
    await scorer.aclose()
    Path(os.environ["PETTA_SOURCE_ROOT"]).parents[1].joinpath("deployment.json").write_text(
        json.dumps(identity, indent=2) + "\n")
    print(json.dumps(identity, indent=2))

asyncio.run(verify())
PY
echo "PLN provisioned. Source $prefix/activate.sh before CLI search."
