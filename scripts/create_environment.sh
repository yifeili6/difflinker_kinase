#!/usr/bin/env bash
# Create the recorded Linux x86-64 environment in a new prefix.
set -Eeuo pipefail
if [[ "${1:-}" == --help ]]; then
    printf 'Usage: bash scripts/create_environment.sh /absolute/path/to/new/environment\nSet KINASE_CONDA to the Conda executable (default: conda).\n'
    exit 0
fi
revision_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
environment_prefix="${1:?Usage: bash scripts/create_environment.sh /absolute/path/to/new/environment}"
[[ "$environment_prefix" == /* && ! -e "$environment_prefix" ]] || {
    echo 'Choose an absolute environment path that does not exist.' >&2
    exit 2
}
kinase_conda="${KINASE_CONDA:-conda}"
cd "$revision_root"
"$kinase_conda" create --yes --prefix "$environment_prefix" --file environment/explicit-lock.txt
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$environment_prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$environment_prefix/bin/python" -m pip install --no-index --no-deps --require-hashes \
    -r environment/pip-requirements.txt -r environment/evaluation-pip-requirements.txt
"$environment_prefix/bin/python" - <<'PY'
import importlib.metadata
import json
import rdkit
import torch
expected = {"torch":"1.11.0", "pytorch-lightning":"1.6.3", "numpy":"1.22.4", "posebusters":"0.2.10", "molsets":"0.3.1", "fcd-torch":"1.0.7"}
observed = {name:importlib.metadata.version(name) for name in expected}
assert observed == expected, (observed, expected)
assert rdkit.__version__ == "2022.03.5"
print(json.dumps({"packages":observed, "rdkit":rdkit.__version__, "cuda_runtime":torch.version.cuda}, indent=2))
PY
printf 'Environment created: %s\n' "$environment_prefix"
