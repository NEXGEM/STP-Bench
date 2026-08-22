#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${1:-.stpbench}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
INSTALL_CUDA_EXTRAS="${INSTALL_CUDA_EXTRAS:-0}"
SKIP_FLASH_ATTN="${SKIP_FLASH_ATTN:-0}"

echo "Creating virtual environment: ${ENV_DIR}"
uv venv --python "${PYTHON_VERSION}" "${ENV_DIR}"

PYTHON_BIN="${ENV_DIR}/bin/python"

# setuptools>=81 dropped the bundled pkg_resources module; scanpy's louvain
# clustering path (used by downstream spatial_domain) still imports it at
# runtime, so pin below that to keep pkg_resources available.
uv pip install --python "${PYTHON_BIN}" --upgrade pip "setuptools<81" wheel packaging ninja

echo "Installing PyTorch CUDA 11.8 stack..."
uv pip install --python "${PYTHON_BIN}" -r requirements/torch-cu118.txt

echo "Installing STpredBench runtime dependencies..."
uv pip install --python "${PYTHON_BIN}" -e .
uv pip install --python "${PYTHON_BIN}" -r requirements/runtime.txt
uv pip install --python "${PYTHON_BIN}" -r requirements/preprocess.txt

if [[ "${INSTALL_CUDA_EXTRAS}" == "1" ]]; then
  echo "Installing optional CUDA dataframe extras..."
  uv pip install --python "${PYTHON_BIN}" -r requirements/cuda.txt
fi

if [[ "${SKIP_FLASH_ATTN}" == "1" ]]; then
  echo "Skipping flash-attn installation (SKIP_FLASH_ATTN=1)."
else
  echo "Installing flash-attn from the pinned prebuilt wheel..."
  uv pip install --python "${PYTHON_BIN}" -r requirements/flash-attn.txt --no-build-isolation
fi

echo "Verifying imports..."
"${PYTHON_BIN}" - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
try:
    import flash_attn
    print("flash_attn:", flash_attn.__version__)
except ImportError:
    print("flash_attn: not installed (optional)")
PY

echo "Done. Activate with: source ${ENV_DIR}/bin/activate"
