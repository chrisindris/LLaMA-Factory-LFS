#!/usr/bin/env bash
# Install h5py into the LLaMA-Factory apptainer overlay (no numpy upgrade).
# Safe for multimodal H5 training (ScanNet_h5 / Spatial-SSRL / 3DThinker).
set -euo pipefail

SIF="${SIF:-/scratch/indrisch/huggingface/hub/datasets--cvis-tmu--compute_canada_sif_files/snapshots/c0b06aa9c1c5df915b12e11e74015483257991b8/llamafactory.sif}"
OVERLAY="${OVERLAY:-/scratch/indrisch/LLaMA-Factory/apptainer/overlay.img}"
WHEELHOUSE="${WHEELHOUSE:-/scratch/indrisch/wheels/llamafactory_py311}"
H5PY_VERSION="${H5PY_VERSION:-3.16.0}"

mkdir -p "${WHEELHOUSE}"
module load apptainer 2>/dev/null || true

echo "SIF: ${SIF}"
echo "OVERLAY: ${OVERLAY}"

if [[ ! -f "${OVERLAY}" ]]; then
    echo "Creating overlay image..."
    mkdir -p $(dirname "${OVERLAY}")
    apptainer overlay create --fakeroot --size 1024 "${OVERLAY}"
fi

echo "Container Python:"
apptainer exec --cleanenv --env PYTHONNOUSERSITE=1 "${SIF}" python -c "import sys; print(sys.version)"

# Download wheel (+ deps metadata) but install h5py with --no-deps so the SIF's
# existing numpy (e.g. 1.26.x for torch) is not replaced by a newer pip pin.
echo "Downloading h5py==${H5PY_VERSION} wheel..."
apptainer exec --cleanenv --env PYTHONNOUSERSITE=1 --bind /scratch/indrisch:/scratch/indrisch "${SIF}" \
  python -m pip download --only-binary=:all: --dest "${WHEELHOUSE}" "h5py==${H5PY_VERSION}"

WHEEL=$(ls -1 "${WHEELHOUSE}"/h5py-"${H5PY_VERSION}"-*.whl | head -1)
if [[ -z "${WHEEL}" || ! -f "${WHEEL}" ]]; then
  echo "ERROR: h5py wheel not found in ${WHEELHOUSE}"
  exit 1
fi
echo "Installing ${WHEEL} into overlay (no deps)..."
apptainer exec --fakeroot --cleanenv --overlay "${OVERLAY}" \
  --bind /scratch/indrisch:/scratch/indrisch \
  --env PYTHONNOUSERSITE=1 \
  "${SIF}" \
  python -m pip install --no-deps "${WHEEL}"

echo "Verify:"
apptainer exec --fakeroot --cleanenv --overlay "${OVERLAY}" \
  --env PYTHONNOUSERSITE=1 \
  "${SIF}" \
  python -c "import h5py, numpy; print('h5py', h5py.__version__, 'numpy', numpy.__version__)"

echo "Done."
