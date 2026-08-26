#!/usr/bin/env bash
# Install h5py (and a few helpers) into the LLaMA-Factory apptainer overlay
# without upgrading the SIF's numpy (required by torch).
# Safe for multimodal H5 training (ScanNet_h5 / Spatial-SSRL / 3DThinker).
#
# Usage (from repo root or scripts/):
#   bash scripts/build_apptainer_overlay.sh
#   OVERLAY=/path/to/overlay.img OVERLAY_SIZE_MB=2048 bash scripts/build_apptainer_overlay.sh
#   FORCE_RECREATE=1 bash scripts/build_apptainer_overlay.sh   # delete and rebuild overlay
#
# Notes:
# - Overlay size is fixed at create time. Changing OVERLAY_SIZE_MB has no effect
#   unless the image is removed/recreated (FORCE_RECREATE=1).
# - 1024 MiB is enough for h5py + wandb + pip upgrades. Use larger sizes only if
#   you plan to install many extra packages into the overlay.
# - export_merge_adapter_job.sh does NOT need packages from this overlay for
#   LoRA merge; it only uses the overlay when present so the env matches training.
set -euo pipefail

. ./utils/env.sh 2>/dev/null || . "$(dirname "$0")/utils/env.sh"

# --- setting environment ---

EXPERIMENT_NAME="qwen2_5vl_lora_sft_CoT_traineval"

if [[ "${CLUSTER:-}" == "RORQUAL" ]]; then
	SCANNET_H5_DIR="/project/def-wangcs/indrisch/scratch_saves/ScanNet_h5/scans"
fi

# --- build apptainer overlay ---

SIF="${SIF:-$HF_HOME/datasets--cvis-tmu--compute_canada_sif_files/snapshots/382a3b3e54a9fa9450c6c99dd83efaa2f0ca4a5a/llamafactory.sif}"
OVERLAY="${OVERLAY:-$PROJECT_DIR/apptainer/overlay.img}"
# Size in MiB for `apptainer overlay create --size`. Only used when creating.
OVERLAY_SIZE_MB="${OVERLAY_SIZE_MB:-1024}"
WHEELHOUSE="${WHEELHOUSE:-/scratch/i/indrisch/wheels/llamafactory_py311}"
H5PY_VERSION="${H5PY_VERSION:-3.16.0}"
FORCE_RECREATE="${FORCE_RECREATE:-0}"

mkdir -p "${WHEELHOUSE}"
module load apptainer 2>/dev/null || true

echo "SIF: ${SIF}"
echo "OVERLAY: ${OVERLAY}"
echo "OVERLAY_SIZE_MB: ${OVERLAY_SIZE_MB}"
echo "WHEELHOUSE: ${WHEELHOUSE}"
echo "H5PY_VERSION: ${H5PY_VERSION}"

if [[ ! -f "${SIF}" ]]; then
	echo "ERROR: SIF not found: ${SIF}"
	exit 1
fi

if [[ "${FORCE_RECREATE}" == "1" && -f "${OVERLAY}" ]]; then
	bak="${OVERLAY}.bak.$(date +%Y%m%d%H%M%S)"
	echo "FORCE_RECREATE=1: moving existing overlay to ${bak}"
	mv "${OVERLAY}" "${bak}"
fi

if [[ ! -f "${OVERLAY}" ]]; then
	echo "Creating overlay image (${OVERLAY_SIZE_MB} MiB)..."
	mkdir -p "$(dirname "${OVERLAY}")"
	apptainer overlay create --fakeroot --size "${OVERLAY_SIZE_MB}" "${OVERLAY}"
else
	echo "Overlay already exists ($(du -h "${OVERLAY}" | awk '{print $1}')). Reusing."
	echo "  To resize/rebuild: FORCE_RECREATE=1 OVERLAY_SIZE_MB=${OVERLAY_SIZE_MB} $0"
fi

echo "Container Python:"
apptainer exec --cleanenv --env PYTHONNOUSERSITE=1 "${SIF}" python -c "import sys; print(sys.version)"

# Download wheel (+ deps metadata) but install h5py with --no-deps so the SIF's
# existing numpy (e.g. 1.26.x for torch) is not replaced by a newer pip pin.
echo "Downloading h5py==${H5PY_VERSION} wheel..."
apptainer exec --cleanenv --env PYTHONNOUSERSITE=1 --bind /scratch/i/indrisch:/scratch/i/indrisch "${SIF}" \
	python -m pip download --only-binary=:all: --dest "${WHEELHOUSE}" "h5py==${H5PY_VERSION}"

WHEEL=$(ls -1 "${WHEELHOUSE}"/h5py-"${H5PY_VERSION}"-*.whl | head -1)
if [[ -z "${WHEEL}" || ! -f "${WHEEL}" ]]; then
	echo "ERROR: h5py wheel not found in ${WHEELHOUSE}"
	exit 1
fi

# get flash_attn and causal_conv1d in the wheelhouse; we are using Py 3.12, torch 2.10, cuda 12.6
# This seems promising: https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
# This seems promising: https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

pushd "${WHEELHOUSE}"
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
popd


echo "Upgrade pip/setuptools/wheel inside overlay..."
apptainer exec --fakeroot --cleanenv --overlay "${OVERLAY}" \
	--bind /scratch/i/indrisch:/scratch/i/indrisch \
	--env PYTHONNOUSERSITE=1 \
	"${SIF}" \
	python -m pip install --upgrade pip setuptools wheel

echo "Installing ${WHEEL} into overlay (no deps) + wandb/sentry-sdk..."
apptainer exec --fakeroot --cleanenv --overlay "${OVERLAY}" \
	--bind /scratch/i/indrisch:/scratch/i/indrisch \
	--env PYTHONNOUSERSITE=1 \
	"${SIF}" \
	python -m pip install --no-deps "${WHEEL}" wandb ray sentry-sdk build pre_commit rapidfuzz language_tool_python flash_linear_attention

echo "Verify:"
apptainer exec --fakeroot --cleanenv --overlay "${OVERLAY}" \
	--env PYTHONNOUSERSITE=1 \
	"${SIF}" \
  pre-commit --version && language_tool_python --version && python -c "import h5py, numpy, wandb, ray, sentry_sdk, build, rapidfuzz, flash_linear_attention, flash_attn, causal_conv1d; print('h5py', h5py.__version__, 'numpy', numpy.__version__, 'wandb', wandb.__version__, 'ray', ray.__version__, 'sentry_sdk', sentry_sdk.VERSION, 'build', build.__version__, 'rapidfuzz', rapidfuzz.__version__, 'flash_linear_attention', flash_linear_attention.__version__, 'flash_attn', flash_attn.__version__, 'causal_conv1d', causal_conv1d.__version__)"

echo "Done. Overlay ready: ${OVERLAY}"
