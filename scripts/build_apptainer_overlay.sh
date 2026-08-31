#!/usr/bin/env bash
# Install h5py (and a few helpers) into the LLaMA-Factory apptainer overlay
# without upgrading the SIF's numpy (required by torch).
# Safe for multimodal H5 training (ScanNet_h5 / Spatial-SSRL / 3DThinker).
#
# Usage (from repo root or scripts/):
#   bash scripts/build_apptainer_overlay.sh
#   OVERLAY=/path/to/overlay.img OVERLAY_SIZE_MB=4096 bash scripts/build_apptainer_overlay.sh
#   FORCE_RECREATE=1 bash scripts/build_apptainer_overlay.sh   # delete and rebuild overlay
#
# Notes:
# - Overlay size is fixed at create time. Changing OVERLAY_SIZE_MB has no effect
#   unless the image is removed/recreated (FORCE_RECREATE=1).
# - Default 4096 MiB covers h5py + wandb + ray + flash_attn + causal_conv1d.
# - Do not create or mount this overlay with --fakeroot on Alliance clusters
#   (Tamia/Ubuntu 24.04 has no /etc/subuid and blocks user-namespace uid maps).
#   Jobs should use: apptainer exec --overlay overlay.img ...
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

SIF="${SIF:-${SIF_FILE:-$HF_HOME/datasets--cvis-tmu--compute_canada_sif_files/snapshots/382a3b3e54a9fa9450c6c99dd83efaa2f0ca4a5a/llamafactory.sif}}"
OVERLAY="${OVERLAY:-$PROJECT_DIR/apptainer/overlay.img}"
# Size in MiB for `apptainer overlay create --size`. Only used when creating.
OVERLAY_SIZE_MB="${OVERLAY_SIZE_MB:-4096}"
WHEELHOUSE="${WHEELHOUSE:-/scratch/i/indrisch/wheels/llamafactory_py311}"
H5PY_VERSION="${H5PY_VERSION:-3.16.0}"
FORCE_RECREATE="${FORCE_RECREATE:-0}"
FLASH_ATTN_WHEEL_URL="${FLASH_ATTN_WHEEL_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl}"
CAUSAL_CONV1D_WHEEL_URL="${CAUSAL_CONV1D_WHEEL_URL:-https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl}"

mkdir -p "${WHEELHOUSE}" "$(dirname "${OVERLAY}")"
WHEELHOUSE="$(cd "${WHEELHOUSE}" && pwd)"
OVERLAY="$(cd "$(dirname "${OVERLAY}")" && pwd)/$(basename "${OVERLAY}")"

module load apptainer 2>/dev/null || true
if ! command -v apptainer >/dev/null 2>&1; then
	echo "ERROR: apptainer not found (module load apptainer failed and it is not on PATH)"
	exit 1
fi

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
	echo "Creating overlay image (${OVERLAY_SIZE_MB} MiB, unprivileged / no --fakeroot)..."
	apptainer overlay create --size "${OVERLAY_SIZE_MB}" "${OVERLAY}"
else
	echo "Overlay already exists ($(du -h "${OVERLAY}" | awk '{print $1}')). Reusing."
	echo "  To resize/rebuild: FORCE_RECREATE=1 OVERLAY_SIZE_MB=${OVERLAY_SIZE_MB} $0"
fi

# Bind the whole scratch tree so relative wheelhouse/overlay paths resolve in-container.
SCRATCH_BIND="${SCRATCH_BIND:-/scratch/i/indrisch:/scratch/i/indrisch}"

run_in_sif() {
	apptainer exec --cleanenv \
		--env PYTHONNOUSERSITE=1 \
		--env TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
		--bind "${SCRATCH_BIND}" \
		"${SIF}" \
		"$@"
}

run_in_overlay() {
	apptainer exec --cleanenv \
		--overlay "${OVERLAY}" \
		--env PYTHONNOUSERSITE=1 \
		--env TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
		--bind "${SCRATCH_BIND}" \
		"${SIF}" \
		"$@"
}

echo "Container Python:"
PY_TAG="$(run_in_sif python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")"
run_in_sif python -c "import sys; print(sys.version)"
echo "PY_TAG: ${PY_TAG}"

# Download wheel (+ deps metadata) but install h5py with --no-deps so the SIF's
# existing numpy (e.g. 1.26.x for torch) is not replaced by a newer pip pin.
echo "Downloading h5py==${H5PY_VERSION} wheel..."
run_in_sif python -m pip download --only-binary=:all: --dest "${WHEELHOUSE}" "h5py==${H5PY_VERSION}"

shopt -s nullglob
h5py_wheels=("${WHEELHOUSE}"/h5py-"${H5PY_VERSION}"-"${PY_TAG}"-*.whl)
shopt -u nullglob
WHEEL="${h5py_wheels[0]:-}"
if [[ -z "${WHEEL}" || ! -f "${WHEEL}" ]]; then
	echo "ERROR: h5py ${PY_TAG} wheel not found in ${WHEELHOUSE}"
	ls -1 "${WHEELHOUSE}"/h5py-"${H5PY_VERSION}"-*.whl 2>/dev/null || true
	exit 1
fi
echo "H5PY_WHEEL: ${WHEEL}"

download_github_wheel() {
	local url="$1"
	local dest="${WHEELHOUSE}/$(basename "${url}")"
	if [[ -f "${dest}" ]]; then
		echo "Already have ${dest}" >&2
	else
		wget -O "${dest}" "${url}" >&2
	fi
	printf '%s\n' "${dest}"
}

FLASH_ATTN_WHEEL=""
CAUSAL_CONV1D_WHEEL=""
if [[ "${PY_TAG}" == "cp311" ]]; then
	# Prebuilt wheels for Py 3.11, torch 2.6, cuda 12.x, cxx11 ABI TRUE.
	echo "Fetching causal_conv1d wheels for ${PY_TAG}..."
	#FLASH_ATTN_WHEEL="$(download_github_wheel "${FLASH_ATTN_WHEEL_URL}")"
	CAUSAL_CONV1D_WHEEL="$(download_github_wheel "${CAUSAL_CONV1D_WHEEL_URL}")"
else
	echo "WARNING: flash_attn/causal_conv1d URLs are cp312; SIF is ${PY_TAG}. Skipping those wheels."
	echo "  Set SIF to the Py 3.12 image (SIF_FILE on TAMIA) or override FLASH_ATTN_WHEEL_URL."
fi

echo "Upgrade pip/setuptools/wheel inside overlay..."
run_in_overlay python -m pip install --upgrade pip setuptools wheel

echo "Installing ${WHEEL} into overlay (no deps)..."
run_in_overlay python -m pip install --no-deps "${WHEEL}"

if [[ -n "${CAUSAL_CONV1D_WHEEL}" ]]; then
	echo "Installing causal_conv1d wheels into overlay (no deps)..."
	run_in_overlay python -m pip install --no-deps "${CAUSAL_CONV1D_WHEEL}"
fi

echo "Installing wandb/ray/helpers into overlay..."
# flash-linear-attention imports as `fla`; --no-deps keeps the SIF torch.
run_in_overlay python -m pip install \
	wandb ray sentry-sdk build pre_commit rapidfuzz language_tool_python liger-kernel 
run_in_overlay python -m pip install --no-deps flash-linear-attention

echo "Verify:"
run_in_overlay bash -c '
set -euo pipefail
pre-commit --version
python - <<"PY"
import importlib.metadata as md
import importlib.util

def ver(*names):
    for name in names:
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            continue
    return "MISSING"

print("h5py", __import__("h5py").__version__)
print("numpy", __import__("numpy").__version__)
print("wandb", __import__("wandb").__version__)
print("ray", __import__("ray").__version__)
print("sentry_sdk", __import__("sentry_sdk").VERSION)
print("build", __import__("build").__version__)
print("rapidfuzz", __import__("rapidfuzz").__version__)
print("language_tool_python", ver("language_tool_python"), "import", "ok" if importlib.util.find_spec("language_tool_python") else "MISSING")
print("flash-linear-attention", ver("flash-linear-attention", "fla-core"), "import", "ok" if importlib.util.find_spec("fla") else "MISSING")
print("flash_attn", ver("flash-attn", "flash_attn"))
print("causal_conv1d", ver("causal-conv1d", "causal_conv1d"))
print("liger_kernel", ver("liger-kernel", "liger_kernel"))
PY
'

echo "Done. Overlay ready: ${OVERLAY}"
echo "Mount with: apptainer exec --overlay ${OVERLAY} ${SIF} ..."
echo "Do not use --fakeroot with this overlay."
