#!/usr/bin/env bash
# Portable body for CoT SFT: Scene30k + SpatialSSRL_coldstart + 3DThinker10k.
#
# Every path is resolved relative to the repo root by
# scripts/utils/portable_env.sh, so this tree can be moved or renamed.
#
# Modes:
#   PREFLIGHT=1 <this script>        check paths and exit (safe on a login node)
#   PORTABLE_STAGE=1 <this script>   create repo-relative symlinks + registry, exit
#   RUNNING_MODE=APPTAINER           run llamafactory-cli inside the SIF (default)
#   RUNNING_MODE=VENV                run llamafactory-cli from VENV_LLAMAFACTORY
#   RUNNING_MODE=SHELL               open a shell inside the SIF
#
# Extra args are forwarded to llamafactory-cli, e.g.:
#   sbatch portable_slurm_...sh num_train_epochs=1.0
set -euo pipefail

BODY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=../../scripts/utils/portable_env.sh
source "${BODY_DIR}/../../scripts/utils/portable_env.sh"

portable_init

EXPERIMENT_NAME="qwen2_5vl_lora_sft_CoT_traineval"
export PORTABLE_YAML_FILE="${PROJECT_DIR}/examples/train_lora/portable_${EXPERIMENT_NAME}.yaml"

mkdir -p "${BODY_DIR}/out" "${WANDB_DIR}"
cd "${PROJECT_DIR}"

if [[ -n "${PORTABLE_STAGE:-}" ]]; then
	portable_stage_assets
	exit $?
fi

if [[ -n "${PREFLIGHT:-}" ]]; then
	portable_preflight
	exit $?
fi

portable_preflight || exit 1

# AllianceCan module stack. Absent on a workstation, which is fine.
if command -v module >/dev/null 2>&1; then
	module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 || true
	module load python/3.12 cuda/12.6 opencv/4.12.0 || true
	module load arrow || true
	[[ "${RUNNING_MODE}" != "VENV" ]] && { module load apptainer || true; }
fi

echo "=== host diagnostics (${CLUSTER}) ==="
echo "HOSTNAME:             $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi || true
echo "====================================="

run_in_apptainer() {
	local program="$1"
	local binds=()
	local nv_lib_dir=""

	# Bind only paths that exist; Apptainer fails on a missing bind source.
	local candidate
	for candidate in "${PROJECT_DIR}" "${HF_HUB_CACHE}" "${SCANNET_H5_DIR}" \
		"${SPATIALSSRL_H5_DIR}" "${THINKER10K_H5_DIR}" "${MEDIA_DIR}" "${HOME}"; do
		[[ -n "${candidate}" && "${candidate}" != "None" && -e "${candidate}" ]] && binds+=(-B "${candidate}")
	done
	[[ -d /dev/shm ]] && binds+=(-B /dev/shm:/dev/shm)
	[[ -d /etc/ssl/certs ]] && binds+=(-B /etc/ssl/certs:/etc/ssl/certs:ro)
	[[ -d /etc/pki ]] && binds+=(-B /etc/pki:/etc/pki:ro)

	# shellcheck disable=SC2206
	[[ -n "${EXTRA_BINDS:-}" ]] && binds+=(${EXTRA_BINDS})

	nv_lib_dir="$(dirname "$(ldconfig -p 2>/dev/null | awk '/libcuda\.so /{print $NF}' | head -1)" 2>/dev/null || true)"
	[[ -n "${nv_lib_dir}" && -d "${nv_lib_dir}" ]] && binds+=(-B "${nv_lib_dir}")

	local overlay=()
	[[ -f "${APPTAINER_OVERLAY}" ]] && overlay=(--overlay "${APPTAINER_OVERLAY}")

	# Use the CUDA toolkit inside the image, not a host module path.
	export APPTAINERENV_CUDA_HOME=/usr/local/cuda

	apptainer run --nv "${overlay[@]}" \
		"${binds[@]}" \
		-W "${SLURM_TMPDIR:-/tmp}" \
		--env HF_HOME="${HF_HOME}" \
		--env HF_HUB_CACHE="${HF_HUB_CACHE}" \
		--env HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" \
		--env TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE}" \
		--env HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE}" \
		--env SCANNET_H5_DIR="${SCANNET_H5_DIR}" \
		--env SPATIALSSRL_H5_DIR="${SPATIALSSRL_H5_DIR}" \
		--env THINKER10K_H5_DIR="${THINKER10K_H5_DIR}" \
		--env MPLCONFIGDIR="${MPLCONFIGDIR}" \
		--env TRITON_CACHE_DIR="${TRITON_CACHE_DIR}" \
		--env TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR}" \
		--env PYTORCH_KERNEL_CACHE_PATH="${PYTORCH_KERNEL_CACHE_PATH}" \
		--env FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE}" \
		--env TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
		--env DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK}" \
		--env FORCE_TORCHRUN="${FORCE_TORCHRUN}" \
		--env WANDB_MODE="${WANDB_MODE}" \
		--env WANDB_DIR="${WANDB_DIR}" \
		--env WANDB_CACHE_DIR="${WANDB_CACHE_DIR}" \
		--env PYTHONNOUSERSITE=1 \
		--env PYTHONUNBUFFERED=1 \
		--env PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}/scripts" \
		--env NCCL_DEBUG=INFO \
		--env NCCL_IB_DISABLE=0 \
		--env NCCL_P2P_DISABLE=0 \
		--env NCCL_SOCKET_IFNAME=^docker0,lo \
		--env CUDA_HOME="${APPTAINERENV_CUDA_HOME}" \
		--pwd "${PROJECT_DIR}" \
		"${SIF_FILE}" \
		${program}
}

case "${RUNNING_MODE}" in
APPTAINER)
	run_in_apptainer "llamafactory-cli train ${PORTABLE_YAML_FILE} $*"
	;;
SHELL)
	run_in_apptainer "bash"
	;;
VENV)
	# shellcheck disable=SC1091
	source "${VENV_LLAMAFACTORY}/bin/activate"
	# Re-assert after activation: an editable install may point at another tree.
	export PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}/scripts"
	llamafactory-cli train "${PORTABLE_YAML_FILE}" "$@"
	;;
*)
	echo "Invalid RUNNING_MODE: ${RUNNING_MODE} (expected APPTAINER, VENV, or SHELL)" >&2
	exit 1
	;;
esac
