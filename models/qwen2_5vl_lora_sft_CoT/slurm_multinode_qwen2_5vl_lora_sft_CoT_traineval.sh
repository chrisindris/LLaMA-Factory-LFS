#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --output=out/%N-multinode_qwen2_5vl_lora_sft_CoT_traineval-%j.out

# RORQUAL:
#SBATCH --cpus-per-task=64
#SBATCH --time=1-12:00:00
#SBATCH --mem=485G
#SBATCH --gpus-per-node=h100:4

# TRILLIUM / NIBI:
#SBATCH --cpus-per-task=96
#SBATCH --time=1-12:00:00
#SBATCH --gpus-per-node=h100:4

# KILLARNEY (outer L40S multinode wrapper sets the real allocation):
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --mem=480GB
#SBATCH --gpus-per-node=l40s:4
# (legacy single-allocation H100 comment: h100:8 / mem=1900G)

# ---------------------------------------------------------------------
# ------------ multinode_qwen2_5vl_lora_sft_CoT_traineval -------------
# ---------------------------------------------------------------------
#
# SFT (LoRA) Qwen2.5-VL on the CoT mix:
#   Scene30k + SpatialSSRL_coldstart + 3DThinker10k  (mix_strategy=concat)
#	In this run, we attempt to use multiple nodes to speed up training.
#
# Images:
#   Scene30k  -> SCANNET_H5_DIR (default /scratch/indrisch/ScanNet_h5/scans)
#   Spatial   -> SPATIALSSRL_H5_DIR
#   3DThinker -> THINKER10K_H5_DIR

# ----- HEADER: ENV VARIABLES -----

# Source env robustly (script lives under models/.../; utils is under scripts/).
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${_SCRIPT_DIR}/../../scripts/utils/env.sh" ]]; then
	. "${_SCRIPT_DIR}/../../scripts/utils/env.sh"
elif [[ -f ./utils/env.sh ]]; then
	. ./utils/env.sh
else
	echo "Error: could not find scripts/utils/env.sh"
	exit 1
fi

EXPERIMENT_NAME="multinode_qwen2_5vl_lora_sft_CoT_traineval"

# --- H5 roots (override per-cluster if needed) ---

if [[ "$CLUSTER" == "RORQUAL" ]]; then
	SCANNET_H5_DIR="/project/def-wangcs/indrisch/scratch_saves/ScanNet_h5/scans"
fi

export SCANNET_H5_DIR="${SCANNET_H5_DIR:-/scratch/indrisch/ScanNet_h5/scans}"
export SPATIALSSRL_H5_DIR="${SPATIALSSRL_H5_DIR:-/scratch/indrisch/Spatial-SSRL_images_h5}"
export THINKER10K_H5_DIR="${THINKER10K_H5_DIR:-/scratch/indrisch/3DThinker10K_images_h5}"
echo "SCANNET_H5_DIR: $SCANNET_H5_DIR"
echo "SPATIALSSRL_H5_DIR: $SPATIALSSRL_H5_DIR"
echo "THINKER10K_H5_DIR: $THINKER10K_H5_DIR"

export WANDB_DIR="${PROJECT_DIR}/wandb/"
# Arch list: L40S is Ada (8.9). Do not trust BEST_GPU=h100 on Killarney sysconfig for L40S jobs.
if [[ "$CLUSTER" == "KILLARNEY" ]]; then
	export TORCH_CUDA_ARCH_LIST="8.9"
elif [[ "$BEST_GPU" == "h100" ]]; then
	export TORCH_CUDA_ARCH_LIST="9.0"
else
	export TORCH_CUDA_ARCH_LIST="8.0"
fi
echo "TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"

YAML_FILE="${PROJECT_DIR}/examples/train_lora/${CLUSTER,,}_${EXPERIMENT_NAME}.yaml"
if [[ "$CLUSTER" == "KILLARNEY" ]]; then
	OUTPUT_DIR="${PROJECT_DIR}/saves/qwen2_5vl-7b/lora/sft/CoT_traineval_multinode_killarney_l40s"
else
	OUTPUT_DIR="${PROJECT_DIR}/saves/qwen2_5vl-7b/lora/sft/CoT_traineval_multinode"
fi
echo "CLUSTER: $CLUSTER"
echo "YAML_FILE: $YAML_FILE"
echo "OUTPUT_DIR: $OUTPUT_DIR"

if [[ ! -f "$YAML_FILE" ]]; then
	echo "Error: YAML config not found: $YAML_FILE"
	exit 1
fi

if [[ -n "$1" ]]; then
	RUNNING_MODE="$1"
fi
echo "RUNNING_MODE: $RUNNING_MODE"

# Shared apptainer env flags for multimodal H5 training
APPTAINER_H5_ENV=(
	--env SCANNET_H5_DIR="${SCANNET_H5_DIR}"
	--env SPATIALSSRL_H5_DIR="${SPATIALSSRL_H5_DIR}"
	--env THINKER10K_H5_DIR="${THINKER10K_H5_DIR}"
)

# Bind H5 trees if they exist (avoid apptainer failure on missing paths)
APPTAINER_H5_BINDS=()
for _h5 in "${SCANNET_H5_DIR}" "${SPATIALSSRL_H5_DIR}" "${THINKER10K_H5_DIR}" "${MEDIA_DIR}"; do
	if [[ -n "${_h5}" && "${_h5}" != "None" && -e "${_h5}" ]]; then
		APPTAINER_H5_BINDS+=(-B "${_h5}")
	fi
done
# Also bind ScanNet_h5 parent when MEDIA_DIR points there
if [[ -d /scratch/indrisch/ScanNet_h5 ]]; then
	APPTAINER_H5_BINDS+=(-B /scratch/indrisch/ScanNet_h5)
elif [[ -d /project/def-wangcs/indrisch/scratch_saves/ScanNet_h5 ]]; then
	APPTAINER_H5_BINDS+=(-B /project/def-wangcs/indrisch/scratch_saves/ScanNet_h5)
fi

MPI_LIB_PATH="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcc12/openmpi/4.1.5/lib"
HWLOC_LIB_PATH="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/hwloc/2.9.1/lib"

# ----- EXPERIMENT -----

# Note: binding host nvcc via -B $(dirname $(which nvcc)) does not work
# reliably on these clusters; use the CUDA toolkit inside the SIF instead.

run_llamafactory_apptainer() {
	# $1 optional: extra NVIDIA driver bind args (e.g. "-B /usr/lib64/nvidia")
	local extra_nv_bind="${1:-}"

	# CUDA toolkit path *inside* the NGC/SIF image (not the host module path).
	# Rules for the container command / env:
	#   - Do NOT pass "export ..." as the command: nvidia_entrypoint.sh does
	#     `exec "$@"` and fails with "exec: export: not found".
	#   - Do NOT force --env PATH=...${PATH}: that expands the *host* PATH and
	#     drops /opt/conda/bin where llamafactory-cli lives in this SIF.
	#   - Prefer leaving PATH/LD_LIBRARY_PATH alone so the image defaults
	#     (/opt/conda/bin, /usr/local/cuda/bin, ...) stay intact.
	export APPTAINERENV_CUDA_HOME=/usr/local/cuda

	if [[ "$RUNNING_MODE" == "SHELL" ]]; then
		PROGRAM="bash"
	elif [[ "$RUNNING_MODE" == "APPTAINER" ]]; then
		PROGRAM="llamafactory-cli train ${YAML_FILE}"
	fi

	# Use node-local cache to avoid NFS contention for datasets; this replaces the cache_dir from the yaml.
	export HF_DATASETS_CACHE="${SLURM_TMPDIR}/hf_datasets"
	mkdir -p "${HF_DATASETS_CACHE}"
	# Avoid NFS lock exhaustion when datasets cache is shared across ranks.
	export HF_DATASETS_DISABLE_FILE_LOCKING=1
	export DATASETS_DISABLE_FILE_LOCKING=1

	export NCCL_ASYNC_ERROR_HANDLING=1 && echo "NCCL_ASYNC_ERROR_HANDLING: ${NCCL_ASYNC_ERROR_HANDLING}"
	export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 && echo "TORCH_NCCL_ASYNC_ERROR_HANDLING: ${TORCH_NCCL_ASYNC_ERROR_HANDLING}"
	export NCCL_DEBUG=INFO && echo "NCCL_DEBUG: ${NCCL_DEBUG}"
	export NCCL_SOCKET_IFNAME=^docker0,lo && echo "NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME}"

	# LLaMA-Factory's launcher reads these variables and then invokes
	# torchrun once per node with the correct --node_rank. The outer
	# 2-node SLURM wrapper must start this script through srun so that
	# one parent process exists on every allocated node.
	export NNODES="${SLURM_NNODES}" && echo "NNODES: ${NNODES}"
	export NODE_RANK="${SLURM_NODEID}" && echo "NODE_RANK: ${NODE_RANK}"
	export MASTER_ADDR="${MASTER_ADDR:-${HEAD_NODE}}" && echo "MASTER_ADDR: ${MASTER_ADDR}"
	export MASTER_PORT="${MASTER_PORT:-29500}" && echo "MASTER_PORT: ${MASTER_PORT}"
	export NPROC_PER_NODE="4" && echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"

	if [[ "$NODE_RANK" == 0 ]]; then
		OVERLAY="${PROJECT_DIR}/apptainer/overlay.img"
	else
		OVERLAY="${PROJECT_DIR}/apptainer/overlay_${NODE_RANK}.img"
	fi

	# Prefer host src so H5 image backends (ScanNet_h5 / Spatial-SSRL / 3DThinker)
	# from this checkout override the older /app/src baked into the SIF.
	apptainer run --nv --fakeroot --overlay "${OVERLAY}" \
		${extra_nv_bind} \
		-B ${PROJECT_DIR} \
		-B ${HF_HOME} \
		${APPTAINER_H5_BINDS[@]+"${APPTAINER_H5_BINDS[@]}"} \
		-B /home/indrisch \
		-B /dev/shm:/dev/shm \
		-B /etc/ssl/certs:/etc/ssl/certs:ro \
		-B /etc/pki:/etc/pki:ro \
		-W ${SLURM_TMPDIR} \
		--env HF_HUB_OFFLINE=1 \
		--env HF_HOME="${HF_HOME}" \
		--env HF_HUB_CACHE="${HF_HUB_CACHE}" \
		--env HF_DATASETS_DISABLE_FILE_LOCKING="${HF_DATASETS_DISABLE_FILE_LOCKING}" \
		--env DATASETS_DISABLE_FILE_LOCKING="${DATASETS_DISABLE_FILE_LOCKING}" \
		--env MPLCONFIGDIR="${SLURM_TMPDIR}/.config/matplotlib" \
		--env TRITON_CACHE_DIR="${SLURM_TMPDIR}/.triton_cache" \
		--env DISABLE_VERSION_CHECK=1 \
		--env FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE}" \
		--env TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
		--env TORCH_EXTENSIONS_DIR="${SLURM_TMPDIR}/.cache/torch_extensions" \
		--env PYTORCH_KERNEL_CACHE_PATH="${SLURM_TMPDIR}/.cache/torch/kernels" \
		--env FORCE_TORCHRUN=1 \
		--env WANDB_MODE=offline \
		--env WANDB_DIR="${WANDB_DIR}" \
		--env WANDB_CACHE_DIR="${SLURM_TMPDIR}/.cache/wandb" \
		--env PYTHONNOUSERSITE=1 \
		--env PYTHONUNBUFFERED=1 \
		--env PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}" \
		--env NCCL_IB_DISABLE=0 \
		--env NCCL_P2P_DISABLE=0 \
		--env NCCL_DEBUG="${NCCL_DEBUG}" \
		--env NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING}" \
		--env TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING}" \
		--env NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
		--env CUDA_HOME="${APPTAINERENV_CUDA_HOME}" \
		--env NNODES="${NNODES}" \
		--env NODE_RANK="${NODE_RANK}" \
		--env MASTER_ADDR="${MASTER_ADDR}" \
		--env MASTER_PORT="${MASTER_PORT}" \
		--env NPROC_PER_NODE="${NPROC_PER_NODE}" \
		"${APPTAINER_H5_ENV[@]}" \
		--pwd ${PROJECT_DIR} \
		${SIF_FILE} \
		${PROGRAM}
}

lmod_preflight() {
	# useful for VENV.
	local lmod_init="/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/init/bash"
	local lmod_exec="/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/libexec/lmod"
	local resolved_init
	local resolved_exec

	if [[ ! -e "$lmod_init" || ! -e "$lmod_exec" ]]; then
		echo "ERROR: Lmod bootstrap path is unavailable on this node."
		echo "  lmod init: $lmod_init"
		echo "  lmod exec: $lmod_exec"
		ls -l "$lmod_init" "$lmod_exec" 2>/dev/null || true
		exit 1
	fi

	resolved_init=$(readlink -f "$lmod_init" 2>/dev/null || true)
	resolved_exec=$(readlink -f "$lmod_exec" 2>/dev/null || true)
	if [[ -z "$resolved_init" || -z "$resolved_exec" ]]; then
		echo "ERROR: Lmod symlink resolution failed before module initialization."
		echo "  lmod init: $lmod_init"
		echo "  lmod exec: $lmod_exec"
		ls -l "$lmod_init" "$lmod_exec" 2>/dev/null || true
		exit 1
	fi
}

if [[ "$CLUSTER" == "NIBI" ]]; then

	# STEP 1: RUN THE TRAINING AND EVALUATION

	# better to have triton cache on a non-nfs file system for speed
	# if we are offline, we need to indicate this

	if [[ "$RUNNING_MODE" == "APPTAINER" ]]; then

		module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
		module load python/3.12 cuda/12.6 opencv/4.12.0
		module load arrow
		module load apptainer

		echo "=== HOST DIAGNOSTICS ==="
		echo "HOSTNAME: $(hostname)"
		echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
		echo "SLURM_GPUS: $SLURM_GPUS"
		echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
		nvidia-smi
		ls -la /dev/nvidia* 2>/dev/null || echo "No /dev/nvidia* devices found on host"
		echo "=== END HOST DIAGNOSTICS ==="

		NVIDIA_LIB_DIR=$(dirname "$(ldconfig -p 2>/dev/null | grep 'libcuda\.so ' | awk '{print $NF}' | head -1)" 2>/dev/null)
		NVIDIA_BIND_ARGS=""
		if [[ -n "$NVIDIA_LIB_DIR" && -d "$NVIDIA_LIB_DIR" ]]; then
			echo "Found NVIDIA driver libs at: $NVIDIA_LIB_DIR"
			NVIDIA_BIND_ARGS="-B ${NVIDIA_LIB_DIR}"
		else
			echo "WARNING: Could not locate NVIDIA driver libs via ldconfig"
		fi

		echo "=== APPTAINER GPU SANITY TEST ==="
		apptainer exec --nv ${NVIDIA_BIND_ARGS} \
			-B ${PROJECT_DIR} \
			${SIF_FILE} \
			nvidia-smi
		echo "=== END APPTAINER GPU SANITY TEST (exit code: $?) ==="

		run_llamafactory_apptainer "${NVIDIA_BIND_ARGS}"

	elif [[ "$RUNNING_MODE" == "VENV" ]]; then

		module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
		module load python/3.12 cuda/12.6 opencv/4.12.0
		module load arrow

		echo "Copying venv to local storage..."
		cp -a /scratch/indrisch/venv_llamafactory_cu126 ${SLURM_TMPDIR}/venv_llamafactory_cu126
		source ${SLURM_TMPDIR}/venv_llamafactory_cu126/bin/activate

		export PYTHONUNBUFFERED=1
		export NCCL_DEBUG=INFO
		export TORCH_CUDA_ARCH_LIST="9.0"
		export FORCE_TORCHRUN=1
		export HF_HUB_OFFLINE=1
		export WANDB_MODE=offline
		export WANDB_DIR="${WANDB_DIR}"
		export WANDB_CACHE_DIR="${SLURM_TMPDIR}/.cache/wandb"
		export TRITON_CACHE_DIR="${SLURM_TMPDIR}/.triton_cache"
		export DISABLE_VERSION_CHECK=1

		echo "=== VENV DIAGNOSTICS ==="
		echo "HOSTNAME: $(hostname)"
		echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
		echo "SLURM_GPUS: $SLURM_GPUS"
		echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
		nvidia-smi
		python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"
		echo "=== END VENV DIAGNOSTICS ==="

		pushd /scratch/indrisch/LLaMA-Factory
		llamafactory-cli train ${YAML_FILE}

	else
		echo "Invalid running mode: $RUNNING_MODE"
		exit 1
	fi

elif [[ "$CLUSTER" == "RORQUAL" ]]; then

	if [[ "$RUNNING_MODE" == "APPTAINER" ]]; then

		module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
		module load python/3.12 cuda/12.6 opencv/4.12.0
		module load arrow
		module load apptainer

		echo "=== HOST DIAGNOSTICS ==="
		echo "HOSTNAME: $(hostname)"
		echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
		nvidia-smi
		echo "=== END HOST DIAGNOSTICS ==="

		run_llamafactory_apptainer

	elif [[ "$RUNNING_MODE" == "VENV" ]]; then

		module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
		module load python/3.12 cuda/12.6 opencv/4.12.0
		module load arrow

		echo "Copying venv to local storage..."
		cp -a /scratch/indrisch/venv_llamafactory_cu126 ${SLURM_TMPDIR}/venv_llamafactory_cu126
		source ${SLURM_TMPDIR}/venv_llamafactory_cu126/bin/activate

		export PYTHONUNBUFFERED=1
		export NCCL_DEBUG=INFO
		export TORCH_CUDA_ARCH_LIST="9.0"
		export FORCE_TORCHRUN=1
		export HF_HUB_OFFLINE=1
		export WANDB_MODE=offline
		export WANDB_DIR="${WANDB_DIR}"
		export WANDB_CACHE_DIR="${SLURM_TMPDIR}/.cache/wandb"
		export TRITON_CACHE_DIR="${SLURM_TMPDIR}/.triton_cache"
		export DISABLE_VERSION_CHECK=1
		export SCANNET_H5_DIR SPATIALSSRL_H5_DIR THINKER10K_H5_DIR

		pushd ${PROJECT_DIR}
		llamafactory-cli train ${YAML_FILE}

	elif [[ "$RUNNING_MODE" == "SHELL" ]]; then

		module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
		module load python/3.12 cuda/12.6 opencv/4.12.0
		module load arrow
		module load apptainer

		echo "=== HOST DIAGNOSTICS ==="
		echo "HOSTNAME: $(hostname)"
		echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
		nvidia-smi
		echo "=== END HOST DIAGNOSTICS ==="

		run_llamafactory_apptainer

	else
		echo "Invalid running mode: $RUNNING_MODE"
		exit 1
	fi

elif [[ "$CLUSTER" == "TRILLIUM" ]]; then

	module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
	module load python/3.12 cuda/12.6 opencv/4.12.0
	module load arrow
	module load apptainer

	# Trillium: apptainer + H5 binds
	if command -v module >/dev/null 2>&1; then
		module load apptainer 2>/dev/null || true
	fi

	echo "=== HOST DIAGNOSTICS (${CLUSTER}) ==="
	echo "HOSTNAME: $(hostname)"
	echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
	nvidia-smi || true
	echo "=== END HOST DIAGNOSTICS ==="

	run_llamafactory_apptainer

elif [[ "$CLUSTER" == "KILLARNEY" ]]; then

	# Killarney default is VENV (see scripts/utils/env.sh). Multinode L40S CoT
	# uses node-local HF dataset cache + torchrun env vars (NNODES/NODE_RANK/...).

	if [[ "$RUNNING_MODE" == "APPTAINER" ]]; then

		module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
		module load python/3.12 cuda/12.6 opencv/4.12.0
		module load arrow
		module load apptainer

		MPI_LIB_PATH="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcc12/openmpi/4.1.5/lib"
		HWLOC_LIB_PATH="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/hwloc/2.9.1/lib"

		APPTAINER_H5_BINDS+=(-B "${MPI_LIB_PATH}:${MPI_LIB_PATH}:ro")
		APPTAINER_H5_BINDS+=(-B "${HWLOC_LIB_PATH}:${HWLOC_LIB_PATH}:ro")

		# APPTAINER_H5_BINDS is a global array already expanded inside
		# run_llamafactory_apptainer; do not pass its name as $1 (that slot is
		# optional NVIDIA bind args, e.g. "-B /usr/lib64/nvidia").
		run_llamafactory_apptainer

		# # Node-local datasets cache (replaces yaml cache_dir).
		# export HF_DATASETS_CACHE="${SLURM_TMPDIR}/hf_datasets"
		# mkdir -p "${HF_DATASETS_CACHE}"
		# export HF_DATASETS_DISABLE_FILE_LOCKING=1
		# export DATASETS_DISABLE_FILE_LOCKING=1

		# export NNODES="${SLURM_NNODES:-1}"
		# export NODE_RANK="${SLURM_NODEID:-0}"
		# export MASTER_ADDR="${MASTER_ADDR:-${HEAD_NODE}}"
		# export MASTER_PORT="${MASTER_PORT:-29500}"
		# export NPROC_PER_NODE="4"

		# apptainer run --nv --overlay ${PROJECT_DIR}/apptainer/overlay.img \
		# 	-C \
		# 	-B ${PROJECT_DIR} \
		# 	-B ${HF_HOME} \
		# 	${APPTAINER_H5_BINDS[@]+"${APPTAINER_H5_BINDS[@]}"} \
		# 	-B /home/indrisch \
		# 	-B /dev/shm:/dev/shm \
		# 	-B /etc/ssl/certs:/etc/ssl/certs:ro \
		# 	-B /etc/pki:/etc/pki:ro \
		# 	-B "${MPI_LIB_PATH}:${MPI_LIB_PATH}:ro" \
		# 	-B "${HWLOC_LIB_PATH}:${HWLOC_LIB_PATH}:ro" \
		# 	-W ${SLURM_TMPDIR} \
		# 	--env LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/lib64:/lib/x86_64-linux-gnu:/lib64:${MPI_LIB_PATH}:${HWLOC_LIB_PATH}:${LD_LIBRARY_PATH}" \
		# 	--env HF_HUB_OFFLINE=1 \
		# 	--env MPLCONFIGDIR="${SLURM_TMPDIR}/.config/matplotlib" \
		# 	--env HF_HOME="${HF_HOME}" \
		# 	--env HF_HUB_CACHE="${HF_HUB_CACHE}" \
		# 	--env HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
		# 	--env HF_DATASETS_DISABLE_FILE_LOCKING=1 \
		# 	--env DATASETS_DISABLE_FILE_LOCKING=1 \
		# 	--env TRITON_CACHE_DIR="${SLURM_TMPDIR}/.triton_cache" \
		# 	--env FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE}" \
		# 	--env TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
		# 	--env TORCH_EXTENSIONS_DIR="${SLURM_TMPDIR}/.cache/torch_extensions" \
		# 	--env PYTORCH_KERNEL_CACHE_PATH="${SLURM_TMPDIR}/.cache/torch/kernels" \
		# 	--env FORCE_TORCHRUN=1 \
		# 	--env WANDB_MODE=offline \
		# 	--env WANDB_DIR="${WANDB_DIR}" \
		# 	--env WANDB_CACHE_DIR="${SLURM_TMPDIR}/.cache/wandb" \
		# 	--env PYTHONNOUSERSITE=1 \
		# 	--env HOME="${SLURM_TMPDIR}" \
		# 	--env PYTHONPATH="${PROJECT_DIR}/src" \
		# 	--env NCCL_IB_DISABLE=0 \
		# 	--env NCCL_P2P_DISABLE=0 \
		# 	--env NCCL_DEBUG=INFO \
		# 	--env NCCL_SOCKET_IFNAME=^docker0,lo \
		# 	--env NNODES="${NNODES}" \
		# 	--env NODE_RANK="${NODE_RANK}" \
		# 	--env MASTER_ADDR="${MASTER_ADDR}" \
		# 	--env MASTER_PORT="${MASTER_PORT}" \
		# 	--env NPROC_PER_NODE="${NPROC_PER_NODE}" \
		# 	"${APPTAINER_H5_ENV[@]}" \
		# 	--pwd ${PROJECT_DIR} \
		# 	${SIF_FILE} \
		# 	llamafactory-cli train ${YAML_FILE}

	elif [[ "$RUNNING_MODE" == "VENV" ]]; then

		lmod_preflight
		module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
		module load python/3.12 cuda/12.6 opencv/4.12.0
		module load arrow

		# Shared-filesystem venvs are slow (metadata); copy a known-good cu126
		# env to node-local storage when available. Qwen2.5-VL does not need the
		# Qwen3.5 cu132 stack. For z2_offload, rebuild with DS_BUILD_CPU_ADAM=1.
		SRC_VENV="${VENV_LLAMAFACTORY_CU126:-/project/aip-wangcs/indrisch/venv_llamafactory_cu126}"
		if [[ ! -f "${SRC_VENV}/bin/activate" ]]; then
			# Fallback: scratch qwen35 cu126 env (also works for Qwen2.5-VL).
			SRC_VENV="/scratch/indrisch/venv_llamafactory_cu126_qwen35"
		fi
		if [[ -f "${SRC_VENV}/bin/activate" ]]; then
			LOCAL_VENV="${SLURM_TMPDIR}/$(basename "${SRC_VENV}")"
			echo "Copying venv ${SRC_VENV} -> ${LOCAL_VENV} ..."
			cp -a "${SRC_VENV}" "${LOCAL_VENV}"
			# shellcheck disable=SC1091
			source "${LOCAL_VENV}/bin/activate"
		else
			echo "WARNING: no prebuilt cu126 venv found; building via install_as_venv.sh"
			export DS_BUILD_CPU_ADAM=1
			export BUILD_UTILS=1
			export DS_BUILD_OPS=1
			export VENV_LLAMAFACTORY="${SLURM_TMPDIR}/venv_llamafactory_cu126_qwen35"
			"${PROJECT_DIR}/install_as_venv.sh" KILLARNEY
			# shellcheck disable=SC1091
			source "${VENV_LLAMAFACTORY}/bin/activate"
		fi

		export PYTHONUNBUFFERED=1
		export FORCE_TORCHRUN=1
		export HF_HUB_OFFLINE=1
		export WANDB_MODE=offline
		export WANDB_DIR="${WANDB_DIR}"
		export WANDB_CACHE_DIR="${SLURM_TMPDIR}/.cache/wandb"
		export TRITON_CACHE_DIR="${SLURM_TMPDIR}/.triton_cache"
		export TORCH_EXTENSIONS_DIR="${SLURM_TMPDIR}/.cache/torch_extensions"
		export PYTORCH_KERNEL_CACHE_PATH="${SLURM_TMPDIR}/.cache/torch/kernels"
		mkdir -p "${TORCH_EXTENSIONS_DIR}"
		export DISABLE_VERSION_CHECK=1
		export SCANNET_H5_DIR SPATIALSSRL_H5_DIR THINKER10K_H5_DIR
		export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
		export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

		# Use node-local cache to avoid NFS contention for datasets; replaces yaml cache_dir.
		export HF_DATASETS_CACHE="${SLURM_TMPDIR}/hf_datasets"
		mkdir -p "${HF_DATASETS_CACHE}"
		export HF_DATASETS_DISABLE_FILE_LOCKING=1
		export DATASETS_DISABLE_FILE_LOCKING=1

		echo "=== VENV DIAGNOSTICS (KILLARNEY) ==="
		echo "HOSTNAME: $(hostname)"
		echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
		echo "SLURM_NNODES: ${SLURM_NNODES:-}"
		echo "SLURM_NODEID: ${SLURM_NODEID:-}"
		echo "HF_DATASETS_CACHE: ${HF_DATASETS_CACHE}"
		nvidia-smi || true
		python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())" || true
		echo "=== END VENV DIAGNOSTICS ==="

		if ! command -v llamafactory-cli >/dev/null 2>&1; then
			echo "ERROR: llamafactory-cli is not available after activating the venv."
			exit 1
		fi

		pushd "${PROJECT_DIR}"

		if [[ "${SLURM_NNODES:-1}" -ge 2 ]]; then
			export NCCL_ASYNC_ERROR_HANDLING=1
			export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
			export NCCL_DEBUG=INFO
			export NCCL_SOCKET_IFNAME=^docker0,lo

			# LLaMA-Factory launcher reads these and invokes torchrun once per node.
			export NNODES="${SLURM_NNODES}"
			export NODE_RANK="${SLURM_NODEID}"
			export MASTER_ADDR="${MASTER_ADDR:-${HEAD_NODE}}"
			export MASTER_PORT="${MASTER_PORT:-29500}"
			export NPROC_PER_NODE="4"

			echo "NNODES: ${NNODES}"
			echo "NODE_RANK: ${NODE_RANK}"
			echo "MASTER_ADDR: ${MASTER_ADDR}"
			echo "MASTER_PORT: ${MASTER_PORT}"
			echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"
		fi

		llamafactory-cli train "${YAML_FILE}"

	else
		echo "Invalid running mode: $RUNNING_MODE"
		exit 1
	fi

else
	echo "Invalid cluster: $CLUSTER"
	exit 1
fi
