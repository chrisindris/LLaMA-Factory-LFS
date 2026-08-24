#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=out/%N-qwen2_5vl_lora_sft_CoT_eval-%j.out

# RORQUAL:
#SBATCH --cpus-per-task=64
#SBATCH --time=1-12:00:00
#SBATCH --mem=485G
#SBATCH --gpus-per-node=h100:4

# TRILLIUM / NIBI:
#SBATCH --cpus-per-task=96
#SBATCH --time=1-12:00:00
#SBATCH --gpus-per-node=h100:4

# KILLARNEY:
#SBATCH --cpus-per-task=48
#SBATCH --time=1-12:00:00
#SBATCH --mem=1900G
#SBATCH --gpus-per-node=h100:8

# ---------------------------------------------------------------------
# ------ qwen2_5vl_lora_sft_CoT_eval ---------------------------------
# ---------------------------------------------------------------------
#
# Eval-only job (no training). Prefer launching via the cluster wrapper:
#   models/qwen2_5vl_lora_sft_CoT/${CLUSTER,,}_slurm_qwen2_5vl_lora_sft_CoT_eval.sh
#
# Uses llamafactory-cli train with a YAML that sets do_train: false and
# do_eval: true (llamafactory-cli eval is unsupported). Expects
# save_eval_predictions so the run writes:
#   ${OUTPUT_DIR}/eval_predictions.json
#
# YAML_FILE / OUTPUT_DIR are normally exported by the wrapper after
# materializing a per-model config from the eval template.
#
# RUNNING_MODE (from env.sh / cluster defaults): APPTAINER | VENV | SHELL
#

# --- for reading cluster-specific settings ---
. ../../scripts/utils/env.sh

# ----- HEADER: ENV VARIABLES -----

EXPERIMENT_NAME="qwen2_5vl_lora_sft_CoT_eval"

export PYTHONUNBUFFERED=1

# Prefer this checkout's src so prediction-dump flags (save_eval_predictions)
# resolve here even when the shared venv / SIF points at another tree.
export PYTHONPATH="${PROJECT_DIR}/src:${SYSCONFIG_DIR_PATH}${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$RUNNING_MODE" == "SHELL" ]]; then
	export SLURM_TMPDIR="/tmp"
fi
echo "SLURM_TMPDIR: ${SLURM_TMPDIR}"

if [[ "$CLUSTER" == "RORQUAL" ]]; then
	export SCANNET_H5_DIR="/project/def-wangcs/indrisch/scratch_saves/ScanNet_h5/scans"
fi

# H5 roots (override per-cluster if needed)
export SCANNET_H5_DIR="${SCANNET_H5_DIR:-/scratch/indrisch/ScanNet_h5/scans}"
export SPATIALSSRL_H5_DIR="${SPATIALSSRL_H5_DIR:-/scratch/indrisch/Spatial-SSRL_images_h5}"
export THINKER10K_H5_DIR="${THINKER10K_H5_DIR:-/scratch/indrisch/3DThinker10K_images_h5}"
echo "SCANNET_H5_DIR: $SCANNET_H5_DIR"
echo "SPATIALSSRL_H5_DIR: $SPATIALSSRL_H5_DIR"
echo "THINKER10K_H5_DIR: $THINKER10K_H5_DIR"

export WANDB_DIR="${PROJECT_DIR}/wandb/"
if [[ "$BEST_GPU" == "h100" ]]; then
	export TORCH_CUDA_ARCH_LIST="9.0"
else
	export TORCH_CUDA_ARCH_LIST="8.0"
fi

YAML_FILE="${YAML_FILE:-${PROJECT_DIR}/examples/train_lora/${CLUSTER,,}_${EXPERIMENT_NAME}.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/saves/qwen2_5vl-7b/lora/sft/CoT_eval}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PROJECT_DIR}/models/qwen2_5vl_lora_sft_CoT/out"

echo "CLUSTER: $CLUSTER"
echo "RUNNING_MODE: $RUNNING_MODE"
echo "YAML_FILE: $YAML_FILE"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "PYTHONPATH: $PYTHONPATH"

if [[ ! -f "$YAML_FILE" ]]; then
	echo "Error: YAML config not found: $YAML_FILE"
	exit 1
fi

# Shared apptainer env flags for multimodal H5 eval
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

# Fail early if a host/venv llamafactory lacks prediction-dump flags.
# Safe no-op when host cannot import llamafactory (pure APPTAINER path).
check_prediction_dump_flags() {
	if ! command -v python3 >/dev/null 2>&1 && ! command -v python >/dev/null 2>&1; then
		echo "Skipping prediction-dump preflight (no host python)."
		return 0
	fi
	local py=python3
	command -v python3 >/dev/null 2>&1 || py=python
	if ! ${py} -c "import llamafactory" 2>/dev/null; then
		echo "Skipping prediction-dump preflight (llamafactory not importable on host)."
		return 0
	fi
	${py} - <<'PY'
import pathlib
import llamafactory

path = pathlib.Path(llamafactory.__file__).resolve()
print("llamafactory=", path)
fa = path.parent / "hparams" / "finetuning_args.py"
text = fa.read_text(encoding="utf-8") if fa.is_file() else ""
if "save_eval_predictions" not in text:
    raise SystemExit(
        f"Loaded llamafactory without prediction-dump flags: {path}\n"
        "Fix: ensure PYTHONPATH starts with $PROJECT_DIR/src"
    )
print("flags_ok=True (save_eval_predictions present)")
PY
}

setup_venv_runtime_env() {
	export PYTHONUNBUFFERED=1
	export NCCL_DEBUG=INFO
	export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
	export FORCE_TORCHRUN=1
	export HF_HUB_OFFLINE=1
	export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
	export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
	export WANDB_MODE=offline
	export WANDB_DIR="${WANDB_DIR}"
	export WANDB_CACHE_DIR="${SLURM_TMPDIR}/.cache/wandb"
	export TRITON_CACHE_DIR="${SLURM_TMPDIR}/.triton_cache"
	export DISABLE_VERSION_CHECK=1
	export SCANNET_H5_DIR SPATIALSSRL_H5_DIR THINKER10K_H5_DIR
	# Re-assert after venv activate (site-packages often reorder PYTHONPATH).
	export PYTHONPATH="${PROJECT_DIR}/src:${SYSCONFIG_DIR_PATH}${PYTHONPATH:+:$PYTHONPATH}"
}

check_eval_prediction_artifact() {
	echo "=== dump artifacts ==="
	if [[ -f "${OUTPUT_DIR}/eval_predictions.json" ]]; then
		ls -la "${OUTPUT_DIR}/eval_predictions.json"
	else
		echo "eval_predictions.json not found under OUTPUT_DIR=${OUTPUT_DIR}"
		echo "(check logs / save_eval_predictions / question_id column)"
	fi
	echo "======================"
}

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
	else
		# APPTAINER (and default): eval-only via train entrypoint + eval YAML
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
	export TORCH_DISTRIBUTED_DEBUG=DETAIL && echo "TORCH_DISTRIBUTED_DEBUG: ${TORCH_DISTRIBUTED_DEBUG}"
	export TORCH_NCCL_TRACE_BUFFER_SIZE=20000 && echo "TORCH_NCCL_TRACE_BUFFER_SIZE: ${TORCH_NCCL_TRACE_BUFFER_SIZE}"
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

	if [ -z "${OVERLAY:-}" ]; then
		if [[ "$NODE_RANK" == 0 ]]; then
			if [ -f "${PROJECT_DIR}/apptainer/overlay_${NODE_RANK}.img" ]; then
				OVERLAY="${PROJECT_DIR}/apptainer/overlay_${NODE_RANK}.img"
			else
				OVERLAY="${PROJECT_DIR}/apptainer/overlay.img"
			fi
		else
			OVERLAY="${PROJECT_DIR}/apptainer/overlay_${NODE_RANK}.img"
		fi
	fi

	export MAX_JOBS="${MAX_JOBS:-16}"

	# Prefer host src so H5 image backends and prediction-dump trainer code
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
		--env TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG}" \
		--env TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE}" \
		--env NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
		--env CUDA_HOME="${APPTAINERENV_CUDA_HOME}" \
		--env NNODES="${NNODES}" \
		--env NODE_RANK="${NODE_RANK}" \
		--env MASTER_ADDR="${MASTER_ADDR}" \
		--env MASTER_PORT="${MASTER_PORT}" \
		--env NPROC_PER_NODE="${NPROC_PER_NODE}" \
		--env MAX_JOBS="${MAX_JOBS}" \
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

	# STEP 1: RUN EVALUATION (eval-only YAML; prediction dumps via save_eval_predictions)

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
		# shellcheck disable=SC1091
		source ${SLURM_TMPDIR}/venv_llamafactory_cu126/bin/activate
		setup_venv_runtime_env
		check_prediction_dump_flags

		echo "=== VENV DIAGNOSTICS ==="
		echo "HOSTNAME: $(hostname)"
		echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
		echo "SLURM_GPUS: $SLURM_GPUS"
		echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
		nvidia-smi
		python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"
		echo "=== END VENV DIAGNOSTICS ==="

		pushd "${PROJECT_DIR}"
		llamafactory-cli train ${YAML_FILE}
		popd

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
		# shellcheck disable=SC1091
		source ${SLURM_TMPDIR}/venv_llamafactory_cu126/bin/activate
		setup_venv_runtime_env
		check_prediction_dump_flags

		pushd "${PROJECT_DIR}"
		llamafactory-cli train ${YAML_FILE}
		popd

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

	if [[ "$RUNNING_MODE" == "VENV" ]]; then

		module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
		module load python/3.12 cuda/12.6 opencv/4.12.0
		module load arrow

		if [[ -f /scratch/indrisch/venv_llamafactory_cu126/bin/activate ]]; then
			echo "Copying venv to local storage..."
			cp -a /scratch/indrisch/venv_llamafactory_cu126 ${SLURM_TMPDIR}/venv_llamafactory_cu126
			# shellcheck disable=SC1091
			source ${SLURM_TMPDIR}/venv_llamafactory_cu126/bin/activate
		elif [[ -n "${VENV_LLAMAFACTORY:-}" && -f "${VENV_LLAMAFACTORY}/bin/activate" ]]; then
			# shellcheck disable=SC1091
			source "${VENV_LLAMAFACTORY}/bin/activate"
		else
			echo "Error: no venv found for RUNNING_MODE=VENV on TRILLIUM"
			exit 1
		fi
		setup_venv_runtime_env
		check_prediction_dump_flags

		echo "=== VENV DIAGNOSTICS (${CLUSTER}) ==="
		echo "HOSTNAME: $(hostname)"
		echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
		nvidia-smi || true
		python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"
		echo "=== END VENV DIAGNOSTICS ==="

		pushd "${PROJECT_DIR}"
		llamafactory-cli train ${YAML_FILE}
		popd

	else
		# Default on Trillium: APPTAINER (or SHELL interactive)
		module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
		module load python/3.12 cuda/12.6 opencv/4.12.0
		module load arrow
		module load apptainer

		if command -v module >/dev/null 2>&1; then
			module load apptainer 2>/dev/null || true
		fi

		echo "=== HOST DIAGNOSTICS (${CLUSTER}) ==="
		echo "HOSTNAME: $(hostname)"
		echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
		echo "RUNNING_MODE: $RUNNING_MODE"
		nvidia-smi || true
		echo "=== END HOST DIAGNOSTICS ==="

		run_llamafactory_apptainer
	fi

elif [[ "$CLUSTER" == "KILLARNEY" ]]; then

	if [[ "$RUNNING_MODE" == "APPTAINER" ]]; then

		module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
		module load python/3.12 cuda/12.6 opencv/4.12.0
		module load arrow
		module load apptainer

		MPI_LIB_PATH="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcc12/openmpi/4.1.5/lib"
		HWLOC_LIB_PATH="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/hwloc/2.9.1/lib"

		apptainer run --nv --overlay ${PROJECT_DIR}/apptainer/overlay.img \
			-C \
			-B ${PROJECT_DIR} \
			-B ${HF_HOME} \
			${APPTAINER_H5_BINDS[@]+"${APPTAINER_H5_BINDS[@]}"} \
			-B /home/indrisch \
			-B /dev/shm:/dev/shm \
			-B /etc/ssl/certs:/etc/ssl/certs:ro \
			-B /etc/pki:/etc/pki:ro \
			-B "${MPI_LIB_PATH}:${MPI_LIB_PATH}:ro" \
			-B "${HWLOC_LIB_PATH}:${HWLOC_LIB_PATH}:ro" \
			-W ${SLURM_TMPDIR} \
			--env LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/lib64:/lib/x86_64-linux-gnu:/lib64:${MPI_LIB_PATH}:${HWLOC_LIB_PATH}:${LD_LIBRARY_PATH}" \
			--env HF_HUB_OFFLINE=1 \
			--env MPLCONFIGDIR="${SLURM_TMPDIR}/.config/matplotlib" \
			--env HF_HOME="${HF_HOME}" \
			--env HF_HUB_CACHE="${HF_HUB_CACHE}" \
			--env TRITON_CACHE_DIR="${SLURM_TMPDIR}/.triton_cache" \
			--env FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE}" \
			--env TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
			--env TORCH_EXTENSIONS_DIR="${SLURM_TMPDIR}/.cache/torch_extensions" \
			--env PYTORCH_KERNEL_CACHE_PATH="${SLURM_TMPDIR}/.cache/torch/kernels" \
			--env FORCE_TORCHRUN=1 \
			--env WANDB_MODE=offline \
			--env WANDB_DIR="${WANDB_DIR}" \
			--env WANDB_CACHE_DIR="${SLURM_TMPDIR}/.cache/wandb" \
			--env PYTHONNOUSERSITE=1 \
			--env HOME="${SLURM_TMPDIR}" \
			--env PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}" \
			--env NCCL_IB_DISABLE=0 \
			--env NCCL_P2P_DISABLE=0 \
			--env NCCL_DEBUG=INFO \
			--env NCCL_SOCKET_IFNAME=^docker0,lo \
			"${APPTAINER_H5_ENV[@]}" \
			--pwd ${PROJECT_DIR} \
			${SIF_FILE} \
			llamafactory-cli train ${YAML_FILE}

	elif [[ "$RUNNING_MODE" == "VENV" ]]; then

		module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
		module load python/3.12 cuda/12.6 opencv/4.12.0
		module load arrow

		# shellcheck disable=SC1091
		source /project/aip-wangcs/indrisch/venv_llamafactory_cu126/bin/activate
		export CUDA_VISIBLE_DEVICES=0,1,2,3
		setup_venv_runtime_env
		check_prediction_dump_flags

		pushd "${PROJECT_DIR}"
		llamafactory-cli train ${YAML_FILE}
		popd

	else
		echo "Invalid running mode: $RUNNING_MODE"
		exit 1
	fi

else
	echo "Invalid cluster: $CLUSTER"
	exit 1
fi

check_eval_prediction_artifact
