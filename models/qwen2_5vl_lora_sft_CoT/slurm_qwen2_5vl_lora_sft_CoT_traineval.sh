#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=out/%N-qwen2_5vl_lora_sft_CoT_traineval-%j.out

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
# ------------ qwen2_5vl_lora_sft_CoT_traineval -----------------------
# ---------------------------------------------------------------------
#
# SFT (LoRA) Qwen2.5-VL on the CoT mix:
#   Scene30k + SpatialSSRL_coldstart + 3DThinker10k  (mix_strategy=concat)
#
# Images:
#   Scene30k  -> SCANNET_H5_DIR (default /scratch/indrisch/ScanNet_h5/scans)
#   Spatial   -> SPATIALSSRL_H5_DIR
#   3DThinker -> THINKER10K_H5_DIR
#

# ----- HEADER: ENV VARIABLES -----

EXPERIMENT_NAME="qwen2_5vl_lora_sft_CoT_traineval"

# --- for reading cluster-specific settings ---

if [[ "$PWD" == *LLaMA-Factory-LFS* ]]; then
    PROJECT_DIR="${PWD%%LLaMA-Factory-LFS*}/LLaMA-Factory-LFS"
elif [[ "$PWD" == *LLaMA-Factory* ]]; then
    PROJECT_DIR="${PWD%%LLaMA-Factory*}/LLaMA-Factory"
else
    echo "Error: Could not find 'LLaMA-Factory' or 'LLaMA-Factory-LFS' in the current path."
    exit 1
fi
SYSCONFIG_DIR_PATH="$PROJECT_DIR/scripts"
export PYTHONPATH="$PYTHONPATH:$SYSCONFIG_DIR_PATH"

# --- setting environment ---

# Detect cluster based on terminal prompt or hostname
if [[ "$PS1" == *"rorqual"* ]] || [[ "$HOSTNAME" == *"rorqual"* ]] || [[ "$PS1" == *"rg"* ]] || [[ "$HOSTNAME" == *"rg"* ]]; then
    CLUSTER="RORQUAL"
    RUNNING_MODE="APPTAINER"
    SCANNET_H5_DIR="/project/def-wangcs/indrisch/scratch_saves/ScanNet_h5/scans"
elif [[ "$PS1" == *"trig"* ]] || [[ "$HOSTNAME" == *"trig"* ]]; then
    CLUSTER="TRILLIUM"
    RUNNING_MODE="APPTAINER"
elif [[ "$PS1" == *"klogin"* ]] || [[ "$HOSTNAME" == *"klogin"* ]] || [[ "$PS1" == *"kn"* ]] || [[ "$HOSTNAME" == *"kn"* ]]; then
    CLUSTER="KILLARNEY"
    RUNNING_MODE="VENV"
elif [[ "$HOSTNAME" == *"nibi"* ]] || [[ "$PS1" == *"nibi"* ]] || [[ "$PS1" == *"g"* ]] || [[ "$HOSTNAME" == *"g"* ]]; then
    CLUSTER="NIBI"
    RUNNING_MODE="APPTAINER"
else
    echo "Warning: Could not detect cluster from PS1 or HOSTNAME. Defaulting to NIBI."
    CLUSTER="NIBI"
    RUNNING_MODE="APPTAINER"
fi

if [[ "$RUNNING_MODE" == "SHELL" ]]; then
    export SLURM_TMPDIR="/tmp"
fi

export HF_HOME="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'HF_HOME'))")" && echo "HF_HOME: $HF_HOME"
export HF_HUB_CACHE="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'HF_HUB_CACHE'))")" && echo "HF_HUB_CACHE: $HF_HUB_CACHE"
export TRITON_CACHE_DIR="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'TRITON_CACHE_DIR'))")" && echo "TRITON_CACHE_DIR: $TRITON_CACHE_DIR"
export FLASHINFER_WORKSPACE_BASE="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'FLASHINFER_WORKSPACE_BASE'))")" && echo "FLASHINFER_WORKSPACE_BASE: $FLASHINFER_WORKSPACE_BASE"
export BEST_GPU="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'BEST_GPU'))")" && echo "BEST_GPU: $BEST_GPU"
export TORCH_EXTENSIONS_DIR="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'TORCH_EXTENSIONS_DIR'))")" && echo "TORCH_EXTENSIONS_DIR: $TORCH_EXTENSIONS_DIR"
export SIF_FILE="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'SIF_FILE'))")" && echo "SIF_FILE: $SIF_FILE"
export MEDIA_DIR="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'media_dir'))")" && echo "MEDIA_DIR: $MEDIA_DIR"

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

YAML_FILE="${PROJECT_DIR}/examples/train_lora/${CLUSTER,,}_${EXPERIMENT_NAME}.yaml"
OUTPUT_DIR="${PROJECT_DIR}/saves/qwen2_5vl-7b/lora/sft/CoT_traineval"
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

    # Prefer host src so H5 image backends (ScanNet_h5 / Spatial-SSRL / 3DThinker)
    # from this checkout override the older /app/src baked into the SIF.
    apptainer run --nv --fakeroot --overlay /scratch/indrisch/LLaMA-Factory/apptainer/overlay.img \
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
        --env NCCL_DEBUG=INFO \
        --env NCCL_SOCKET_IFNAME=^docker0,lo \
        --env CUDA_HOME="${APPTAINERENV_CUDA_HOME}" \
        "${APPTAINER_H5_ENV[@]}" \
        --pwd ${PROJECT_DIR} \
        ${SIF_FILE} \
        llamafactory-cli train ${YAML_FILE}
}


if [[ "$CLUSTER" == "NIBI" ]]; then

    # STEP 1: RUN THE TRAINING AND EVALUATION

    # better to have triton cache on a non-nfs file system for speed
    # if we are offline, we need to indicate this

    if [[ "$RUNNING_MODE" == "APPTAINER" ]]; then

        module load StdEnv/2023  gcc/12.3  openmpi/4.1.5
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

        module load StdEnv/2023  gcc/12.3  openmpi/4.1.5
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

        module load StdEnv/2023  gcc/12.3  openmpi/4.1.5
        module load python/3.12 cuda/12.6 opencv/4.12.0
        module load arrow
        module load apptainer

        echo "=== HOST DIAGNOSTICS ==="
        echo "HOSTNAME: $(hostname)"
        echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
        nvidia-smi
        echo "=== END HOST DIAGNOSTICS ==="

        run_llamafactory_apptainer

        # NVIDIA_LIB_DIR=$(dirname "$(ldconfig -p 2>/dev/null | grep 'libcuda\.so ' | awk '{print $NF}' | head -1)" 2>/dev/null)
        # NVIDIA_BIND_ARGS=""
        # if [[ -n "$NVIDIA_LIB_DIR" && -d "$NVIDIA_LIB_DIR" ]]; then
        #     echo "Found NVIDIA driver libs at: $NVIDIA_LIB_DIR"
        #     NVIDIA_BIND_ARGS="-B ${NVIDIA_LIB_DIR}"
        # fi

        # apptainer run --nv --fakeroot --overlay /scratch/indrisch/LLaMA-Factory/apptainer/overlay.img \
        #     ${NVIDIA_BIND_ARGS} \
        #     -B ${PROJECT_DIR} \
        #     -B ${HF_HOME} \
        #     ${APPTAINER_H5_BINDS[@]+"${APPTAINER_H5_BINDS[@]}"} \
        #     -B /home/indrisch \
        #     -B /dev/shm:/dev/shm \
        #     -B /etc/ssl/certs:/etc/ssl/certs:ro \
        #     -B /etc/pki:/etc/pki:ro \
        #     -W ${SLURM_TMPDIR} \
        #     --env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
        #     --env PYTHONUNBUFFERED=1 \
        #     --env NCCL_DEBUG=INFO \
        #     --env HF_HUB_OFFLINE=1 \
        #     --env MPLCONFIGDIR="${SLURM_TMPDIR}/.config/matplotlib" \
        #     --env HF_HOME="${HF_HOME}" \
        #     --env HF_HUB_CACHE="${HF_HUB_CACHE}" \
        #     --env TRITON_CACHE_DIR="${SLURM_TMPDIR}/.triton_cache" \
        #     --env FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE}" \
        #     --env TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
        #     --env TORCH_EXTENSIONS_DIR="${SLURM_TMPDIR}/.cache/torch_extensions" \
        #     --env PYTORCH_KERNEL_CACHE_PATH="${SLURM_TMPDIR}/.cache/torch/kernels" \
        #     --env FORCE_TORCHRUN=1 \
        #     --env WANDB_MODE=offline \
        #     --env WANDB_DIR="${WANDB_DIR}" \
        #     --env WANDB_CACHE_DIR="${SLURM_TMPDIR}/.cache/wandb" \
        #     "${APPTAINER_H5_ENV[@]}" \
        #     --pwd ${PROJECT_DIR} \
        #     ${SIF_FILE} \
        #     llamafactory-cli train ${YAML_FILE}

    elif [[ "$RUNNING_MODE" == "VENV" ]]; then

        module load StdEnv/2023  gcc/12.3  openmpi/4.1.5
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

        module load StdEnv/2023  gcc/12.3  openmpi/4.1.5
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

    module load StdEnv/2023  gcc/12.3  openmpi/4.1.5
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

    if [[ "$RUNNING_MODE" == "APPTAINER" ]]; then

        module load StdEnv/2023  gcc/12.3  openmpi/4.1.5
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
            --env PYTHONPATH="${PROJECT_DIR}/src" \
            --env NCCL_IB_DISABLE=0 \
            --env NCCL_P2P_DISABLE=0 \
            --env NCCL_DEBUG=INFO \
            --env NCCL_SOCKET_IFNAME=^docker0,lo \
            "${APPTAINER_H5_ENV[@]}" \
            --pwd ${PROJECT_DIR} \
            ${SIF_FILE} \
            llamafactory-cli train ${YAML_FILE}

    elif [[ "$RUNNING_MODE" == "VENV" ]]; then

        module load StdEnv/2023  gcc/12.3  openmpi/4.1.5
        module load python/3.12 cuda/12.6 opencv/4.12.0
        module load arrow

        source /project/aip-wangcs/indrisch/venv_llamafactory_cu126/bin/activate
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        export FORCE_TORCHRUN=1
        export HF_HUB_OFFLINE=1
        export WANDB_MODE=offline
        export WANDB_DIR="${WANDB_DIR}"
        export WANDB_CACHE_DIR="${SLURM_TMPDIR}/.cache/wandb"
        export TRITON_CACHE_DIR="${SLURM_TMPDIR}/.triton_cache"
        export DISABLE_VERSION_CHECK=1
        export SCANNET_H5_DIR SPATIALSSRL_H5_DIR THINKER10K_H5_DIR
        pushd /project/aip-wangcs/indrisch/LLaMA-Factory
        llamafactory-cli train ${YAML_FILE}

    else
        echo "Invalid running mode: $RUNNING_MODE"
        exit 1
    fi

else
    echo "Invalid cluster: $CLUSTER"
    exit 1
fi
