#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=0-00:10:00
#SBATCH --mem=495GB
#SBATCH --gpus-per-node=l40s:4
#SBATCH --output=out/SQA3D_h5py/%N-qwen2_5vl_lora_sft_SQA3Devery24_h5py-%j.out


RUNNING_MODE=$1 # this is optional; generally, we wouldn't use this
YAML_FILE="/project/aip-wangcs/indrisch/LLaMA-Factory/examples/train_lora/qwen2_5vl_lora_sft_SQA3Devery24_h5py.yaml"

if [[ "$RUNNING_MODE" == "APPTAINER" ]]; then

    module load apptainer

    TORCH_CUDA_ARCH_LIST="9.0" # for clusters with h100 GPUs

    # STEP 1: RUN THE TRAINING AND EVALUATION

    # better to have triton cache on a non-nfs file system for speed
    # if we are offline, we need to indicate this
    apptainer run --nv --writable-tmpfs \
        -B /project/aip-wangcs/indrisch/LLaMA-Factory \
        -B /home/indrisch \
        -B /dev/shm:/dev/shm \
        -B /etc/ssl/certs:/etc/ssl/certs:ro \
        -B /etc/pki:/etc/pki:ro \
        -W ${SLURM_TMPDIR} \
        --env HF_HUB_OFFLINE=1 \
        --env MPLCONFIGDIR="${SLURM_TMPDIR}/.config/matplotlib" \
        --env HF_HOME="/scratch/indrisch/huggingface/hub" \
        --env HF_HUB_CACHE="/scratch/indrisch/huggingface/hub" \
        --env TRITON_CACHE_DIR="${SLURM_TMPDIR}/.triton_cache" \
        --env FLASHINFER_WORKSPACE_BASE="/scratch/indrisch/" \
        --env TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
        --env TORCH_EXTENSIONS_DIR="${SLURM_TMPDIR}/.cache/torch_extensions" \
        --env PYTORCH_KERNEL_CACHE_PATH="${SLURM_TMPDIR}/.cache/torch/kernels" \
        --env FORCE_TORCHRUN=1 \
        --env WANDB_MODE=offline \
        --env WANDB_DIR="/project/aip-wangcs/indrisch/LLaMA-Factory/wandb/" \
        --pwd /project/aip-wangcs/indrisch/LLaMA-Factory \
        /project/aip-wangcs/indrisch/easyr1_verl_sif/llamafactory.sif \
        llamafactory-cli train ${YAML_FILE}

elif [[ "$RUNNING_MODE" == "VENV" ]]; then

    module load StdEnv/2023  gcc/12.3  openmpi/4.1.5
    module load python/3.12 cuda/12.6 opencv/4.12.0
    module load arrow

    export DS_BUILD_CPU_ADAM=1
    export DS_BUILD_AIO=1
    export DS_BUILD_UTILS=1
    export LIBRARY_PATH="${CUDA_HOME}/lib64:$LIBRARY_PATH"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"
    # Enable AVX512 compilation flags for DeepSpeed CPU Adam extension
    # DeepSpeed detects AVX512 and uses -D__AVX512__, but compiler needs -mavx512f to actually compile AVX512 code
    export CXXFLAGS="-mavx512f ${CXXFLAGS:-}"
    export CFLAGS="-mavx512f ${CFLAGS:-}"

    #if /project/aip-wangcs/indrisch/venv_llamafactory_cu126/bin/activate exists, use it; otherwise, build from scratch
    if [ -f /project/aip-wangcs/indrisch/venv_llamafactory_cu126/bin/activate ]; then
        source /project/aip-wangcs/indrisch/venv_llamafactory_cu126/bin/activate
    else
        pushd /project/aip-wangcs/indrisch/
        virtualenv --no-download venv_llamafactory_cu126
        source venv_llamafactory_cu126/bin/activate
        pushd /project/aip-wangcs/indrisch/LLaMA-Factory
        pip install --upgrade pip setuptools wheel
        # Note that using CPU ADAM doesn't seem to work on Killarney; hence, avoid using deepspeed versions which use offload.
        # DS_BUILD_CPU_ADAM=1 DS_BUILD_AIO=1 DS_BUILD_UTILS=1 pip install -e ".[torch,metrics,deepspeed,liger-kernel]"
        # DS_BUILD_CPU_ADAM=1 DS_BUILD_AIO=1 DS_BUILD_UTILS=1 pip install wandb ray
        pip install -e ".[torch,metrics,deepspeed,liger-kernel]"
        pip install wandb ray
    fi
    
    # Set environment variables AFTER venv activation to ensure they persist
    # Create cache directories before they're needed
    mkdir -p "${SLURM_TMPDIR}/.cache/torch_extensions"
    mkdir -p "${SLURM_TMPDIR}/.cache/torch/kernels"
    mkdir -p "${SLURM_TMPDIR}/.config/matplotlib"
    mkdir -p "${SLURM_TMPDIR}/.triton_cache"
    
    export HF_HUB_OFFLINE=1 
    export MPLCONFIGDIR="${SLURM_TMPDIR}/.config/matplotlib"
    export HF_HOME="/scratch/indrisch/huggingface/hub"
    export HF_HUB_CACHE="/scratch/indrisch/huggingface/hub"
    export TRITON_CACHE_DIR="${SLURM_TMPDIR}/.triton_cache"
    export FLASHINFER_WORKSPACE_BASE="/scratch/indrisch/"
    export TORCH_CUDA_ARCH_LIST="9.0" # for clusters with a100 GPUs
    export TORCH_EXTENSIONS_DIR="${SLURM_TMPDIR}/.cache/torch_extensions" # needed for cpu_adam
    export PYTORCH_KERNEL_CACHE_PATH="${SLURM_TMPDIR}/.cache/torch/kernels"
    export FORCE_TORCHRUN=1 
    export WANDB_MODE=offline 
    export WANDB_DIR="/project/aip-wangcs/indrisch/LLaMA-Factory/wandb/" 
    export DISABLE_VERSION_CHECK=1 # since the automatic detector doesn't automatically see that transformers==4.57.1+computecanada is the same as transformers==4.57.1
    # giving the slow tokenizer a try: https://github.com/hiyouga/LLaMA-Factory/issues/8600#issuecomment-3227071979


    pushd /project/aip-wangcs/indrisch/LLaMA-Factory
    llamafactory-cli train ${YAML_FILE}

else
    echo "Invalid running mode: $RUNNING_MODE"
    exit 1
fi