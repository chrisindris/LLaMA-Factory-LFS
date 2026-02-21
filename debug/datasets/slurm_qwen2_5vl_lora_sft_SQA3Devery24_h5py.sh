#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=out/SQA3D_h5py/%N-qwen2_5vl_lora_sft_SQA3Devery24_h5py-%j.out

# SBATCH directives - Default to RORQUAL settings
# For TRILLIUM: Uncomment the TRILLIUM section below and comment out conflicting RORQUAL directives,
# or override via sbatch command line arguments
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16  
#SBATCH --time=0-00:10:00
#SBATCH --mem=120GB
#SBATCH --gpus-per-node=h100:1

# For TRILLIUM (uncomment these and comment out the conflicting RORQUAL directives above):
# #SBATCH --cpus-per-task=24
# #SBATCH --time=0-00:15:00

# For KILLARNEY (uncomment these and comment out the conflicting RORQUAL directives above):
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=0-00:10:00
#SBATCH --mem=495GB
#SBATCH --gpus-per-node=l40s:4

# ----- HEADER: ENV VARIABLES -----

EXPERIMENT_NAME="qwen2_5vl_lora_sft_SQA3Devery24_h5py"

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
elif [[ "$PS1" == *"trig"* ]] || [[ "$HOSTNAME" == *"trig"* ]]; then
    CLUSTER="TRILLIUM"
elif [[ "$PS1" == *"klogin"* ]] || [[ "$HOSTNAME" == *"klogin"* ]] || [[ "$PS1" == *"kn"* ]] || [[ "$HOSTNAME" == *"kn"* ]]; then
    CLUSTER="KILLARNEY"
else
    echo "Warning: Could not detect cluster from PS1 or HOSTNAME. Defaulting to RORQUAL."
    CLUSTER="RORQUAL"
fi

# Set cluster-specific variables
echo "Detected cluster: $CLUSTER"
if [[ "$CLUSTER" == "RORQUAL" ]]; then
    # RORQUAL-specific paths and settings
    SIF_PATH="/scratch/indrisch/huggingface/hub/datasets--cvis-tmu--easyr1_verl_sif/snapshots/382a3b3e54a9fa9450c6c99dd83efaa2f0ca4a5a/llamafactory.sif"
    YAML_FILE="/scratch/indrisch/LLaMA-Factory/examples/train_lora/qwen2_5vl_lora_sft_SQA3Devery24_h5py.yaml"
    APPTAINER_EXTRA_FLAGS="-C"
    APPTAINER_NCCL_ENV="--env NCCL_IB_DISABLE=0 --env NCCL_P2P_DISABLE=0 --env NCCL_DEBUG=INFO --env NCCL_SOCKET_IFNAME=^docker0,lo"
    USE_CUDA_VISIBLE_DEVICES=1
    RUNNING_MODE="APPTAINER"
elif [[ "$CLUSTER" == "TRILLIUM" ]]; then
    # TRILLIUM-specific paths and settings
    SIF_PATH="/scratch/indrisch/easyr1_verl_sif/llamafactory.sif"
    YAML_FILE="/scratch/indrisch/LLaMA-Factory/examples/train_lora/qwen2_5vl_lora_sft_SQA3Devery24_h5py.yaml"
    APPTAINER_EXTRA_FLAGS=""
    APPTAINER_NCCL_ENV=""
    USE_CUDA_VISIBLE_DEVICES=0
    RUNNING_MODE="APPTAINER"
elif [[ "$CLUSTER" == "KILLARNEY" ]]; then
    # KILLARNEY-specific paths and settings
    SIF_PATH="/project/aip-wangcs/indrisch/easyr1_verl_sif/llamafactory.sif"
    YAML_FILE="/project/aip-wangcs/indrisch/LLaMA-Factory/examples/train_lora/${CLUSTER,,}_qwen2_5vl_lora_sft_SQA3Devery24_h5py.yaml"
    APPTAINER_EXTRA_FLAGS=""
    APPTAINER_NCCL_ENV=""
    USE_CUDA_VISIBLE_DEVICES=0
    RUNNING_MODE="VENV"
fi

# Get MPI library paths for bind mounting (RORQUAL-specific, commented out)
# MPI_LIB_PATH="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcc12/openmpi/4.1.5/lib"
# HWLOC_LIB_PATH="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/hwloc/2.9.1/lib"
# # Gentoo system libraries (contains libpciaccess and other system deps)
# GENTOO_LIB64_PATH="/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/lib64"

export HF_HOME="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'HF_HOME'))")" && echo "HF_HOME: $HF_HOME"
export HF_HUB_CACHE="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'HF_HUB_CACHE'))")" && echo "HF_HUB_CACHE: $HF_HUB_CACHE"
export TRITON_CACHE_DIR="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'TRITON_CACHE_DIR'))")" && echo "TRITON_CACHE_DIR: $TRITON_CACHE_DIR"
export FLASHINFER_WORKSPACE_BASE="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'FLASHINFER_WORKSPACE_BASE'))")" && echo "FLASHINFER_WORKSPACE_BASE: $FLASHINFER_WORKSPACE_BASE"
export BEST_GPU="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'BEST_GPU'))")" && echo "BEST_GPU: $BEST_GPU"
export TORCH_EXTENSIONS_DIR="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'TORCH_EXTENSIONS_DIR'))")" && echo "TORCH_EXTENSIONS_DIR: $TORCH_EXTENSIONS_DIR"
export SIF_FILE="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'SIF_FILE'))")" && echo "SIF_FILE: $SIF_FILE"

export WANDB_DIR="${PROJECT_DIR}/wandb/"
if [[ "$BEST_GPU" == "h100" ]]; then
    export TORCH_CUDA_ARCH_LIST="9.0"
else
    export TORCH_CUDA_ARCH_LIST="8.0"
fi

YAML_FILE="${PROJECT_DIR}/examples/train_lora/${CLUSTER,,}_${EXPERIMENT_NAME}.yaml"
OUTPUT_DIR="${PROJECT_DIR}/saves/videor1/lora/sft/SQA3Devery24_traineval"

if [[ -z "$RUNNING_MODE" ]]; then
    RUNNING_MODE=$1
fi

# ----- EXPERIMENT -----

if [[ "$CLUSTER" == "RORQUAL" ]]; then

    if [[ "$RUNNING_MODE" == "APPTAINER" ]]; then

        module load apptainer

        TORCH_CUDA_ARCH_LIST="9.0" # for clusters with h100 GPUs
        
        # Set CUDA_VISIBLE_DEVICES: use SLURM's value if set, otherwise default to 0,1,2,3
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

        apptainer run --nv --writable-tmpfs \
            -C \
            -B ${PROJECT_DIR} \
            -B /home/indrisch \
            -B /dev/shm:/dev/shm \
            -B /etc/ssl/certs:/etc/ssl/certs:ro \
            -B /etc/pki:/etc/pki:ro \
            -W ${SLURM_TMPDIR} \
            --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
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
            --env PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}" \
            --env NCCL_IB_DISABLE=0 \
            --env NCCL_P2P_DISABLE=0 \
            --env NCCL_DEBUG=INFO \
            --env NCCL_SOCKET_IFNAME=^docker0,lo \
            --pwd ${PROJECT_DIR} \
            ${SIF_FILE} \
            llamafactory-cli train ${YAML_FILE}

    elif [[ "$RUNNING_MODE" == "SHELL" ]]; then

        module load apptainer

        TORCH_CUDA_ARCH_LIST="9.0" # for clusters with h100 GPUs

        SLURM_TMPDIR="${SLURM_TMPDIR}/tmp"
        
        # Create directories for pip cache and temporary files on scratch
        mkdir -p ${SLURM_TMPDIR}/tmp
        mkdir -p ${SLURM_TMPDIR}/.cache/pip
        mkdir -p ${SLURM_TMPDIR}/.cache/torch_extensions
        mkdir -p ${SLURM_TMPDIR}/.cache/torch/kernels
        mkdir -p ${SLURM_TMPDIR}/.config/matplotlib
        mkdir -p ${SLURM_TMPDIR}/.triton_cache

        apptainer run --nv --writable-tmpfs \
            -C \
            -B ${PROJECT_DIR} \
            -B /home/indrisch \
            -B /dev/shm:/dev/shm \
            -B /etc/ssl/certs:/etc/ssl/certs:ro \
            -B /etc/pki:/etc/pki:ro \
            -W ${SLURM_TMPDIR} \
            --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
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
            --env PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}" \
            --env NCCL_IB_DISABLE=0 \
            --env NCCL_P2P_DISABLE=0 \
            --env NCCL_DEBUG=INFO \
            --env NCCL_SOCKET_IFNAME=^docker0,lo \
            --pwd ${PROJECT_DIR} \
            ${SIF_FILE} \
            bash

        # apptainer run --nv --writable-tmpfs \
        #     -C \
        #     -B /scratch/indrisch \
        #     -B /home/indrisch \
        #     -B /dev/shm:/dev/shm \
        #     -B /etc/ssl/certs:/etc/ssl/certs:ro \
        #     -B /etc/pki:/etc/pki:ro \
        #     -W ${SLURM_TMPDIR} \
        #     --env CUDA_VISIBLE_DEVICES=0,1,2,3 \
        #     --env HF_HUB_OFFLINE=1 \
        #     --env TMPDIR="/scratch/indrisch/tmp" \
        #     --env PIP_CACHE_DIR="/scratch/indrisch/.cache/pip" \
        #     --env MPLCONFIGDIR="/scratch/indrisch/.config/matplotlib" \
        #     --env HF_HOME="/scratch/indrisch/huggingface/hub" \
        #     --env HF_HUB_CACHE="/scratch/indrisch/huggingface/hub" \
        #     --env TRITON_CACHE_DIR="/scratch/indrisch/.triton_cache" \
        #     --env FLASHINFER_WORKSPACE_BASE="/scratch/indrisch/" \
        #     --env TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
        #     --env TORCH_EXTENSIONS_DIR="/scratch/indrisch/.cache/torch_extensions" \
        #     --env PYTORCH_KERNEL_CACHE_PATH="/scratch/indrisch/.cache/torch/kernels" \
        #     --env FORCE_TORCHRUN=1 \
        #     --env WANDB_MODE=offline \
        #     --env WANDB_DIR="/scratch/indrisch/LLaMA-Factory/wandb/" \
        #     --env PYTHONPATH="/scratch/indrisch/LLaMA-Factory/src:${PYTHONPATH:-}" \
        #     --pwd /scratch/indrisch/LLaMA-Factory \
        #     /scratch/indrisch/huggingface/hub/datasets--cvis-tmu--easyr1_verl_sif/snapshots/382a3b3e54a9fa9450c6c99dd83efaa2f0ca4a5a/llamafactory.sif \
        #     bash


    elif [[ "$RUNNING_MODE" == "VENV" ]]; then

        module load StdEnv/2023  gcc/12.3  openmpi/4.1.5
        module load python/3.12 cuda/12.6 opencv/4.12.0
        module load arrow

        source /scratch/indrisch/venv_llamafactory_cu126/bin/activate

        # pushd /scratch/indrisch/
        # module load StdEnv/2023  gcc/12.3  openmpi/4.1.5
        # module load python/3.12 cuda/12.6 opencv/4.12.0
        # module load arrow
        # virtualenv --no-download venv_llamafactory_cu126
        # source venv_llamafactory_cu126/bin/activate
        # popd
        
        # Set environment variables AFTER venv activation to ensure they persist
        # Create cache directories before they're needed
        mkdir -p "${SLURM_TMPDIR}/.cache/torch_extensions"
        mkdir -p "${SLURM_TMPDIR}/.cache/torch/kernels"
        mkdir -p "${SLURM_TMPDIR}/.config/matplotlib"
        mkdir -p "${SLURM_TMPDIR}/.triton_cache"
        
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        export HF_HUB_OFFLINE=1 
        export MPLCONFIGDIR="${SLURM_TMPDIR}/.config/matplotlib"
        export HF_HOME="${HF_HOME}"
        export HF_HUB_CACHE="${HF_HUB_CACHE}"
        export TRITON_CACHE_DIR="${SLURM_TMPDIR}/.triton_cache"
        export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE}"
        export TORCH_CUDA_ARCH_LIST="9.0" # for clusters with a100 GPUs
        export TORCH_EXTENSIONS_DIR="${SLURM_TMPDIR}/.cache/torch_extensions" # needed for cpu_adam
        export PYTORCH_KERNEL_CACHE_PATH="${SLURM_TMPDIR}/.cache/torch/kernels"
        export FORCE_TORCHRUN=1 
        export WANDB_MODE=offline 
        export WANDB_DIR="${WANDB_DIR}" 
        export DISABLE_VERSION_CHECK=1 # since the automatic detector doesn't automatically see that transformers==4.57.1+computecanada is the same as transformers==4.57.1
        # giving the slow tokenizer a try: https://github.com/hiyouga/LLaMA-Factory/issues/8600#issuecomment-3227071979

        # pushd /scratch/indrisch/LLaMA-Factory
        # pip install --upgrade pip setuptools wheel
        # pip install packaging psutil pandas pillow decorator scipy matplotlib platformdirs pyarrow sympy wandb ray -e ".[torch,metrics,deepspeed,liger-kernel]"


        pushd ${PROJECT_DIR}
        llamafactory-cli train $YAML_FILE
        # llamafactory-cli train \
        #     --model_name_or_path Video-R1/Video-R1-7B \
        #     --no_use_fast_tokenizer \
        #     --cache_dir /scratch/indrisch/huggingface/hub \
        #     --image_max_pixels 65536 \
        #     --video_max_pixels 16384 \
        #     --trust_remote_code \
        #     --stage sft \
        #     --do_train \
        #     --finetuning_type lora \
        #     --lora_rank 8 \
        #     --lora_target all \
        #     --dataset SQA3Devery24 \
        #     --media_dir /scratch/indrisch/data/ \
        #     --template videor1 \
        #     --cutoff_len 131072 \
        #     --preprocessing_num_workers 32 \
        #     --dataloader_num_workers 0 \
        #     --dataloader_pin_memory false \
        #     --low_cpu_mem_usage \
        #     --output_dir /scratch/indrisch/LLaMA-Factory/saves/videor1/lora/sft/SQA3Devery24_traineval \
        #     --logging_steps 10 \
        #     --save_steps 200 \
        #     --plot_loss \
        #     --overwrite_output_dir \
        #     --save_only_model false \
        #     --report_to wandb \
        #     --per_device_train_batch_size 2 \
        #     --gradient_accumulation_steps 8 \
        #     --learning_rate 1.0e-4 \
        #     --num_train_epochs 1.0 \
        #     --lr_scheduler_type cosine \
        #     --warmup_ratio 0.1 \
        #     --bf16 \
        #     --ddp_timeout 180000000 \
        #     --debug underflow_overflow \
        #     --log_level debug \
        #     --log_level_replica debug \
        #     --print_param_status \
        #     --flash_attn fa2 \
        #     --enable_liger_kernel \
        #     --gradient_checkpointing \
        #     --deepspeed /scratch/indrisch/LLaMA-Factory/examples/deepspeed/ds_z2_config.json \
        #     --val_size 0.1 \
        #     --per_device_eval_batch_size 1 \
        #     --eval_strategy steps \
        #     --eval_steps 200

    else
        echo "Invalid running mode: $RUNNING_MODE"
        exit 1
    fi

elif [[ "$CLUSTER" == "TRILLIUM" ]]; then

    module load apptainer

    TORCH_CUDA_ARCH_LIST="9.0" # for clusters with h100 GPUs

    # STEP 1: RUN THE TRAINING AND EVALUATION

    # Note: we need to set the PYTHONPATH so that llama-factory will use the latest code in the src directory
    apptainer run --nv --writable-tmpfs \
        -B ${PROJECT_DIR} \
        -B /home/indrisch \
        -B /dev/shm:/dev/shm \
        -B /etc/ssl/certs:/etc/ssl/certs:ro \
        -B /etc/pki:/etc/pki:ro \
        -W ${SLURM_TMPDIR} \
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
        --env PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}" \
        --pwd ${PROJECT_DIR} \
        ${SIF_FILE} \
        llamafactory-cli train ${YAML_FILE}

elif [[ "$CLUSTER" == "KILLARNEY" ]]; then

    if [[ "$RUNNING_MODE" == "APPTAINER" ]]; then

        module load apptainer

        TORCH_CUDA_ARCH_LIST="9.0" # for clusters with h100 GPUs

        # STEP 1: RUN THE TRAINING AND EVALUATION

        # better to have triton cache on a non-nfs file system for speed
        # if we are offline, we need to indicate this
        apptainer run --nv --writable-tmpfs \
            -B ${PROJECT_DIR} \
            -B /home/indrisch \
            -B /dev/shm:/dev/shm \
            -B /etc/ssl/certs:/etc/ssl/certs:ro \
            -B /etc/pki:/etc/pki:ro \
            -W ${SLURM_TMPDIR} \
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
            --env PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}" \
            --pwd ${PROJECT_DIR} \
            ${SIF_FILE} \
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
            pushd ${PROJECT_DIR}
            virtualenv --no-download venv_llamafactory_cu126
            source venv_llamafactory_cu126/bin/activate
            pushd ${PROJECT_DIR}
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
        export HF_HOME="${HF_HOME}"
        export HF_HUB_CACHE="${HF_HUB_CACHE}"
        export TRITON_CACHE_DIR="${SLURM_TMPDIR}/.triton_cache"
        export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE}"
        export TORCH_CUDA_ARCH_LIST="9.0" # for clusters with a100 GPUs
        export TORCH_EXTENSIONS_DIR="${SLURM_TMPDIR}/.cache/torch_extensions" # needed for cpu_adam
        export PYTORCH_KERNEL_CACHE_PATH="${SLURM_TMPDIR}/.cache/torch/kernels"
        export FORCE_TORCHRUN=1 
        export WANDB_MODE=offline 
        export WANDB_DIR="${WANDB_DIR}" 
        export DISABLE_VERSION_CHECK=1 # since the automatic detector doesn't automatically see that transformers==4.57.1+computecanada is the same as transformers==4.57.1
        # giving the slow tokenizer a try: https://github.com/hiyouga/LLaMA-Factory/issues/8600#issuecomment-3227071979


        pushd ${PROJECT_DIR}
        llamafactory-cli train ${YAML_FILE}

    else
        echo "Invalid running mode: $RUNNING_MODE"
        exit 1
    fi

else
    echo "Invalid cluster: $CLUSTER"
    exit 1
fi