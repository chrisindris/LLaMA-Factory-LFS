#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=out/%N-qwen3vl_lora_sft_SQA3Devery24_traineval-%j.out

# RORQUAL:
#SBATCH --cpus-per-task=64
#SBATCH --time=0-18:00:00
#SBATCH --mem=485G
#SBATCH --gpus-per-node=h100:4

# TRILLIUM:
#SBATCH --cpus-per-task=96
#SBATCH --time=0-22:00:00
#SBATCH --gpus-per-node=h100:4

# KILLARNEY:
#SBATCH --cpus-per-task=48
#SBATCH --time=0-18:00:00
#SBATCH --mem=1900G
#SBATCH --gpus-per-node=h100:8

# ----- HEADER: ENV VARIABLES -----

EXPERIMENT_NAME="qwen3vl_lora_sft_SQA3Devery24_traineval"

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
    RUNNING_MODE="APPTAINER" # running mode for RORQUAL
elif [[ "$PS1" == *"trig"* ]] || [[ "$HOSTNAME" == *"trig"* ]]; then
    CLUSTER="TRILLIUM"
    RUNNING_MODE="APPTAINER" # running mode for TRILLIUM
elif [[ "$PS1" == *"klogin"* ]] || [[ "$HOSTNAME" == *"klogin"* ]] || [[ "$PS1" == *"kn"* ]] || [[ "$HOSTNAME" == *"kn"* ]]; then
    CLUSTER="KILLARNEY"
    RUNNING_MODE="VENV" # running mode for KILLARNEY
else
    echo "Warning: Could not detect cluster from PS1 or HOSTNAME. Defaulting to RORQUAL."
    CLUSTER="RORQUAL"
    RUNNING_MODE="APPTAINER" # running mode for unknown cluster
fi

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
OUTPUT_DIR="${PROJECT_DIR}/saves/qwen3vl-8b/lora/sft/SQA3Devery24_traineval"

if [[ -z "$RUNNING_MODE" ]]; then
    RUNNING_MODE=$1
fi

# ----- EXPERIMENT -----

if [[ "$CLUSTER" == "RORQUAL" ]]; then

    # STEP 1: RUN THE TRAINING AND EVALUATION

    # better to have triton cache on a non-nfs file system for speed
    # if we are offline, we need to indicate this

    if [[ "$RUNNING_MODE" == "APPTAINER" ]]; then

        module load apptainer

        apptainer run --nv --writable-tmpfs \
            -C \
            -B ${PROJECT_DIR} \
            -B /home/indrisch \
            -B /dev/shm:/dev/shm \
            -B /etc/ssl/certs:/etc/ssl/certs:ro \
            -B /etc/pki:/etc/pki:ro \
            -W ${SLURM_TMPDIR} \
            --env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
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
            --pwd ${PROJECT_DIR} \
            ${SIF_FILE} \
            pip freeze && llamafactory-cli train ${YAML_FILE}

    elif [[ "$RUNNING_MODE" == "VENV" ]]; then

        module load StdEnv/2023  gcc/12.3  openmpi/4.1.5
        module load python/3.12 cuda/12.6 opencv/4.12.0
        module load arrow

        source /project/aip-wangcs/indrisch/venv_llamafactory_cu126/bin/activate
        export TORCH_CUDA_ARCH_LIST="9.0" # for clusters with a100 GPUs
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        export FORCE_TORCHRUN=1 
        export HF_HUB_OFFLINE=1 
        export WANDB_MODE=offline 
        export TRITON_CACHE_DIR="${SLURM_TMPDIR}/.triton_cache"
        export DISABLE_VERSION_CHECK=1 # since the automatic detector doesn't automatically see that transformers==4.57.1+computecanada is the same as transformers==4.57.1
        # giving the slow tokenizer a try: https://github.com/hiyouga/LLaMA-Factory/issues/8600#issuecomment-3227071979
        pushd /project/aip-wangcs/indrisch/LLaMA-Factory
        llamafactory-cli train ${YAML_FILE}
        # llamafactory-cli train \
        #     --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
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
        #     --media_dir /project/aip-wangcs/shared/data/ \
        #     --template qwen3_vl \
        #     --cutoff_len 131072 \
        #     --preprocessing_num_workers 32 \
        #     --dataloader_num_workers 0 \
        #     --dataloader_pin_memory false \
        #     --low_cpu_mem_usage \
        #     --output_dir /project/aip-wangcs/indrisch/LLaMA-Factory/saves/qwen3vl-8b/lora/sft/SQA3Devery24_traineval \
        #     --logging_steps 10 \
        #     --save_steps 200 \
        #     --plot_loss \
        #     --overwrite_output_dir \
        #     --save_only_model false \
        #     --report_to wandb \
        #     --per_device_train_batch_size 2 \
        #     --gradient_accumulation_steps 8 \
        #     --learning_rate 1.0e-4 \
        #     --num_train_epochs 2.0 \
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
        #     --deepspeed /project/aip-wangcs/indrisch/LLaMA-Factory/examples/deepspeed/ds_z3_offload_config.json \
        #     --val_size 0.1 \
        #     --per_device_eval_batch_size 1 \
        #     --eval_strategy steps \
        #     --eval_steps 200

    else
        echo "Invalid running mode: $RUNNING_MODE"
        exit 1
    fi
elif [[ "$CLUSTER" == "TRILLIUM" ]]; then

    apptainer run --nv --writable-tmpfs \
        -B /scratch/indrisch/LLaMA-Factory \
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
        --pwd ${PROJECT_DIR} \
        ${SIF_FILE} \
        llamafactory-cli train ${YAML_FILE}

elif [[ "$CLUSTER" == "KILLARNEY" ]]; then

    if [[ "$RUNNING_MODE" == "APPTAINER" ]]; then

        module load apptainer

        apptainer run --nv --writable-tmpfs \
            -C \
            -B ${PROJECT_DIR} \
            -B /home/indrisch \
            -B /dev/shm:/dev/shm \
            -B /etc/ssl/certs:/etc/ssl/certs:ro \
            -B /etc/pki:/etc/pki:ro \
            -W ${SLURM_TMPDIR} \
            --env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
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
            --pwd ${PROJECT_DIR} \
            ${SIF_FILE} \
            pip freeze && llamafactory-cli train ${YAML_FILE}

    elif [[ "$RUNNING_MODE" == "VENV" ]]; then

        module load StdEnv/2023  gcc/12.3  openmpi/4.1.5
        module load python/3.12 cuda/12.6 opencv/4.12.0
        module load arrow

        source /project/aip-wangcs/indrisch/venv_llamafactory_cu126/bin/activate
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        export FORCE_TORCHRUN=1 
        export HF_HUB_OFFLINE=1 
        export WANDB_MODE=offline 
        export TRITON_CACHE_DIR="${SLURM_TMPDIR}/.triton_cache"
        export DISABLE_VERSION_CHECK=1 # since the automatic detector doesn't automatically see that transformers==4.57.1+computecanada is the same as transformers==4.57.1
        # giving the slow tokenizer a try: https://github.com/hiyouga/LLaMA-Factory/issues/8600#issuecomment-3227071979
        pushd /project/aip-wangcs/indrisch/LLaMA-Factory
        llamafactory-cli train ${YAML_FILE}
        # llamafactory-cli train \
        #     --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
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
        #     --media_dir /project/aip-wangcs/shared/data/ \
        #     --template qwen3_vl \
        #     --cutoff_len 131072 \
        #     --preprocessing_num_workers 32 \
        #     --dataloader_num_workers 0 \
        #     --dataloader_pin_memory false \
        #     --low_cpu_mem_usage \
        #     --output_dir /project/aip-wangcs/indrisch/LLaMA-Factory/saves/qwen3vl-8b/lora/sft/SQA3Devery24_traineval \
        #     --logging_steps 10 \
        #     --save_steps 200 \
        #     --plot_loss \
        #     --overwrite_output_dir \
        #     --save_only_model false \
        #     --report_to wandb \
        #     --per_device_train_batch_size 2 \
        #     --gradient_accumulation_steps 8 \
        #     --learning_rate 1.0e-4 \
        #     --num_train_epochs 2.0 \
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
        #     --deepspeed /project/aip-wangcs/indrisch/LLaMA-Factory/examples/deepspeed/ds_z3_offload_config.json \
        #     --val_size 0.1 \
        #     --per_device_eval_batch_size 1 \
        #     --eval_strategy steps \
        #     --eval_steps 200

    else
        echo "Invalid running mode: $RUNNING_MODE"
        exit 1
    fi

else
    echo "Invalid cluster: $CLUSTER"
    exit 1
fi