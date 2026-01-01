#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64  
#SBATCH --time=0-02:00:00
#SBATCH --mem=485GB
#SBATCH --gpus-per-node=h100:4
#SBATCH --output=out/%N-qwen2_5vl_eval_SQA3Devery24_on_R0C0F0X1-%j.out

# Get MPI library paths for bind mounting
# MPI_LIB_PATH="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcc12/openmpi/4.1.5/lib"
# HWLOC_LIB_PATH="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/hwloc/2.9.1/lib"
# # Gentoo system libraries (contains libpciaccess and other system deps)
# GENTOO_LIB64_PATH="/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/lib64"

# STEP 1: RUN THE TRAINING AND EVALUATION

# better to have triton cache on a non-nfs file system for speed
# if we are offline, we need to indicate this

RUNNING_MODE=$1 # this is optional; generally, we wouldn't use this
YAML_FILE="/scratch/indrisch/LLaMA-Factory/examples/train_lora/qwen2_5vl_eval_SQA3Devery24_on_R0C0F0X1.yaml"

# STEP 1: RUN THE TRAINING AND EVALUATION

# better to have triton cache on a non-nfs file system for speed
# if we are offline, we need to indicate this

if [[ "$RUNNING_MODE" == "APPTAINER" ]]; then
    # use for rorqual

    module load apptainer

    TORCH_CUDA_ARCH_LIST="9.0" # for clusters with h100 GPUs
    
    # Set CUDA_VISIBLE_DEVICES: use SLURM's value if set, otherwise default to 0,1,2,3
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

    apptainer run --nv --writable-tmpfs \
        -C \
        -B /scratch/indrisch/LLaMA-Factory \
        -B /home/indrisch \
        -B /dev/shm:/dev/shm \
        -B /etc/ssl/certs:/etc/ssl/certs:ro \
        -B /etc/pki:/etc/pki:ro \
        -W ${SLURM_TMPDIR} \
        --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
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
        --env WANDB_DIR="/scratch/indrisch/LLaMA-Factory/wandb/" \
        --env WANDB_CACHE_DIR="${SLURM_TMPDIR}/.cache/wandb" \
        --env PYTHONPATH="/scratch/indrisch/LLaMA-Factory/src:${PYTHONPATH:-}" \
        --env NCCL_IB_DISABLE=0 \
        --env NCCL_P2P_DISABLE=0 \
        --env NCCL_DEBUG=INFO \
        --env NCCL_SOCKET_IFNAME=^docker0,lo \
        --pwd /scratch/indrisch/LLaMA-Factory \
        /scratch/indrisch/huggingface/hub/datasets--cvis-tmu--easyr1_verl_sif/snapshots/382a3b3e54a9fa9450c6c99dd83efaa2f0ca4a5a/llamafactory.sif \
        llamafactory-cli train ${YAML_FILE}

elif [[ "$RUNNING_MODE" == "SHELL" ]]; then

    module load apptainer

    TORCH_CUDA_ARCH_LIST="9.0" # for clusters with h100 GPUs

    SLURM_TMPDIR="/scratch/indrisch/tmp"
    
    # Create directories for pip cache and temporary files on scratch
    mkdir -p /scratch/indrisch/tmp
    mkdir -p /scratch/indrisch/.cache/pip
    mkdir -p /scratch/indrisch/.cache/torch_extensions
    mkdir -p /scratch/indrisch/.cache/torch/kernels
    mkdir -p /scratch/indrisch/.config/matplotlib
    mkdir -p /scratch/indrisch/.triton_cache

    apptainer shell --nv --writable \
        -B /scratch/indrisch \
        -B /home/indrisch \
        -B /dev/shm:/dev/shm \
        -B /etc/ssl/certs:/etc/ssl/certs:ro \
        -B /etc/pki:/etc/pki:ro \
        -W ${SLURM_TMPDIR} \
        --env CUDA_VISIBLE_DEVICES=0,1,2,3 \
        --env HF_HUB_OFFLINE=1 \
        --env TMPDIR="/scratch/indrisch/tmp" \
        --env PIP_CACHE_DIR="/scratch/indrisch/.cache/pip" \
        --env MPLCONFIGDIR="/scratch/indrisch/.config/matplotlib" \
        --env HF_HOME="/scratch/indrisch/huggingface/hub" \
        --env HF_HUB_CACHE="/scratch/indrisch/huggingface/hub" \
        --env TRITON_CACHE_DIR="/scratch/indrisch/.triton_cache" \
        --env FLASHINFER_WORKSPACE_BASE="/scratch/indrisch/" \
        --env TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
        --env TORCH_EXTENSIONS_DIR="/scratch/indrisch/.cache/torch_extensions" \
        --env PYTORCH_KERNEL_CACHE_PATH="/scratch/indrisch/.cache/torch/kernels" \
        --env FORCE_TORCHRUN=1 \
        --env WANDB_MODE=offline \
        --env WANDB_DIR="/scratch/indrisch/LLaMA-Factory/wandb/" \
        --env PYTHONPATH="/scratch/indrisch/LLaMA-Factory/src:${PYTHONPATH:-}" \
        --pwd /scratch/indrisch/LLaMA-Factory \
        /scratch/indrisch/huggingface/hub/datasets--cvis-tmu--easyr1_verl_sif/snapshots/382a3b3e54a9fa9450c6c99dd83efaa2f0ca4a5a/llamafactory.sif


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
    export HF_HOME="/scratch/indrisch/huggingface/hub"
    export HF_HUB_CACHE="/scratch/indrisch/huggingface/hub"
    export TRITON_CACHE_DIR="${SLURM_TMPDIR}/.triton_cache"
    export FLASHINFER_WORKSPACE_BASE="/scratch/indrisch/"
    export TORCH_CUDA_ARCH_LIST="9.0" # for clusters with a100 GPUs
    export TORCH_EXTENSIONS_DIR="${SLURM_TMPDIR}/.cache/torch_extensions" # needed for cpu_adam
    export PYTORCH_KERNEL_CACHE_PATH="${SLURM_TMPDIR}/.cache/torch/kernels"
    export FORCE_TORCHRUN=1 
    export WANDB_MODE=offline 
    export WANDB_DIR="/scratch/indrisch/LLaMA-Factory/wandb/" 
    export DISABLE_VERSION_CHECK=1 # since the automatic detector doesn't automatically see that transformers==4.57.1+computecanada is the same as transformers==4.57.1
    # giving the slow tokenizer a try: https://github.com/hiyouga/LLaMA-Factory/issues/8600#issuecomment-3227071979

    # pushd /scratch/indrisch/LLaMA-Factory
    # pip install --upgrade pip setuptools wheel
    # pip install packaging psutil pandas pillow decorator scipy matplotlib platformdirs pyarrow sympy wandb ray -e ".[torch,metrics,deepspeed,liger-kernel]"


    pushd /scratch/indrisch/LLaMA-Factory
    llamafactory-cli train ${YAML_FILE}
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