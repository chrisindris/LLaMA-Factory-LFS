#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
<<<<<<<< HEAD:preliminaries/videor1/slurm_videor1_lora_sft_SQA3Devery24_traineval_continued.sh
#SBATCH --time=1-00:00:00
#SBATCH --mem=950GB
#SBATCH --gpus-per-node=h100:4
#SBATCH --output=out/%N-videor1_lora_sft_SQA3Devery24_traineval_continued-%j.out
========
#SBATCH --time=0-18:00:00
#SBATCH --mem=950GB
#SBATCH --gpus-per-node=h100:4
#SBATCH --output=out/%N-videor1_lora_sft_SQA3Devery24_traineval-%j.out
>>>>>>>> killarney-SFT-SQA3D:preliminaries/videor1/slurm_videor1_lora_sft_SQA3Devery24_traineval.sh

# Get MPI library paths for bind mounting
# MPI_LIB_PATH="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcc12/openmpi/4.1.5/lib"
# HWLOC_LIB_PATH="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/hwloc/2.9.1/lib"
# # Gentoo system libraries (contains libpciaccess and other system deps)
# GENTOO_LIB64_PATH="/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/lib64"

# STEP 1: RUN THE TRAINING AND EVALUATION

# better to have triton cache on a non-nfs file system for speed
# if we are offline, we need to indicate this

RUNNING_MODE=$1 # this is optional; generally, we wouldn't use this

# STEP 1: RUN THE TRAINING AND EVALUATION

# better to have triton cache on a non-nfs file system for speed
# if we are offline, we need to indicate this

if [[ "$RUNNING_MODE" == "APPTAINER" ]]; then

    module load apptainer

    TORCH_CUDA_ARCH_LIST="9.0" # for clusters with h100 GPUs

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
        llamafactory-cli train /project/aip-wangcs/indrisch/LLaMA-Factory/examples/train_lora/videor1_lora_sft_SQA3Devery24_traineval_continued.yaml

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
<<<<<<<< HEAD:preliminaries/videor1/slurm_videor1_lora_sft_SQA3Devery24_traineval_continued.sh
    llamafactory-cli train /project/aip-wangcs/indrisch/LLaMA-Factory/examples/train_lora/videor1_lora_sft_SQA3Devery24_traineval_continued.yaml
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
    #     --media_dir /project/aip-wangcs/shared/data/ \
    #     --template videor1 \
    #     --cutoff_len 131072 \
    #     --preprocessing_num_workers 32 \
    #     --dataloader_num_workers 0 \
    #     --dataloader_pin_memory false \
    #     --low_cpu_mem_usage \
    #     --output_dir /project/aip-wangcs/indrisch/LLaMA-Factory/saves/videor1/lora/sft/SQA3Devery24_traineval_continued \
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
    #     --deepspeed /project/aip-wangcs/indrisch/LLaMA-Factory/examples/deepspeed/ds_z2_config.json \
    #     --val_size 0.1 \
    #     --per_device_eval_batch_size 1 \
    #     --eval_strategy steps \
    #     --eval_steps 200 \
    #     --adapter_name_or_path /project/aip-wangcs/indrisch/LLaMA-Factory/saves/videor1/lora/sft/SQA3Devery24_traineval_continued/checkpoint-400/
========
    llamafactory-cli train \
        --model_name_or_path Video-R1/Video-R1-7B \
        --no_use_fast_tokenizer \
        --cache_dir /scratch/indrisch/huggingface/hub \
        --image_max_pixels 65536 \
        --video_max_pixels 16384 \
        --trust_remote_code \
        --stage sft \
        --do_train \
        --finetuning_type lora \
        --lora_rank 8 \
        --lora_target all \
        --dataset SQA3Devery24 \
        --media_dir /project/aip-wangcs/shared/data/ \
        --template videor1 \
        --cutoff_len 131072 \
        --preprocessing_num_workers 32 \
        --dataloader_num_workers 0 \
        --dataloader_pin_memory false \
        --low_cpu_mem_usage \
        --output_dir /project/aip-wangcs/indrisch/LLaMA-Factory/saves/videor1/lora/sft/SQA3Devery24_traineval \
        --logging_steps 10 \
        --save_steps 200 \
        --plot_loss \
        --overwrite_output_dir \
        --save_only_model false \
        --report_to wandb \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --learning_rate 1.0e-4 \
        --num_train_epochs 2.0 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.1 \
        --bf16 \
        --ddp_timeout 180000000 \
        --debug underflow_overflow \
        --log_level debug \
        --log_level_replica debug \
        --print_param_status \
        --flash_attn fa2 \
        --enable_liger_kernel \
        --gradient_checkpointing \
        --deepspeed /project/aip-wangcs/indrisch/LLaMA-Factory/examples/deepspeed/ds_z2_config.json \
        --val_size 0.1 \
        --per_device_eval_batch_size 1 \
        --eval_strategy steps \
        --eval_steps 200
>>>>>>>> killarney-SFT-SQA3D:preliminaries/videor1/slurm_videor1_lora_sft_SQA3Devery24_traineval.sh

else
    echo "Invalid running mode: $RUNNING_MODE"
    exit 1
fi