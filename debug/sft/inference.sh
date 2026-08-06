#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=0-00:20:00
#SBATCH --mem=400GB
#SBATCH --gpus-per-node=l40s:4
#SBATCH --output=out/%N-inference-%j.out


RUNNING_MODE=$1 # this is optional; generally, we wouldn't use this
#YAML_FILE="/project/aip-wangcs/indrisch/LLaMA-Factory/examples/train_lora/qwen2_5vl_lora_sft_SQA3Devery24_h5py.yaml"

# if $SLURM_TMPDIR is not set, set it to /tmp. This is primarily if we want to run on login node (i.e. short debugs)
if [ -z "$SLURM_TMPDIR" ]; then
    SLURM_TMPDIR="/tmp"
fi

if [[ "$RUNNING_MODE" == "VENV" ]]; then

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

    pushd /project/aip-wangcs/indrisch/LLaMA-Factory/scripts/
    
    # It seems like the HfArgumentParser doesn't support dataloader_num_workers and dataloader_pin_memory unless we allow extra args (although this may not actually do anything)
    # export ALLOW_EXTRA_ARGS=1


    python vllm_infer.py \
        --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
        --adapter_name_or_path /scratch/indrisch/huggingface/hub/models--cvis-tmu--qwen2_5vl-7b-lora-sft-SQA3Devery24_ep1/snapshots/0bdc7d9fb51e700a889b40dbabf01c929e31d43c/checkpoint-465/ \
        --media_dir /project/aip-wangcs/shared/data/ \
        --image_max_pixels 65536 \
        --dataset SQA3Devery24 \
        --template qwen2_vl \
        --cutoff_len 126076 \
        --max_samples 100 \
        --overwrite_cache true \
        --preprocessing_num_workers 32 \
        --low_cpu_mem_usage true \
        --enable_thinking \
        --temperature 0.95 \
        --top_p 0.7 \
        --vllm_config "{\"limit_mm_per_prompt\": {\"image\": 320}}" \
        --save_name /project/aip-wangcs/indrisch/LLaMA-Factory/debug/sft/out/generated_predictions_Qwen25VL7BInstruct_SQA3Dep1.jsonl \
        # --pipeline_parallel_size -1 # if we want to debug on 0 GPUs, this gets a little bit into the code.

    python vllm_infer.py \
        --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
        --media_dir /project/aip-wangcs/shared/data/ \
        --image_max_pixels 65536 \
        --dataset SQA3Devery24 \
        --template qwen2_vl \
        --cutoff_len 126076 \
        --max_samples 100 \
        --overwrite_cache true \
        --preprocessing_num_workers 32 \
        --low_cpu_mem_usage true \
        --enable_thinking \
        --temperature 0.95 \
        --top_p 0.7 \
        --vllm_config "{\"limit_mm_per_prompt\": {\"image\": 320}}" \
        --save_name /project/aip-wangcs/indrisch/LLaMA-Factory/debug/sft/out/generated_predictions_Qwen25VL7BInstruct.jsonl \
        # --pipeline_parallel_size -1 # if we want to debug on 0 GPUs, this gets a little bit into the code.

else
    echo "Invalid running mode: $RUNNING_MODE"
    exit 1
fi