#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:03:00
#SBATCH --mem=256GB
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=out/%N-qwen3vl_lora_sft_SQA3Devery24_traineval-%j.out



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
    /project/aip-wangcs/indrisch/easyr1_verl_sif/llamafactory_wandb.sif \
    pip freeze && llamafactory-cli train /project/aip-wangcs/indrisch/LLaMA-Factory/examples/train_lora/qwen3vl_lora_sft_SQA3Devery24_traineval.yaml