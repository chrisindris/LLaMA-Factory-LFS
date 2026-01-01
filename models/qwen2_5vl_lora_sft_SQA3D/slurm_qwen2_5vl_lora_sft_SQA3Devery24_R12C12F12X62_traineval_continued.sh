#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=0-18:00:00
#SBATCH --gpus-per-node=h100:4
#SBATCH --output=%N-qwen2_5vl_lora_sft_SQA3Devery24_R12C12F12X62_traineval_continued-%j.out

module load apptainer

TORCH_CUDA_ARCH_LIST="9.0" # for clusters with h100 GPUs

# STEP 1: RUN THE TRAINING AND EVALUATION

# better to have triton cache on a non-nfs file system for speed
# if we are offline, we need to indicate this
# Use --tmpfs to create a larger /dev/shm inside container (300GB) instead of binding host's /dev/shm
# This avoids SIGBUS errors from shared memory exhaustion during long training runs
apptainer run --nv --writable-tmpfs \
    -B /scratch/indrisch/LLaMA-Factory \
    -B /home/indrisch \
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
    --env WANDB_DIR="/scratch/indrisch/LLaMA-Factory/wandb/" \
    --pwd /scratch/indrisch/LLaMA-Factory \
    /scratch/indrisch/easyr1_verl_sif/llamafactory.sif \
    llamafactory-cli train /scratch/indrisch/LLaMA-Factory/examples/train_lora/qwen2_5vl_lora_sft_SQA3Devery24_R12C12F12X62_traineval_continued.yaml