#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=0-03:00:00
#SBATCH --gpus-per-node=h100:4
#SBATCH --output=%N-qwen2_5vl_lora_sft_SQA3D_unsloth-%j.out

# 100 examples (per_device_train_batch_size: 2) takes 25 minutes; used 14.49% of CPU memory (108.97 of 751.95 GB), no GPU OOM errors
# 500 examples (per_device_train_batch_size: 2) on a full node (96 CPUs, 4 GPUs, 32 preproc workers and 32 dataloader workers) took 1.75 hours; used 30.80% of CPU memory (231.57 of 751.95 GB)

module load apptainer

TORCH_CUDA_ARCH_LIST="9.0" # for clusters with h100 GPUs

# Container SIF file
#SIF_FILE="/scratch/indrisch/easyr1_spatial_understanding/easyr1_spatial_understanding.sif"
SIF_FILE="/scratch/indrisch/easyr1_verl_sif/llamafactory.sif"

# Bind mounts - following run_apptainer.sh pattern
BIND_MOUNTS=(
    -B /scratch/indrisch/LLaMA-Factory
    -B /home/indrisch
    -B /project/def-wangcs/indrisch
    -B /dev/shm:/dev/shm
    -B /etc/ssl/certs:/etc/ssl/certs:ro
    -B /etc/pki:/etc/pki:ro
)

# better to have triton cache on a non-nfs file system for speed
# if we are offline, we need to indicate this
apptainer exec --nv "${BIND_MOUNTS[@]}" \
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
    --pwd /scratch/indrisch/LLaMA-Factory \
    ${SIF_FILE} \
    llamafactory-cli train /scratch/indrisch/LLaMA-Factory/examples/train_lora/qwen2_5vl_lora_sft_SQA3D_unsloth.yaml

