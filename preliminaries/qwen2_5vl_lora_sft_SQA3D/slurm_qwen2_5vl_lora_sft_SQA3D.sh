#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24   
#SBATCH --time=0-00:10:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=%N-qwen2_5vl_lora_sft_SQA3D-%j.out

# 100 examples (per_device_train_batch_size: 2) takes 25 minutes; used 14.49% of memory (108.97 of 751.95 GB)
# 500 examples (per_device_train_batch_size: 2) takes about 120 minutes; used 22.57% of memory (169.68 of 751.95 GB)

module load apptainer

TORCH_CUDA_ARCH_LIST="9.0" # for clusters with h100 GPUs

# better to have triton cache on a non-nfs file system for speed
# if we are offline, we need to indicate this
apptainer run --nv --writable-tmpfs \
    -B /scratch/indrisch/LLaMA-Factory \
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
    --pwd /scratch/indrisch/LLaMA-Factory \
    /scratch/indrisch/easyr1_verl_sif/llamafactory.sif \
    llamafactory-cli train /scratch/indrisch/LLaMA-Factory/examples/train_lora/qwen2_5vl_lora_sft_SQA3D.yaml