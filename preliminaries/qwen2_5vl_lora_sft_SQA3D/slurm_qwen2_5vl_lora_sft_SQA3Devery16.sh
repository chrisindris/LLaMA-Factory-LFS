#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=0-23:59:59
#SBATCH --gpus-per-node=h100:4
#SBATCH --output=%N-qwen2_5vl_lora_sft_SQA3Devery16-%j.out

# 33047 examples (per_device_train_batch_size: 2) on a full node (96 CPUs, 4 GPUs, 32 preproc workers and 4 dataloader workers) took 19 hours; used 15.59% of CPU memory (117.26 of 751.95 GB)

module load apptainer

TORCH_CUDA_ARCH_LIST="9.0" # for clusters with h100 GPUs

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
    --pwd /project/aip-wangcs/indrisch/LLaMA-Factory \
    /scratch/indrisch/huggingface/hub/datasets--cvis-tmu--easyr1_verl_sif/snapshots/382a3b3e54a9fa9450c6c99dd83efaa2f0ca4a5a/llamafactory.sif \
    llamafactory-cli train /project/aip-wangcs/indrisch/LLaMA-Factory/examples/train_lora/qwen2_5vl_lora_sft_SQA3Devery16.yaml