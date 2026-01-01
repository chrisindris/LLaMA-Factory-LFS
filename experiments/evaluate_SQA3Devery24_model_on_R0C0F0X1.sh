#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=0-01:15:00
#SBATCH --gpus-per-node=h100:4
#SBATCH --output=out/%N-qwen2_5vl_eval_SQA3Devery24_on_R0C0F0X1-%j.out

module load apptainer

TORCH_CUDA_ARCH_LIST="9.0" # for clusters with h100 GPUs

apptainer run --nv --writable-tmpfs \
    -C \
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
    --env WANDB_CACHE_DIR="${SLURM_TMPDIR}/.cache/wandb" \
    --env PYTHONPATH="/scratch/indrisch/LLaMA-Factory/src:${PYTHONPATH:-}" \
    --env NCCL_IB_DISABLE=0 \
    --env NCCL_P2P_DISABLE=0 \
    --env NCCL_DEBUG=INFO \
    --env NCCL_SOCKET_IFNAME=^docker0,lo \
    --pwd /scratch/indrisch/LLaMA-Factory \
    /scratch/indrisch/easyr1_verl_sif/llamafactory.sif \
    llamafactory-cli train /scratch/indrisch/LLaMA-Factory/examples/train_lora/qwen2_5vl_eval_SQA3Devery24_on_R0C0F0X1.yaml