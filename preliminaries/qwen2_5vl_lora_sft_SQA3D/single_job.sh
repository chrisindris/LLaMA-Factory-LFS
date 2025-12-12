#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --time=0-08:00:00
#SBATCH --gpus-per-node=h100:4
#SBATCH --output=vgllm_3d_8b_lora_sft_SQA3Devery24_traineval_single-%j.out

# Define the YAML file
YAML_FILE="examples/train_lora/vgllm_lora_sft_SQA3Devery24_traineval_resumefromcheckpoint.yaml"

# Some environmental variables for Apptainer
HF_HOME="/scratch/leihan/huggingface/hub"
HF_HUB_CACHE="/scratch/leihan/huggingface/hub"
FLASHINFER_WORKSPACE_BASE="/scratch/leihan/"   
TRANSFORMERS_CACHE="/scratch/leihan/huggingface/hub"
WANDB_MODE="offline"
WANDB_API_KEY="34be47c188879e2f3feecda78088b42d0f6c4562"
WANDB_PROJECT="llama_factory_sft"
WANDB_ENTITY="cvis_tmu"
WANDB_NAME="vgllm_3d_8b_sqa3d_sft_lora_$(date +%Y%m%d_%H%M%S)"

# Run the command
module load apptainer
TORCH_CUDA_ARCH_LIST="9.0"
apptainer run --nv --writable-tmpfs \
    -B /scratch/leihan/LLaMA-Factory-LFS \
    -B /dev/shm:/dev/shm \
    -B /etc/ssl/certs:/etc/ssl/certs:ro \
    -B /etc/pki:/etc/pki:ro \
    -B /project/def-wangcs/shared/data \
    -W ${SLURM_TMPDIR} \
    --env HF_HUB_OFFLINE=1 \
    --env MPLCONFIGDIR="${SLURM_TMPDIR}/.config/matplotlib" \
    --env HOME="${SLURM_TMPDIR}" \
    --env HF_HOME="${HF_HOME}" \
    --env HF_HUB_CACHE="${HF_HUB_CACHE}" \
    --env TRITON_CACHE_DIR="${SLURM_TMPDIR}/.triton_cache" \
    --env FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE}" \
    --env TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    --env TORCH_EXTENSIONS_DIR="${SLURM_TMPDIR}/.cache/torch_extensions" \
    --env PYTORCH_KERNEL_CACHE_PATH="${SLURM_TMPDIR}/.cache/torch/kernels" \
    --env FORCE_TORCHRUN=1 \
    --env WANDB_CACHE_DIR="${SLURM_TMPDIR}/.wandb" \
    --env WANDB_MODE="${WANDB_MODE}" \
    --env WANDB_API_KEY="${WANDB_API_KEY}" \
    --env WANDB_PROJECT="${WANDB_PROJECT}" \
    --env WANDB_NAME="${WANDB_NAME}" \
    --env WANDB_ENTITY="${WANDB_ENTITY}" \
    --pwd /scratch/leihan/LLaMA-Factory-LFS \
    llama_factory.sif bash -c "llamafactory-cli train ${YAML_FILE}"
