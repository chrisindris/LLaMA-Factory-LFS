#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=out/%N-qwen2_5vl_lora_sft_CoT_traineval_resume_epoch2-%j.out
#SBATCH --cpus-per-task=96
#SBATCH --time=0-16:00:00
#SBATCH --gpus-per-node=h100:4
#SBATCH --mail-user=christopher.indris@torontomu.ca
#SBATCH --mail-type=ALL

# Trillium wrapper: continuous resume CoT SFT from checkpoint-620 (~epoch 1 -> 2).
# Submit from models/qwen2_5vl_lora_sft_CoT/ so SLURM out/ lands next to this script:
#   sbatch trillium_slurm_qwen2_5vl_lora_sft_CoT_traineval_resume_epoch2.sh

if [[ "$PWD" == *LLaMA-Factory-LFS* ]]; then
    PROJECT_DIR="${PWD%%LLaMA-Factory-LFS*}/LLaMA-Factory-LFS"
elif [[ "$PWD" == *LLaMA-Factory* ]]; then
    PROJECT_DIR="${PWD%%LLaMA-Factory*}/LLaMA-Factory"
else
    echo "Error: Could not find 'LLaMA-Factory' or 'LLaMA-Factory-LFS' in the current path."
    exit 1
fi
SYSCONFIG_DIR_PATH="$PROJECT_DIR/scripts"
export PYTHONPATH="$PYTHONPATH:$SYSCONFIG_DIR_PATH"
export PYTHONUNBUFFERED=1

mkdir -p "${PROJECT_DIR}/models/qwen2_5vl_lora_sft_CoT/out"

${PROJECT_DIR}/models/qwen2_5vl_lora_sft_CoT/slurm_qwen2_5vl_lora_sft_CoT_traineval_resume_epoch2.sh "$@"
