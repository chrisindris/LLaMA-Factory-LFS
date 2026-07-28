#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=out/%N-qwen2_5vl_lora_sft_CoT_traineval-%j.out
#SBATCH --cpus-per-task=112
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
#SBATCH --gpus-per-node=h100:8

# Nibi wrapper for CoT SFT (Scene30k + SpatialSSRL_coldstart + 3DThinker10k).
# Submit from models/qwen2_5vl_lora_sft_CoT/ so SLURM out/ lands next to this script:
#   sbatch nibi_slurm_qwen2_5vl_lora_sft_CoT_traineval.sh

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

${PROJECT_DIR}/models/qwen2_5vl_lora_sft_CoT/slurm_qwen2_5vl_lora_sft_CoT_traineval.sh "$@"
