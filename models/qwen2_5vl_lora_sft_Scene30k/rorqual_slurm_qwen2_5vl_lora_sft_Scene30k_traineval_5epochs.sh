#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=out/%N-qwen2_5vl_lora_sft_Scene30k_traineval_5epochs-%j.out
#SBATCH --cpus-per-task=64
#SBATCH --time=2-00:00:00
#SBATCH --mem=485G
#SBATCH --gpus-per-node=h100:4

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

${PROJECT_DIR}/models/qwen2_5vl_lora_sft_Scene30k/slurm_qwen2_5vl_lora_sft_Scene30k_traineval_5epochs.sh "$@"