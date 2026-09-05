#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=out/%N-qwen2_5vl_lora_sft_CoT_traineval-%j.out
#SBATCH --cpus-per-task=96
#SBATCH --time=1-00:00:00
#SBATCH --gpus-per-node=h100:4

# Portable wrapper for CoT SFT (Scene30k + SpatialSSRL_coldstart + 3DThinker10k).
#
# Unlike the cluster-specific wrappers, this one derives the repo root from its
# OWN location, so the checkout can be renamed or moved anywhere.
#
# Submit from this directory so SLURM out/ lands next to the script:
#   sbatch portable_slurm_qwen2_5vl_lora_sft_CoT_traineval.sh
# Add site flags as needed:
#   sbatch -A <account> --mail-user=<you> --mail-type=ALL portable_slurm_...sh
#
# One-time setup on a login node:
#   PORTABLE_STAGE=1 ./portable_body_qwen2_5vl_lora_sft_CoT_traineval.sh
#   PREFLIGHT=1      ./portable_body_qwen2_5vl_lora_sft_CoT_traineval.sh
set -euo pipefail

WRAPPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
mkdir -p "${WRAPPER_DIR}/out"

exec "${WRAPPER_DIR}/portable_body_qwen2_5vl_lora_sft_CoT_traineval.sh" "$@"
