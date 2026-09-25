#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=out/%N-qwen2_5vl_lora_sft_CoT_traineval-%j.out
#SBATCH --cpus-per-task=96
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=h100:2

# Portable wrapper for CoT SFT (Scene30k + SpatialSSRL_coldstart + 3DThinker10k).
#
# Unlike the cluster-specific wrappers, this one derives the repo root from its
# OWN location, so the checkout can be renamed or moved anywhere.
#
# SLURM opens --output before this script runs, so create out/ first and submit
# from this directory (paths are relative to the submit cwd, not WRAPPER_DIR):
#   mkdir -p out
#   sbatch portable_slurm_qwen2_5vl_lora_sft_CoT_traineval.sh
#
# Site flags as needed. Defaults are Trillium-shaped (h100:2, no --mem); override
# on clusters that differ, e.g. Killarney L40S:
#   sbatch -A <account> --gpus-per-node=l40s:2 --mem=0 \
#     --mail-user=<you> --mail-type=ALL portable_slurm_...sh
#
# One-time setup on a login node:
#   PORTABLE_STAGE=1 ./portable_body_qwen2_5vl_lora_sft_CoT_traineval.sh
#   PREFLIGHT=1      ./portable_body_qwen2_5vl_lora_sft_CoT_traineval.sh
set -euo pipefail

# SLURM copies the batch script to /var/spool/slurmd/job*/slurm_script, so
# BASH_SOURCE points at the spool copy — not the repo. Prefer SLURM_SUBMIT_DIR
# (cwd at sbatch time) when the body lives next to the submitted wrapper.
BODY_NAME="portable_body_qwen2_5vl_lora_sft_CoT_traineval.sh"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/${BODY_NAME}" ]]; then
	WRAPPER_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
	WRAPPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
fi

exec "${WRAPPER_DIR}/${BODY_NAME}" "$@"
