#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --output=out/%N-qwen2_5vl_lora_sft_CoT_traineval-%j.out
#SBATCH --cpus-per-task=24
#SBATCH --time=1-00:00:00
#SBATCH --gpus-per-node=h100:4
#SBATCH --mail-user=christopher.indris@torontomu.ca
#SBATCH --mail-type=ALL

# Trillium wrapper for CoT SFT (Scene30k + SpatialSSRL_coldstart + 3DThinker10k).
# Submit from models/qwen2_5vl_lora_sft_CoT/ so SLURM out/ lands next to this script:
#   sbatch trillium_slurm_multinode_qwen2_5vl_lora_sft_CoT_traineval.sh
#
# Node-local dataset staging is ON by default in the shared multinode worker
# (STAGE_DATASETS_LOCAL=1). Set STAGE_DATASETS_LOCAL=0 to read shared paths.

. ../../scripts/utils/env.sh

mkdir -p "${PROJECT_DIR}/models/qwen2_5vl_lora_sft_CoT/out"

export HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) # store head node's address
export MASTER_ADDR="${MASTER_ADDR:-${HEAD_NODE:-$(hostname)}}"
export MASTER_PORT="${MASTER_PORT:-29500}"

echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST}"
echo "HEAD_NODE: ${HEAD_NODE}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"

# Launch one parent task per node. Each parent task then lets LLaMA-Factory
# start one torchrun worker per visible GPU on that node.
srun \
	--nodes "${SLURM_NNODES}" \
	--ntasks "${SLURM_NNODES}" \
	--ntasks-per-node 1 \
	--kill-on-bad-exit=1 \
	bash ${PROJECT_DIR}/models/qwen2_5vl_lora_sft_CoT/slurm_multinode_qwen2_5vl_lora_sft_CoT_traineval.sh "$@"
