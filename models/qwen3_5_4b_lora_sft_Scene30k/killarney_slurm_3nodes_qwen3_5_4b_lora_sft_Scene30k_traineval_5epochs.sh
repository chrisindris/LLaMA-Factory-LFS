#!/bin/bash
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --output=out/%N-3nodes_qwen3_5_4b_lora_sft_Scene30k_traineval_5epochs-%j.out
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --mem=480GB
#SBATCH --gpus-per-node=l40s:4
#SBATCH --mail-user=christopher.indris@torontomu.ca
#SBATCH --mail-type=ALL

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
	bash ${PROJECT_DIR}/models/qwen3_5_4b_lora_sft_Scene30k/slurm_3nodes_qwen3_5_4b_lora_sft_Scene30k_traineval_5epochs.sh "$@"
