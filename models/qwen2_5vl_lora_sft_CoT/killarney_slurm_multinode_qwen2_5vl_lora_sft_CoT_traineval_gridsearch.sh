#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=out/%N-qwen2_5vl_lora_sft_CoT_traineval-%j.out
#SBATCH --cpus-per-task=64
#SBATCH --time=0-02:30:00
#SBATCH --mem=480GB
#SBATCH --gpus-per-node=l40s:4
#SBATCH --mail-user=christopher.indris@torontomu.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-8

# Calls killarney_slurm_multinode_qwen2_5vl_lora_sft_CoT_traineval.sh to see if we can find working settings for:
# - L40S_IMAGE_SAMPLE_COUNT
# - L40S_CUTOFF_LEN

# basics

export RUNNING_MODE="APPTAINER"

if [[ "$SLURM_ARRAY_TASK_ID" -eq 0 ]]; then
  export OVERLAY="/project/aip-wangcs/indrisch/LLaMA-Factory/apptainer/overlay.img"
else
  export OVERLAY="/project/aip-wangcs/indrisch/LLaMA-Factory/apptainer/overlay_${SLURM_ARRAY_TASK_ID}.img"
fi 
echo "OVERLAY: ${OVERLAY}"

export YAML_FILE="/project/aip-wangcs/indrisch/LLaMA-Factory/examples/train_lora/killarney_qwen2_5vl_lora_sft_CoT_traineval_resume_epoch1_arrayjob${SLURM_ARRAY_TASK_ID}.yaml"
echo "YAML_FILE: ${YAML_FILE}"

# experiments

L40S_IMAGE_SAMPLE_COUNTS=("120" "240" "360")
L40S_CUTOFF_LENS=("32768" "65536" "131072")

L40S_IMAGE_SAMPLE_COUNTS_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
L40S_CUTOFF_LENS_INDEX=$((SLURM_ARRAY_TASK_ID % 3))

export L40S_IMAGE_SAMPLE_COUNT=${L40S_IMAGE_SAMPLE_COUNTS[$L40S_IMAGE_SAMPLE_COUNTS_INDEX]}
export L40S_CUTOFF_LEN=${L40S_CUTOFF_LENS[$L40S_CUTOFF_LENS_INDEX]}

# launch!

./killarney_slurm_multinode_qwen2_5vl_lora_sft_CoT_traineval.sh "$@"
