#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=out/%N-qwen2_5vl_lora_sft_CoT_traineval_resume-%j.out
#SBATCH --cpus-per-task=96
#SBATCH --time=0-16:00:00
#SBATCH --gpus-per-node=h100:4
#SBATCH --mail-user=christopher.indris@torontomu.ca
#SBATCH --mail-type=ALL

# Trillium wrapper: continuous resume CoT SFT from checkpoint-SN (~epoch N -> N+1), where S is the number of steps per epoch.
# Submit from models/qwen2_5vl_lora_sft_CoT/ so SLURM out/ lands next to this script:
#   sbatch trillium_slurm_qwen2_5vl_lora_sft_CoT_traineval_resume.sh
#   recommended time: SBATCH --time=0-16:00:00

. ../../scripts/utils/env.sh

# ----- DEFAULT ARGUMENTS -----
# we can either set directly outside the script, or use the defaults.

export STARTING_EPOCH="${STARTING_EPOCH:-3}"
export ENDING_EPOCH="${ENDING_EPOCH:-4}"
export STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-620}"

# ----- ARGUMENT PARSING -----
# we can explicitly override the above by setting them with flags.

while [[ $# -gt 0 ]]; do
  case "$1" in
    --starting-epoch)
      export STARTING_EPOCH="${2}"
      shift 2
      ;;
    --ending-epoch)
      export ENDING_EPOCH="${2}"
      shift 2
      ;;
    --steps-per-epoch)
      export STEPS_PER_EPOCH="${2}"
      shift 2
      ;;
    -h|--help)
      echo "Usage:"
      echo "<set other vars here as desired> $0 --running-mode <RUNNING_MODE> --starting-epoch <STARTING_EPOCH> --ending-epoch <ENDING_EPOCH> --steps-per-epoch <STEPS_PER_EPOCH>"
      exit 0
      ;;
    *)
      echo "Error: Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

# --- further cluster-specific settings ---

export PYTHONUNBUFFERED=1

if [[ "$RUNNING_MODE" == "SHELL" ]]; then
    export SLURM_TMPDIR="/tmp"
fi

if [[ "$CLUSTER" == "RORQUAL" ]]; then
    export SCANNET_H5_DIR="/project/def-wangcs/indrisch/scratch_saves/ScanNet_h5/scans"
fi

echo "RUNNING_MODE: $RUNNING_MODE"

# --- setting python environment ---

module load StdEnv/2023  gcc/12.3  openmpi/4.1.5
module load python/3.12 cuda/12.6 opencv/4.12.0
module load arrow
module load apptainer
source $VENV_LLAMAFACTORY/bin/activate
pip install --upgrade pip setuptools wheel
pip install packaging
pip install --no-index huggingface_hub ruamel.yaml

# --- using the python environment, use ruamel.yaml to make and modify the yaml needed for the run. ---

EXPERIMENT_NAME="qwen2_5vl_lora_sft_CoT_traineval_resume"
mkdir -p "${PROJECT_DIR}/models/qwen2_5vl_lora_sft_CoT/out"

TEMPLATE_YAML="${PROJECT_DIR}/examples/train_lora/trillium_qwen2_5vl_lora_sft_CoT_traineval_resume_epoch2.yaml"

# |------------
# | Create a copy of TEMPLATE_YAML that will live at TEMPLATE_YAML but at ...epoch${ENDING_EPOCH}.yaml
# | Set the following fields in the copied yaml:
# | output_dir: saves/qwen2_5vl-7b/lora/sft/CoT_traineval_resume_ep${ENDING_EPOCH}/
# | resume_from_checkpoint: ${PROJECT_DIR}saves/qwen2_5vl-7b/lora/sft/CoT_traineval/checkpoint-($((STARTING_EPOCH * STEPS_PER_EPOCH)))
# | adapter_name_or_path: ${PROJECT_DIR}saves/qwen2_5vl-7b/lora/sft/CoT_traineval/checkpoint-($((STARTING_EPOCH * STEPS_PER_EPOCH)))
# | stop_at_global_step: ($((ENDING_EPOCH * STEPS_PER_EPOCH)))
# |-----------------

export YAML_FILE="${TEMPLATE_YAML/epoch2/epoch${ENDING_EPOCH}}"
export OUTPUT_DIR_SAVES="saves/qwen2_5vl-7b/lora/sft/CoT_traineval_resume_ep${ENDING_EPOCH}/"
export OUTPUT_DIR="${PROJECT_DIR}/${OUTPUT_DIR_SAVES}"
export RESUME_CKPT="${PROJECT_DIR}/saves/qwen2_5vl-7b/lora/sft/CoT_traineval_resume_ep${STARTING_EPOCH}/checkpoint-$((STARTING_EPOCH * STEPS_PER_EPOCH))"

python "${PROJECT_DIR}/scripts/utils/modify_yaml.py" \
  --yaml-template-path "${TEMPLATE_YAML}" \
  --yaml-output-path "${TEMPLATE_YAML/epoch2/epoch${ENDING_EPOCH}}" \
  --output_dir "${OUTPUT_DIR_SAVES}" \
  --resume_from_checkpoint "${RESUME_CKPT}" \
  --adapter_name_or_path "${RESUME_CKPT}" \
  --stop_at_global_step $((ENDING_EPOCH * STEPS_PER_EPOCH))

deactivate

# ----- launch! -----

${PROJECT_DIR}/models/qwen2_5vl_lora_sft_CoT/slurm_qwen2_5vl_lora_sft_CoT_traineval_resume.sh "$@"
