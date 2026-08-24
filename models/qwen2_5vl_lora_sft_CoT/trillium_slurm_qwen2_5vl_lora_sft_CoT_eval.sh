#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=out/%N-qwen2_5vl_lora_sft_CoT_traineval_eval-%j.out
#SBATCH --cpus-per-task=96
#SBATCH --time=0-04:00:00
#SBATCH --gpus-per-node=h100:4
#SBATCH --mail-user=christopher.indris@torontomu.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-3

# Trillium wrapper to launch array jobs for evaluating models on datasets.

. ../../scripts/utils/env.sh

# --- models for experiments ---

MODEL_PATHS=(
	Qwen/Qwen2.5-VL-7B-Instruct
	cvis-tmu/qwen2_5vl-7b-lora-sft-CoT_traineval_1epochs_merged
	cvis-tmu/qwen2_5vl-7b-lora-sft-CoT_traineval_2epochs_merged
	cvis-tmu/qwen2_5vl-7b-lora-sft-CoT_traineval_3epochs_merged
)

ADAPTER_PATHS=(
)

# # ----- DEFAULT ARGUMENTS -----
# # we can either set directly outside the script, or use the defaults.

# STARTING_EPOCH="${STARTING_EPOCH:-4}"
# ENDING_EPOCH="${ENDING_EPOCH:-5}"
# STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-620}"

# # ----- ARGUMENT PARSING -----
# # we can explicitly override the above by setting them with flags.

# while [[ $# -gt 0 ]]; do
# 	case "$1" in
# 	--starting-epoch)
# 		export STARTING_EPOCH="${2}"
# 		shift 2
# 		;;
# 	--ending-epoch)
# 		export ENDING_EPOCH="${2}"
# 		shift 2
# 		;;
# 	--steps-per-epoch)
# 		export STEPS_PER_EPOCH="${2}"
# 		shift 2
# 		;;
# 	-h | --help)
# 		echo "Usage:"
# 		echo "<set other vars here as desired> $0 --running-mode <RUNNING_MODE> --starting-epoch <STARTING_EPOCH> --ending-epoch <ENDING_EPOCH> --steps-per-epoch <STEPS_PER_EPOCH>"
# 		exit 0
# 		;;
# 	*)
# 		echo "Error: Unknown argument: $1" >&2
# 		exit 1
# 		;;
# 	esac
# done

# --- model selection (via SLURM_ARRAY_TASK_ID) ---

COMBINED_PATHS=("${MODEL_PATHS[@]}" "${ADAPTER_PATHS[@]}")
IDX=${SLURM_ARRAY_TASK_ID:-0}

# - set ADAPTER_PATH and BASE_MODEL_PATH -
if [[ "${IDX}" -ge ${#MODEL_PATHS[@]} ]]; then
	ADAPTER_PATH="${COMBINED_PATHS[${IDX}]}"
	MODEL_NAME="$(basename "$ADAPTER_PATH")"
	echo "Using adapter: $ADAPTER_PATH"

	if [[ "$ADAPTER_PATH" == *"qwen2_5vl-7b"* ]]; then
		BASE_MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
	elif [[ "$ADAPTER_PATH" == *"videor1"* ]]; then
		BASE_MODEL_PATH="Video-R1/Video-R1-7B"
	else
		echo "Error: Could not determine base model path from adapter path: $ADAPTER_PATH"
		exit 1
	fi

else
	BASE_MODEL_PATH="${COMBINED_PATHS[${IDX}]}"
	MODEL_NAME="$(basename "$BASE_MODEL_PATH")"
	echo "Using base model: $BASE_MODEL_PATH"
	unset ADAPTER_PATH
fi

# - set the template -
if [[ "$BASE_MODEL_PATH" == *"qwen2_5vl-7b"* ]] || [[ "$BASE_MODEL_PATH" == *"Qwen2.5-VL-7B"* ]]; then
	BASE_MODEL_PATH_TEMPLATE="qwen2_vl"
elif [[ "$BASE_MODEL_PATH" == *"videor1"* ]] || [[ "$BASE_MODEL_PATH" == *"Video-R1"* ]]; then
	BASE_MODEL_PATH_TEMPLATE="videor1"
else
	echo "Error: Could not determine template for base model: $BASE_MODEL_PATH"
	exit 1
fi

echo "ADAPTER_PATH: $ADAPTER_PATH"
echo "BASE_MODEL_PATH: $BASE_MODEL_PATH"
echo "BASE_MODEL_PATH_TEMPLATE: $BASE_MODEL_PATH_TEMPLATE"
echo "MODEL_NAME: ${MODEL_NAME}"

# --- further cluster-specific settings ---

export PYTHONUNBUFFERED=1

if [[ "$RUNNING_MODE" == "SHELL" ]]; then
	export SLURM_TMPDIR="/tmp"
fi
echo "SLURM_TMPDIR: ${SLURM_TMPDIR}"

if [[ "$CLUSTER" == "RORQUAL" ]]; then
	export SCANNET_H5_DIR="/project/def-wangcs/indrisch/scratch_saves/ScanNet_h5/scans"
fi

echo "RUNNING_MODE: $RUNNING_MODE"

# --- setting python environment ---

module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
module load python/3.12 cuda/12.6 opencv/4.12.0
module load arrow
module load apptainer
source $VENV_LLAMAFACTORY/bin/activate
pip install --upgrade pip setuptools wheel
pip install packaging
pip install --no-index huggingface_hub ruamel.yaml

# --- using the python environment, use ruamel.yaml to make and modify the yaml needed for the run. ---

# EXPERIMENT_NAME="qwen2_5vl_lora_sft_CoT_eval"
# mkdir -p "${PROJECT_DIR}/models/qwen2_5vl_lora_sft_CoT/out"

TEMPLATE_YAML="${PROJECT_DIR}/examples/train_lora/trillium_qwen2_5vl_lora_sft_CoT_eval.yaml"

# |------------
# | Create a copy of TEMPLATE_YAML that will live at TEMPLATE_YAML but at ...epoch${MODEL_NAME}.yaml
# | Set the following fields in the copied yaml:
# | model_name_or_path: needs to be the path to the latest local snapshot (base model)
# | output_dir: saves/qwen2_5vl-7b/lora/sft/CoT_traineval_resume_ep${ENDING_EPOCH}/
# | resume_from_checkpoint: ${PROJECT_DIR}saves/qwen2_5vl-7b/lora/sft/CoT_traineval/checkpoint-($((STARTING_EPOCH * STEPS_PER_EPOCH)))
# | adapter_name_or_path: ${PROJECT_DIR}saves/qwen2_5vl-7b/lora/sft/CoT_traineval/checkpoint-($((STARTING_EPOCH * STEPS_PER_EPOCH)))
# | stop_at_global_step: ($((ENDING_EPOCH * STEPS_PER_EPOCH)))
# |-----------------

export YAML_FILE="${TEMPLATE_YAML/.yaml/_${MODEL_NAME}.yaml}"
echo "YAML_FILE: $YAML_FILE"

MODEL_NAME_OR_PATH=$(find "$(hf cache scan | grep "${BASE_MODEL_PATH}" | awk '{print $NF}')/snapshots/" -maxdepth 1 -mindepth 1 -printf "%T+ %p\n" | sort | tail -n 1 | awk '{print $NF}') # get the most recent snapshot
echo "MODEL_NAME_OR_PATH: $MODEL_NAME_OR_PATH"

export OUTPUT_DIR="${PROJECT_DIR}/models/${MODEL_NAME}/lora/eval/"
mkdir -p "${OUTPUT_DIR}"
echo "OUTPUT_DIR: $OUTPUT_DIR"

MODIFY_YAML_ARGS=(
	--yaml-template-path "${TEMPLATE_YAML}"
	--yaml-output-path "${YAML_FILE}"
	--model_name_or_path "${MODEL_NAME_OR_PATH}"
	--template "${BASE_MODEL_PATH_TEMPLATE}"
	--output_dir "${OUTPUT_DIR}"
	--deepspeed "examples/deepspeed/ds_z3_config.json" # this is necessary to use
)
# Only set adapter/resume when evaluating a LoRA adapter path (not base/merged).
if [[ -n "${ADAPTER_PATH:-}" ]]; then
	MODIFY_YAML_ARGS+=(
		--adapter_name_or_path "${ADAPTER_PATH}"
		--resume_from_checkpoint "${ADAPTER_PATH}"
	)
fi

python "${PROJECT_DIR}/scripts/utils/modify_yaml.py" "${MODIFY_YAML_ARGS[@]}"

deactivate

# ----- launch! -----

${PROJECT_DIR}/models/qwen2_5vl_lora_sft_CoT/slurm_qwen2_5vl_lora_sft_CoT_eval.sh "$@"
