#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=out/%N-qwen2_5vl_lora_sft_CoT_traineval_eval-%j.out
#SBATCH --cpus-per-task=48
#SBATCH --time=0-12:00:00
#SBATCH --mem=0
#SBATCH --gpus-per-node=h100:4
#SBATCH --mail-user=christopher.indris@torontomu.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-3

# TamIA wrapper to launch array jobs for evaluating models on datasets.

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
elif [[ "$CLUSTER" == "TAMIA" ]]; then
	# TamIA scratch is /scratch/i/<user>, not /scratch/<user> (Trillium/Nibi default).
	export SCANNET_H5_DIR="${SCANNET_H5_DIR:-/scratch/i/indrisch/ScanNet_h5/scans}"
	export SPATIALSSRL_H5_DIR="${SPATIALSSRL_H5_DIR:-/scratch/i/indrisch/Spatial-SSRL_images_h5}"
	export THINKER10K_H5_DIR="${THINKER10K_H5_DIR:-/scratch/i/indrisch/3DThinker10K_images_h5}"
fi

echo "RUNNING_MODE: $RUNNING_MODE"
echo "SCANNET_H5_DIR: ${SCANNET_H5_DIR:-}"
echo "SPATIALSSRL_H5_DIR: ${SPATIALSSRL_H5_DIR:-}"
echo "THINKER10K_H5_DIR: ${THINKER10K_H5_DIR:-}"

# --- setting python environment ---

# Match the TamIA VENV in slurm_qwen2_5vl_lora_sft_CoT_eval.sh (py313 venv).
# Compute nodes have no pypi.org; pip must stay --no-index (wheelhouse only).
module load StdEnv gcc openmpi python/3.13 cuda/12.6 opencv arrow apptainer hwloc/2.9.1
source $VENV_LLAMAFACTORY/bin/activate
export PIP_NO_INDEX=1
pip install --no-index --upgrade pip setuptools wheel
pip install --no-index packaging
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

# huggingface_hub>=1.0 dropped `hf cache scan` (now `hf cache list`). Resolve the
# local snapshot from HF_HUB_CACHE so offline compute nodes get a real path.
resolve_local_hf_snapshot() {
	local repo_id="$1"
	local cache_root="${HF_HUB_CACHE:-${HF_HOME:-}}"
	if [[ -d "${repo_id}" ]]; then
		printf '%s\n' "${repo_id}"
		return 0
	fi
	if [[ -z "${cache_root}" ]]; then
		echo "Error: HF_HUB_CACHE/HF_HOME is unset; cannot resolve ${repo_id}" >&2
		return 1
	fi
	local repo_dir="${cache_root}/models--${repo_id//\//--}"
	local snapshots_dir="${repo_dir}/snapshots"
	if [[ ! -d "${snapshots_dir}" ]]; then
		echo "Error: no local HF snapshot for ${repo_id}" >&2
		echo "Expected snapshots under: ${snapshots_dir}" >&2
		return 1
	fi
	local latest=""
	if [[ -f "${repo_dir}/refs/main" ]]; then
		latest="${snapshots_dir}/$(tr -d '[:space:]' <"${repo_dir}/refs/main")"
	fi
	if [[ -z "${latest}" || ! -d "${latest}" ]]; then
		latest=$(find "${snapshots_dir}" -maxdepth 1 -mindepth 1 -type d -printf "%T+ %p\n" | sort | tail -n 1 | awk '{print $NF}')
	fi
	if [[ -z "${latest}" || ! -d "${latest}" ]]; then
		echo "Error: snapshots dir is empty: ${snapshots_dir}" >&2
		return 1
	fi
	printf '%s\n' "${latest}"
}

if ! MODEL_NAME_OR_PATH="$(resolve_local_hf_snapshot "${BASE_MODEL_PATH}")"; then
	exit 1
fi
echo "MODEL_NAME_OR_PATH: $MODEL_NAME_OR_PATH"

# Trillium template hard-codes cache_dir=/scratch/indrisch/huggingface/hub.
export CACHE_DIR="${HF_HUB_CACHE:-${HF_HOME}}"
echo "CACHE_DIR: $CACHE_DIR"

export OUTPUT_DIR="${PROJECT_DIR}/models/${MODEL_NAME}/lora/eval/"
mkdir -p "${OUTPUT_DIR}"
echo "OUTPUT_DIR: $OUTPUT_DIR"

MODIFY_YAML_ARGS=(
	--yaml-template-path "${TEMPLATE_YAML}"
	--yaml-output-path "${YAML_FILE}"
	--model_name_or_path "${MODEL_NAME_OR_PATH}"
	--cache_dir "${CACHE_DIR}"
	--template "${BASE_MODEL_PATH_TEMPLATE}"
	--output_dir "${OUTPUT_DIR}"
	--deepspeed "examples/deepspeed/ds_z3_config.json" # this is necessary to use
)
if [[ -n "${SCANNET_H5_DIR:-}" ]]; then
	# Template media_dir is the ScanNet_h5 parent (not .../scans).
	MODIFY_YAML_ARGS+=(--media_dir "${SCANNET_H5_DIR%/scans}")
fi
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
