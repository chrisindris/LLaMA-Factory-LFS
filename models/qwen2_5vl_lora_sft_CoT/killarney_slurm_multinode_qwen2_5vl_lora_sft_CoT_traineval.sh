#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --output=out/%N-qwen2_5vl_lora_sft_CoT_traineval-%j.out
#SBATCH --cpus-per-task=64
#SBATCH --time=0-05:00:00
#SBATCH --mem=0
#SBATCH --gpus-per-node=l40s:4
#SBATCH --mail-user=christopher.indris@torontomu.ca
#SBATCH --mail-type=ALL

# Killarney wrapper for CoT SFT (Scene30k + SpatialSSRL_coldstart + 3DThinker10k)
# on L40S (48GB) GPUs. Submit from models/qwen2_5vl_lora_sft_CoT/ so SLURM out/
# lands next to this script:
#   sbatch killarney_slurm_multinode_qwen2_5vl_lora_sft_CoT_traineval.sh
#
# Uses killarney_multinode_qwen2_5vl_lora_sft_CoT_traineval.yaml via the shared
# worker (CLUSTER-detected path).
#
# Per-node dataset staging (default ON in the shared multinode worker):
#   Each srun task copies annotations + H5 packs to $SLURM_TMPDIR/cot_stage
#   (parallel CPUs) so workers do not thrash shared /scratch during training.
#   Disable:  STAGE_DATASETS_LOCAL=0 sbatch ...
#   Tuning:   STAGE_COPY_JOBS, STAGE_STAGGER_SEC — see
#             scripts/utils/stage_node_local_datasets.sh

# --- for reading cluster-specific settings ---
. $(find $(REGEX="(.*LLaMA-Factory[^/]*).*" && [[ $PWD =~ $REGEX ]] && echo "${BASH_REMATCH[1]}") -name "env.sh")

# ----- DEFAULT ARGUMENTS -----
export STARTING_EPOCH="${STARTING_EPOCH:-0}"
export ENDING_EPOCH="${ENDING_EPOCH:-1}"
export STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-310}" # IMPORTANT NOTE: the default value of this should be equal to 4 / num_of_gpus_used * 620

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
	-h | --help)
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

module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
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
# | Create a copy of TEMPLATE_YAML at ...epoch${ENDING_EPOCH}.yaml (cluster-prefixed).
# | Always set:
# |   output_dir: saves/qwen2_5vl-7b/lora/sft/CoT_traineval_resume_ep${ENDING_EPOCH}/
# |   stop_at_global_step: $((ENDING_EPOCH * STEPS_PER_EPOCH))
# |
# | If STARTING_EPOCH > 0 (resume):
# |   resume_from_checkpoint / adapter_name_or_path:
# |     ${PROJECT_DIR}/saves/.../CoT_traineval_resume_ep${STARTING_EPOCH}/checkpoint-$((STARTING_EPOCH * STEPS_PER_EPOCH))
# |   allow_warm_start_resume / require_resume_bundle as warm-start defaults
# |
# | If STARTING_EPOCH == 0 (fresh start, like trillium_*_CoT_traineval.yaml):
# |   resume_from_checkpoint: null
# |   adapter_name_or_path: omitted entirely
# |-----------------

if [ -z "${YAML_FILE:-}" ]; then
	export YAML_FILE="${TEMPLATE_YAML/epoch2/epoch${ENDING_EPOCH}}"
	export YAML_FILE="${YAML_FILE/trillium/${CLUSTER,,}}" && echo "YAML_FILE: ${YAML_FILE}"
fi

export OUTPUT_DIR_SAVES="saves/qwen2_5vl-7b/lora/sft/CoT_traineval_resume_ep${ENDING_EPOCH}/" && echo "OUTPUT_DIR_SAVES: ${OUTPUT_DIR_SAVES}"
export OUTPUT_DIR="${PROJECT_DIR}/${OUTPUT_DIR_SAVES}" && echo "OUTPUT_DIR: ${OUTPUT_DIR}"

if [[ "${STARTING_EPOCH}" -gt 0 ]]; then
	export RESUME_CKPT="${PROJECT_DIR}/saves/qwen2_5vl-7b/lora/sft/CoT_traineval_resume_ep${STARTING_EPOCH}/checkpoint-$((STARTING_EPOCH * STEPS_PER_EPOCH))"
else
	export RESUME_CKPT=null
fi
echo "RESUME_CKPT: ${RESUME_CKPT}"

MODIFY_EXTRA=()
if [[ "${STARTING_EPOCH}" -eq 0 ]]; then
	MODIFY_EXTRA+=(--allow_warm_start_resume true --require_resume_bundle false)
fi

# settings for different gpu types; *"l"* refers to l40s (48GB), otherwise they are A100/H100/H200 which are all 80GB+

GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n 1 | awk '{print $NF}')
echo "GPU TYPE: $GPU_TYPE"

CUTOFF_LEN=$([[ "$GPU_TYPE" == "L40S" ]] && echo ${CUTOFF_LEN:-65536} || echo 131072)                               # 131072 shown to work on l40s, though 65536 may help if batch_size=2
IMAGE_SAMPLE_COUNT=$([[ "$GPU_TYPE" == "L40S" ]] && echo ${L40S_IMAGE_SAMPLE_COUNT:-360} || echo "-1")              # large values shown to work on l40s; 360 should prevent all but the most massive loads
PER_DEVICE_TRAIN_BATCH_SIZE=$([[ "$GPU_TYPE" == "L40S" ]] && echo ${L40S_PER_DEVICE_TRAIN_BATCH_SIZE:-2} || echo 2) # prevents GPU OOM on l40s
GRADIENT_ACCUMULATION_STEPS=$([[ "$GPU_TYPE" == "L40S" ]] && echo 16 || echo 8)
DEEPSPEED=$([[ "$GPU_TYPE" == "L40S" ]] && echo "examples/deepspeed/ds_z2_offload_config.json" || echo "examples/deepspeed/ds_z2_config.json")
PREPROCESSING_NUM_WORKERS=$([[ "$GPU_TYPE" == "L40S" ]] && echo 64 || echo 32) # With large multimodal data on some systems (seen on Rorqual), 32 may deadlock with large multimodal data. However, if we have the data on each compute node, even 64 might be acceptable.
DATALOADER_NUM_WORKERS=$([[ "$GPU_TYPE" == "L40S" ]] && echo 2 || echo 4)      # experiments 4667851_[N] showed that our loaders are running out of memory; additionally, Killarney's l40s nodes only have 512GB of memory.

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=10800

# ----- create the yaml (i.e. set settings) -----

# Define your command arguments in an array
cmd_args=(
	--yaml-template-path "${TEMPLATE_YAML}"
	--yaml-output-path "${YAML_FILE}"
	--output_dir "${OUTPUT_DIR_SAVES}"
	--resume_from_checkpoint "${RESUME_CKPT}"
	--adapter_name_or_path "${RESUME_CKPT}"
	--stop_at_global_step $((ENDING_EPOCH * STEPS_PER_EPOCH))
	--cutoff_len "${CUTOFF_LEN}"
	--image_sample_count "${IMAGE_SAMPLE_COUNT}"
	--per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
	--gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
	--deepspeed "${DEEPSPEED}"
	--preprocessing_num_workers "${PREPROCESSING_NUM_WORKERS}"
	--dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
	--ddp_timeout "${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC}" # avoid NCCL timeouts
)

python "${PROJECT_DIR}/scripts/utils/modify_yaml.py" \
	"${cmd_args[@]}" \
	"${MODIFY_EXTRA[@]}"

deactivate

# ----- multi-node setup -----

echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST}"
export HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) && echo "HEAD_NODE: ${HEAD_NODE}" # store head node's address
export MASTER_ADDR="${MASTER_ADDR:-${HEAD_NODE:-$(hostname)}}" && echo "HEAD_NODE: ${HEAD_NODE}" && echo "MASTER_ADDR: ${MASTER_ADDR}"
export MASTER_PORT="${MASTER_PORT:-29500}" && echo "MASTER_PORT: ${MASTER_PORT}"

# Launch one parent task per node. Each parent task then lets LLaMA-Factory
# start one torchrun worker per visible GPU on that node.
srun \
	--nodes "${SLURM_NNODES}" \
	--ntasks "${SLURM_NNODES}" \
	--ntasks-per-node 1 \
	--kill-on-bad-exit=1 \
	bash ${PROJECT_DIR}/models/qwen2_5vl_lora_sft_CoT/slurm_multinode_qwen2_5vl_lora_sft_CoT_traineval.sh "$@"
