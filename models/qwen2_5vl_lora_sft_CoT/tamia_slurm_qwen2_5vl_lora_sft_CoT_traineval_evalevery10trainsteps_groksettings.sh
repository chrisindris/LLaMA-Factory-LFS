#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=out/%N-qwen2_5vl_lora_sft_CoT_traineval_evalevery10trainsteps_groksettings-%j.out
#SBATCH --cpus-per-task=48
#SBATCH --time=0-19:00:00
#SBATCH --mem=0
#SBATCH --gpus-per-node=h100:4
#SBATCH --mail-user=christopher.indris@torontomu.ca
#SBATCH --mail-type=ALL

# ===  tamia_slurm_qwen2_5vl_lora_sft_CoT_traineval_evalevery10trainsteps_groksettings.sh  ===
# This is identical to tamia_slurm_qwen2_5vl_lora_sft_CoT_traineval_evalevery10trainsteps_groksettings.sh, but using the settings suggested by grok in debug/logging_analysis/out/cot_sft_remedy_report.md.
# - This will be followed up by grok's suggested ablation study (array job).
#  
#  Prereqs:
#  - Create ${PROJECT_DIR}/data/control_tokens.yaml (token -> description dict for desc_init) --> DONE!
#  - Ensure all datasets have only the think and answer tags -> DONE!
#  - Ensure that the 16 eval samples we use are the SAME (8 Scene30k + 4 SpatialSSRL + 4 3DThinker). Train on the parent mixes minus those frozen question_ids (`exclude_eval_from_train`).
#  - Check with AI (give it the llamafactory output instructions and the settings we are using) to suggest alternative optimizers (adam/badam/galore/apollo) [though this is more for memory] or lora settings. Perhaps nonzero --lora-dropout could help? -> keep adam, use nonzero --lora-dropout
#
# --- TamIA wrapper for CoT SFT (Scene30k + SpatialSSRL_coldstart + 3DThinker10k) on H100 (80GB) GPUs. Identical to tamia_qwen2_5vl_lora_sft_CoT_traineval.sh, but: ---
# Changes (eval):
# 1. Every 10 training steps we perform evaluation on the SAME 16 eval examples.; will involve --eval_steps=10, --eval_on_start=True (both of those should use only 16 examples), possibly --eval_strategy=steps, prediction_loss_only=false? Maybe --do-predict=True to do predictions on the test set also, although this would likely want to do predictions on the whole test set which we don't want? 
# --> Eval of 4384 eval examples (batch size 1, 4 GPUs => 1024 steps) takes ~90 mins; specifically, it was 1:34:55 for the training and 1:35:14 total so it takes about 20 seconds for overhead
# --> therefore 16 examples (4 steps) should take ~20 seconds each and about 20 seconds for overhead (40 seconds); let's be generous and give 1 min per eval
# --> evaluating every 10 steps means (for 616 steps per epoch with val_size_equivalent=0.1) 62 rounds
# --> therefore, eval 16 examples every 10 steps should only take 1 extra hour
# 2. We could change --repetition_penalty=1.1 to slightly prevent repetition, though this does not affect the training, only the eval output. We'd want to adjust this for when we are benchmarking.
# Changes (tokens):
# 2. --compute_accuracy set to True to compute token-level accuracy (good for evaluating fine-grained model predictions, tracking lang model performance, diagnosing prediction errors)
# 3. --new_special_tokens_config: ${PROJECT_DIR}/data/control_tokens.yaml: token->description dict (not the AddedToken list in control_tokens.json)
# 4. --init_special_tokens: desc_init_w_noise 
# 5. --additional_target: embed_tokens,lm_head: we do this because we will define LoRA freezes base model weights, but since the new tokens have uninitialized embeddings we need to ensure that the embed_tokens and lm_head are unfrozen to adjust to them.
# 6. --skip_special_tokens False
# Changes (SFT hyperparams):
# 7. --warmup_ratio: 0.02: reduce from 0.1 so that instead of 342 steps (0.5 epochs) we only warm up for 68 steps. This will reduce the AoC of the learning rate, and therefore we don't have a huge shock to the weights while keeping the same max_learning_rate.
# 8. We will reduce the lr decay via modifying num_cycles (HuggingFace cosine, NOT Megatron lr_decay_iters).
# --> transformers.get_cosine_schedule_with_warmup lets you pass in num_cycles (the # of cosine periods to cover over the course of the training). 
# --> Currently it is 0.5 so we go from the max down to 0 learning rate. This follows curve \left\{342<x<3085:0.0005\cdot\cos\left(\frac{0.5\cdot\pi\left(x-342\right)}{1542.5}\right)+0.0005\right\}.
# --> We can make this 0.4 so that the learning rate doesn't decay as much.
# --> at step 685 = epoch 1 (with warmup of 68), the lr with 0.4 is 97% of the lr with 0.5.
# --> at step 1240 = epoch 2 (with warmup of 68), the lr with 0.4 is 87% of the lr with 0.5.
# --> Must quote the JSON: --lr_scheduler_kwargs '{"num_cycles": 0.4}' or bash splits it and YAML becomes '{lr_decay_iters:'.
# 9. Extra lora modifications
#
#
# Ideas:
# - we could have a separate run where --eval_on_each_dataset = True?
# - is there a better loss function we can use?
# 
#
# Submit from models/qwen2_5vl_lora_sft_CoT/ so SLURM out/
# lands next to this script:
#   sbatch tamia_slurm_qwen2_5vl_lora_sft_CoT_traineval_evalevery10trainsteps_groksettings.sh
#
# Uses tamia_qwen2_5vl_lora_sft_CoT_traineval.yaml via the shared
# worker (CLUSTER-detected path).
#
# Per-node dataset staging (default ON in the shared multinode worker):
#   Each srun task copies annotations + H5 packs to $SLURM_TMPDIR/cot_stage
#   (parallel CPUs) so workers do not thrash shared /scratch during training.
#   Disable:  STAGE_DATASETS_LOCAL=0 sbatch ...
#   Tuning:   STAGE_COPY_JOBS, STAGE_STAGGER_SEC — see
#             scripts/utils/stage_node_local_datasets.sh
#
#
# TODO for this run:
# - modify the SLURM_TMPDIR to be SLURM_TMPDIR/experiment_name (mkdir -p) to prevent collisions. -> done!
# - make the venv from scratch in the new SLURM_TMPDIR, which will also stave off collisions. -> done!
# - we have some hardcoded paths in use, perhaps we can put them into env.sh? -> done!
# - make sure that the YAML we write to is named according to the experiment. -> done!

EXPERIMENT_NAME="qwen2_5vl_lora_sft_CoT_traineval_evalevery10trainsteps_groksettings"

# --- for reading cluster-specific settings ---
. $(find $(REGEX="(.*LLaMA-Factory[^/]*).*" && [[ $PWD =~ $REGEX ]] && echo "${BASH_REMATCH[1]}") -name "env.sh")

# ----- DEFAULT ARGUMENTS -----
export STARTING_EPOCH="${STARTING_EPOCH:-0}"
export ENDING_EPOCH="${ENDING_EPOCH:-1}"
export STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-616}" # val_size_equivalent=0.1 on 43835 -> 39451, minus X eval16 overlaps; 4 GPU, bs=2, ga=8 -> 616 if X=0. Scale as 4/num_gpus * 616
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}" # 

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

# on TAMIA, compute nodes always use /tmp by default; this helps for organization, in case they use the same global /tmp.
if [[ "$CLUSTER" == "TAMIA" && "$RUNNING_MODE" != "SHELL" ]] ; then
    export SLURM_TMPDIR="${SLURM_TMPDIR}/${EXPERIMENT_NAME}"
    mkdir -p "$SLURM_TMPDIR"
fi

echo "SLURM_TMPDIR: $SLURM_TMPDIR"
echo "RUNNING_MODE: $RUNNING_MODE"

# these should be exported through sysconfig.json 
# export SCANNET_H5_DIR="/scratch/i/indrisch/ScanNet_h5/scans"
# export SPATIALSSRL_H5_DIR="/scratch/i/indrisch/Spatial-SSRL_images_h5"
# export THINKER10K_H5_DIR="/scratch/i/indrisch/3DThinker10K_images_h5/"

# Must match data/dataset_info.json (formatted CoT mix). Job 445260 staged the
# unformatted Scene30k parquet, which has no formatting_instruction column.
# export SCENE30K_ANN_SRC="${HF_HUB_CACHE}/datasets--cvis-tmu--Scene30K/snapshots/84a202a417f455f197879495c81b1095f1cf8f53/train-00000-of-00001.with_question_id.formatted.parquet"
# export SPATIALSSRL_ANN_SRC="${HF_HUB_CACHE}/datasets--cvis-tmu--Spatial-SSRL-81k/snapshots/d8e2fabf27e68f41c997b4e7532c67758668c0bd/SFT-coldstart.with_question_id.formatted.json"
# export THINKER10K_ANN_SRC="${HF_HUB_CACHE}/datasets--cvis-tmu--3dthinker-10k-mcq/snapshots/4dd9eb7f24b03c4e9f1265c7e177325cadec9d2d/3dthinker10k_cot.with_question_id.formatted.jsonl"
export DATASET_INFO_PATH=$(find $(REGEX="(.*LLaMA-Factory[^/]*).*" && [[ $PWD =~ $REGEX ]] && echo "${BASH_REMATCH[1]}") -name "dataset_info.json")
export SCENE30K_ANN_SRC="$(jq -r '.["Scene30k"].file_name' "${DATASET_INFO_PATH}" | envsubst)" && echo "SCENE30K_ANN_SRC: ${SCENE30K_ANN_SRC}"
export SPATIALSSRL_ANN_SRC="$(jq -r '.["SpatialSSRL_coldstart"].file_name' "${DATASET_INFO_PATH}" | envsubst)" && echo "SPATIALSSRL_ANN_SRC: ${SPATIALSSRL_ANN_SRC}"
export THINKER10K_ANN_SRC="$(jq -r '.["3DThinker10k"].file_name' "${DATASET_INFO_PATH}" | envsubst)" && echo "THINKER10K_ANN_SRC: ${THINKER10K_ANN_SRC}"

# --- setting python environment ---

module load StdEnv gcc openmpi python/3.13 cuda/12.6 opencv arrow apptainer hwloc/2.9.1

echo "Copying venv to local storage..."
COPIED_VENV="${SLURM_TMPDIR}/venv"
cp -a "${VENV_LLAMAFACTORY}" "${COPIED_VENV}"
export VENV_LLAMAFACTORY="${COPIED_VENV}"

echo "VENV_LLAMAFACTORY: ${VENV_LLAMAFACTORY}"

source "${VENV_LLAMAFACTORY}/bin/activate"
pip install --no-index --upgrade pip setuptools wheel
pip install --no-index packaging
pip install --no-index huggingface_hub ruamel.yaml

# --- using the python environment, use ruamel.yaml to make and modify the yaml needed for the run. ---

mkdir -p "${PROJECT_DIR}/models/qwen2_5vl_lora_sft_CoT/out"

TEMPLATE_YAML="${PROJECT_DIR}/examples/train_lora/trillium_qwen2_5vl_lora_sft_CoT_traineval_resume_epoch2.yaml"

# |------------
# | Create a copy of TEMPLATE_YAML at ...epoch${ENDING_EPOCH}.yaml (cluster-prefixed).
# | Always set:
# |   output_dir: saves/qwen2_5vl-7b/lora/sft/CoT_traineval_evalevery10trainsteps_groksettings_ep${ENDING_EPOCH}/
# |   stop_at_global_step: $((ENDING_EPOCH * STEPS_PER_EPOCH))
# |
# | If STARTING_EPOCH > 0 (resume):
# |   resume_from_checkpoint / adapter_name_or_path:
# |     ${PROJECT_DIR}/saves/.../CoT_traineval_evalevery10trainsteps_groksettings_ep${STARTING_EPOCH}/checkpoint-$((STARTING_EPOCH * STEPS_PER_EPOCH))
# |   allow_warm_start_resume / require_resume_bundle as warm-start defaults
# |
# | If STARTING_EPOCH == 0 (fresh start, like trillium_*_CoT_traineval.yaml):
# |   resume_from_checkpoint: null
# |   adapter_name_or_path: omitted entirely
# |-----------------

if [ -z "${YAML_FILE:-}" ]; then
  #export YAML_FILE="${TEMPLATE_YAML/epoch2/epoch${ENDING_EPOCH}}"
  #export YAML_FILE="${YAML_FILE/trillium/${CLUSTER,,}}"
  export YAML_FILE="$(dirname "$TEMPLATE_YAML")/${CLUSTER,,}_${EXPERIMENT_NAME}_epoch${ENDING_EPOCH}_job${SLURM_JOB_ID}.yaml"
  echo "YAML_FILE: ${YAML_FILE}"
fi

export OUTPUT_DIR_SAVES="saves/qwen2_5vl-7b/lora/sft/CoT_traineval_evalevery10trainsteps_groksettings_ep${ENDING_EPOCH}/" && echo "OUTPUT_DIR_SAVES: ${OUTPUT_DIR_SAVES}"
export OUTPUT_DIR="${PROJECT_DIR}/${OUTPUT_DIR_SAVES}" && echo "OUTPUT_DIR: ${OUTPUT_DIR}"

# The Trillium template hard-codes cache_dir=/scratch/indrisch/huggingface/hub.
# That path does not exist on TamIA; transformers then cannot resolve
# Qwen/Qwen2.5-VL-7B-Instruct under HF_HUB_OFFLINE=1.
export CACHE_DIR="${HF_HUB_CACHE}" && echo "CACHE_DIR: ${CACHE_DIR}"
QWEN_CACHE="${CACHE_DIR}/models--Qwen--Qwen2.5-VL-7B-Instruct"
if [[ ! -d "${QWEN_CACHE}/snapshots" ]]; then
  echo "Error: Qwen2.5-VL-7B-Instruct not found under ${CACHE_DIR}" >&2
  echo "Expected: ${QWEN_CACHE}" >&2
  exit 1
fi

if [[ "${STARTING_EPOCH}" -gt 0 ]]; then
  export RESUME_CKPT="${PROJECT_DIR}/saves/qwen2_5vl-7b/lora/sft/CoT_traineval_evalevery10trainsteps_groksettings_ep${STARTING_EPOCH}/checkpoint-$((STARTING_EPOCH * STEPS_PER_EPOCH))"
else
  export RESUME_CKPT=null
fi
echo "RESUME_CKPT: ${RESUME_CKPT}"

MODIFY_EXTRA=()
if [[ "${STARTING_EPOCH}" -eq 0 ]]; then
  MODIFY_EXTRA+=(--allow_warm_start_resume true --require_resume_bundle false)
fi

# settings for different gpu types; *"l"* refers to l40s (48GB), otherwise they are A100/H100/H200 which are all 80GB+

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n 1)
echo "GPU NAME: $GPU_NAME"
# nvidia-smi last field is "HBM3" on H100 80GB HBM3, not "H100". Match L40S by name.
if echo "$GPU_NAME" | grep -qi 'L40S'; then
  GPU_TYPE=L40S
else
  GPU_TYPE=80GB
fi
echo "GPU TYPE: $GPU_TYPE"

CUTOFF_LEN=$([[ "$GPU_TYPE" == "L40S" ]] && echo ${CUTOFF_LEN:-32768} || echo 134272) # 65536 is too high if batch_size=2.
IMAGE_SAMPLE_COUNT=$([[ "$GPU_TYPE" == "L40S" ]] && echo ${L40S_IMAGE_SAMPLE_COUNT:-300} || echo "-1") # large values shown to work on l40s; 360 should prevent all but the most massive loads
PER_DEVICE_TRAIN_BATCH_SIZE=$([[ "$GPU_TYPE" == "L40S" ]] && echo ${L40S_PER_DEVICE_TRAIN_BATCH_SIZE:-1} || echo 2) # prevents GPU OOM on l40s
GRADIENT_ACCUMULATION_STEPS=$([[ "$GPU_TYPE" == "L40S" ]] && echo 16 || echo 8)
DEEPSPEED=$([[ "$GPU_TYPE" == "L40S" ]] && echo "examples/deepspeed/ds_z2_offload_config.json" || echo "examples/deepspeed/ds_z2_config.json")
PREPROCESSING_NUM_WORKERS=$([[ "$GPU_TYPE" == "L40S" ]] && echo 64 || echo 32) # With large multimodal data on some systems (seen on Rorqual), 32 may deadlock with large multimodal data. However, if we have the data on each compute node, even 64 might be acceptable.
DATALOADER_NUM_WORKERS=$([[ "$GPU_TYPE" == "L40S" ]] && echo 2 || echo 4) # experiments 4667851_[N] showed that our loaders are running out of memory; additionally, Killarney's l40s nodes only have 512GB of memory.

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=10800

# ----- create the yaml (i.e. set settings) -----

# Define your command arguments in an array
cmd_args=(
    --yaml-template-path "${TEMPLATE_YAML}"
    --yaml-output-path "${YAML_FILE}"
    --cache_dir "${CACHE_DIR}"
    --output_dir "${OUTPUT_DIR_SAVES}"
    --resume_from_checkpoint "${RESUME_CKPT}"
    --adapter_name_or_path "${RESUME_CKPT}"
    --save_steps "${STEPS_PER_EPOCH}"
    --stop_at_global_step $((ENDING_EPOCH * STEPS_PER_EPOCH))
    --cutoff_len "${CUTOFF_LEN}"
    --image_sample_count "${IMAGE_SAMPLE_COUNT}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --deepspeed "${DEEPSPEED}"
    --preprocessing_num_workers "${PREPROCESSING_NUM_WORKERS}"
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
    --ddp_timeout "${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC}" # avoid NCCL timeouts
    --train_prediction_interval 10 # save the training every 10 steps
    --train_prediction_max_samples 0 # no cap
    --eval_steps 10 # to run an evaluation (and log it) every 10 training steps.
    --eval_on_start true # to eval on the base qwen
    --eval_strategy steps # to ensure that we eval every 10 steps not every 10 epochs
    --eval_dataset Scene30k_eval16,SpatialSSRL_eval16,3DThinker10k_eval16 # frozen 8+4+4 probe
    --val_size 0 # eval_dataset cannot be combined with val_size>0
    --val_size_equivalent 0.1 # train size as if val_size=0.1 (39451 before dropping X eval16 overlaps)
    --exclude_eval_from_train true # drop the 16 holdout question_ids from Scene30k/SpatialSSRL/3DThinker train mixes
    --lora_dropout 0.05 # reduce LoRA adapter overfitting
    --compute_accuracy true # we will compute the token accuracy too
    --repetition_penalty 1.1 # slightly prevents repetition; this is only for eval, we'd want to set this when benchmarking.
    --new_special_tokens_config "${PROJECT_DIR}/data/control_tokens.yaml" # token->description dict for desc_init; not the AddedToken list in control_tokens.json
    --init_special_tokens desc_init_w_noise # initialize special tokens with semantic + random noise
    --skip_special_tokens false # ensure that the special tokens are not ignored
    --additional_target embed_tokens,lm_head # need to unfreeze some model (non-LoRA) weights to adapt to the special tokens
    --warmup_ratio 0.03 # use 102 steps (0.15 epoch) rather than 342 steps (0.5 epoch) for warmup
    --lr_scheduler_type cosine_with_min_lr
    --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' # min lr is 10% of max; quote JSON so bash does not split on the colon/space
    --eval_prediction_mode generate # more accurate to external benchmark behaviour, though it would take longer
    --eval_dump_max_new_tokens 1024 # hard cap for dump generate; keeps NCCL eval gathers from waiting on 2048-token loops
    --do_sample false # greedy dump generate; sampling looped <|im_start|> and timed out NCCL
    --learning_rate 2.0e-5 # half of the previous max lr
)

python "${PROJECT_DIR}/scripts/utils/modify_yaml.py" \
  "${cmd_args[@]}" \
  "${MODIFY_EXTRA[@]}"

# Fail fast: new_special_tokens_config must be a token->description dict (job 445127
# crashed because control_tokens.json is a HuggingFace AddedToken list).
python - "${PROJECT_DIR}/data/control_tokens.yaml" <<'PY'
import sys
from ruamel.yaml import YAML

path = sys.argv[1]
cfg = YAML(typ="safe").load(open(path, encoding="utf-8"))
if not isinstance(cfg, dict):
    raise SystemExit(
        f"new_special_tokens_config must be a dict mapping tokens to descriptions. "
        f"Got {type(cfg).__name__} from {path}. Use data/control_tokens.yaml, not control_tokens.json."
    )
print(f"Special-token descriptions ({len(cfg)}): {list(cfg)}")
PY

EVAL16_IDS="${PROJECT_DIR}/data/cot_eval16/question_ids.json"
if [[ -f "${EVAL16_IDS}" ]]; then
  echo "Frozen 16-sample eval IDs (${EVAL16_IDS}):"
  python - "${EVAL16_IDS}" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
for name, spec in manifest.get("datasets", {}).items():
    print(f"  {name} ({spec.get('n', 0)}):")
    for qid in spec.get("question_ids", []):
        print(f"    {qid}")
PY
else
  echo "Warning: frozen eval ID manifest missing: ${EVAL16_IDS}" >&2
fi

deactivate

# ----- multi-node setup -----

echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST}"
export NNODES="${SLURM_NNODES:-1}" && echo "SLURM_NNODES: ${SLURM_NNODES}"
export HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) && export HEAD_NODE="${HEAD_NODE:-$(hostname)}" && echo "HEAD_NODE: ${HEAD_NODE}" # store head node's address
export MASTER_ADDR="${HEAD_NODE}" && echo "HEAD_NODE: ${HEAD_NODE}" && echo "MASTER_ADDR: ${MASTER_ADDR}"
export MASTER_PORT="${MASTER_PORT:-29500}" && echo "MASTER_PORT: ${MASTER_PORT}"

# Launch one parent task per node. Each parent task then lets LLaMA-Factory
# start one torchrun worker per visible GPU on that node.
if [[ ! "${RUNNING_MODE}" == "SHELL" ]]; then
  srun \
    --nodes "${NNODES}" \
    --ntasks "${NNODES}" \
    --ntasks-per-node 1 \
    --kill-on-bad-exit=1 \
    bash ${PROJECT_DIR}/models/qwen2_5vl_lora_sft_CoT/slurm_multinode_qwen2_5vl_lora_sft_CoT_traineval.sh "$@"
else
  bash ${PROJECT_DIR}/models/qwen2_5vl_lora_sft_CoT/slurm_multinode_qwen2_5vl_lora_sft_CoT_traineval.sh "$@"
fi
