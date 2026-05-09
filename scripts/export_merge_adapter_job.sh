#!/bin/bash
#SBATCH --account=def-wangcs
#SBATCH --job-name=llamafactory_export_merge
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=err/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100_2g.20gb:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=00:15:00
#SBATCH --array=3-3
#SBATCH --mail-type=ALL
#SBATCH --mail-user=christopher.indris@torontomu.ca

# to run: sbatch export_merge_adapter_job.sh [cluster]

set -euo pipefail

module load StdEnv/2023  gcc/12.3  openmpi/4.1.5
module load python/3.12 cuda/12.6 opencv/4.12.0
module load arrow
module load apptainer

# User inputs (override with environment variables or edit below)
# Define arrays for array job
BASE_MODEL_PATHS=(
    "Video-R1/Video-R1-7B"
    "Qwen/Qwen2.5-VL-7B-Instruct"
)

BASE_MODEL_PATH_TEMPLATES=(
    "videor1"
    "qwen2_vl"
)

ADAPTER_PATHS=(
    # "cvis-tmu/videor1-lora-sft-SQA3Devery24_800steps"   # 2 epochs of VideoR1 on SQA3D
    # "cvis-tmu/qwen2_5vl-7b-lora-sft-SQA3Devery24_R12C12F12X62_865steps" # 2 epochs of Qwen2.5VL on X62
    # "cvis-tmu/qwen2_5vl-7b-lora-sft-SQA3Devery24_ep2"   # 2 epochs of Qwen2.5VL on SQA3D
    # "cvis-tmu/qwen2_5vl-7b-lora-sft-SQA3Devery24_X1_465steps" # 1 epoch of Qwen2.5VL on X1 version of SQA3D (corrections made)
    # "cvis-tmu/qwen2_5vl-7b-lora-sft-SQA3Devery24_C1_465steps" # 1 epoch of Qwen2.5VL on C1 version of SQA3D (corrections made)
    # "cvis-tmu/videor1-lora-sft-Scene30k_traineval_852steps" # 2 epochs of VideoR1 on Scene30k dataset
    # "cvis-tmu/qwen2_5vl-7b-lora-sft-Scene30k_traineval_2130steps" # 5 epochs of Qwen2.5VL on Scene30k dataset
    cvis-tmu/videor1sft-lora-sft-Scene30k_traineval_426steps 
    cvis-tmu/videor1sft-lora-sft-Scene30k_traineval_852steps
    cvis-tmu/videor1-lora-sft-Scene30k_traineval_426steps
    cvis-tmu/videor1-lora-sft-Scene30k_traineval_852steps
)

# Get the index from SLURM_ARRAY_TASK_ID, default to 0
IDX=${SLURM_ARRAY_TASK_ID:-1}
ADAPTER_PATH=${ADAPTER_PATHS[$IDX]}
ADAPTER_NAME=$(basename "$ADAPTER_PATH")
# if ADAPTER_NAME contains 'videor1', use BASE_MODEL_PATHS[0], else use BASE_MODEL_PATHS[1]
if [[ "$ADAPTER_NAME" == *"videor1"* ]]; then
    BASE_MODEL_PATH=${BASE_MODEL_PATHS[0]}
    TEMPLATE=${BASE_MODEL_PATH_TEMPLATES[0]}
else
    BASE_MODEL_PATH=${BASE_MODEL_PATHS[1]}
    TEMPLATE=${BASE_MODEL_PATH_TEMPLATES[1]}
fi
USER_ACCOUNT=$(whoami)

# --- for reading cluster-specific settings ---

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

# --- setting environment ---

CLUSTER=${1}
if [[ -z "${CLUSTER:-}" ]]; then
    # Detect cluster based on terminal prompt or hostname
    if [[ "$PS1" == *"rorqual"* ]] || [[ "$HOSTNAME" == *"rorqual"* ]] || [[ "$PS1" == *"rg"* ]] || [[ "$HOSTNAME" == *"rg"* ]]; then
        CLUSTER="RORQUAL"
        RUNNING_MODE="APPTAINER" # running mode for RORQUAL
        OFFLINE=1
    elif [[ "$PS1" == *"trig"* ]] || [[ "$HOSTNAME" == *"trig"* ]]; then
        CLUSTER="TRILLIUM"
        RUNNING_MODE="APPTAINER" # running mode for TRILLIUM
        OFFLINE=1
    elif [[ "$PS1" == *"klogin"* ]] || [[ "$HOSTNAME" == *"klogin"* ]] || [[ "$PS1" == *"kn"* ]] || [[ "$HOSTNAME" == *"kn"* ]]; then
        CLUSTER="KILLARNEY"
        RUNNING_MODE="VENV" # running mode for KILLARNEY
        OFFLINE=1
    elif [[ "$PS1" == *"nibi"* ]] || [[ "$HOSTNAME" == *"nibi"* ]] || [[ "$PS1" == *"g"* ]] || [[ "$HOSTNAME" == *"g"* ]]; then
        CLUSTER="NIBI"
        RUNNING_MODE="APPTAINER" # running mode for NIBI
    else
        echo "Warning: Could not detect cluster from PS1 or HOSTNAME. Defaulting to RORQUAL."
        CLUSTER="RORQUAL"
        RUNNING_MODE="APPTAINER" # running mode for unknown cluster
        OFFLINE=1
    fi
fi

OFFLINE=${OFFLINE:-0} # by default, run in online mode
TRANSFORMERS_OFFLINE=$OFFLINE
HUGGINGFACE_HUB_OFFLINE=$OFFLINE

# --- read per-cluster settings from sysconfig.json ---
export HF_HOME="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'HF_HOME'))")" && echo "HF_HOME: $HF_HOME"
export HF_HUB_CACHE="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'HF_HUB_CACHE'))")" && echo "HF_HUB_CACHE: $HF_HUB_CACHE"
export TRITON_CACHE_DIR="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'TRITON_CACHE_DIR'))")" && echo "TRITON_CACHE_DIR: $TRITON_CACHE_DIR"
export FLASHINFER_WORKSPACE_BASE="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'FLASHINFER_WORKSPACE_BASE'))")" && echo "FLASHINFER_WORKSPACE_BASE: $FLASHINFER_WORKSPACE_BASE"
export BEST_GPU="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'BEST_GPU'))")" && echo "BEST_GPU: $BEST_GPU"
export TORCH_EXTENSIONS_DIR="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'TORCH_EXTENSIONS_DIR'))")" && echo "TORCH_EXTENSIONS_DIR: $TORCH_EXTENSIONS_DIR"
export SIF_FILE="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'SIF_FILE'))")" && echo "SIF_FILE: $SIF_FILE"
export MEDIA_DIR="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'media_dir'))")" && echo "MEDIA_DIR: $MEDIA_DIR"
export VENV_LLAMAFACTORY="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'VENV_LLAMAFACTORY'))")" && echo "VENV_LLAMAFACTORY: $VENV_LLAMAFACTORY"

# ----------

# Extract adapter name for unique output directory and repo name
WORKDIR=${WORKDIR:-$PROJECT_DIR}
EXPORT_DIR=${EXPORT_DIR:-"${WORKDIR}/models/merged-model/${ADAPTER_NAME}"}
TEMPLATE=${TEMPLATE:-"qwen2_vl"}
INFER_DTYPE=${INFER_DTYPE:-"bfloat16"}
EXPORT_SIZE=${EXPORT_SIZE:-2}          # shard size in GB
EXPORT_DEVICE=${EXPORT_DEVICE:-"auto"}  # use auto to place export on GPU if available
CONTAINER=${CONTAINER:-$SIF_FILE}

HF_TOKEN=${HF_TOKEN:-$(cat /home/indrisch/TOKENS/cvis-tmu-organization-token.txt)}                # optional Hugging Face token for private checkpoints

DISABLE_VERSION_CHECK=${DISABLE_VERSION_CHECK:-"1"}  # disable version check for faster startup
HF_HOME=${HF_HOME:-$HF_HOME}


# ========= run the export and merge process inside the container =========

# apptainer run --nv --writable-tmpfs \
#   -B ${WORKDIR} \
#   -B /dev/shm:/dev/shm \
#   -B /etc/ssl/certs:/etc/ssl/certs:ro \
#   -B /etc/pki:/etc/pki:ro \
#   -W "${SLURM_TMPDIR:-/tmp}" \
#   --env HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}" \
#   --env HF_HOME="${HF_HOME}" \
#   --env HF_TOKEN="${HF_TOKEN}" \
#   --env TRANSFORMERS_CACHE="${HF_HOME}" \
#   --env TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE} \
#   --env HUGGINGFACE_HUB_OFFLINE=${HUGGINGFACE_HUB_OFFLINE} \
#   --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
#   --env DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK}" \
#   --pwd "${WORKDIR}" \
#   "${CONTAINER}" bash -lc "set -euo pipefail; \
#     llamafactory-cli export \
#       --model_name_or_path \"${BASE_MODEL_PATH}\" \
#       --adapter_name_or_path \"${ADAPTER_PATH}\" \
#       --export_dir \"${EXPORT_DIR}\" \
#       --template \"${TEMPLATE}\" \
#       --finetuning_type lora \
#       --export_size ${EXPORT_SIZE} \
#       --export_device ${EXPORT_DEVICE} \
#       --infer_dtype ${INFER_DTYPE} \
#       --export_legacy_format false"


# on the login node, run "apptainer overlay create --fakeroot --size 20000 ./apptainer/overlay.img"
# apptainer run --nv --fakeroot --overlay /scratch/indrisch/LLaMA-Factory/apptainer/overlay.img \

# apptainer run --nv --fakeroot --overlay /scratch/indrisch/LLaMA-Factory/apptainer/overlay.img \
#     -C \
#     -B /scratch/indrisch/ \
#     -B /dev/shm:/dev/shm \
#     -B /etc/ssl/certs:/etc/ssl/certs:ro \
#     -B /etc/pki:/etc/pki:ro \
#     /scratch/indrisch/huggingface/hub/datasets--cvis-tmu--compute_canada_sif_files/snapshots/382a3b3e54a9fa9450c6c99dd83efaa2f0ca4a5a/llamafactory.sif \
#     bash

apptainer run --nv --writable-tmpfs \
    -C \
    -B /scratch/indrisch/ \
    -B ${WORKDIR} \
    -B /dev/shm:/dev/shm \
    -B /etc/ssl/certs:/etc/ssl/certs:ro \
    -B /etc/pki:/etc/pki:ro \
    -W "${SLURM_TMPDIR:-/tmp}" \
    --env HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}" \
    --env HF_HOME="${HF_HOME}" \
    --env HF_TOKEN="${HF_TOKEN}" \
    --env TRANSFORMERS_CACHE="${HF_HOME}" \
    --env TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE} \
    --env HUGGINGFACE_HUB_OFFLINE=${HUGGINGFACE_HUB_OFFLINE} \
    --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
    --env DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK}" \
    --env PYTHONPATH="${PROJECT_DIR}/src" \
    --pwd "${WORKDIR}" \
    "${CONTAINER}" bash -lc "set -euo pipefail; \
        llamafactory-cli export \
        --model_name_or_path \"${BASE_MODEL_PATH}\" \
        --adapter_name_or_path \"${ADAPTER_PATH}\" \
        --export_dir \"${EXPORT_DIR}\" \
        --template \"${TEMPLATE}\" \
        --finetuning_type lora \
        --export_size ${EXPORT_SIZE} \
        --export_device ${EXPORT_DEVICE} \
        --infer_dtype ${INFER_DTYPE} \
        --export_legacy_format false"

echo "Model exported successfully to ${EXPORT_DIR}"

# Export the output merged model to HF Hub (optional)
# Uncomment and set the following variables if you want to push to HF Hub
# module load python/3.12.4
# module load arrow/21.0.0

python -m pip install --upgrade huggingface_hub

hf upload "${ADAPTER_PATH}_merged" "${EXPORT_DIR}" \
  --repo-type model \
  --token "${HF_TOKEN}" \
  --commit-message "Upload merged model from LLaMA-Factory export ${ADAPTER_NAME}"

echo "Model uploaded successfully to Hugging Face Hub"