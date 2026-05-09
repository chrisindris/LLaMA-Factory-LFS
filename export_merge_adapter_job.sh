#!/bin/bash
#SBATCH --account=def-wangcs
#SBATCH --job-name=llamafactory_export_merge
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100_2g.20gb:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=00:15:00
#SBATCH --array=0-1 # 2 jobs for each model-adapter pair
#SBATCH --mail-type=ALL
#SBATCH --mail-user=christopher.indris@torontomu.ca

set -euo pipefail

module load gcc apptainer

# User inputs (override with environment variables or edit below)
# Define arrays for array job
BASE_MODEL_PATHS=(
    "Video-R1/Video-R1-7B"
    "Qwen/Qwen2.5-VL-7B-Instruct"
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

BASE_MODEL_PATH=${BASE_MODEL_PATHS[$IDX]}
ADAPTER_PATH=${ADAPTER_PATHS[$IDX]}

# Extract adapter name for unique output directory and repo name
ADAPTER_NAME=$(basename "$ADAPTER_PATH")

USER_ACCOUNT=$(whoami)
EXPORT_DIR=${EXPORT_DIR:-"output/merged-model/${ADAPTER_NAME}"}
TEMPLATE=${TEMPLATE:-"qwen2_vl"}
INFER_DTYPE=${INFER_DTYPE:-"bfloat16"}
EXPORT_SIZE=${EXPORT_SIZE:-2}          # shard size in GB
EXPORT_DEVICE=${EXPORT_DEVICE:-"auto"}  # use auto to place export on GPU if available
CONTAINER=${CONTAINER:-"/scratch/${USER_ACCOUNT}/LLaMA-Factory/llamafactory.sif"}
WORKDIR=${WORKDIR:-"/scratch/${USER_ACCOUNT}/LLaMA-Factory"}

_HF_TOKEN=$(cat /scratch/indrisch/TOKENS/huggingface/cvis-tmu-organization-token.txt)
HF_TOKEN=${HF_TOKEN:-_HF_TOKEN}                # optional Hugging Face token for private checkpoints

DISABLE_VERSION_CHECK=${DISABLE_VERSION_CHECK:-"1"}  # disable version check for faster startup
HF_HOME=${HF_HOME:-"huggingface"}

apptainer run --nv --writable-tmpfs \
  -B /scratch/${USER_ACCOUNT}/LLaMA-Factory \
  -B /dev/shm:/dev/shm \
  -B /etc/ssl/certs:/etc/ssl/certs:ro \
  -B /etc/pki:/etc/pki:ro \
  -W "${SLURM_TMPDIR:-/tmp}" \
  --env HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}" \
  --env HF_HOME="${HF_HOME}" \
  --env TRANSFORMERS_CACHE="${HF_HOME}" \
  --env TRANSFORMERS_OFFLINE=0 \
  --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
  --env HF_TOKEN="${HF_TOKEN}" \
  --env DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK}" \
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
module load python/3.12.4
module load arrow/21.0.0

python -m pip install --upgrade huggingface_hub

hf upload "${ADAPTER_PATH}_merged" "${EXPORT_DIR}" \
  --repo-type model \
  --token "${HF_TOKEN}" \
  --commit-message "Upload merged model from LLaMA-Factory export ${ADAPTER_NAME}"

echo "Model uploaded successfully to Hugging Face Hub"