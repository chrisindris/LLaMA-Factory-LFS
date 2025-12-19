#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=0-00:15:00
#SBATCH --mem=1024GB
#SBATCH --gpus-per-node=h100:4
#SBATCH --array=1-5
#SBATCH --output=%N-qwen2_5vl_lora_sft_SQA3Devery24_traineval_resumefromcheckpoint_array-%j.out

TASK_NUMBER=${SLURM_ARRAY_TASK_ID}
CLUSTER_NAME=$1

# Auto-detect cluster name from hostname if not provided
if [[ -z "$CLUSTER_NAME" ]]; then
    HOSTNAME=$(hostname)
    if [[ "$HOSTNAME" == kn* ]]; then
        CLUSTER_NAME="KILLARNEY"
    elif [[ "$HOSTNAME" == *rorqual* ]] || [[ "$HOSTNAME" == rorqual* ]]; then
        CLUSTER_NAME="RORQUAL"
    elif [[ "$HOSTNAME" == *fir* ]] || [[ "$HOSTNAME" == fir* ]]; then
        CLUSTER_NAME="FIR"
    elif [[ "$HOSTNAME" == *nibi* ]] || [[ "$HOSTNAME" == nibi* ]]; then
        CLUSTER_NAME="NIBI"
    elif [[ "$HOSTNAME" == *narval* ]] || [[ "$HOSTNAME" == narval* ]]; then
        CLUSTER_NAME="NARVAL"
    elif [[ "$HOSTNAME" == *trillium* ]] || [[ "$HOSTNAME" == trillium* ]]; then
        CLUSTER_NAME="TRILLIUM"
    elif [[ "$HOSTNAME" == *tamia* ]] || [[ "$HOSTNAME" == tamia* ]]; then
        CLUSTER_NAME="TAMIA"
    else
        echo "Error: Could not auto-detect cluster name from hostname '$HOSTNAME'. Please provide CLUSTER_NAME as first argument or set it as environment variable."
        exit 1
    fi
    echo "Auto-detected cluster name: $CLUSTER_NAME (from hostname: $HOSTNAME)"
fi

echo "Task number: $TASK_NUMBER"

# ----- GET VALUES FOR THIS CLUSTER -----

module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
module load cuda/12.6 opencv/4.11 python/3.11
module load arrow

if [[ "$PWD" == *LLaMA-Factory-LFS* ]]; then
    PROJECT_DIR="${PWD%%LLaMA-Factory-LFS*}/LLaMA-Factory-LFS"
elif [[ "$PWD" == *LLaMA-Factory* ]]; then
    PROJECT_DIR="${PWD%%LLaMA-Factory*}/LLaMA-Factory"
else
    echo "Error: Could not find 'LLaMA-Factory' or 'LLaMA-Factory-LFS' in the current path."
    exit 1
fi

export PYTHONPATH="$PYTHONPATH:${PROJECT_DIR}/scripts"

# Helper function to read config and handle None values (both Python None and string "None")
read_config() {
    local key=$1
    local value=$(python -c "import sysconfigtool; result = sysconfigtool.read('$CLUSTER_NAME', '$key'); print(result if result is not None and str(result) != 'None' else '')")
    # Filter out literal "None" string
    [[ "$value" == "None" ]] && value=""
    echo "$value"
}

export MEDIA_DIR="$(read_config 'media_dir')"
export HF_HOME="$(read_config 'HF_HOME')"
export HF_HUB_CACHE="$(read_config 'HF_HUB_CACHE')"
export TRITON_CACHE_DIR="$(read_config 'TRITON_CACHE_DIR')"
export FLASHINFER_WORKSPACE_BASE="$(read_config 'FLASHINFER_WORKSPACE_BASE')"
export TORCH_CUDA_ARCH_LIST="$(read_config 'TORCH_CUDA_ARCH_LIST')"
# export TORCH_EXTENSIONS_DIR="$(read_config 'TORCH_EXTENSIONS_DIR')"
# export PYTORCH_KERNEL_CACHE_PATH="$(read_config 'PYTORCH_KERNEL_CACHE_PATH')"
# export FORCE_TORCHRUN="$(read_config 'FORCE_TORCHRUN')"
export BEST_GPU="$(read_config 'BEST_GPU')"
export SIF_FILE="$(read_config 'SIF_FILE')"

# Validate critical variables
if [[ -z "$SIF_FILE" ]] || [[ "$SIF_FILE" == "None" ]]; then
    echo "Error: SIF_FILE is not set or is None for cluster '$CLUSTER_NAME'. Please check sysconfig.json."
    exit 1
fi

if [[ "$BEST_GPU" == "h100" ]]; then
    export TORCH_CUDA_ARCH_LIST="9.0"
else
    export TORCH_CUDA_ARCH_LIST="8.0"
fi

YAML_DIR="${PROJECT_DIR}/examples/train_lora/"
BASE_YAML="qwen2_5vl_lora_sft_SQA3Devery24_traineval_resumefromcheckpoint"
SAVE_DIR="${PROJECT_DIR}/saves/qwen2_5vl-7b/lora/sft/SQA3Devery24_traineval/"

# ----- EDIT THE YAML -----

# get the number (epoch) of the most recent saved checkpoint.
MOST_RECENT_SAVE=$(ls -larth ${SAVE_DIR} | grep checkpoint | grep -v converted | tail -n 1 | grep -Eo '[0-9]+$')
echo "Most Recent Save: ${MOST_RECENT_SAVE}"
NEW_EPOCH=$(expr 234 + ${TASK_NUMBER} \* 2) # This will be $(expr ${MOST_RECENT_SAVE} + 1) when we run it for real
echo "New Epoch: ${NEW_EPOCH}"

# starting from the base yaml, generate the yaml for this run.
NEW_YAML=${YAML_DIR}/${BASE_YAML}_${TASK_NUMBER}.yaml
echo "New YAML: ${NEW_YAML}"

source /scratch/indrisch/venv_llamafactory/bin/activate && python ${PROJECT_DIR}/scripts/modify_yaml.py \
    --input_yaml ${YAML_DIR}/${BASE_YAML}.yaml \
    --output_yaml ${NEW_YAML} \
    --keys \
    "num_train_epochs" \
    "resume_from_checkpoint" \
    "media_dir" \
    --values \
    "${NEW_EPOCH}" \
    "${SAVE_DIR}checkpoint-${MOST_RECENT_SAVE}/" \
    "${MEDIA_DIR}" && deactivate


# ----- RUN THE COMMAND -----

# echo all variables currently set:
echo "HF_HOME: ${HF_HOME}"
echo "HF_HUB_CACHE: ${HF_HUB_CACHE}"
echo "TRITON_CACHE_DIR: ${TRITON_CACHE_DIR}"
echo "FLASHINFER_WORKSPACE_BASE: ${FLASHINFER_WORKSPACE_BASE}"
echo "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"
# echo "TORCH_EXTENSIONS_DIR: ${TORCH_EXTENSIONS_DIR}"
# echo "PYTORCH_KERNEL_CACHE_PATH: ${PYTORCH_KERNEL_CACHE_PATH}"
# echo "FORCE_TORCHRUN: ${FORCE_TORCHRUN}"
echo "BEST_GPU: ${BEST_GPU}"
echo "SIF_FILE: ${SIF_FILE}"
echo "MEDIA_DIR: ${MEDIA_DIR}"
echo "PROJECT_DIR: ${PROJECT_DIR}"
echo "SLURM_TMPDIR: ${SLURM_TMPDIR}"
echo "NEW_YAML: ${NEW_YAML}"
echo "MOST_RECENT_SAVE: ${MOST_RECENT_SAVE}"
echo "NEW_EPOCH: ${NEW_EPOCH}"
echo "TASK_NUMBER: ${TASK_NUMBER}"
echo "CLUSTER_NAME: ${CLUSTER_NAME}"


module load apptainer

# Unset CUDA-related environment variables from host modules to prevent conflicts
# The container has its own CUDA installation, and host CUDA paths can cause DeepSpeed
# to look for nvcc at incorrect locations (e.g., /cvmfs/... paths that don't exist in container)
unset CUDA_HOME CUDA_PATH CUDA_ROOT CUDA_BIN_PATH CUDA_LIB_PATH
unset CUDA_INC_PATH CUDA_LD_LIBRARY_PATH
unset EBROOTCUDA EBVERSIONCUDA EBDEVELCUDA

apptainer run --nv --writable-tmpfs \
    -B ${PROJECT_DIR} \
    -B /home/indrisch \
    -B /dev/shm:/dev/shm \
    -B /etc/ssl/certs:/etc/ssl/certs:ro \
    -B /etc/pki:/etc/pki:ro \
    -W ${SLURM_TMPDIR} \
    --env HF_HUB_OFFLINE=1 \
    --env MPLCONFIGDIR="${SLURM_TMPDIR}/.config/matplotlib" \
    --env HF_HOME="${HF_HOME}" \
    --env HF_HUB_CACHE="${HF_HUB_CACHE}" \
    --env TRITON_CACHE_DIR="${TRITON_CACHE_DIR}" \
    --env FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE}" \
    --env TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    --env TORCH_EXTENSIONS_DIR="${SLURM_TMPDIR}/.cache/torch_extensions" \
    --env PYTORCH_KERNEL_CACHE_PATH="${SLURM_TMPDIR}/.cache/torch/kernels" \
    --env FORCE_TORCHRUN=1 \
    --pwd ${PROJECT_DIR} \
    ${SIF_FILE} \
    llamafactory-cli train ${NEW_YAML}
