#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=0-00:15:00
#SBATCH --gpus-per-node=h100:4
#SBATCH --output=%N-qwen2_5vl_lora_sft_SQA3Devery24_traineval_resumefromcheckpoint-%j.out

TASK_NUMBER=$1
CLUSTER_NAME=$2

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
export MEDIA_DIR="$(python -c "import sysconfigtool; print(sysconfigtool.read('$CLUSTER_NAME', 'media_dir'))")"
export HF_HOME="$(python -c "import sysconfigtool; print(sysconfigtool.read('$CLUSTER_NAME', 'HF_HOME'))")"
export HF_HUB_CACHE="$(python -c "import sysconfigtool; print(sysconfigtool.read('$CLUSTER_NAME', 'HF_HUB_CACHE'))")"
export TRITON_CACHE_DIR="$(python -c "import sysconfigtool; print(sysconfigtool.read('$CLUSTER_NAME', 'TRITON_CACHE_DIR'))")"
export FLASHINFER_WORKSPACE_BASE="$(python -c "import sysconfigtool; print(sysconfigtool.read('$CLUSTER_NAME', 'FLASHINFER_WORKSPACE_BASE'))")"
export TORCH_CUDA_ARCH_LIST="$(python -c "import sysconfigtool; print(sysconfigtool.read('$CLUSTER_NAME', 'TORCH_CUDA_ARCH_LIST'))")"
export TORCH_EXTENSIONS_DIR="$(python -c "import sysconfigtool; print(sysconfigtool.read('$CLUSTER_NAME', 'TORCH_EXTENSIONS_DIR'))")"
export PYTORCH_KERNEL_CACHE_PATH="$(python -c "import sysconfigtool; print(sysconfigtool.read('$CLUSTER_NAME', 'PYTORCH_KERNEL_CACHE_PATH'))")"
export FORCE_TORCHRUN="$(python -c "import sysconfigtool; print(sysconfigtool.read('$CLUSTER_NAME', 'FORCE_TORCHRUN'))")"
export BEST_GPU="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('$CLUSTER_NAME', 'BEST_GPU'))")"
export SIF_FILE="$(python -c "import sysconfigtool; print(sysconfigtool.read('$CLUSTER_NAME', 'SIF_FILE'))")"

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

python modify_yaml.py \
    --input_path ${YAML_DIR}/${BASE_YAML}.yaml \
    --output_path ${NEW_YAML} \
    --keys \ 
    "num_train_epochs" \
    "resume_from_checkpoint" \
    "media_dir" \
    --values \
    "${NEW_EPOCH}" \
    "${SAVE_DIR}checkpoint-${MOST_RECENT_SAVE}/" \
    "${MEDIA_DIR}" \


# ----- RUN THE COMMAND -----

module load apptainer
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