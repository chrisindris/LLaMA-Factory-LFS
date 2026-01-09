#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64  
#SBATCH --time=0-02:00:00
#SBATCH --mem=485GB
#SBATCH --gpus-per-node=h100:4
#SBATCH --array=0-19
#SBATCH --output=out/%N-qwen2_5vl_eval_1epochmodel_on_dataset-%j.out

# we are going to evaluate 1epoch models [X62, R1, C1, X1] on the datasets [SQA3D, X62, R1, C1, X1]
# SLURM_ARRAY_TASK_ID=19 # temporary, for debugging

# Since $SLURM_ARRAY_TASK_ID goes from 0 to 19 inclusive, the model will be determined by $SLURM_ARRAY_TASK_ID // 5 and the dataset will be determined by $SLURM_ARRAY_TASK_ID % 5
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
DATASET_INDEX=$((SLURM_ARRAY_TASK_ID % 5))

MODEL_SHORTNAMES=("X62" "R1" "C1" "X1")
DATASET_SHORTNAMES=("SQA3D" "X62" "R1" "C1" "X1")
DATASET_NAMES=("SQA3Devery24" "SQA3Devery24_R12C12F12X62" "SQA3Devery24_R1C0F0X0" "SQA3Devery24_R0C1F0X0" "SQA3Devery24_R0C0F0X1")
MODEL_CHECKPOINTS=("/scratch/indrisch/huggingface/hub/models--cvis-tmu--qwen2_5vl-7b-lora-sft-SQA3Devery24_R12C12F12X62_465steps/snapshots/d9721b66286ae0d8e0e3b213fb5950cbfec94678/" "/scratch/indrisch/huggingface/hub/models--cvis-tmu--qwen2_5vl-7b-lora-sft-SQA3Devery24_R1_465steps/snapshots/f623329af098faf8b79db69bd779640419581827/" "/scratch/indrisch/huggingface/hub/models--cvis-tmu--qwen2_5vl-7b-lora-sft-SQA3Devery24_C1_465steps/snapshots/8d20c34098248e3b686eb673743e1d4b9845eb0d/" "/scratch/indrisch/huggingface/hub/models--cvis-tmu--qwen2_5vl-7b-lora-sft-SQA3Devery24_X1_465steps/snapshots/b5d0cb4f4bdaa2d0dc7639e1da80927467404bf6/")

MODEL_SHORTNAME=${MODEL_SHORTNAMES[$MODEL_INDEX]}
DATASET_SHORTNAME=${DATASET_SHORTNAMES[$DATASET_INDEX]}
DATASET_NAME=${DATASET_NAMES[$DATASET_INDEX]}
MODEL_CHECKPOINT=${MODEL_CHECKPOINTS[$MODEL_INDEX]}

RUNNING_MODE=$1 # on Rorqual, use APPTAINER

echo "Model index: $MODEL_INDEX"
echo "Dataset index: $DATASET_INDEX"
echo "Running mode: $RUNNING_MODE"
echo "Model shortname: $MODEL_SHORTNAME"
echo "Dataset shortname: $DATASET_SHORTNAME"
echo "Dataset name: $DATASET_NAME"
echo "Model checkpoint: $MODEL_CHECKPOINT"

# Before we run the code that follows in this file, we need to create the YAML file at the path $YAML_FILE below.
# To build the YAML file:
# 1. Use /scratch/indrisch/LLaMA-Factory/examples/train_lora/qwen2_5vl_eval_SQA3Devery24_on_R0C0F0X1.yaml as the base yaml file, so we can create a copy of it at location $YAML_FILE.
# 2. Replace the text "SQA3Devery24_R0C0F0X1" with the value of $DATASET_NAME (this will set "dataset: $DATASET_NAME").
# 3. Replace the text "saves/qwen2_5vl-7b/lora/eval/SQA3Devery24_on_R0C0F0X1" with "saves/qwen2_5vl-7b/lora/eval/${MODEL_SHORTNAME}_on_${DATASET_SHORTNAME}" (this will set "output_dir: saves/qwen2_5vl-7b/lora/eval/${MODEL_SHORTNAME}_on_${DATASET_SHORTNAME}").
# 4. Replace the text "/scratch/indrisch/huggingface/hub/models--cvis-tmu--qwen2_5vl-7b-lora-sft-SQA3Devery24_ep1/snapshots/0bdc7d9fb51e700a889b40dbabf01c929e31d43c/checkpoint-465/" with the value of $MODEL_CHECKPOINT (this will set "adapter_name_or_path: $MODEL_CHECKPOINT").
YAML_FILE="/scratch/indrisch/LLaMA-Factory/examples/train_lora/qwen2_5vl_eval_${MODEL_SHORTNAME}_on_${DATASET_SHORTNAME}.yaml"
cp /scratch/indrisch/LLaMA-Factory/examples/train_lora/qwen2_5vl_eval_SQA3Devery24_on_R0C0F0X1.yaml $YAML_FILE
sed -i "s/SQA3Devery24_R0C0F0X1/${DATASET_NAME}/g" $YAML_FILE
sed -i "s/saves\/qwen2_5vl-7b\/lora\/eval\/SQA3Devery24_on_R0C0F0X1/saves\/qwen2_5vl-7b\/lora\/eval\/${MODEL_SHORTNAME}_on_${DATASET_SHORTNAME}/g" $YAML_FILE
sed -i "s|/scratch/indrisch/huggingface/hub/models--cvis-tmu--qwen2_5vl-7b-lora-sft-SQA3Devery24_ep1/snapshots/0bdc7d9fb51e700a889b40dbabf01c929e31d43c/checkpoint-465/|${MODEL_CHECKPOINT}|g" $YAML_FILE





# STEP 1: RUN THE TRAINING AND EVALUATION

# better to have triton cache on a non-nfs file system for speed
# if we are offline, we need to indicate this

if [[ "$RUNNING_MODE" == "APPTAINER" ]]; then
    # use for rorqual

    module load apptainer

    TORCH_CUDA_ARCH_LIST="9.0" # for clusters with h100 GPUs
    
    # Set CUDA_VISIBLE_DEVICES: use SLURM's value if set, otherwise default to 0,1,2,3
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

    apptainer run --nv --writable-tmpfs \
        -C \
        -B /scratch/indrisch/LLaMA-Factory \
        -B /home/indrisch \
        -B /dev/shm:/dev/shm \
        -B /etc/ssl/certs:/etc/ssl/certs:ro \
        -B /etc/pki:/etc/pki:ro \
        -W ${SLURM_TMPDIR} \
        --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
        --env HF_HUB_OFFLINE=1 \
        --env MPLCONFIGDIR="${SLURM_TMPDIR}/.config/matplotlib" \
        --env HF_HOME="/scratch/indrisch/huggingface/hub" \
        --env HF_HUB_CACHE="/scratch/indrisch/huggingface/hub" \
        --env TRITON_CACHE_DIR="${SLURM_TMPDIR}/.triton_cache" \
        --env FLASHINFER_WORKSPACE_BASE="/scratch/indrisch/" \
        --env TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
        --env TORCH_EXTENSIONS_DIR="${SLURM_TMPDIR}/.cache/torch_extensions" \
        --env PYTORCH_KERNEL_CACHE_PATH="${SLURM_TMPDIR}/.cache/torch/kernels" \
        --env FORCE_TORCHRUN=1 \
        --env WANDB_MODE=offline \
        --env WANDB_DIR="/scratch/indrisch/LLaMA-Factory/wandb/" \
        --env WANDB_CACHE_DIR="${SLURM_TMPDIR}/.cache/wandb" \
        --env PYTHONPATH="/scratch/indrisch/LLaMA-Factory/src:${PYTHONPATH:-}" \
        --env NCCL_IB_DISABLE=0 \
        --env NCCL_P2P_DISABLE=0 \
        --env NCCL_DEBUG=INFO \
        --env NCCL_SOCKET_IFNAME=^docker0,lo \
        --pwd /scratch/indrisch/LLaMA-Factory \
        /scratch/indrisch/huggingface/hub/datasets--cvis-tmu--easyr1_verl_sif/snapshots/382a3b3e54a9fa9450c6c99dd83efaa2f0ca4a5a/llamafactory.sif \
        llamafactory-cli train ${YAML_FILE}

elif [[ "$RUNNING_MODE" == "SHELL" ]]; then

    module load apptainer

    TORCH_CUDA_ARCH_LIST="9.0" # for clusters with h100 GPUs

    SLURM_TMPDIR="/scratch/indrisch/tmp"
    
    # Create directories for pip cache and temporary files on scratch
    mkdir -p /scratch/indrisch/tmp
    mkdir -p /scratch/indrisch/.cache/pip
    mkdir -p /scratch/indrisch/.cache/torch_extensions
    mkdir -p /scratch/indrisch/.cache/torch/kernels
    mkdir -p /scratch/indrisch/.config/matplotlib
    mkdir -p /scratch/indrisch/.triton_cache

    apptainer shell --nv --writable \
        -B /scratch/indrisch \
        -B /home/indrisch \
        -B /dev/shm:/dev/shm \
        -B /etc/ssl/certs:/etc/ssl/certs:ro \
        -B /etc/pki:/etc/pki:ro \
        -W ${SLURM_TMPDIR} \
        --env CUDA_VISIBLE_DEVICES=0,1,2,3 \
        --env HF_HUB_OFFLINE=1 \
        --env TMPDIR="/scratch/indrisch/tmp" \
        --env PIP_CACHE_DIR="/scratch/indrisch/.cache/pip" \
        --env MPLCONFIGDIR="/scratch/indrisch/.config/matplotlib" \
        --env HF_HOME="/scratch/indrisch/huggingface/hub" \
        --env HF_HUB_CACHE="/scratch/indrisch/huggingface/hub" \
        --env TRITON_CACHE_DIR="/scratch/indrisch/.triton_cache" \
        --env FLASHINFER_WORKSPACE_BASE="/scratch/indrisch/" \
        --env TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
        --env TORCH_EXTENSIONS_DIR="/scratch/indrisch/.cache/torch_extensions" \
        --env PYTORCH_KERNEL_CACHE_PATH="/scratch/indrisch/.cache/torch/kernels" \
        --env FORCE_TORCHRUN=1 \
        --env WANDB_MODE=offline \
        --env WANDB_DIR="/scratch/indrisch/LLaMA-Factory/wandb/" \
        --env PYTHONPATH="/scratch/indrisch/LLaMA-Factory/src:${PYTHONPATH:-}" \
        --pwd /scratch/indrisch/LLaMA-Factory \
        /scratch/indrisch/huggingface/hub/datasets--cvis-tmu--easyr1_verl_sif/snapshots/382a3b3e54a9fa9450c6c99dd83efaa2f0ca4a5a/llamafactory.sif


elif [[ "$RUNNING_MODE" == "VENV" ]]; then

    module load StdEnv/2023  gcc/12.3  openmpi/4.1.5
    module load python/3.12 cuda/12.6 opencv/4.12.0
    module load arrow

    source /scratch/indrisch/venv_llamafactory_cu126/bin/activate

    # pushd /scratch/indrisch/
    # module load StdEnv/2023  gcc/12.3  openmpi/4.1.5
    # module load python/3.12 cuda/12.6 opencv/4.12.0
    # module load arrow
    # virtualenv --no-download venv_llamafactory_cu126
    # source venv_llamafactory_cu126/bin/activate
    # popd
    
    # Set environment variables AFTER venv activation to ensure they persist
    # Create cache directories before they're needed
    mkdir -p "${SLURM_TMPDIR}/.cache/torch_extensions"
    mkdir -p "${SLURM_TMPDIR}/.cache/torch/kernels"
    mkdir -p "${SLURM_TMPDIR}/.config/matplotlib"
    mkdir -p "${SLURM_TMPDIR}/.triton_cache"
    
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export HF_HUB_OFFLINE=1 
    export MPLCONFIGDIR="${SLURM_TMPDIR}/.config/matplotlib"
    export HF_HOME="/scratch/indrisch/huggingface/hub"
    export HF_HUB_CACHE="/scratch/indrisch/huggingface/hub"
    export TRITON_CACHE_DIR="${SLURM_TMPDIR}/.triton_cache"
    export FLASHINFER_WORKSPACE_BASE="/scratch/indrisch/"
    export TORCH_CUDA_ARCH_LIST="9.0" # for clusters with a100 GPUs
    export TORCH_EXTENSIONS_DIR="${SLURM_TMPDIR}/.cache/torch_extensions" # needed for cpu_adam
    export PYTORCH_KERNEL_CACHE_PATH="${SLURM_TMPDIR}/.cache/torch/kernels"
    export FORCE_TORCHRUN=1 
    export WANDB_MODE=offline 
    export WANDB_DIR="/scratch/indrisch/LLaMA-Factory/wandb/" 
    export DISABLE_VERSION_CHECK=1 # since the automatic detector doesn't automatically see that transformers==4.57.1+computecanada is the same as transformers==4.57.1
    # giving the slow tokenizer a try: https://github.com/hiyouga/LLaMA-Factory/issues/8600#issuecomment-3227071979

    # pushd /scratch/indrisch/LLaMA-Factory
    # pip install --upgrade pip setuptools wheel
    # pip install packaging psutil pandas pillow decorator scipy matplotlib platformdirs pyarrow sympy wandb ray -e ".[torch,metrics,deepspeed,liger-kernel]"


    pushd /scratch/indrisch/LLaMA-Factory
    llamafactory-cli train ${YAML_FILE}

else
    echo "Invalid running mode: $RUNNING_MODE"
    exit 1
fi





