#!/bin/bash

# RUN ON: Login Node
# RUN AS: ./scripts/merge_all_videor1_checkpoints.sh <cluster_name>

if [ -z "$1" ]; then
    echo "Usage: $0 <cluster_name>"
    echo "Example: $0 TRILLIUM"
    exit 1
fi

# module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 arrow cuda/12.6
# module load python/3.12 opencv/4.12.0
# module load git-lfs/3.4.0 
# git-lfs install

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

export VENV_LLAMAFACTORY="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('$1', 'VENV_LLAMAFACTORY'))")"
export HF_HOME="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('$1', 'HF_HOME'))")"
echo "HF_HOME: $HF_HOME"
export HF_HUB_CACHE="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('$1', 'HF_HUB_CACHE'))")"
echo "HF_HUB_CACHE: $HF_HUB_CACHE"
export HF_HUB_OFFLINE=1
unset PYTHONPATH

if [ -z "$VENV_LLAMAFACTORY" ]; then
    echo "Failed to fetch VENV_LLAMAFACTORY for cluster $1. Check scripts/init_cluster_config.py or sysconfigtool.py"
    exit 1
fi

# source /scratch/indrisch/venv_llamafactory_cu126/bin/activate
# source ENV/bin/activate # before running this script, in the command line, load the modules above and then activate it and run `pip install --no-index -r requirements.txt`
# pip install --no-index packaging sympy scipy pytz python-dateutil

# CHECKPOINTS=(
#     "cvis-tmu/videor1-lora-sft-Scene30k_traineval_426steps"
#     "cvis-tmu/videor1-lora-sft-Scene30k_traineval_852steps"
#     "cvis-tmu/videor1-lora-sft-Scene30k_traineval_5epochs"
#     "cvis-tmu/videor1sft-lora-sft-Scene30k_traineval_426steps"
#     "cvis-tmu/videor1sft-lora-sft-Scene30k_traineval_852steps"
#     "cvis-tmu/videor1sft-lora-sft-Scene30k_traineval_5epochs"
# )

CHECKPOINTS=(
    "cvis-tmu/videor1-lora-sft-Scene30k_traineval_852steps"
)

BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"


name_to_path() {
    FOLDER=$(find /scratch/indrisch/huggingface/hub/ -maxdepth 1 -name "*$(basename "$1")*" | head -n 1)
    SNAPSHOT=$(dirname $(find "$FOLDER" -name README.md))
    echo $SNAPSHOT
}


for CKPT in "${CHECKPOINTS[@]}"; do
    echo "================================================================="
    echo "Processing Checkpoint: $CKPT"
    echo "================================================================="
    
    # Fetch local cache path using huggingface_hub
    # ADAPTER_PATH=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='$CKPT', local_files_only=True))")
    
    # if [ -z "$ADAPTER_PATH" ]; then
    #     echo "Error: Could not resolve local cache path for $CKPT. Please ensure it is downloaded via get_data.sh."
    #     continue
    # fi
    
    # echo "Resolved adapter path: $ADAPTER_PATH"
    
    # Strip org prefix to get the clean model name
    CLEAN_NAME=$(basename "$CKPT")
    EXPORT_DIR="$PROJECT_DIR/models/${CLEAN_NAME}_merged"
    
    echo "Merging into: $EXPORT_DIR"
    
    python "$PROJECT_DIR/scripts/merge_lora_for_resume.py" \
        --base-model $(name_to_path "$BASE_MODEL") \
        --adapter-checkpoint $(name_to_path "$CKPT") \
        --output-dir "$EXPORT_DIR"
        
    echo "Successfully finished processing $CKPT"
    echo ""
done

echo "All checkpoints processed!"

# deactivate
