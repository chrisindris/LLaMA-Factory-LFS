#!/bin/bash

# --- variables for the run commands ---
# run this script as "./upload_merged_checkpoint.sh [flag argumens]"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-id)
      REPO_ID="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --commit-message)
      COMMIT_MESSAGE="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage:"
      echo "  $0 --repo-id <REPO_ID> --checkpoint <CHECKPOINT_DIR> [--commit-message <MESSAGE>]"
      exit 0
      ;;
    *)
      echo "Error: Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$REPO_ID" || -z "$CHECKPOINT" ]]; then
    echo "Error: --repo-id and --checkpoint are required." >&2
    exit 1
fi

if [[ -z "$COMMIT_MESSAGE" ]]; then
    COMMIT_MESSAGE="Upload merged checkpoint: $CHECKPOINT"
fi

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
if [[ "$PS1" == *"rorqual"* ]] || [[ "$HOSTNAME" == *"rorqual"* ]] || [[ "$PS1" == *"rg"* ]] || [[ "$HOSTNAME" == *"rg"* ]]; then
    CLUSTER="RORQUAL"
elif [[ "$PS1" == *"trig"* ]] || [[ "$HOSTNAME" == *"trig"* ]]; then
    CLUSTER="TRILLIUM"
elif [[ "$PS1" == *"klogin"* ]] || [[ "$HOSTNAME" == *"klogin"* ]] || [[ "$PS1" == *"kn"* ]] || [[ "$HOSTNAME" == *"kn"* ]]; then
    CLUSTER="KILLARNEY"
else
    echo "Warning: Could not detect cluster from PS1 or HOSTNAME. Defaulting to RORQUAL."
    CLUSTER="RORQUAL"
fi

export VENV_DATASET_UPLOAD="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'VENV_DATASET_UPLOAD'))")" && echo "VENV_DATASET_UPLOAD: $VENV_DATASET_UPLOAD"

module load StdEnv/2023  gcc/12.3  openmpi/4.1.5
module load python/3.12 cuda/12.6 opencv/4.12.0
module load arrow

source $VENV_DATASET_UPLOAD/bin/activate
export HF_TOKEN=$(cat /home/indrisch/TOKENS/cvis-tmu-organization-token.txt)

# --- run commands ---

python upload_merged_checkpoint.py \
  --checkpoint "$CHECKPOINT" \
  --repo-id "$REPO_ID" \
  --commit-message "$COMMIT_MESSAGE"

deactivate
