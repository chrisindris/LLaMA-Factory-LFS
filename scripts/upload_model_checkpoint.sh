#!/bin/bash

# --- variables for the run commands ---
# run this script as "./upload_model_checkpoint.sh [flag argumens]", where the following possibilities are valid:
# --id <ID> --checkpoint <CHECKPOINT> --wandb-log <WANDB_LOG> --commit-message <COMMIT MESSAGE>
# --out <OUT_FILE> --commit-message <COMMIT MESSAGE>
# note that if <COMMIT MESSAGE> is not provided, set it equal either to "<ID> CHECKPOINT> <WANDB_LOG" or "<OUT_FILE>", depending on which flags are provided.

# --- parse run flags ---

while [[ $# -gt 0 ]]; do
  case "$1" in
    --id)
      ID="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --out)
      OUT_FILE="$2"
      shift 2
      ;;
    --commit-message)
      COMMIT_MESSAGE="$2"
      shift 2
      ;;
    --wandb-log)
      WANDB_LOG="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage:"
      echo "  $0 --id <ID> --checkpoint <CHECKPOINT_DIR> --wandb-log <WANDB_LOG> [--commit-message <MESSAGE>]"
      echo "  $0 --out <OUT_FILE_OR_DIR> [--commit-message <MESSAGE>]"
      exit 0
      ;;
    *)
      echo "Error: Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -n "$OUT_FILE" && ( -n "$ID" || -n "$CHECKPOINT_DIR" || -n "$WANDB_LOG" ) ]]; then
  echo "Error: Use either --out OR (--id and --checkpoint and --wandb-log), not both." >&2
  exit 1
fi


if [[ -z "$COMMIT_MESSAGE" ]]; then
  if [[ -n "$ID" ]] && [[ -n "$CHECKPOINT" ]] && [[ -n "$WANDB_LOG" ]]; then
    COMMIT_MESSAGE="id: $ID, checkpoint: $CHECKPOINT, wandb log: $WANDB_LOG"
  elif [[ -n "$OUT_FILE" ]]; then
    COMMIT_MESSAGE="out file: $OUT_FILE"
  else
    echo "Error: --commit-message must be provided." >&2
    exit 1
  fi
fi

# --- HANDLING ASSIGNMENT IN THE --out CASE ---
# if --out is provided, we will:
# 1. check if the file exists
# 2. Set CHECKPOINT to Y, given that there is a line in $OUT_FILE that has the form "X Saving model checkpoint to Y" where X is any string. Note that if we get CHECKPOINT as empty, exit with an error.
# 3. Set WANDB_LOG to Z, given that there is a line in $OUT_FILE that has the form "wandb: Run data is saved locally in Z". Note that if we get WANDB_LOG as empty, exit with an error.
# 4. Set ID to B, where the name $OUT_FILE is equal to A-B-C.out where A is an alphanumeric string, B is any string, and C is numeric. Note that if we get ID as empty, exit with an error.

if [[ -n "$OUT_FILE" ]]; then
    # 1) check if the file exists
    if [[ ! -f "$OUT_FILE" ]]; then
        echo "Error: --out file does not exist: $OUT_FILE" >&2
        exit 1
    fi

    # 2) Set CHECKPOINT to Y from: "X Saving model checkpoint to Y"
    CHECKPOINT="$(awk '
        /Saving model checkpoint to / {
            line = $0
            sub(/^.*Saving model checkpoint to /, "", line)
            gsub(/\r/, "", line)
            sub(/[[:space:]]+$/, "", line)
            checkpoint = line
        }
        END { if (checkpoint != "") print checkpoint }
    ' "$OUT_FILE")"

    if [[ -z "$CHECKPOINT" ]]; then
        echo "Error: Could not parse CHECKPOINT from --out file: $OUT_FILE" >&2
        exit 1
    fi

    # 3) Set WANDB_LOG to Z from: "wandb: Run data is saved locally in Z"
    WANDB_LOG="$(awk '
        /wandb: Run data is saved locally in / {
            line = $0
            sub(/^.*wandb: Run data is saved locally in /, "", line)
            gsub(/\r/, "", line)
            sub(/[[:space:]]+$/, "", line)
            wandb = line
        }
        END { if (wandb != "") print wandb }
    ' "$OUT_FILE")"

    if [[ -z "$WANDB_LOG" ]]; then
        echo "Error: Could not parse WANDB_LOG from --out file: $OUT_FILE" >&2
        exit 1
    fi

    # 4) Set ID to B from: A-B-C.out (ID = B)
    out_base="$(basename "$OUT_FILE")"
    if [[ "$out_base" =~ ^[[:alnum:]]+-(.+)-([0-9]+)\.out$ ]]; then
        ID="${BASH_REMATCH[1]}"
    else
        echo "Error: Could not parse ID from --out filename: $out_base (expected A-B-C.out)" >&2
        exit 1
    fi

    if [[ -z "$ID" ]]; then
        echo "Error: Parsed ID is empty from --out filename: $out_base" >&2
        exit 1
    fi
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

# Detect cluster based on terminal prompt or hostname
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

python upload_model_checkpoint.py \
  --checkpoint "$CHECKPOINT" \
  --repo-id "cvis-tmu/$ID" \
  --commit-message "$COMMIT_MESSAGE"

wandb sync "$WANDB_LOG" --id "$ID"

deactivate
