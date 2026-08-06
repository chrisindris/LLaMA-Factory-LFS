#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=out/%N-MODEL_on_DATASET-%j.out
#SBATCH --cpus-per-task=96
#SBATCH --time=0-01:15:00
#SBATCH --gpus-per-node=h100:4

# $1 is the MODEL, $2 is the DATASET. Both $1 and $2 are required. $1 is in {SQA3Devery24, X62}, $2 is in {SQA3Devery24, X62, R1C0F0X0, R0C1F0X0, R0C0F0X1}.

MODEL=$1
DATASET=$2

if [[ "$MODEL" != "SQA3Devery24" ]] && [[ "$MODEL" != "X62" ]]; then
    echo "Error: MODEL must be one of {SQA3Devery24, X62}."
    exit 1
fi

if [[ "$DATASET" != "SQA3Devery24" ]] && [[ "$DATASET" != "X62" ]] && [[ "$DATASET" != "R1C0F0X0" ]] && [[ "$DATASET" != "R0C1F0X0" ]] && [[ "$DATASET" != "R0C0F0X1" ]]; then
    echo "Error: DATASET must be one of {SQA3Devery24, X62, R1C0F0X0, R0C1F0X0, R0C0F0X1}."
    exit 1
fi

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

${PROJECT_DIR}/experiments/killarney_evaluate_${MODEL}_on_${DATASET}.sh