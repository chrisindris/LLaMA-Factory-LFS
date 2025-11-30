#!/bin/bash

TASK_NUMBER=$1

echo "Task number: $TASK_NUMBER"

YAML_DIR="/project/aip-wangcs/indrisch/LLaMA-Factory/examples/train_lora/"
BASE_YAML="qwen2_5vl_lora_sft_SQA3Devery24_traineval_resumefromcheckpoint"
SAVE_DIR="/project/aip-wangcs/indrisch/LLaMA-Factory/saves/qwen2_5vl-7b/lora/sft/SQA3Devery24_traineval/"

# ----- EDIT THE YAML -----

# get the number (epoch) of the most recent saved checkpoint.
MOST_RECENT_SAVE=$(ls -larth ${SAVE_DIR} | grep checkpoint | grep -v converted | tail -n 1 | grep -Eo '[0-9]+$')
echo "Most Recent Save: ${MOST_RECENT_SAVE}"
NEW_EPOCH=$(expr 234 + ${TASK_NUMBER} \* 2) # This will be $(expr ${MOST_RECENT_SAVE} + 1) when we run it for real
echo "New Epoch: ${NEW_EPOCH}"

# starting from the base yaml, generate the yaml for this run.
NEW_YAML=${YAML_DIR}/${BASE_YAML}_${TASK_NUMBER}.yaml
TMP_YAML=${YAML_DIR}/${BASE_YAML}_${TASK_NUMBER}.tmp.yaml

echo "New YAML: ${NEW_YAML}"
sed "s/num_train_epochs[^#]*#/num_train_epochs: ${NEW_EPOCH} #/g" ${YAML_DIR}/${BASE_YAML}.yaml > ${TMP_YAML}
sed "s|resume_from_checkpoint[^#]*#|resume_from_checkpoint: ${SAVE_DIR}checkpoint-${MOST_RECENT_SAVE}/ #|g" ${TMP_YAML} > ${NEW_YAML}
rm ${TMP_YAML}

# run the command
module load apptainer
TORCH_CUDA_ARCH_LIST="9.0"
apptainer run --nv --writable-tmpfs \
    -B /project/aip-wangcs/indrisch/LLaMA-Factory \
    -B /home/indrisch \
    -B /dev/shm:/dev/shm \
    -B /etc/ssl/certs:/etc/ssl/certs:ro \
    -B /etc/pki:/etc/pki:ro \
    -W ${SLURM_TMPDIR} \
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
    --pwd /project/aip-wangcs/indrisch/LLaMA-Factory \
    /project/aip-wangcs/indrisch/easyr1_verl_sif/llamafactory.sif \
    llamafactory-cli train ${NEW_YAML}
