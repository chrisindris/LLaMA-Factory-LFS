#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=1-00:00:00
#SBATCH --mem=2000GB
#SBATCH --gpus-per-node=h100:8
#SBATCH --output=%N-qwen2_5vl_lora_sft_SQA3Devery24_traineval-%j.out

# Note, for prediction:
# --- to do inference (NOT an evaluation of the model after training): examples/inference/llama3_lora_sft.yaml
# - predict_with_generate: true # this is required
# - predict_with_generate: true is incompatible with DeepSpeed ZeRO-3; hence, we need a separate script and separate run for evaluation (when we would use a separate dataset for eval; see adgen_train in dataset_info.json and on huggingface)

#Note, for evaluation:
# --- to evaluate the model after training (general competency) [https://llamafactory.readthedocs.io/en/latest/getting_started/eval.html]: see examples/train_lora/llama3_lora_eval.yaml
# ----> Question1a: Why does this not refer to a dataset?
# ----> Question1b: What are [mmlu_test, ceval_validation, cmmlu_test]?
# --- to evaluate the model after training (NLG assessment; BLEU and ROUGE scores for quality of generation) [https://llamafactory.readthedocs.io/en/latest/getting_started/eval.html]: see examples/extras/nlg_eval/llama3_lora_predict.yaml
# ----> Question2a: What is NLG, BLEU, ROUGE? (note that this uses alpaca and identity datasets, for which the answer is a full sentence not just a final answer)

# questions regarding evaluation:
# 1. do I need to specify train vs. eval examples in the dataset?
# - 1.ANSWER: see adgen_train/adgen_eval (in dataset_info.json) and on huggingface; the idea is to have a separate train and eval set.
# 1a. how do I use "subset" to split the dataset into train and eval?
# - 1a.ANSWER: see adgen_train/adgen_eval (in dataset_info.json); there is an adgen_train and a separate adgen_eval dataset, where the split indicates the split shown on huggingface.
# - 1a.NOTE: to have it set up like that on huggingface with separate train and eval sets [in my dataset creation script, ask Cursor]
# 2. if I set val_size = 0.1, will that set train_size = 0.9?
# - 2.NOTE: on https://llamafactory.readthedocs.io/en/latest/getting_started/sft.html, it seems like you can do both training and evaluation in one script, but in examples/train_lora/llama3_lora_sft.yaml it is commented out.
# - 2.NOTE: it might help if you can do train and eval in the same script as shown in https://llamafactory.readthedocs.io/en/latest/getting_started/sft.html
# - 2.ANSWER: yes (see files named *SQA3Devery24_traineval*)
# 3. should the training ground truth be the traces or the ground truth answers?
# - 3.ANSWER: for NLG it seems like the same full answer used for training is used here for evaluation (see examples/extras/nlg_eval/llama3_lora_predict.yaml)
# - 3.ANSWER: [answer in general terms, such as for general competency]


# 500 examples took about 5 minutes
# 1 epoch for 33047 examples (train and test 90-10 split) took under 13 hours

module load apptainer

TORCH_CUDA_ARCH_LIST="9.0" # for clusters with h100 GPUs

# STEP 1: RUN THE TRAINING AND EVALUATION

# better to have triton cache on a non-nfs file system for speed
# if we are offline, we need to indicate this
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
    llamafactory-cli train /project/aip-wangcs/indrisch/LLaMA-Factory/examples/train_lora/qwen2_5vl_lora_sft_SQA3Devery24_traineval.yaml