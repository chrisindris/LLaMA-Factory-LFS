#!/bin/bash

source /scratch/indrisch/venv_dataset_upload/bin/activate
export HF_TOKEN=$(cat /home/indrisch/TOKENS/cvis-tmu-organization-token.txt)

# python upload_model_checkpoint.py \
#   --checkpoint /scratch/indrisch/LLaMA-Factory/saves/qwen2_5vl-7b/lora/sft/SQA3Devery24_R12C12F12X62_traineval/checkpoint-465/ \
#   --repo-id cvis-tmu/qwen2_5vl-7b-lora-sft-SQA3Devery24_R12C12F12X62_865steps \
#   --commit-message "Upload LoRA adapter checkpoint for SQA3Devery24 on Qwen2.5-VL-7B model after 865 steps (465 steps after resuming from 400 step checkpoint, namely cvis-tmu/qwen2_5vl-7b-lora-sft-SQA3Devery24_R12C12F12X62_400steps) of training"

# python upload_model_checkpoint.py \
#   --checkpoint saves/qwen2_5vl-7b/lora/sft/SQA3Devery24_R0C0F0X1_traineval_eval200_corrected/checkpoint-465/ \
#   --repo-id cvis-tmu/qwen2_5vl-7b-lora-sft-SQA3Devery24_X1_465steps \
#   --commit-message "Qwen2.5-VL-7B LoRA adapter checkpoint trained for 465 steps (1 epoch) total on the X1 version of the SQA3D dataset"


# python upload_model_checkpoint.py \
#   --checkpoint saves/qwen2_5vl-7b/lora/sft/SQA3Devery24_R0C1F0X0_traineval_eval200_corrected/checkpoint-465/ \
#   --repo-id cvis-tmu/qwen2_5vl-7b-lora-sft-SQA3Devery24_C1_465steps \
#   --commit-message "Qwen2.5-VL-7B LoRA adapter checkpoint trained for 465 steps (1 epoch) total on the C1 version of the SQA3D dataset"

python upload_model_checkpoint.py \
  --checkpoint saves/qwen2_5vl-7b/lora/sft/SQA3Devery24_R1C0F0X0_traineval_eval200_corrected/checkpoint-465/ \
  --repo-id cvis-tmu/qwen2_5vl-7b-lora-sft-SQA3Devery24_R1_465steps \
  --commit-message "Qwen2.5-VL-7B LoRA adapter checkpoint trained for 465 steps (1 epoch) total on the R1 version of the SQA3D dataset"

python upload_model_checkpoint.py \
  --checkpoint saves/qwen2_5vl-7b/lora/sft/SQA3Devery24_R12C12F12X62_traineval_eval200_corrected/checkpoint-465/ \
  --repo-id cvis-tmu/qwen2_5vl-7b-lora-sft-SQA3Devery24_R12C12F12X62_465steps \
  --commit-message "Qwen2.5-VL-7B LoRA adapter checkpoint trained for 465 steps (1 epoch) total on the R12C12F12X62 (aka X62) version of the SQA3D dataset"

# python upload_model_checkpoint.py \
#   --checkpoint saves/qwen2_5vl-7b/lora/sft/SQA3Devery24_traineval_native8gpu_epoch3plus/checkpoint-400/ \
#   --repo-id cvis-tmu/qwen2_5vl-7b-lora-sft-SQA3Devery24_traineval_native8gpu_ep4 \
#   --commit-message "Qwen2.5-VL-7B-Instruct trained with 8gpus for a total of 866 steps (about 4 epochs); re-committing just in case due to 504 errors during previous upload"
