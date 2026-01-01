#!/bin/bash

source /scratch/indrisch/venv_dataset_upload/bin/activate
export HF_TOKEN=$(cat /home/indrisch/TOKENS/cvis-tmu-organization-token.txt)

python upload_model_checkpoint.py \
  --checkpoint /scratch/indrisch/LLaMA-Factory/saves/qwen2_5vl-7b/lora/sft/SQA3Devery24_R12C12F12X62_traineval/checkpoint-465/ \
  --repo-id cvis-tmu/qwen2_5vl-7b-lora-sft-SQA3Devery24_R12C12F12X62_865steps \
  --commit-message "Upload LoRA adapter checkpoint for SQA3Devery24 on Qwen2.5-VL-7B model after 865 steps (465 steps after resuming from 400 step checkpoint, namely cvis-tmu/qwen2_5vl-7b-lora-sft-SQA3Devery24_R12C12F12X62_400steps) of training"

