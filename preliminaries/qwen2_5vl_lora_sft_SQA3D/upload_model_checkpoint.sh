#!/bin/bash

source /scratch/indrisch/venv_dataset_upload/bin/activate
export HF_TOKEN=$(cat /home/indrisch/TOKENS/cvis-tmu-organization-token.txt)

python upload_model_checkpoint.py \
  --checkpoint /scratch/indrisch/LLaMA-Factory/saves/qwen3vl-8b/lora/sft/SQA3Devery24_traineval/checkpoint-465 \
  --repo-id cvis-tmu/qwen3vl-8b-lora-sft-SQA3Devery24_465steps \
  --commit-message "Upload LoRA adapter checkpoint for SQA3Devery24 on Qwen3VL-8B model after 465 steps of training"

python upload_model_checkpoint.py \
  --checkpoint /scratch/indrisch/LLaMA-Factory/saves/videor1sft/lora/sft/SQA3Devery24_traineval/checkpoint-465 \
  --repo-id cvis-tmu/videor1-qwen2_5vl-7b-cot-sft-lora-sft-SQA3Devery24_465steps \
  --commit-message "Upload LoRA adapter checkpoint for SQA3Devery24 on Video-R1/Qwen2.5-VL-7B-COT-SFT model after 465 steps of training"

