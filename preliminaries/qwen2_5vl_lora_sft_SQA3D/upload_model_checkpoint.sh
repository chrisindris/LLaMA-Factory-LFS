#!/bin/bash

source /scratch/indrisch/venv_dataset_upload/bin/activate
export HF_TOKEN=$(cat /home/indrisch/TOKENS/cvis-tmu-organization-token.txt)

python upload_model_checkpoint.py \
  --checkpoint saves/qwen2_5vl-7b/lora/sft/SQA3Devery24_traineval \
  --repo-id cvis-tmu/qwen2_5vl-7b-lora-sft-SQA3Devery24_ep1