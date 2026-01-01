#!/bin/bash

source /scratch/indrisch/venv_dataset_upload/bin/activate
export HF_TOKEN=$(cat /home/indrisch/TOKENS/cvis-tmu-organization-token.txt)

# python upload_model_checkpoint.py \
#   --checkpoint saves/qwen2_5vl-7b/lora/sft/SQA3Devery24_traineval_native8gpu/checkpoint-200/ \
#   --repo-id cvis-tmu/qwen2_5vl-7b-lora-sft-SQA3Devery24_200steps-native8gpu

# python upload_model_checkpoint.py \
#   --checkpoint saves/qwen2_5vl-7b/lora/sft/SQA3Devery24_traineval/checkpoint-600/ \
#   --repo-id cvis-tmu/qwen2_5vl-7b-lora-sft-SQA3Devery24_ep5

# python upload_model_checkpoint.py \
#   --checkpoint saves/qwen2_5vl-7b/lora/sft/SQA3Devery24_traineval_native8gpu/checkpoint-233/ \
#   --repo-id cvis-tmu/qwen2_5vl-7b-lora-sft-SQA3Devery24_traineval_native8gpu_ep2

# python upload_model_checkpoint.py \
#   --checkpoint saves/videor1/lora/sft/SQA3Devery24_traineval/checkpoint-400/ \
#   --repo-id cvis-tmu/videor1-lora-sft-SQA3Devery24_400steps


# --==-- NOTE FOR VideoR1: in the README.md of the saved checkpoint, you may need to set the base_model to Video-R1/Qwen2.5-VL-7B-COT-SFT.  --==--

# python upload_model_checkpoint.py \
#   --checkpoint saves/videor1/lora/sft/SQA3Devery24_traineval_continued/checkpoint-400/ \
#   --repo-id cvis-tmu/videor1-lora-sft-SQA3Devery24_800steps

python upload_model_checkpoint.py \
  --checkpoint saves/videor1/lora/sft/SQA3Devery24_traineval_continued/checkpoint-400/ \
  --repo-id cvis-tmu/videor1-lora-sft-SQA3Devery24_1200steps