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

# python upload_model_checkpoint.py \
#   --checkpoint saves/videor1/lora/sft/SQA3Devery24_traineval_continued/checkpoint-400/ \
#   --repo-id cvis-tmu/videor1-lora-sft-SQA3Devery24_1200steps

# python upload_model_checkpoint.py \
#   --checkpoint saves/qwen2_5vl-7b/lora/sft/SQA3Devery24_traineval_native8gpu_epoch3plus/checkpoint-400/ \
#   --repo-id cvis-tmu/qwen2_5vl-7b-lora-sft-SQA3Devery24_traineval_native8gpu_ep4 \
#   --commit-message "Qwen2.5-VL-7B-Instruct trained with 8gpus for a total of 866 steps (about 4 epochs)"

# python upload_model_checkpoint.py \
#   --checkpoint saves/videor1/lora/sft/SQA3Devery24_traineval_continued/checkpoint-400/ \
#   --repo-id cvis-tmu/videor1-lora-sft-SQA3Devery24_1600steps \
#   --commit-message "VideoR1 trained with 8gpus for a total of 1600 steps (3.44 epochs)"


# --- changed the base model to Video-R1/Video-R1-7B in the README.md of the saved checkpoint. ---

# python upload_model_checkpoint.py \
#   --checkpoint /scratch/indrisch/huggingface/hub/models--cvis-tmu--videor1-lora-sft-SQA3Devery24_1200steps/snapshots/3ff1e2834079106e878900ef1529dcbaca5490f5/ \
#   --repo-id cvis-tmu/videor1-lora-sft-SQA3Devery24_1200steps \
#   --commit-message "VideoR1 trained with 8gpus for a total of 1200 steps (2.67 epochs); updating the base model to Video-R1/Video-R1-7B in the README.md of the saved checkpoint."

# python upload_model_checkpoint.py \
#   --checkpoint /scratch/indrisch/huggingface/hub/models--cvis-tmu--videor1-lora-sft-SQA3Devery24_800steps/snapshots/58bd8d18068d5cfb6f864e8e1a0d98cacf4e16b6/ \
#   --repo-id cvis-tmu/videor1-lora-sft-SQA3Devery24_800steps \
#   --commit-message "VideoR1 trained with 8gpus for a total of 800 steps (1.72 epochs); updating the base model to Video-R1/Video-R1-7B in the README.md of the saved checkpoint."

# python upload_model_checkpoint.py \
#   --checkpoint /scratch/indrisch/huggingface/hub/models--cvis-tmu--videor1-lora-sft-SQA3Devery24_400steps/snapshots/34544894c271d880730426493b6eefa4d87fa0a1/ \
#   --repo-id cvis-tmu/videor1-lora-sft-SQA3Devery24_400steps \
#   --commit-message "VideoR1 trained with 8gpus for a total of 400 steps (0.83 epochs); updating the base model to Video-R1/Video-R1-7B in the README.md of the saved checkpoint."


python upload_model_checkpoint.py \
  --checkpoint saves/qwen2_5vl-7b/lora/sft/SQA3Devery24_traineval_native8gpu_evalsteps200_continued/checkpoint-400/ \
  --repo-id cvis-tmu/qwen2_5vl-7b-lora-sft-SQA3Devery24_200steps-native8gpu_ep2 \
  --commit-message "Qwen2.5-VL-7B-Instruct trained (evaluating every 200 steps) with 8gpus for a total of 400 steps (about 2 epochs)"
