#!/bin/bash

# ./upload_model_checkpoint.sh \
#     --id qwen2_5vl-7b-lora-sft-Scene30k_traineval_1278steps \
#     --checkpoint /scratch/indrisch/LLaMA-Factory/saves/qwen2_5vl-7b/lora/sft/Scene30k_traineval_5epochs/checkpoint-1278/ \
#     --wandb-log /scratch/indrisch/LLaMA-Factory/wandb/wandb/offline-run-20260406_075716-4qql81z4 \
#     --commit-message "Qwen2.5-VL-7B LoRA adapter checkpoint trained for 1278 steps (3 epoch) total on the Scene30k dataset (corrected)"

# ./upload_model_checkpoint.sh \
#     --id qwen2_5vl-7b-lora-sft-Scene30k_traineval_1704steps \
#     --checkpoint /scratch/indrisch/LLaMA-Factory/saves/qwen2_5vl-7b/lora/sft/Scene30k_traineval_5epochs/checkpoint-1704/ \
#     --wandb-log /scratch/indrisch/LLaMA-Factory/wandb/wandb/offline-run-20260406_075716-4qql81z4 \
#     --commit-message "Qwen2.5-VL-7B LoRA adapter checkpoint trained for 1704 steps (4 epoch) total on the Scene30k dataset (corrected)"

./upload_model_checkpoint.sh \
    --id qwen2_5vl-7b-lora-sft-Scene30k_traineval_2130steps \
    --checkpoint /scratch/indrisch/LLaMA-Factory/saves/qwen2_5vl-7b/lora/sft/Scene30k_traineval_5epochs/checkpoint-2130/ \
    --wandb-log /scratch/indrisch/LLaMA-Factory/wandb/wandb/offline-run-20260406_075716-4qql81z4 \
    --commit-message "Qwen2.5-VL-7B LoRA adapter checkpoint trained for 2130 steps (5 epoch) total on the Scene30k dataset (corrected)"