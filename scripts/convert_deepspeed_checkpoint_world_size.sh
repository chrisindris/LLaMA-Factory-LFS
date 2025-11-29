#!/bin/bash

module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
module load cuda/12.6 opencv/4.11 python/3.11
module load arrow

# if /scratch/indrisch/venv_llamafactory does not exist, create it
if [ ! -d "/scratch/indrisch/venv_llamafactory" ]; then
    virtualenv --no-download /scratch/indrisch/venv_llamafactory
    source /scratch/indrisch/venv_llamafactory/bin/activate
    pip install --no-index --upgrade pip setuptools wheel
    pip install --no-index -r /scratch/indrisch/LLaMA-Factory/requirements.txt
    pip install --no-index deepspeed
else
    source /scratch/indrisch/venv_llamafactory/bin/activate
fi

python /scratch/indrisch/LLaMA-Factory/scripts/convert_deepspeed_checkpoint_world_size.py \
    --checkpoint_path saves/qwen2_5vl-7b/lora/sft/SQA3Devery24_traineval/checkpoint-465 \
    --original_world_size 4 \
    --target_world_sizes 1 8 \
    --output_dir saves/qwen2_5vl-7b/lora/sft/SQA3Devery24_traineval/checkpoint-465_converted