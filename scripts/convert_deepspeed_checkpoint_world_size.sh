#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:03:00
#SBATCH --mem=8GB
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=%N-convert_deepspeed_checkpoint_world_size-%j.out

module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
module load cuda/12.6 opencv/4.11 python/3.11
module load arrow

# if /scratch/indrisch/venv_llamafactory does not exist, create it
if [ ! -d "/scratch/indrisch/venv_llamafactory" ]; then
    virtualenv --no-download /scratch/indrisch/venv_llamafactory
    source /scratch/indrisch/venv_llamafactory/bin/activate
    pip install --no-index --upgrade pip setuptools wheel
    pip install --no-index typing-extensions packaging pandas pillow decorator scipy psutil sympy deepspeed -r /project/aip-wangcs/indrisch/LLaMA-Factory/requirements.txt
else
    source /scratch/indrisch/venv_llamafactory/bin/activate
fi

# python /project/aip-wangcs/indrisch/LLaMA-Factory/scripts/convert_deepspeed_checkpoint_world_size.py \
#     --checkpoint_path /project/aip-wangcs/indrisch/LLaMA-Factory/saves/qwen2_5vl-7b/lora/sft/SQA3Devery24_traineval/checkpoint-465 \
#     --original_world_size 4 \
#     --target_world_sizes 1 8 \
#     --output_dir /project/aip-wangcs/indrisch/LLaMA-Factory/saves/qwen2_5vl-7b/lora/sft/SQA3Devery24_traineval/checkpoint-465_converted

python /project/aip-wangcs/indrisch/LLaMA-Factory/scripts/ds_to_universal.py \ 
    --input_folder /project/aip-wangcs/indrisch/LLaMA-Factory/saves/qwen2_5vl-7b/lora/sft/SQA3Devery24_traineval/checkpoint-465 \
    --output_folder /project/aip-wangcs/indrisch/LLaMA-Factory/saves/qwen2_5vl-7b/lora/sft/SQA3Devery24_traineval/checkpoint-465_converted