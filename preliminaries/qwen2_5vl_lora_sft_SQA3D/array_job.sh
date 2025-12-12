#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --time=0-01:00:00
#SBATCH --gpus-per-node=h100:4
#SBATCH --array=1-5
#SBATCH --output=vgllm_3d_8b_lora_sft_SQA3Devery24_traineval_resumefromcheckpoint_array-%A-%a-%j.out

preliminaries/qwen2_5vl_lora_sft_SQA3D/multi_runner.sh $SLURM_ARRAY_TASK_ID