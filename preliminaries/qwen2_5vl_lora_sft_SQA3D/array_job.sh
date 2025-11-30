#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=0-00:15:00
#SBATCH --gpus-per-node=h100:4
#SBATCH --array=1-5
#SBATCH --output=%N-qwen2_5vl_lora_sft_SQA3Devery24_traineval_resumefromcheckpoint_array-%j.out

# run as sbatch array_job.sh <cluster_name>

CLUSTER_NAME=$1
./multi_runner.sh $SLURM_ARRAY_TASK_ID $CLUSTER_NAME