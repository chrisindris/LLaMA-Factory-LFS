#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4   
#SBATCH --time=0-00:25:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=%N-qwen2_5vl_lora_sft-%j.out

module load apptainer

# better to have triton cache on a non-nfs file system for speed
# if we are offline, we need to indicate this
apptainer run --nv --writable-tmpfs \
    -B /scratch/indrisch/LLaMA-Factory \
    -B /home/indrisch \
    -B /dev/shm:/dev/shm \
    -B /etc/ssl/certs:/etc/ssl/certs:ro \
    -B /etc/pki:/etc/pki:ro \
    -W ${SLURM_TMPDIR} \
    --env HF_HUB_OFFLINE=1 \
    --env MPLCONFIGDIR="${SLURM_TMPDIR}/.config/matplotlib" \
    --env HF_HOME="/scratch/indrisch/huggingface/hub" \
    --env HF_HUB_CACHE="/scratch/indrisch/huggingface/hub" \
    --env TRITON_CACHE_DIR="${SLURM_TMPDIR}/.triton_cache" \
    --env FLASHINFER_WORKSPACE_BASE="/scratch/indrisch/" \
    --pwd /scratch/indrisch/LLaMA-Factory \
    /scratch/indrisch/easyr1_verl_sif/llamafactory.sif \
    llamafactory-cli train /scratch/indrisch/LLaMA-Factory/examples/train_lora/qwen2_5vl_lora_sft.yaml