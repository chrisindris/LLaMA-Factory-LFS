#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:15:00
#SBATCH --gpus-per-node=h100_2g.20gb:1
#SBATCH --mem=64GB
#SBATCH --output=out/%N-merge_wrapper-%j.out
#SBATCH --mail-user=christopher.indris@torontomu.ca
#SBATCH --mail-type=ALL

# run this in a CPU job.

module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
module load python/3.12 cuda/12.6 opencv/4.12.0
module load arrow

source /scratch/indrisch/venv_llamafactory_cu126/bin/activate

# DISABLE_VERSION_CHECK=1 python merge_lora_for_resume.py --base-model /scratch/indrisch/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5/ --adapter-checkpoint /scratch/indrisch/huggingface/hub/models--cvis-tmu--videor1-lora-sft-Scene30k_traineval_426steps/snapshots/41cf574c5e5dbb2fff362c9b64a310f9f457dda2/ --output-dir ../models/videor1-lora-sft-Scene30k_traineval_426steps_merged/
# DISABLE_VERSION_CHECK=1 python merge_lora_for_resume.py --base-model /scratch/indrisch/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5/ --adapter-checkpoint /scratch/indrisch/huggingface/hub/models--cvis-tmu--videor1-lora-sft-Scene30k_traineval_852steps/snapshots/a6362521d38c04bbced5da3168d5b7c52ad65375/ --output-dir ../models/videor1-lora-sft-Scene30k_traineval_852steps_merged/
# DISABLE_VERSION_CHECK=1 python merge_lora_for_resume.py --base-model /scratch/indrisch/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5/ --adapter-checkpoint /scratch/indrisch/huggingface/hub/models--cvis-tmu--videor1-lora-sft-Scene30k_traineval_5epochs/snapshots/d5e1b585c36610d1ce79487bd00c1ff5ef022540/ --output-dir ../models/videor1-lora-sft-Scene30k_traineval_5epochs_merged/

# DISABLE_VERSION_CHECK=1 python merge_lora_for_resume.py --base-model /scratch/indrisch/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5/ --adapter-checkpoint /scratch/indrisch/huggingface/hub/models--cvis-tmu--videor1sft-lora-sft-Scene30k_traineval_426steps/snapshots/55f6fcee800cb45453beedea1f3f07f3c579d6db/ --output-dir ../models/videor1sft-lora-sft-Scene30k_traineval_426steps_merged/
# DISABLE_VERSION_CHECK=1 python merge_lora_for_resume.py --base-model /scratch/indrisch/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5/ --adapter-checkpoint /scratch/indrisch/huggingface/hub/models--cvis-tmu--videor1sft-lora-sft-Scene30k_traineval_852steps/snapshots/adbf135bfa7f46bed08dac6f9526ef25a1048cd9/ --output-dir ../models/videor1sft-lora-sft-Scene30k_traineval_852steps_merged/
# DISABLE_VERSION_CHECK=1 python merge_lora_for_resume.py --base-model /scratch/indrisch/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5/ --adapter-checkpoint /scratch/indrisch/huggingface/hub/models--cvis-tmu--videor1sft-lora-sft-Scene30k_traineval_5epochs/snapshots/dce24826d1db8902e81c31cc7c907b66771be500/ --output-dir ../models/videor1sft-lora-sft-Scene30k_traineval_5epochs_merged/

DISABLE_VERSION_CHECK=1 python merge_lora_for_resume.py --base-model /scratch/indrisch/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5/ --adapter-checkpoint /scratch/indrisch/huggingface/hub/models--cvis-tmu--qwen2_5vl-7b-lora-sft-CoT_traineval_1epochs/snapshots/586c43def596efe688aa2d9f262959fd209de425/ --output-dir ../models/qwen2_5vl_lora_sft_CoT_merged/
