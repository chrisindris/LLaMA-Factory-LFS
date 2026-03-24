#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=out/%N-add_images_to_dataset-%j.out
#SBATCH --cpus-per-task=64
#SBATCH --time=0-00:01:00
#SBATCH --mem=4G

module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
module load python/3.12 cuda/12.6 opencv/4.12.0
module load arrow
source /scratch/indrisch/venv_llamafactory_cu126/bin/activate

ONLINE=false
UPLOAD_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --online)      ONLINE=true ;;
        --upload_only) UPLOAD_ONLY=true ;;
    esac
done

export HF_TOKEN=$(cat $HOME/TOKENS/cvis-tmu-organization-token.txt)

cd /scratch/indrisch/LLaMA-Factory
if [ "$UPLOAD_ONLY" = true ]; then
    python data/Scene30k/add_images_to_dataset.py --upload-only
elif [ "$ONLINE" = true ]; then
    python data/Scene30k/add_images_to_dataset.py --online
else
    python data/Scene30k/add_images_to_dataset.py
fi
