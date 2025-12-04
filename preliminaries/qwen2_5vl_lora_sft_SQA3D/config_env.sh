#!/bin/bash

# set the environment to use apptainer

module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
module load cuda/12.6 opencv/4.11 python/3.11
module load arrow
module load apptainer

TORCH_CUDA_ARCH_LIST="9.0" # for clusters with h100 GPUs

virtualenv --no-download $SLURM_TMPDIR/ENV
source $SLURM_TMPDIR/ENV/bin/activate

pip install --no-index --upgrade pip
pip install --no-index ray