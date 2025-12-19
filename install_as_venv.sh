#! /bin/bash

# ensure that cuda 12.8 is being used; ln -sfn /usr/local/cuda-12.8 /etc/alternatives/cuda

pushd /scratch/indrisch/
module load StdEnv/2023  gcc/12.3  openmpi/4.1.5
module load python/3.12 cuda/12.6 opencv/4.12.0
module load arrow
virtualenv --no-download venv_llamafactory_cu126
source venv_llamafactory_cu126/bin/activate
popd
pip install --upgrade pip setuptools wheel
pip install packaging psutil pandas pillow decorator scipy matplotlib platformdirs pyarrow sympy wandb ray -e ".[torch,metrics,deepspeed,liger-kernel]" --no-build-isolation