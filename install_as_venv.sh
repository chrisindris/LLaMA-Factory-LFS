#! /bin/bash

# ensure that cuda 12.8 is being used; ln -sfn /usr/local/cuda-12.8 /etc/alternatives/cuda

export HF_HUB_OFFLINE=1 
export MPLCONFIGDIR="${SLURM_TMPDIR}/.config/matplotlib"
export HF_HOME="/scratch/indrisch/huggingface/hub"
export HF_HUB_CACHE="/scratch/indrisch/huggingface/hub"
export TRITON_CACHE_DIR="${SLURM_TMPDIR}/.triton_cache"
export FLASHINFER_WORKSPACE_BASE="/scratch/indrisch/"
export TORCH_CUDA_ARCH_LIST="9.0" # for clusters with a100 GPUs
export TORCH_EXTENSIONS_DIR="${SLURM_TMPDIR}/.cache/torch_extensions" # needed for cpu_adam
export PYTORCH_KERNEL_CACHE_PATH="${SLURM_TMPDIR}/.cache/torch/kernels"
export FORCE_TORCHRUN=1 
export WANDB_MODE=offline 
export WANDB_DIR="/scratch/indrisch/LLaMA-Factory/wandb/" 
export DISABLE_VERSION_CHECK=1 # since the automatic detector doesn't automatically see that transformers==4.57.1+computecanada is the same as transformers==4.57.1

pushd /scratch/indrisch/
module load StdEnv/2023  gcc/12.3  openmpi/4.1.5
module load python/3.12 cuda/12.6 opencv/4.12.0
module load arrow
virtualenv --no-download venv_llamafactory_cu126
source venv_llamafactory_cu126/bin/activate
popd
pip install --upgrade pip setuptools wheel
pip install packaging psutil pandas pillow decorator scipy matplotlib platformdirs pyarrow sympy wandb ray -e ".[torch,metrics,deepspeed,liger-kernel]"