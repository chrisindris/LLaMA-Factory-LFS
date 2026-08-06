#! /bin/bash

# ensure that cuda 12.8 is being used; ln -sfn /usr/local/cuda-12.8 /etc/alternatives/cuda

usage() {
    echo "Usage: $0 (RORQUAL|FIR|NIBI|NARVAL|TRILLIUM)"
    exit 1
}

if [ $# -ne 1 ]; then
    usage
fi

if [ $1 != "RORQUAL" ] && [ $1 != "FIR" ] && [ $1 != "NIBI" ] && [ $1 != "NARVAL" ] && [ $1 != "TRILLIUM" ]; then
    usage
fi

# get the parent of the project directory
if [[ "$PWD" == *LLaMA-Factory-LFS* ]]; then
    PROJECT_PARENT="${PWD%%LLaMA-Factory-LFS*}"
    PROJECT_DIR="$PROJECT_PARENT/LLaMA-Factory-LFS"
    sysconfigtool_DIR_PATH="$PROJECT_DIR/scripts"
elif [[ "$PWD" == *LLaMA-Factory* ]]; then
    PROJECT_PARENT="${PWD%%LLaMA-Factory*}"
    PROJECT_DIR="$PROJECT_PARENT/LLaMA-Factory"
    sysconfigtool_DIR_PATH="$PROJECT_DIR/scripts"
else
    echo "Error: Could not find 'LLaMA-Factory-LFS' or 'LLaMA-Factory' in the current path."
    exit 1
fi

export PYTHONPATH="$PYTHONPATH:$sysconfigtool_DIR_PATH"

# if SLURM_TMPDIR is not set, set it to /tmp
if [ -z "$SLURM_TMPDIR" ]; then
    SLURM_TMPDIR="/tmp"
fi

export HF_HUB_OFFLINE=1 
export MPLCONFIGDIR="${SLURM_TMPDIR}/.config/matplotlib"
export HF_HOME="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${1}', 'HF_HOME'))")" 
export HF_HUB_CACHE="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${1}', 'HF_HUB_CACHE'))")" 
export TRITON_CACHE_DIR="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${1}', 'TRITON_CACHE_DIR'))")" 
export FLASHINFER_WORKSPACE_BASE="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${1}', 'FLASHINFER_WORKSPACE_BASE'))")" 
export TORCH_CUDA_ARCH_LIST="9.0" # for clusters with a100 GPUs
export TORCH_EXTENSIONS_DIR="${SLURM_TMPDIR}/.cache/torch_extensions" # needed for cpu_adam
export PYTORCH_KERNEL_CACHE_PATH="${SLURM_TMPDIR}/.cache/torch/kernels"
export FORCE_TORCHRUN=1 
export WANDB_MODE=offline 
export WANDB_DIR="${PROJECT_DIR}/wandb/" 
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
