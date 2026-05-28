#! /bin/bash

# ensure that cuda 12.8 is being used; ln -sfn /usr/local/cuda-12.8 /etc/alternatives/cuda

usage() {
    echo "Usage: $0 (RORQUAL|FIR|NIBI|NARVAL|TRILLIUM|KILLARNEY)"
    exit 1
}

if [ $# -ne 1 ]; then
    usage
fi

if [ $1 != "RORQUAL" ] && [ $1 != "FIR" ] && [ $1 != "NIBI" ] && [ $1 != "NARVAL" ] && [ $1 != "TRILLIUM" ] && [ $1 != "KILLARNEY" ]; then
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
if [ -z "$VENV_LLAMAFACTORY" ]; then
    export VENV_LLAMAFACTORY="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${1}', 'VENV_LLAMAFACTORY'))")"
fi
echo "VENV_LLAMAFACTORY: $VENV_LLAMAFACTORY"
export TORCH_CUDA_ARCH_LIST="9.0" # for clusters with a100 GPUs
export TORCH_EXTENSIONS_DIR="${SLURM_TMPDIR}/.cache/torch_extensions" # needed for cpu_adam
export PYTORCH_KERNEL_CACHE_PATH="${SLURM_TMPDIR}/.cache/torch/kernels"
export FORCE_TORCHRUN=1 
export WANDB_MODE=offline 
export WANDB_DIR="${PROJECT_DIR}/wandb/" 
export DISABLE_VERSION_CHECK=1 # since the automatic detector doesn't automatically see that transformers==4.57.1+computecanada is the same as transformers==4.57.1

# --- build CPU Adam if we have set DS_BUILD_CPU_ADAM, BUILD_UTILS and DS_BUILD_OPS to 1 ---

# needed when we get AttributeError: 'DeepSpeedCPUAdam' object has no attribute 'ds_opt_adam'
export DS_BUILD_CPU_ADAM=${DS_BUILD_CPU_ADAM:-0}
export BUILD_UTILS=${BUILD_UTILS:-0}
export DS_BUILD_OPS=${DS_BUILD_OPS:-0}

# Auto-detect AVX-512 support and set compiler flags for building CPU extensions.
# If `DS_FORCE_BUILD_CPU_ADAM=1` is set in the environment, force build regardless
# of detection. If AVX-512 is not present, disable building CPU Adam to avoid
# JIT compile failures on machines without AVX-512.
if [ -z "${DS_FORCE_BUILD_CPU_ADAM}" ]; then
    if command -v lscpu >/dev/null 2>&1; then
        if lscpu | grep -qi avx512; then
            echo "AVX-512 support detected — enabling AVX-512 compile flags for native extensions"
            gcc --version
            g++ --version
            export CFLAGS="-O3 -march=native -mavx512f -mavx512dq -mavx512bw"
            export CXXFLAGS="-O3 -std=c++17 -march=native -mavx512f -mavx512dq -mavx512bw"
        else
            echo "No AVX-512 support detected — disabling DS_BUILD_CPU_ADAM to avoid build errors"
            export DS_BUILD_CPU_ADAM=0
        fi
    else
        echo "lscpu not found — leaving DS_BUILD_CPU_ADAM=${DS_BUILD_CPU_ADAM} (set DS_FORCE_BUILD_CPU_ADAM=1 to override)"
    fi
else
    echo "DS_FORCE_BUILD_CPU_ADAM is set — forcing CPUAdam build (ensure your CPU and toolchain support AVX-512)"
fi


# --- decide on the cuda version to use ---

if [[ "$VENV_LLAMAFACTORY" == *cu12* ]]; then
    echo "Setting up environment for CUDA 12.x"
    module load StdEnv/2023  gcc/12.3  openmpi/4.1.5
    module load python/3.12 cuda/12.6 opencv/4.12.0
    module load arrow
elif [[ "$VENV_LLAMAFACTORY" == *cu13* ]]; then
    echo "Setting up environment for CUDA 13.x"
    module load StdEnv gcc openmpi python/3.12 cuda/13.2 opencv arrow
else
    echo "Error: The specified VENV_LLAMAFACTORY at $VENV_LLAMAFACTORY does not appear to be configured for a supported CUDA version. Please set VENV_LLAMAFACTORY to a virtual environment that has been set up with supported CUDA support."
    exit 1
fi

pushd "$PROJECT_DIR" >/dev/null
# module load StdEnv/2023  gcc/12.3  openmpi/4.1.5
# module load python/3.12 cuda/12.6 opencv/4.12.0
# module load arrow
# module load StdEnv gcc openmpi python/3.12 cuda/13.2 opencv arrow
echo "CREATING VENV at ${VENV_LLAMAFACTORY}..."
virtualenv --no-download ${VENV_LLAMAFACTORY}
echo "SOURCING VENV at ${VENV_LLAMAFACTORY}..."
source ${VENV_LLAMAFACTORY}/bin/activate
echo "UPGRADING PIP, SETUPTOOLS and WHEEL in the VENV..."
pip install --no-index --upgrade pip setuptools wheel


# --- if we want to use Qwen3.5, we need to use "transformers>=5.2.0"; otherwise, "transformers==4.57.1" is fine ---
if [[ "$VENV_LLAMAFACTORY" == *qwen35* ]]; then
    echo "Installing transformers>=5.2.0 for Qwen3.5 compatibility"
    pip install packaging psutil pandas pillow decorator scipy matplotlib platformdirs pyarrow sympy wandb ray h5py flash_attn "transformers>=5.2.0" flash_linear_attention causal_conv1d -e ".[torch,metrics,deepspeed,liger-kernel]"
else
    echo "Installing transformers==4.57.1 for compatibility with models like Qwen2.5 and LLaVa-3D"
    pip install packaging psutil pandas pillow decorator scipy matplotlib platformdirs pyarrow sympy wandb ray h5py flash_attn "transformers==4.57.1" flash_linear_attention causal_conv1d -e ".[torch,metrics,deepspeed,liger-kernel]"
fi

# DeepSpeed's CPUAdam builder defaults to -march=x86-64-v3, which is too weak
# for the AVX-512 intrinsics used by cpu_adam.cpp on Killarney. Rewrite the
# installed builder so AVX-512-capable nodes compile CPUAdam with native CPU
# flags and can produce cpu_adam.so.
python3 - <<'PY'
from pathlib import Path
import site

replacement = "        if 'avx512' in cpu_info['flags'] or 'avx512f' in cpu_info['flags']:\n            return '-march=native'\n        return '-march=x86-64-v3'"

for site_dir in site.getsitepackages():
    builder_path = Path(site_dir) / 'deepspeed' / 'ops' / 'op_builder' / 'builder.py'
    if not builder_path.is_file():
        continue
    builder_text = builder_path.read_text()
    old_line = "        return '-march=x86-64-v3'"
    if old_line in builder_text and replacement not in builder_text:
        builder_path.write_text(builder_text.replace(old_line, replacement, 1))
        print(f'Patched {builder_path}')
    break
PY
popd >/dev/null
