#! /bin/bash

# ensure that cuda 12.8 is being used; ln -sfn /usr/local/cuda-12.8 /etc/alternatives/cuda

# --- for reading cluster-specific settings ---
. $(find $(REGEX="(.*LLaMA-Factory[^/]*).*" && [[ $PWD =~ $REGEX ]] && echo "${BASH_REMATCH[1]}") -name "env.sh")

# usage() {
#     echo "Usage: $0 (RORQUAL|FIR|NIBI|NARVAL|TRILLIUM|KILLARNEY)"
#     exit 1
# }
#
# if [ $# -ne 1 ]; then
#     usage
# fi
#
# if [ $1 != "RORQUAL" ] && [ $1 != "FIR" ] && [ $1 != "NIBI" ] && [ $1 != "NARVAL" ] && [ $1 != "TRILLIUM" ] && [ $1 != "KILLARNEY" ]; then
#     usage
# fi
#
# # get the parent of the project directory
# if [[ "$PWD" == *LLaMA-Factory-LFS* ]]; then
#     PROJECT_PARENT="${PWD%%LLaMA-Factory-LFS*}"
#     PROJECT_DIR="$PROJECT_PARENT/LLaMA-Factory-LFS"
#     sysconfigtool_DIR_PATH="$PROJECT_DIR/scripts"
# elif [[ "$PWD" == *LLaMA-Factory* ]]; then
#     PROJECT_PARENT="${PWD%%LLaMA-Factory*}"
#     PROJECT_DIR="$PROJECT_PARENT/LLaMA-Factory"
#     sysconfigtool_DIR_PATH="$PROJECT_DIR/scripts"
# else
#     echo "Error: Could not find 'LLaMA-Factory-LFS' or 'LLaMA-Factory' in the current path."
#     exit 1
# fi
#
# export PYTHONPATH="$PYTHONPATH:$sysconfigtool_DIR_PATH"

lmod_preflight() {
	local lmod_init="/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/init/bash"
	local lmod_exec="/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/libexec/lmod"
	local resolved_init
	local resolved_exec

	if [[ ! -e "$lmod_init" || ! -e "$lmod_exec" ]]; then
		echo "ERROR: Lmod bootstrap path is unavailable on this node."
		echo "  lmod init: $lmod_init"
		echo "  lmod exec: $lmod_exec"
		ls -l "$lmod_init" "$lmod_exec" 2>/dev/null || true
		exit 1
	fi

	resolved_init=$(readlink -f "$lmod_init" 2>/dev/null || true)
	resolved_exec=$(readlink -f "$lmod_exec" 2>/dev/null || true)
	if [[ -z "$resolved_init" || -z "$resolved_exec" ]]; then
		echo "ERROR: Lmod symlink resolution failed before module initialization."
		echo "  lmod init: $lmod_init"
		echo "  lmod exec: $lmod_exec"
		ls -l "$lmod_init" "$lmod_exec" 2>/dev/null || true
		exit 1
	fi
}

# DeepSpeed 0.18 cpu_arch() always returns -march=x86-64-v3 on x86, while
# simd_width() sets -D__AVX512__ from CPU flags. AVX-512 intrinsics in
# cpu_adam then fail to inline (target specific option mismatch). CFLAGS /
# CXXFLAGS do not reach this JIT: DeepSpeed passes cpu_arch() as extra_cflags.
# Patch after pip install; the Alliance wheel does not prebuild CPUAdam.
patch_deepspeed_cpu_arch() {
	python - <<'PY'
from pathlib import Path
import site

old_line = "        return '-march=x86-64-v3'"
replacement = (
    "        flags = cpu_info.get('flags', '')\n"
    "        flags_str = ' '.join(flags) if isinstance(flags, (list, tuple, set)) else str(flags)\n"
    "        if 'avx512' in flags_str:\n"
    "            return '-march=x86-64-v4'\n"
    "        return '-march=x86-64-v3'"
)
marker = "return '-march=x86-64-v4'"

patched = False
for site_dir in site.getsitepackages():
    builder_path = Path(site_dir) / "deepspeed" / "ops" / "op_builder" / "builder.py"
    if not builder_path.is_file():
        continue
    builder_text = builder_path.read_text()
    if marker in builder_text:
        print(f"Already patched {builder_path}")
        patched = True
        break
    if old_line not in builder_text:
        raise SystemExit(f"ERROR: expected cpu_arch() return not found in {builder_path}")
    builder_path.write_text(builder_text.replace(old_line, replacement, 1))
    print(f"Patched {builder_path}")
    patched = True
    break
if not patched:
    raise SystemExit("ERROR: DeepSpeed builder.py not found in site-packages")
PY
}

prebuild_cpu_adam() {
	if [[ "${DS_BUILD_CPU_ADAM}" != "1" ]]; then
		echo "Skipping CPUAdam prebuild (DS_BUILD_CPU_ADAM=${DS_BUILD_CPU_ADAM})"
		return 0
	fi
	if [[ -z "${VENV_LLAMAFACTORY}" ]]; then
		echo "ERROR: VENV_LLAMAFACTORY is not set; cannot prebuild CPUAdam"
		exit 1
	fi

	export TORCH_EXTENSIONS_DIR="${VENV_LLAMAFACTORY%/}/torch_extensions"
	mkdir -p "${TORCH_EXTENSIONS_DIR}"

	local root
	for root in \
		"${TORCH_EXTENSIONS_DIR}" \
		"${HOME}/.cache/torch_extensions" \
		"/tmp/torch_extensions" \
		"${SLURM_TMPDIR}/.cache/torch_extensions"; do
		if [[ -d "$root" ]]; then
			find "$root" -type d -name "cpu_adam" -prune -exec rm -rf {} +
		fi
	done

	echo "Prebuilding DeepSpeed CPUAdam into ${TORCH_EXTENSIONS_DIR}"
	python -c "import deepspeed; deepspeed.ops.op_builder.CPUAdamBuilder().load(); print('cpu_adam OK')"
}

# if SLURM_TMPDIR is not set, set it to /tmp
if [ -z "$SLURM_TMPDIR" ]; then
	SLURM_TMPDIR="/tmp"
fi

export HF_HUB_OFFLINE=1
export MPLCONFIGDIR="${SLURM_TMPDIR}/.config/matplotlib"
# export HF_HOME="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${1}', 'HF_HOME'))")"
# export HF_HUB_CACHE="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${1}', 'HF_HUB_CACHE'))")"
# export TRITON_CACHE_DIR="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${1}', 'TRITON_CACHE_DIR'))")"
# export FLASHINFER_WORKSPACE_BASE="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${1}', 'FLASHINFER_WORKSPACE_BASE'))")"
# if [ -z "$VENV_LLAMAFACTORY" ]; then
#     export VENV_LLAMAFACTORY="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${1}', 'VENV_LLAMAFACTORY'))")"
# fi
echo "VENV_LLAMAFACTORY: $VENV_LLAMAFACTORY"
# export TORCH_CUDA_ARCH_LIST="9.0" # for clusters with a100 GPUs
# export TORCH_EXTENSIONS_DIR="${SLURM_TMPDIR}/.cache/torch_extensions" # needed for cpu_adam
export PYTORCH_KERNEL_CACHE_PATH="${SLURM_TMPDIR}/.cache/torch/kernels"
export FORCE_TORCHRUN=1
export WANDB_MODE=offline
export WANDB_DIR="${PROJECT_DIR}/wandb/"
export DISABLE_VERSION_CHECK=1 # since the automatic detector doesn't automatically see that transformers==4.57.1+computecanada is the same as transformers==4.57.1

# --- build CPU Adam if we have set DS_BUILD_CPU_ADAM, BUILD_UTILS and DS_BUILD_OPS to 1 ---

# needed when we get AttributeError: 'DeepSpeedCPUAdam' object has no attribute 'ds_opt_adam'
export DS_BUILD_CPU_ADAM=${DS_BUILD_CPU_ADAM:-1}
export BUILD_UTILS=${BUILD_UTILS:-1}
export DS_BUILD_OPS=${DS_BUILD_OPS:-1}

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
if [[ "$VENV_LLAMAFACTORY" == *py313* ]]; then
	module load StdEnv gcc openmpi python/3.13 cuda/12.6 opencv arrow apptainer hwloc/2.9.1
	virtualenv --no-download "$VENV_LLAMAFACTORY"
	source "${VENV_LLAMAFACTORY}/bin/activate"
	python -m pip install --no-cache-dir --upgrade pip packaging wheel setuptools
	mkdir -p wheels
	pushd wheels
	wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.7cxx11abiFALSE-cp313-cp313-linux_x86_64.whl
	wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2/causal_conv1d-1.6.1+cu12torch2.6cxx11abiFALSE-cp313-cp313-linux_x86_64.whl
	popd
	python -m pip install -e . -r requirements/metrics.txt -r requirements/deepspeed.txt -r requirements/dev.txt -r requirements/logging_analysis.txt h5py wandb ray sentry-sdk liger-kernel flash_linear_attention wheels/causal_conv1d-1.6.1+cu12torch2.6cxx11abiFALSE-cp313-cp313-linux_x86_64.whl wheels/flash_attn-2.8.3.post1+cu12torch2.7cxx11abiFALSE-cp313-cp313-linux_x86_64.whl
	patch_deepspeed_cpu_arch
	prebuild_cpu_adam
	exit 0
elif [[ "$VENV_LLAMAFACTORY" == *cu12* ]]; then
	echo "Setting up environment for CUDA 12.x"
	lmod_preflight
	module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
	module load python/3.12 cuda/12.6 opencv/4.12.0
	module load arrow
elif [[ "$VENV_LLAMAFACTORY" == *cu13* ]]; then
	echo "Setting up environment for CUDA 13.x"
	lmod_preflight
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
if ! command -v virtualenv >/dev/null 2>&1; then
	echo "ERROR: virtualenv is not available after module initialization."
	exit 1
fi
virtualenv --no-download ${VENV_LLAMAFACTORY}
if [[ ! -f "${VENV_LLAMAFACTORY}/bin/activate" ]]; then
	echo "ERROR: virtualenv did not create ${VENV_LLAMAFACTORY}/bin/activate"
	exit 1
fi
source ${VENV_LLAMAFACTORY}/bin/activate
python3 -m pip install --upgrade pip setuptools wheel

# --- if we want to use Qwen3.5, we need to use "transformers>=5.2.0"; otherwise, "transformers==4.57.1" is fine ---
if [[ "$VENV_LLAMAFACTORY" == *qwen35* ]]; then
	echo "Installing transformers>=5.2.0 for Qwen3.5 compatibility"
	python3 -m pip install packaging psutil pandas pillow decorator scipy matplotlib platformdirs pyarrow sympy wandb ray h5py "transformers>=5.2.0" flash_linear_attention causal_conv1d -e ".[torch,metrics,deepspeed,liger-kernel]"
else
	echo "Installing transformers==4.57.1 for compatibility with models like Qwen2.5 and LLaVa-3D"
	python3 -m pip install packaging psutil pandas pillow decorator scipy matplotlib platformdirs pyarrow sympy wandb ray h5py "transformers==4.57.1" flash_linear_attention causal_conv1d -e ".[torch,metrics,deepspeed,liger-kernel]"
fi

patch_deepspeed_cpu_arch
prebuild_cpu_adam
popd >/dev/null
