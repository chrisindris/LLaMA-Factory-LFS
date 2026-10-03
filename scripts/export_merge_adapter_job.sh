#!/bin/bash
#SBATCH --account=def-wangcs
#SBATCH --job-name=llamafactory_export_merge
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=err/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=128GB
#SBATCH --time=00:10:00
#SBATCH --array=0-0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=christopher.indris@torontomu.ca

# to run: sbatch export_merge_adapter_job.sh [cluster]
# if we use GPUs: SBATCH --gpus-per-node=h100_2g.20gb:1
#
# Optional env overrides:
#   UPLOAD_TO_HF=0|1   (default 1)  skip Hub push after export
#   OVERLAY=path       apptainer overlay image (default: $PROJECT_DIR/apptainer/overlay.img)
#   EXPORT_DIR, WORKDIR, HF_TOKEN, etc.

set -euo pipefail

module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
module load python/3.12 cuda/12.6 opencv/4.12.0
module load arrow
module load apptainer

# User inputs (override with environment variables or edit below)
# Define arrays for array job
BASE_MODEL_PATHS=(
	"Video-R1/Video-R1-7B"
	"Qwen/Qwen2.5-VL-7B-Instruct"
)

BASE_MODEL_PATH_TEMPLATES=(
	"videor1"
	"qwen2_vl"
)

ADAPTER_PATHS=(
	cvis-tmu/qwen2_5vl-7b-lora-sft-CoT_traineval_3epochs
	# "cvis-tmu/videor1-lora-sft-SQA3Devery24_800steps"   # 2 epochs of VideoR1 on SQA3D
	# "cvis-tmu/qwen2_5vl-7b-lora-sft-SQA3Devery24_R12C12F12X62_865steps" # 2 epochs of Qwen2.5VL on X62
	# "cvis-tmu/qwen2_5vl-7b-lora-sft-SQA3Devery24_ep2"   # 2 epochs of Qwen2.5VL on SQA3D
	# "cvis-tmu/qwen2_5vl-7b-lora-sft-SQA3Devery24_X1_465steps" # 1 epoch of Qwen2.5VL on X1 version of SQA3D (corrections made)
	# "cvis-tmu/qwen2_5vl-7b-lora-sft-SQA3Devery24_C1_465steps" # 1 epoch of Qwen2.5VL on C1 version of SQA3D (corrections made)
	# "cvis-tmu/videor1-lora-sft-Scene30k_traineval_852steps" # 2 epochs of VideoR1 on Scene30k dataset
	# "cvis-tmu/qwen2_5vl-7b-lora-sft-Scene30k_traineval_2130steps" # 5 epochs of Qwen2.5VL on Scene30k dataset
	# cvis-tmu/videor1sft-lora-sft-Scene30k_traineval_426steps
	# cvis-tmu/videor1sft-lora-sft-Scene30k_traineval_852steps
	# cvis-tmu/videor1-lora-sft-Scene30k_traineval_426steps
	# cvis-tmu/videor1-lora-sft-Scene30k_traineval_852steps
	# cvis-tmu/videor1sft-lora-sft-Scene30k_traineval_5epochs
	# cvis-tmu/videor1-lora-sft-Scene30k_traineval_5epochs
)

# Get the index from SLURM_ARRAY_TASK_ID, default to 0
IDX=${SLURM_ARRAY_TASK_ID:-0}
ADAPTER_PATH=${ADAPTER_PATHS[$IDX]}
# Keep the original Hub id (e.g. cvis-tmu/foo) for upload REPO_ID even if we
# later rewrite ADAPTER_PATH to a local snapshot for offline export.
ADAPTER_HUB_ID="${ADAPTER_PATH}"
ADAPTER_NAME=$(basename "$ADAPTER_PATH")
# if ADAPTER_NAME contains 'videor1', use BASE_MODEL_PATHS[0], else use BASE_MODEL_PATHS[1]
if [[ "$ADAPTER_NAME" == *"videor1"* ]]; then
	BASE_MODEL_PATH=${BASE_MODEL_PATHS[0]}
	TEMPLATE=${BASE_MODEL_PATH_TEMPLATES[0]}
else
	BASE_MODEL_PATH=${BASE_MODEL_PATHS[1]}
	TEMPLATE=${BASE_MODEL_PATH_TEMPLATES[1]}
fi
USER_ACCOUNT=$(whoami)

# --- for reading cluster-specific settings ---
# Avoid double slash: ${PWD%%LLaMA-Factory*} leaves a trailing / when PWD ends at the project root.
if [[ "$PWD" == *LLaMA-Factory-LFS* ]]; then
	PROJECT_DIR="${PWD%%LLaMA-Factory-LFS*}LLaMA-Factory-LFS"
elif [[ "$PWD" == *LLaMA-Factory* ]]; then
	PROJECT_DIR="${PWD%%LLaMA-Factory*}LLaMA-Factory"
else
	echo "Error: Could not find 'LLaMA-Factory' or 'LLaMA-Factory-LFS' in the current path."
	exit 1
fi
# Normalize // that can appear depending on PWD shape
PROJECT_DIR="${PROJECT_DIR//\/\//\/}"
SYSCONFIG_DIR_PATH="$PROJECT_DIR/scripts"
export PYTHONPATH="${PYTHONPATH:-}:$SYSCONFIG_DIR_PATH"

# --- setting environment ---
# Batch jobs often have empty PS1; short node names like "m1" also miss naive substring checks.
# Prefer FQDN / domain (e.g. m1.nibi.sharcnet) and SLURM_CLUSTER_NAME when present.
HOST_SHORT="${HOSTNAME:-$(hostname 2>/dev/null || true)}"
HOST_FQDN="$(hostname -f 2>/dev/null || true)"
HOST_HINT="${HOST_SHORT} ${HOST_FQDN} ${SLURM_CLUSTER_NAME:-} ${SLURM_SUBMIT_HOST:-} ${PS1:-}"

CLUSTER="${1:-}"
if [[ -z "${CLUSTER}" ]]; then
	# Detect cluster based on hostname / FQDN / SLURM / prompt
	if [[ "${HOST_HINT}" == *"rorqual"* ]] || [[ "${HOST_HINT}" == *"rg"* ]]; then
		CLUSTER="RORQUAL"
		RUNNING_MODE="APPTAINER"
		OFFLINE=1
	elif [[ "${HOST_HINT}" == *"trillium"* ]] || [[ "${HOST_HINT}" == *"trig"* ]]; then
		CLUSTER="TRILLIUM"
		RUNNING_MODE="APPTAINER"
		OFFLINE=1
	elif [[ "${HOST_HINT}" == *"killarney"* ]] || [[ "${HOST_HINT}" == *"klogin"* ]] || [[ "${HOST_HINT}" == *"kn"* ]]; then
		CLUSTER="KILLARNEY"
		RUNNING_MODE="VENV"
		OFFLINE=1
	elif [[ "${HOST_HINT}" == *"nibi"* ]]; then
		# NIBI GPU nodes (g##) and CPU/login nodes (m##, l##.nibi.sharcnet)
		CLUSTER="NIBI"
		RUNNING_MODE="APPTAINER"
		# Prefer local HF cache; compute nodes may lack egress.
		OFFLINE=1
	elif [[ "${HOST_SHORT}" == g* ]] || [[ "${HOST_SHORT}" == m* ]]; then
		# Bare short names on NIBI when FQDN is unavailable inside the job
		CLUSTER="NIBI"
		RUNNING_MODE="APPTAINER"
		OFFLINE=1
	else
		echo "Warning: Could not detect cluster from HOST_HINT='${HOST_HINT}'. Defaulting to NIBI."
		CLUSTER="NIBI"
		RUNNING_MODE="APPTAINER"
		OFFLINE=1
	fi
fi

OFFLINE=${OFFLINE:-0} # by default, run in online mode
# Keep both legacy and current hub offline flags in sync (PEFT uses HF_HUB_OFFLINE).
export TRANSFORMERS_OFFLINE=$OFFLINE
export HUGGINGFACE_HUB_OFFLINE=$OFFLINE
# export HF_HUB_OFFLINE=$OFFLINE
unset HF_HUB_OFFLINE

echo "CLUSTER: $CLUSTER  HOST_SHORT: $HOST_SHORT  HOST_FQDN: $HOST_FQDN  OFFLINE: $OFFLINE"

# --- read per-cluster settings from sysconfig.json ---
export HF_HOME="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'HF_HOME'))")" && echo "HF_HOME: $HF_HOME"
export HF_HUB_CACHE="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'HF_HUB_CACHE'))")" && echo "HF_HUB_CACHE: $HF_HUB_CACHE"
export TRITON_CACHE_DIR="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'TRITON_CACHE_DIR'))")" && echo "TRITON_CACHE_DIR: $TRITON_CACHE_DIR"
export FLASHINFER_WORKSPACE_BASE="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'FLASHINFER_WORKSPACE_BASE'))")" && echo "FLASHINFER_WORKSPACE_BASE: $FLASHINFER_WORKSPACE_BASE"
export BEST_GPU="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'BEST_GPU'))")" && echo "BEST_GPU: $BEST_GPU"
export TORCH_EXTENSIONS_DIR="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'TORCH_EXTENSIONS_DIR'))")" && echo "TORCH_EXTENSIONS_DIR: $TORCH_EXTENSIONS_DIR"
export SIF_FILE="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'SIF_FILE'))")" && echo "SIF_FILE: $SIF_FILE"
export MEDIA_DIR="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'media_dir'))")" && echo "MEDIA_DIR: $MEDIA_DIR"
export VENV_LLAMAFACTORY="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'VENV_LLAMAFACTORY'))")" && echo "VENV_LLAMAFACTORY: $VENV_LLAMAFACTORY"
export VENV_DATASET_UPLOAD="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'VENV_DATASET_UPLOAD'))")" && echo "VENV_DATASET_UPLOAD: $VENV_DATASET_UPLOAD"

# Resolve Hub repo ids to a local snapshot when present so PEFT does not require network
# even if cache layout / offline flags are wrong. Models live under HF_HUB_CACHE/models--org--name.
resolve_hf_snapshot() {
	local repo_id="$1"
	local cache_root="${HF_HUB_CACHE:-${HF_HOME:-}}"
	# Already a filesystem path
	if [[ -d "${repo_id}" ]]; then
		echo "${repo_id}"
		return 0
	fi
	# Hub id: org/name
	if [[ "${repo_id}" != */* ]]; then
		echo "${repo_id}"
		return 0
	fi
	local org name dir snap
	org="${repo_id%%/*}"
	name="${repo_id#*/}"
	dir="${cache_root}/models--${org}--${name}"
	if [[ -f "${dir}/refs/main" ]]; then
		snap="$(tr -d '[:space:]' <"${dir}/refs/main")"
		if [[ -n "${snap}" && -f "${dir}/snapshots/${snap}/adapter_config.json" ]]; then
			echo "${dir}/snapshots/${snap}"
			return 0
		fi
		if [[ -n "${snap}" && -d "${dir}/snapshots/${snap}" ]]; then
			echo "${dir}/snapshots/${snap}"
			return 0
		fi
	fi
	echo "${repo_id}"
}

ADAPTER_PATH_RESOLVED="$(resolve_hf_snapshot "${ADAPTER_PATH}")"
BASE_MODEL_PATH_RESOLVED="$(resolve_hf_snapshot "${BASE_MODEL_PATH}")"
if [[ "${ADAPTER_PATH_RESOLVED}" != "${ADAPTER_PATH}" ]]; then
	echo "Resolved adapter to local snapshot: ${ADAPTER_PATH_RESOLVED}"
	ADAPTER_PATH="${ADAPTER_PATH_RESOLVED}"
fi
if [[ "${BASE_MODEL_PATH_RESOLVED}" != "${BASE_MODEL_PATH}" ]]; then
	echo "Resolved base model to local snapshot: ${BASE_MODEL_PATH_RESOLVED}"
	BASE_MODEL_PATH="${BASE_MODEL_PATH_RESOLVED}"
fi

# ----------

# Extract adapter name for unique output directory and repo name
WORKDIR=${WORKDIR:-$PROJECT_DIR}
EXPORT_DIR=${EXPORT_DIR:-"${WORKDIR}/models/merged-model/${ADAPTER_NAME}"}
TEMPLATE=${TEMPLATE:-"qwen2_vl"}
INFER_DTYPE=${INFER_DTYPE:-"bfloat16"}
EXPORT_SIZE=${EXPORT_SIZE:-2}         # shard size in GB
EXPORT_DEVICE=${EXPORT_DEVICE:-"cpu"} # cpu is enough for LoRA merge; use auto/cuda if a GPU is allocated
CONTAINER=${CONTAINER:-$SIF_FILE}
OVERLAY=${OVERLAY:-"${PROJECT_DIR}/apptainer/overlay.img"}
UPLOAD_TO_HF=${UPLOAD_TO_HF:-1}

HF_TOKEN=${HF_TOKEN:-$(cat /home/indrisch/TOKENS/cvis-tmu-organization-token.txt)} # optional Hugging Face token for private checkpoints

DISABLE_VERSION_CHECK=${DISABLE_VERSION_CHECK:-"1"} # disable version check for faster startup

echo "PROJECT_DIR: $PROJECT_DIR"
echo "ADAPTER_PATH: $ADAPTER_PATH"
echo "BASE_MODEL_PATH: $BASE_MODEL_PATH"
echo "EXPORT_DIR: $EXPORT_DIR"
echo "OVERLAY: $OVERLAY"
echo "CONTAINER: $CONTAINER"
echo "UPLOAD_TO_HF: $UPLOAD_TO_HF"

# ========= run the export and merge process inside the container =========
#
# Overlay is optional for pure export (writable-tmpfs is enough), but using the
# same overlay as training keeps the Python env consistent (h5py/wandb/pip).
# Build with: bash scripts/build_apptainer_overlay.sh

APPTAINER_EXTRA_ARGS=()
if [[ -f "${OVERLAY}" ]]; then
	# Overlay is built without --fakeroot (Tamia has no subuid / blocked uid maps).
	APPTAINER_EXTRA_ARGS+=(--overlay "${OVERLAY}")
	echo "Using overlay: ${OVERLAY}"
else
	# Fall back when overlay was not built yet
	APPTAINER_EXTRA_ARGS+=(--writable-tmpfs)
	echo "WARNING: overlay not found at ${OVERLAY}; using --writable-tmpfs"
fi

unset HF_HUB_OFFLINE

apptainer run --nv "${APPTAINER_EXTRA_ARGS[@]}" \
	-C \
	-B /scratch/indrisch/ \
	-B "${WORKDIR}" \
	-B /dev/shm:/dev/shm \
	-B /etc/ssl/certs:/etc/ssl/certs:ro \
	-B /etc/pki:/etc/pki:ro \
	-W "${SLURM_TMPDIR:-/tmp}" \
	--env HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}" \
	--env HF_HOME="${HF_HOME}" \
	--env HF_HUB_CACHE="${HF_HUB_CACHE}" \
	--env HF_TOKEN="${HF_TOKEN}" \
	--env TRANSFORMERS_CACHE="${HF_HOME}" \
	--env TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE}" \
	--env HUGGINGFACE_HUB_OFFLINE="${HUGGINGFACE_HUB_OFFLINE}" \
	--env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
	--env DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK}" \
	--env PYTHONPATH="${PROJECT_DIR}/src" \
	--pwd "${WORKDIR}" \
	"${CONTAINER}" bash -lc "set -euo pipefail; unset HF_HUB_OFFLINE; \
        llamafactory-cli export \
        --model_name_or_path \"${BASE_MODEL_PATH}\" \
        --adapter_name_or_path \"${ADAPTER_PATH}\" \
        --export_dir \"${EXPORT_DIR}\" \
        --template \"${TEMPLATE}\" \
        --finetuning_type lora \
        --export_size ${EXPORT_SIZE} \
        --export_device ${EXPORT_DEVICE} \
        --infer_dtype ${INFER_DTYPE} \
        --export_legacy_format false"

echo "Model exported successfully to ${EXPORT_DIR}"

# ========= upload merged model to HF Hub (host venv, NOT broken user-site pip) =========
# Job 19170537 failed here: host `python -m pip` resolves to
# ~/.local/lib/python3.12/site-packages/pip which is broken
# (ModuleNotFoundError: pip._vendor.rich). Use VENV_DATASET_UPLOAD instead,
# matching scripts/upload_merged_checkpoint.sh.

if [[ "${UPLOAD_TO_HF}" == "1" ]]; then
	if [[ ! -d "${EXPORT_DIR}" ]] || [[ ! -f "${EXPORT_DIR}/config.json" ]]; then
		echo "ERROR: export dir incomplete: ${EXPORT_DIR}"
		exit 1
	fi

	if [[ ! -x "${VENV_DATASET_UPLOAD}/bin/hf" ]] && [[ ! -x "${VENV_DATASET_UPLOAD}/bin/python" ]]; then
		echo "ERROR: VENV_DATASET_UPLOAD missing or incomplete: ${VENV_DATASET_UPLOAD}"
		echo "Create it or set UPLOAD_TO_HF=0 to skip Hub upload."
		exit 1
	fi

	# Export/merge runs offline (local cache), but Hub upload needs network.
	# Job 19409391 failed here with OfflineModeIsEnabled on create_repo.
	unset HF_HUB_OFFLINE HUGGINGFACE_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_HUB_DISABLE_TELEMETRY || true
	export HF_HUB_OFFLINE=0
	export HUGGINGFACE_HUB_OFFLINE=0
	export TRANSFORMERS_OFFLINE=0

	# shellcheck disable=SC1091
	source "${VENV_DATASET_UPLOAD}/bin/activate"
	# Do not run `python -m pip install` against the system/user site; venv already has hub.
	python -c "import huggingface_hub; print('huggingface_hub', huggingface_hub.__version__)"

	# Always derive Hub repo from the original adapter id, not a local snapshot path.
	# e.g. cvis-tmu/foo -> cvis-tmu/foo_merged
	if [[ "${ADAPTER_HUB_ID}" == */* && "${ADAPTER_HUB_ID}" != /* ]]; then
		REPO_ID="${ADAPTER_HUB_ID}_merged"
	else
		REPO_ID="cvis-tmu/${ADAPTER_NAME}_merged"
	fi
	echo "Uploading ${EXPORT_DIR} -> ${REPO_ID} (online; HF_HUB_OFFLINE=${HF_HUB_OFFLINE})"
	hf upload "${REPO_ID}" "${EXPORT_DIR}" \
		--repo-type model \
		--token "${HF_TOKEN}" \
		--commit-message "Upload merged model from LLaMA-Factory export ${ADAPTER_NAME}"

	deactivate
	echo "Model uploaded successfully to Hugging Face Hub: ${REPO_ID}"
else
	echo "Skipping HF upload (UPLOAD_TO_HF=${UPLOAD_TO_HF}). Local model at: ${EXPORT_DIR}"
fi
