#!/bin/bash

set -euo pipefail

usage() {
	cat <<'EOF'
Usage: ./upload_custom_videor1_base_versions.sh

Uploads locally edited Video-R1 base model snapshots from the Hugging Face cache
to the cvis-tmu organization:
	- cvis-tmu/Video-R1-7B
	- cvis-tmu/Qwen2.5-VL-7B-COT-SFT

Optional environment variables:
	CLUSTER                  Cluster override for sysconfig lookup.
	HF_HOME                  Hugging Face home directory.
	HF_HUB_CACHE             Hugging Face hub cache directory.
	VENV_DATASET_UPLOAD      Python venv with huggingface_hub installed.
	HF_TOKEN                 Hugging Face token (defaults to token file).
	SOURCE_VIDEOR1_DIR       Override source directory for Video-R1-7B.
	SOURCE_QWEN_COT_DIR       Override source directory for Qwen2.5-VL-7B-COT-SFT.
	COMMIT_MESSAGE_VIDEOR1    Commit message for Video-R1-7B upload.
	COMMIT_MESSAGE_QWEN_COT   Commit message for Qwen2.5-VL-7B-COT-SFT upload.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
	usage
	exit 0
fi

# --- for reading cluster-specific settings ---

if [[ "$PWD" == *LLaMA-Factory-LFS* ]]; then
		PROJECT_DIR="${PWD%%LLaMA-Factory-LFS*}/LLaMA-Factory-LFS"
elif [[ "$PWD" == *LLaMA-Factory* ]]; then
		PROJECT_DIR="${PWD%%LLaMA-Factory*}/LLaMA-Factory"
else
		echo "Error: Could not find 'LLaMA-Factory' or 'LLaMA-Factory-LFS' in the current path." >&2
		exit 1
fi
SYSCONFIG_DIR_PATH="$PROJECT_DIR/scripts"
export PYTHONPATH="$PYTHONPATH:$SYSCONFIG_DIR_PATH"

# --- setting environment ---
if [[ -z "${CLUSTER:-}" ]]; then
	if [[ "$HOSTNAME" == *"rorqual"* ]] || [[ "$HOSTNAME" == *"rg"* ]]; then
			export CLUSTER="RORQUAL"
	elif [[ "$HOSTNAME" == *"trig"* ]]; then
			export CLUSTER="TRILLIUM"
	elif [[ "$HOSTNAME" == *"klogin"* ]] || [[ "$HOSTNAME" == *"kn"* ]]; then
			export CLUSTER="KILLARNEY"
	elif [[ "$HOSTNAME" == *"nibi"* ]]; then
			export CLUSTER="NIBI"
	else
			echo "Warning: Could not detect cluster from PS1 or HOSTNAME. Defaulting to RORQUAL." >&2
			export CLUSTER="RORQUAL"
	fi
fi

echo "Detected cluster: $CLUSTER"

if [[ -z "${HF_HOME:-}" ]]; then
	HF_HOME="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'HF_HOME') or '')")"
fi
if [[ -z "${HF_HUB_CACHE:-}" ]]; then
	HF_HUB_CACHE="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'HF_HUB_CACHE') or '')")"
fi
if [[ -z "${VENV_DATASET_UPLOAD:-}" ]]; then
    export VENV_DATASET_UPLOAD="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'VENV_DATASET_UPLOAD'))")" && echo "VENV_DATASET_UPLOAD: $VENV_DATASET_UPLOAD"
fi

if [[ -z "${HF_HUB_CACHE:-}" ]]; then
	if [[ -n "${HF_HOME:-}" ]]; then
		HF_HUB_CACHE="$HF_HOME"
	else
		HF_HUB_CACHE="/scratch/indrisch/huggingface/hub"
	fi
fi
if [[ -z "${HF_HOME:-}" ]]; then
	HF_HOME="$HF_HUB_CACHE"
fi

if [[ -z "${VENV_DATASET_UPLOAD:-}" ]]; then
	echo "Error: VENV_DATASET_UPLOAD is not set and could not be read from sysconfig." >&2
	exit 1
fi

export HF_HOME
export HF_HUB_CACHE

module load StdEnv/2023  gcc/12.3  openmpi/4.1.5
module load python/3.12 cuda/12.6 opencv/4.12.0
module load arrow

source "$VENV_DATASET_UPLOAD/bin/activate"

if [[ -z "${HF_TOKEN:-}" ]]; then
	if [[ -f /home/indrisch/TOKENS/cvis-tmu-organization-token.txt ]]; then
		HF_TOKEN="$(cat /home/indrisch/TOKENS/cvis-tmu-organization-token.txt)"
	fi
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
	echo "Error: HF_TOKEN is not set and token file is missing." >&2
	exit 1
fi

if ! command -v hf >/dev/null 2>&1; then
	python -m pip install --upgrade huggingface_hub
fi

resolve_snapshot_dir() {
	local cache_root="$1"
	local repo_slug="$2"
	local repo_dir="${cache_root}/models--${repo_slug//\//--}"
	local snapshot_id=""

	if [[ ! -d "$repo_dir" ]]; then
		echo "Error: Cache directory not found: $repo_dir" >&2
		return 1
	fi

	if [[ -f "$repo_dir/refs/main" ]]; then
		snapshot_id="$(tr -d '\r\n' < "$repo_dir/refs/main")"
		if [[ -n "$snapshot_id" && -d "$repo_dir/snapshots/$snapshot_id" ]]; then
			echo "$repo_dir/snapshots/$snapshot_id"
			return 0
		fi
	fi

	snapshot_id="$(ls -1t "$repo_dir/snapshots" 2>/dev/null | head -n 1)"
	if [[ -n "$snapshot_id" && -d "$repo_dir/snapshots/$snapshot_id" ]]; then
		echo "$repo_dir/snapshots/$snapshot_id"
		return 0
	fi

	echo "Error: No snapshots found under $repo_dir" >&2
	return 1
}

SOURCE_VIDEOR1_DIR="${SOURCE_VIDEOR1_DIR:-$(resolve_snapshot_dir "$HF_HUB_CACHE" "Video-R1/Video-R1-7B")}" || exit 1
SOURCE_QWEN_COT_DIR="${SOURCE_QWEN_COT_DIR:-$(resolve_snapshot_dir "$HF_HUB_CACHE" "Video-R1/Qwen2.5-VL-7B-COT-SFT")}" || exit 1

TARGET_VIDEOR1_REPO="cvis-tmu/Video-R1-7B"
TARGET_QWEN_COT_REPO="cvis-tmu/Qwen2.5-VL-7B-COT-SFT"

COMMIT_MESSAGE_VIDEOR1="${COMMIT_MESSAGE_VIDEOR1:-Upload custom Video-R1-7B base model which uses 1) Qwen2VLImageProcessor and 2) does not set the path in config.json}"
COMMIT_MESSAGE_QWEN_COT="${COMMIT_MESSAGE_QWEN_COT:-Upload custom Qwen2.5-VL-7B-COT-SFT base model which uses 1) Qwen2VLImageProcessor}"

echo "Uploading from: $SOURCE_VIDEOR1_DIR -> $TARGET_VIDEOR1_REPO"
hf upload "$TARGET_VIDEOR1_REPO" "$SOURCE_VIDEOR1_DIR" \
	--repo-type model \
	--token "$HF_TOKEN" \
	--commit-message "$COMMIT_MESSAGE_VIDEOR1"

echo "Uploading from: $SOURCE_QWEN_COT_DIR -> $TARGET_QWEN_COT_REPO"
hf upload "$TARGET_QWEN_COT_REPO" "$SOURCE_QWEN_COT_DIR" \
	--repo-type model \
	--token "$HF_TOKEN" \
	--commit-message "$COMMIT_MESSAGE_QWEN_COT"

deactivate
