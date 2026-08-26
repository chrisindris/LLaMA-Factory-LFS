#!/bin/bash
# Upload QUESTION_ID annotation files to the cvis-tmu dataset repos on the Hub.
# Hugging Face equivalent of git add + commit + push (hf upload → one commit per file).

set -euo pipefail

usage() {
	cat <<'EOF'
Usage: ./scripts/upload_question_id_datasets.sh [--dry-run]

Uploads the with_question_id annotation files to the matching Hugging Face
dataset repos (already cached locally under HF_HUB_CACHE):

  1. data/3DThinker-10K/out/3dthinker10k_cot.with_question_id.jsonl
     -> cvis-tmu/3dthinker-10k-mcq  (path_in_repo: 3dthinker10k_cot.with_question_id.jsonl)
  2. .../datasets--internlm--Spatial-SSRL-81k/.../SFT-coldstart.with_question_id.json
     -> cvis-tmu/Spatial-SSRL-81k   (path_in_repo: SFT-coldstart.with_question_id.json)
  3. .../datasets--cvis-tmu--Scene30K/.../data/train-00000-of-00001.with_question_id.parquet
     -> cvis-tmu/Scene30K           (path_in_repo: data/train-00000-of-00001.with_question_id.parquet)

Each call is `hf upload <repo_id> <local_file> <path_in_repo> --repo-type dataset`.

Optional flags:
	--dry-run                Print the hf upload commands without running them.

Optional environment variables:
	CLUSTER                  Cluster override for sysconfig lookup.
	HF_HOME / HF_HUB_CACHE   Hugging Face cache roots (default: sysconfig).
	VENV_DATASET_UPLOAD      Python venv with huggingface_hub / hf CLI.
	HF_TOKEN                 Hugging Face token (defaults to token file).
	COMMIT_MESSAGE           Shared commit message prefix override.
EOF
}

DRY_RUN=0
while [[ $# -gt 0 ]]; do
	case "$1" in
		-h|--help)
			usage
			exit 0
			;;
		--dry-run)
			DRY_RUN=1
			shift
			;;
		*)
			echo "Error: Unknown argument: $1" >&2
			usage >&2
			exit 1
			;;
	esac
done

# --- project + sysconfig ---

if [[ "$PWD" == *LLaMA-Factory-LFS* ]]; then
	PROJECT_DIR="${PWD%%LLaMA-Factory-LFS*}/LLaMA-Factory-LFS"
elif [[ "$PWD" == *LLaMA-Factory* ]]; then
	PROJECT_DIR="${PWD%%LLaMA-Factory*}/LLaMA-Factory"
else
	echo "Error: Could not find 'LLaMA-Factory' or 'LLaMA-Factory-LFS' in the current path." >&2
	exit 1
fi
SYSCONFIG_DIR_PATH="$PROJECT_DIR/scripts"
export PYTHONPATH="${PYTHONPATH:-}:$SYSCONFIG_DIR_PATH"

if [[ -z "${CLUSTER:-}" ]]; then
	if [[ "$HOSTNAME" == *"rorqual"* ]] || [[ "$HOSTNAME" == *"rg"* ]]; then
		export CLUSTER="RORQUAL"
	elif [[ "$HOSTNAME" == *"trig"* ]] || [[ "$HOSTNAME" == *"trillium"* ]]; then
		export CLUSTER="TRILLIUM"
	elif [[ "$HOSTNAME" == *"klogin"* ]] || [[ "$HOSTNAME" == *"kn"* ]]; then
		export CLUSTER="KILLARNEY"
	elif [[ "$HOSTNAME" == *"tamia"* ]] || [[ "$HOSTNAME" == *"tg"* ]]; then
		export CLUSTER="TAMIA"
	elif [[ "$HOSTNAME" == *"nibi"* ]]; then
		export CLUSTER="NIBI"
	else
		echo "Warning: Could not detect cluster from HOSTNAME. Defaulting to TAMIA." >&2
		export CLUSTER="TAMIA"
	fi
fi

echo "Detected cluster: $CLUSTER"
echo "PROJECT_DIR: $PROJECT_DIR"

if [[ -z "${HF_HOME:-}" ]]; then
	HF_HOME="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'HF_HOME') or '')")"
fi
if [[ -z "${HF_HUB_CACHE:-}" ]]; then
	HF_HUB_CACHE="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'HF_HUB_CACHE') or '')")"
fi
if [[ -z "${VENV_DATASET_UPLOAD:-}" ]]; then
	VENV_DATASET_UPLOAD="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'VENV_DATASET_UPLOAD') or '')")"
fi

if [[ -z "${HF_HUB_CACHE:-}" ]]; then
	if [[ -n "${HF_HOME:-}" ]]; then
		HF_HUB_CACHE="$HF_HOME"
	else
		HF_HUB_CACHE="/project/aip-wangcs/indrisch/huggingface/hub"
	fi
fi
if [[ -z "${HF_HOME:-}" ]]; then
	HF_HOME="$HF_HUB_CACHE"
fi

export HF_HOME
export HF_HUB_CACHE
echo "HF_HOME: $HF_HOME"
echo "HF_HUB_CACHE: $HF_HUB_CACHE"

if [[ -z "${VENV_DATASET_UPLOAD:-}" ]]; then
	echo "Error: VENV_DATASET_UPLOAD is not set and could not be read from sysconfig." >&2
	exit 1
fi
echo "VENV_DATASET_UPLOAD: $VENV_DATASET_UPLOAD"

# --- venv + token ---

# shellcheck disable=SC1091
source "$VENV_DATASET_UPLOAD/bin/activate"

if [[ -z "${HF_TOKEN:-}" ]]; then
	if [[ -f "${HOME}/TOKENS/cvis-tmu-organization-token.txt" ]]; then
		HF_TOKEN="$(cat "${HOME}/TOKENS/cvis-tmu-organization-token.txt")"
	elif [[ -f /home/indrisch/TOKENS/cvis-tmu-organization-token.txt ]]; then
		HF_TOKEN="$(cat /home/indrisch/TOKENS/cvis-tmu-organization-token.txt)"
	fi
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
	echo "Error: HF_TOKEN is not set and token file is missing." >&2
	exit 1
fi
export HF_TOKEN

if ! command -v hf >/dev/null 2>&1; then
	python -m pip install --upgrade huggingface_hub
fi

# --- upload helper ---

repo_id_from_cache_dir() {
	local cache_dir="$1"
	local slug
	slug="$(basename "$cache_dir")"
	slug="${slug#datasets--}"
	# datasets--org--name  ->  org/name  (first remaining -- only)
	echo "${slug/--//}"
}

upload_file_to_dataset() {
	local local_file="$1"
	local dest_cache="$2"
	local path_in_repo="$3"
	local commit_message="$4"
	local repo_id
	repo_id="$(repo_id_from_cache_dir "$dest_cache")"

	if [[ ! -f "$local_file" ]]; then
		echo "Error: Source file not found: $local_file" >&2
		return 1
	fi
	if [[ ! -d "$dest_cache" ]]; then
		echo "Error: Destination Hub cache dir not found: $dest_cache" >&2
		return 1
	fi

	echo
	echo "==== $repo_id ===="
	echo "local file:    $local_file"
	echo "dest cache:    $dest_cache"
	echo "path_in_repo:  $path_in_repo"
	echo "commit:        $commit_message"

	local cmd=(
		hf upload "$repo_id" "$local_file" "$path_in_repo"
		--repo-type dataset
		--token "$HF_TOKEN"
		--commit-message "$commit_message"
	)
	echo "command: hf upload ${repo_id} ${local_file} ${path_in_repo} --repo-type dataset --token <redacted> --commit-message ${commit_message}"

	if [[ "$DRY_RUN" -eq 1 ]]; then
		echo "[dry-run] skipped"
		return 0
	fi

	"${cmd[@]}"
	echo "Uploaded https://huggingface.co/datasets/${repo_id}/blob/main/${path_in_repo}"
}

COMMIT_PREFIX="${COMMIT_MESSAGE:-Add QUESTION_ID annotations}"

FILE_3DTHINKER="${PROJECT_DIR}/data/3DThinker-10K/out/3dthinker10k_cot.with_question_id.jsonl"
DEST_3DTHINKER="${HF_HUB_CACHE}/datasets--cvis-tmu--3dthinker-10k-mcq"

FILE_SPATIAL="${HF_HUB_CACHE}/datasets--internlm--Spatial-SSRL-81k/snapshots/54b82086060a5612f95588b4979446da2282bcd9/SFT-coldstart.with_question_id.json"
DEST_SPATIAL="${HF_HUB_CACHE}/datasets--cvis-tmu--Spatial-SSRL-81k"

FILE_SCENE30K="${HF_HUB_CACHE}/datasets--cvis-tmu--Scene30K/snapshots/13b41da710700aed32c928c81b8f5e433134eb75/data/train-00000-of-00001.with_question_id.parquet"
DEST_SCENE30K="${HF_HUB_CACHE}/datasets--cvis-tmu--Scene30K"

upload_file_to_dataset \
	"$FILE_3DTHINKER" \
	"$DEST_3DTHINKER" \
	"3dthinker10k_cot.with_question_id.jsonl" \
	"${COMMIT_PREFIX}: 3dthinker10k_cot.with_question_id.jsonl"

upload_file_to_dataset \
	"$FILE_SPATIAL" \
	"$DEST_SPATIAL" \
	"SFT-coldstart.with_question_id.json" \
	"${COMMIT_PREFIX}: SFT-coldstart.with_question_id.json"

upload_file_to_dataset \
	"$FILE_SCENE30K" \
	"$DEST_SCENE30K" \
	"data/train-00000-of-00001.with_question_id.parquet" \
	"${COMMIT_PREFIX}: data/train-00000-of-00001.with_question_id.parquet"

echo
if [[ "$DRY_RUN" -eq 1 ]]; then
	echo "Dry run finished. Re-run without --dry-run to upload."
else
	echo "All three dataset files uploaded."
fi

deactivate
