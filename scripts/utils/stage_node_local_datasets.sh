#!/usr/bin/env bash
# Stage CoT default datasets (annotations + H5 media packs) onto node-local disk.
#
# Intended to be sourced by multi-node SLURM workers (one process per node), or
# invoked as a CLI for dry-runs:
#   bash scripts/utils/stage_node_local_datasets.sh --cot-defaults
#
# Offline only: copies from existing cluster paths; never downloads.
#
# On success exports:
#   NODE_LOCAL_DATA_ROOT, LOCAL_STAGE_ROOT, LOCAL_DATASET_DIR, LOCAL_MEDIA_DIR
#   SCANNET_H5_DIR, SPATIALSSRL_H5_DIR, THINKER10K_H5_DIR  (repointed to local)
#   SCENE30K_ANN_LOCAL, SPATIALSSRL_ANN_LOCAL, THINKER10K_ANN_LOCAL
#
# When sourced, do not enable `set -e` here (would alter the caller's shell).

# ---------------------------------------------------------------------------
# Defaults (overridable via env before source/invoke)
# ---------------------------------------------------------------------------

# Shared annotation sources (match data/dataset_info.json defaults).
# HF_HUB_CACHE / HF_HOME come from scripts/utils/env.sh (per-cluster sysconfig).
_hf_cache="${HF_HUB_CACHE:-${HF_HOME:-}}"
: "${SCENE30K_ANN_SRC:=${_hf_cache}/datasets--cvis-tmu--Scene30K/snapshots/13b41da710700aed32c928c81b8f5e433134eb75/data/train-00000-of-00001.with_question_id.parquet}"
: "${SPATIALSSRL_ANN_SRC:=${_hf_cache}/datasets--internlm--Spatial-SSRL-81k/snapshots/54b82086060a5612f95588b4979446da2282bcd9/SFT-coldstart.with_question_id.json}"

# 3DThinker annotation is relative under data/ in the repo; resolve via PROJECT_DIR when set.
if [[ -z "${THINKER10K_ANN_SRC:-}" ]]; then
	if [[ -n "${PROJECT_DIR:-}" && -f "${PROJECT_DIR}/data/3DThinker-10K/out/3dthinker10k_cot.jsonl" ]]; then
		THINKER10K_ANN_SRC="${PROJECT_DIR}/data/3DThinker-10K/out/3dthinker10k_cot.jsonl"
	else
		THINKER10K_ANN_SRC="data/3DThinker-10K/out/3dthinker10k_cot.jsonl"
	fi
fi

# H5 source roots (workers usually export these before sourcing).
: "${SCANNET_H5_DIR:=/scratch/indrisch/ScanNet_h5/scans}"
: "${SPATIALSSRL_H5_DIR:=/scratch/indrisch/Spatial-SSRL_images_h5}"
: "${THINKER10K_H5_DIR:=/scratch/indrisch/3DThinker10K_images_h5}"

# Shared dataset_info used as the schema template.
if [[ -z "${SOURCE_DATASET_INFO:-}" ]]; then
	if [[ -n "${PROJECT_DIR:-}" && -f "${PROJECT_DIR}/data/dataset_info.json" ]]; then
		SOURCE_DATASET_INFO="${PROJECT_DIR}/data/dataset_info.json"
	else
		SOURCE_DATASET_INFO="data/dataset_info.json"
	fi
fi

# Staging root on node-local storage.
: "${NODE_LOCAL_DATA_ROOT:=${SLURM_TMPDIR:-/tmp}/cot_stage}"

# Parallelism: half of allocated CPUs, clamped to [4, 32].
_stage_default_jobs() {
	local cpus="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-$(nproc 2>/dev/null || echo 8)}}"
	# Keep only digits.
	cpus="${cpus//[^0-9]/}"
	[[ -n "$cpus" ]] || cpus=8
	local half=$((cpus / 2))
	if ((half < 4)); then half=4; fi
	if ((half > 32)); then half=32; fi
	echo "$half"
}
: "${STAGE_COPY_JOBS:=$(_stage_default_jobs)}"
: "${STAGE_STAGGER_SEC:=0}"
# Estimated need ~110G media + headroom (bytes). Override if your tree differs.
: "${STAGE_NEED_BYTES:=$((120 * 1024 * 1024 * 1024))}"
: "${STAGE_SKIP_IF_COMPLETE:=1}"

# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

_stage_log() {
	echo "[stage $(hostname) node=${SLURM_NODEID:-?} $(date -Is)] $*"
}

_stage_seconds() {
	date +%s
}

stage_check_space() {
	local need_bytes="${1:-$STAGE_NEED_BYTES}"
	local target="${2:-$NODE_LOCAL_DATA_ROOT}"
	mkdir -p "$target"
	local avail
	avail="$(df -B1 --output=avail "$target" 2>/dev/null | tail -n 1 | tr -d ' ')"
	if [[ -z "$avail" || ! "$avail" =~ ^[0-9]+$ ]]; then
		_stage_log "WARNING: could not determine free space on $target; continuing"
		return 0
	fi
	_stage_log "Free space on $target: $avail bytes (need ~$need_bytes)"
	if ((avail < need_bytes)); then
		_stage_log "ERROR: insufficient local space. Free=$avail need>=$need_bytes"
		_stage_log "Set STAGE_DATASETS_LOCAL=0 to train from shared paths, or free disk."
		return 1
	fi
}

stage_require_src() {
	local path="$1"
	local label="${2:-$path}"
	if [[ ! -e "$path" ]]; then
		_stage_log "ERROR: missing source for $label: $path"
		return 1
	fi
}

# Copy a single file, always materializing HF hub symlinks (cp -L).
stage_copy_file() {
	local src="$1"
	local dest="$2"
	mkdir -p "$(dirname "$dest")"
	# -L: dereference; -a-ish: preserve times when possible
	cp -L --preserve=timestamps "$src" "$dest"
	if [[ -L "$dest" ]]; then
		_stage_log "ERROR: destination still a symlink (expected real file): $dest"
		return 1
	fi
	if [[ ! -s "$dest" ]]; then
		_stage_log "ERROR: empty annotation after copy: $dest"
		return 1
	fi
}

# Parallel copy of direct children (files and/or dirs) under src into dest/.
stage_copy_tree_parallel() {
	local src="$1"
	local dest="$2"
	local jobs="${3:-$STAGE_COPY_JOBS}"
	mkdir -p "$dest"
	if [[ ! -d "$src" ]]; then
		_stage_log "ERROR: not a directory: $src"
		return 1
	fi

	# Prefer rsync per top-level entry (good for both many scenes and few large files).
	local -a entries=()
	local ent
	while IFS= read -r -d '' ent; do
		entries+=("$ent")
	done < <(find "$src" -mindepth 1 -maxdepth 1 -print0)

	if ((${#entries[@]} == 0)); then
		_stage_log "WARNING: no entries under $src"
		return 0
	fi

	_stage_log "Copying ${#entries[@]} entries from $src -> $dest (jobs=$jobs)"
	# xargs -P: one rsync per top-level entry
	printf '%s\0' "${entries[@]}" | xargs -0 -P "$jobs" -I{} \
		rsync -a --info=stats0,progress0 {} "$dest/"
}

stage_write_dataset_info() {
	local out_json="$1"
	local scene_ann="$2"
	local spatial_ann="$3"
	local thinker_ann="$4"
	local builder="${PROJECT_DIR:-}/scripts/utils/build_local_dataset_info.py"
	if [[ ! -f "$builder" ]]; then
		# Fallback relative to this script.
		builder="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build_local_dataset_info.py"
	fi
	if [[ ! -f "$builder" ]]; then
		_stage_log "ERROR: build_local_dataset_info.py not found"
		return 1
	fi
	python3 "$builder" \
		--source-dataset-info "$SOURCE_DATASET_INFO" \
		--output-dataset-info "$out_json" \
		--datasets "Scene30k,SpatialSSRL_coldstart,3DThinker10k" \
		--file-name "Scene30k=${scene_ann}" \
		--file-name "SpatialSSRL_coldstart=${spatial_ann}" \
		--file-name "3DThinker10k=${thinker_ann}"
}

# Pure-stdlib YAML patch for dataset_dir + media_dir (no ruamel required).
stage_patch_yaml_paths() {
	local src_yaml="$1"
	local dest_yaml="$2"
	local dataset_dir="$3"
	local media_dir="$4"
	python3 - "$src_yaml" "$dest_yaml" "$dataset_dir" "$media_dir" <<'PY'
import re
import sys
from pathlib import Path

src, dest, dataset_dir, media_dir = sys.argv[1:5]
text = Path(src).read_text(encoding="utf-8")

def upsert(key: str, value: str, body: str) -> str:
    # Match a top-level key line: "key: ..." (optional spaces)
    pat = re.compile(rf"(?m)^({re.escape(key)}\s*:\s*).*$")
    line = f"{key}: {value}"
    if pat.search(body):
        return pat.sub(rf"\g<1>{value}", body, count=1)
    # Insert after media_dir if present, else after dataset: line, else append.
    for anchor in ("media_dir", "dataset", "mix_strategy"):
        apat = re.compile(rf"(?m)^({re.escape(anchor)}\s*:.*)$")
        m = apat.search(body)
        if m:
            insert_at = m.end()
            return body[:insert_at] + "\n" + line + body[insert_at:]
    return body.rstrip() + "\n" + line + "\n"

text = upsert("media_dir", media_dir, text)
text = upsert("dataset_dir", dataset_dir, text)
Path(dest).parent.mkdir(parents=True, exist_ok=True)
Path(dest).write_text(text, encoding="utf-8")
print(f"Patched YAML -> {dest}")
print(f"  dataset_dir: {dataset_dir}")
print(f"  media_dir: {media_dir}")
PY
}

stage_verify_cot() {
	local root="${1:-$NODE_LOCAL_DATA_ROOT}"
	local ann_dir="${root}/annotations"
	local media="${root}/media"
	local scannet="${media}/ScanNet_h5/scans"
	local spatial="${media}/Spatial-SSRL_images_h5"
	local thinker="${media}/3DThinker10K_images_h5"
	local dataset_info="${root}/dataset_dir/dataset_info.json"

	local ok=1
	for f in \
		"${ann_dir}/Scene30k.parquet" \
		"${ann_dir}/SpatialSSRL_coldstart.json" \
		"${ann_dir}/3dthinker10k_cot.jsonl" \
		"$dataset_info"
	do
		if [[ ! -s "$f" ]]; then
			_stage_log "VERIFY FAIL: missing/empty $f"
			ok=0
		fi
	done

	if [[ ! -f "${scannet}/scene0000_00/images.hdf5" ]]; then
		# scene0000_00 may not always exist; accept any scene pack.
		if ! find "$scannet" -mindepth 2 -maxdepth 2 -name 'images.hdf5' 2>/dev/null | head -n 1 | grep -q .; then
			_stage_log "VERIFY FAIL: no ScanNet images.hdf5 under $scannet"
			ok=0
		fi
	fi
	if [[ ! -f "${spatial}/coldstart_SFT.h5" && ! -f "${spatial}/spatial_ssrl_images_index.json" ]]; then
		_stage_log "VERIFY FAIL: Spatial-SSRL H5/index missing under $spatial"
		ok=0
	fi
	if [[ ! -f "${thinker}/3dthinker10k_images.h5" ]]; then
		_stage_log "VERIFY FAIL: missing ${thinker}/3dthinker10k_images.h5"
		ok=0
	fi

	if ((ok != 1)); then
		return 1
	fi
	_stage_log "VERIFY OK under $root"
}

stage_cot_default_datasets() {
	local t0 t1 t_comp
	t0="$(_stage_seconds)"

	export NODE_LOCAL_DATA_ROOT
	export LOCAL_STAGE_ROOT="${NODE_LOCAL_DATA_ROOT}"
	local root="$NODE_LOCAL_DATA_ROOT"
	local ann_dir="${root}/annotations"
	local media_root="${root}/media"
	local dataset_dir="${root}/dataset_dir"
	local scannet_dst="${media_root}/ScanNet_h5/scans"
	local spatial_dst="${media_root}/Spatial-SSRL_images_h5"
	local thinker_dst="${media_root}/3DThinker10K_images_h5"
	local sentinel="${root}/.stage_complete"

	_stage_log "=== Node-local CoT dataset staging ==="
	_stage_log "HOST=$(hostname) SLURM_NODEID=${SLURM_NODEID:-} SLURM_NNODES=${SLURM_NNODES:-}"
	_stage_log "NODE_LOCAL_DATA_ROOT=$root"
	_stage_log "STAGE_COPY_JOBS=$STAGE_COPY_JOBS STAGE_STAGGER_SEC=$STAGE_STAGGER_SEC"
	_stage_log "Sources:"
	_stage_log "  SCENE30K_ANN_SRC=$SCENE30K_ANN_SRC"
	_stage_log "  SPATIALSSRL_ANN_SRC=$SPATIALSSRL_ANN_SRC"
	_stage_log "  THINKER10K_ANN_SRC=$THINKER10K_ANN_SRC"
	_stage_log "  SCANNET_H5_DIR(src)=$SCANNET_H5_DIR"
	_stage_log "  SPATIALSSRL_H5_DIR(src)=$SPATIALSSRL_H5_DIR"
	_stage_log "  THINKER10K_H5_DIR(src)=$THINKER10K_H5_DIR"

	if [[ "${STAGE_SKIP_IF_COMPLETE}" == "1" && -f "$sentinel" ]]; then
		_stage_log "Found $sentinel — verifying existing stage and reusing"
		if stage_verify_cot "$root"; then
			export SCENE30K_ANN_LOCAL="${ann_dir}/Scene30k.parquet"
			export SPATIALSSRL_ANN_LOCAL="${ann_dir}/SpatialSSRL_coldstart.json"
			export THINKER10K_ANN_LOCAL="${ann_dir}/3dthinker10k_cot.jsonl"
			export LOCAL_DATASET_DIR="$dataset_dir"
			export LOCAL_MEDIA_DIR="${media_root}/ScanNet_h5"
			export SCANNET_H5_DIR="$scannet_dst"
			export SPATIALSSRL_H5_DIR="$spatial_dst"
			export THINKER10K_H5_DIR="$thinker_dst"
			_stage_log "Reusing staged data; H5 env repointed to local"
			return 0
		fi
		_stage_log "Existing stage incomplete; re-staging"
		rm -f "$sentinel"
	fi

	# Optional stagger to reduce multi-node NFS stampede at job start.
	if [[ "${STAGE_STAGGER_SEC}" =~ ^[0-9]+$ ]] && ((STAGE_STAGGER_SEC > 0)); then
		local delay=$(( ${SLURM_NODEID:-0} * STAGE_STAGGER_SEC ))
		if ((delay > 0)); then
			_stage_log "Stagger sleep ${delay}s"
			sleep "$delay"
		fi
	fi

	stage_require_src "$SCENE30K_ANN_SRC" "Scene30k annotation" || return 1
	stage_require_src "$SPATIALSSRL_ANN_SRC" "SpatialSSRL annotation" || return 1
	stage_require_src "$THINKER10K_ANN_SRC" "3DThinker annotation" || return 1
	stage_require_src "$SCANNET_H5_DIR" "ScanNet H5" || return 1
	stage_require_src "$SPATIALSSRL_H5_DIR" "Spatial-SSRL H5" || return 1
	stage_require_src "$THINKER10K_H5_DIR" "3DThinker H5" || return 1
	stage_require_src "$SOURCE_DATASET_INFO" "dataset_info.json" || return 1
	stage_check_space "$STAGE_NEED_BYTES" "$root" || return 1

	mkdir -p "$ann_dir" "$scannet_dst" "$spatial_dst" "$thinker_dst" "$dataset_dir"

	# --- annotations (small; always dereference) ---
	t_comp="$(_stage_seconds)"
	_stage_log "Copying annotations..."
	stage_copy_file "$SCENE30K_ANN_SRC" "${ann_dir}/Scene30k.parquet" || return 1
	stage_copy_file "$SPATIALSSRL_ANN_SRC" "${ann_dir}/SpatialSSRL_coldstart.json" || return 1
	stage_copy_file "$THINKER10K_ANN_SRC" "${ann_dir}/3dthinker10k_cot.jsonl" || return 1
	_stage_log "Annotations done in $((_stage_seconds - t_comp))s"

	# --- H5 media (large; parallel) ---
	# Snapshot source paths before we overwrite the H5 env exports.
	local scannet_src="$SCANNET_H5_DIR"
	local spatial_src="$SPATIALSSRL_H5_DIR"
	local thinker_src="$THINKER10K_H5_DIR"

	t_comp="$(_stage_seconds)"
	_stage_log "Copying ScanNet H5 tree..."
	stage_copy_tree_parallel "$scannet_src" "$scannet_dst" "$STAGE_COPY_JOBS" || return 1
	_stage_log "ScanNet done in $((_stage_seconds - t_comp))s"

	t_comp="$(_stage_seconds)"
	_stage_log "Copying Spatial-SSRL H5 tree..."
	stage_copy_tree_parallel "$spatial_src" "$spatial_dst" "$STAGE_COPY_JOBS" || return 1
	_stage_log "Spatial-SSRL done in $((_stage_seconds - t_comp))s"

	t_comp="$(_stage_seconds)"
	_stage_log "Copying 3DThinker H5 tree..."
	stage_copy_tree_parallel "$thinker_src" "$thinker_dst" "$STAGE_COPY_JOBS" || return 1
	_stage_log "3DThinker done in $((_stage_seconds - t_comp))s"

	# --- local dataset_info ---
	export SCENE30K_ANN_LOCAL="${ann_dir}/Scene30k.parquet"
	export SPATIALSSRL_ANN_LOCAL="${ann_dir}/SpatialSSRL_coldstart.json"
	export THINKER10K_ANN_LOCAL="${ann_dir}/3dthinker10k_cot.jsonl"
	stage_write_dataset_info \
		"${dataset_dir}/dataset_info.json" \
		"$SCENE30K_ANN_LOCAL" \
		"$SPATIALSSRL_ANN_LOCAL" \
		"$THINKER10K_ANN_LOCAL" || return 1

	stage_verify_cot "$root" || return 1

	# Repoint H5 env to local copies (training uses these).
	export SCANNET_H5_DIR="$scannet_dst"
	export SPATIALSSRL_H5_DIR="$spatial_dst"
	export THINKER10K_H5_DIR="$thinker_dst"
	export LOCAL_DATASET_DIR="$dataset_dir"
	export LOCAL_MEDIA_DIR="${media_root}/ScanNet_h5"
	export LOCAL_STAGE_ROOT="$root"
	export NODE_LOCAL_DATA_ROOT="$root"

	touch "$sentinel"
	t1="$(_stage_seconds)"
	_stage_log "=== Staging complete in $((t1 - t0))s ==="
	_stage_log "LOCAL_DATASET_DIR=$LOCAL_DATASET_DIR"
	_stage_log "LOCAL_MEDIA_DIR=$LOCAL_MEDIA_DIR"
	_stage_log "SCANNET_H5_DIR=$SCANNET_H5_DIR"
	_stage_log "SPATIALSSRL_H5_DIR=$SPATIALSSRL_H5_DIR"
	_stage_log "THINKER10K_H5_DIR=$THINKER10K_H5_DIR"
	df -h "$root" || true
}

# Rebuild APPTAINER bind/env arrays after H5 paths change (call from worker).
# Uses namerefs when bash supports them; otherwise prints guidance.
stage_refresh_apptainer_h5() {
	# shellcheck disable=SC2034
	APPTAINER_H5_ENV=(
		--env "SCANNET_H5_DIR=${SCANNET_H5_DIR}"
		--env "SPATIALSSRL_H5_DIR=${SPATIALSSRL_H5_DIR}"
		--env "THINKER10K_H5_DIR=${THINKER10K_H5_DIR}"
	)
	APPTAINER_H5_BINDS=()
	local _h5
	for _h5 in "${SCANNET_H5_DIR}" "${SPATIALSSRL_H5_DIR}" "${THINKER10K_H5_DIR}" "${LOCAL_MEDIA_DIR:-}" "${NODE_LOCAL_DATA_ROOT:-}" "${SLURM_TMPDIR:-}"; do
		if [[ -n "${_h5}" && "${_h5}" != "None" && -e "${_h5}" ]]; then
			APPTAINER_H5_BINDS+=(-B "${_h5}")
		fi
	done
	_stage_log "Refreshed APPTAINER_H5_ENV/BINDS for local H5 roots"
}

# ---------------------------------------------------------------------------
# CLI entry (when executed, not sourced)
# ---------------------------------------------------------------------------

_stage_is_sourced() {
	[[ "${BASH_SOURCE[0]}" != "${0}" ]]
}

if ! _stage_is_sourced; then
	set -euo pipefail
	# Resolve PROJECT_DIR when run standalone.
	if [[ -z "${PROJECT_DIR:-}" ]]; then
		_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
		# scripts/utils -> repo root
		if [[ -f "${_here}/../../data/dataset_info.json" ]]; then
			export PROJECT_DIR="$(cd "${_here}/../.." && pwd)"
		fi
	fi
	case "${1:-}" in
		--cot-defaults|"")
			stage_cot_default_datasets
			;;
		-h|--help)
			cat <<'EOF'
Usage:
  bash scripts/utils/stage_node_local_datasets.sh --cot-defaults

  # or source and call:
  source scripts/utils/stage_node_local_datasets.sh
  stage_cot_default_datasets

Env knobs: NODE_LOCAL_DATA_ROOT, STAGE_COPY_JOBS, STAGE_STAGGER_SEC,
  STAGE_NEED_BYTES, STAGE_SKIP_IF_COMPLETE, SCENE30K_ANN_SRC,
  SPATIALSSRL_ANN_SRC, THINKER10K_ANN_SRC, SCANNET_H5_DIR,
  SPATIALSSRL_H5_DIR, THINKER10K_H5_DIR, SOURCE_DATASET_INFO
EOF
			;;
		*)
			echo "Unknown arg: $1 (try --help)" >&2
			exit 2
			;;
	esac
fi
