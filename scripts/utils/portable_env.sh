#!/usr/bin/env bash
# Portable, repo-relative path resolver for LLaMA-Factory-LFS SLURM jobs.
#
# Source this file; do not execute it:
#   source "$(dirname "${BASH_SOURCE[0]}")/../../scripts/utils/portable_env.sh"
#   portable_resolve_project_dir
#
# Design: this is the ONLY component that maps a machine to a set of paths.
# Every path has a repo-relative default and an environment override, so the
# checkout can be moved or renamed and still run. See
# docs/superpowers/specs/2026-09-05-portable-slurm-wrapper-design.md

# A directory is the repo root when it holds both of these.
PORTABLE_ROOT_SENTINEL_FILE="setup.py"
PORTABLE_ROOT_SENTINEL_DIR="src/llamafactory"

_portable_is_root() {
	local candidate="$1"
	[[ -f "${candidate}/${PORTABLE_ROOT_SENTINEL_FILE}" ]] &&
		[[ -d "${candidate}/${PORTABLE_ROOT_SENTINEL_DIR}" ]]
}

# Sets and exports PROJECT_DIR. Returns 1 when no repo root can be found.
portable_resolve_project_dir() {
	local candidate=""

	if [[ -n "${LFS_PROJECT_DIR:-}" ]]; then
		if ! candidate="$(cd "${LFS_PROJECT_DIR}" 2>/dev/null && pwd -P)"; then
			echo "portable_env: LFS_PROJECT_DIR does not exist: ${LFS_PROJECT_DIR}" >&2
			return 1
		fi
		# Validate even an explicit override: every path in this library is derived
		# from PROJECT_DIR, so a wrong root yields a whole set of wrong paths.
		if ! _portable_is_root "${candidate}"; then
			echo "portable_env: LFS_PROJECT_DIR is not a repo root: ${candidate}" >&2
			return 1
		fi
		PROJECT_DIR="${candidate}"
		export PROJECT_DIR
		return 0
	fi

	# This file lives at <root>/scripts/utils/, so the root is two levels up.
	local here
	if ! here="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd -P)"; then
		echo "portable_env: cannot locate this script" >&2
		return 1
	fi

	if candidate="$(cd "${here}/../.." 2>/dev/null && pwd -P)" && _portable_is_root "${candidate}"; then
		PROJECT_DIR="${candidate}"
		export PROJECT_DIR
		return 0
	fi

	# Fallback for unusual layouts: ask git, then validate the same way.
	local git_root
	if git_root="$(git -C "${here}" rev-parse --show-toplevel 2>/dev/null)" && [[ -n "${git_root}" ]]; then
		if candidate="$(cd "${git_root}" 2>/dev/null && pwd -P)" && _portable_is_root "${candidate}"; then
			PROJECT_DIR="${candidate}"
			export PROJECT_DIR
			return 0
		fi
	fi

	echo "portable_env: could not find repo root (no ${PORTABLE_ROOT_SENTINEL_FILE} + ${PORTABLE_ROOT_SENTINEL_DIR})" >&2
	return 1
}

# Detect the cluster and default running mode. Never overwrites a value the
# caller already set, so CLUSTER=X in the environment always wins.
portable_detect_cluster() {
	local host="${HOSTNAME:-$(hostname 2>/dev/null || echo unknown)}"

	if [[ -z "${CLUSTER:-}" ]]; then
		case "${host}" in
		*rorqual* | rg* | rc*) CLUSTER="RORQUAL" ;;
		*trillium* | trig* | tri*) CLUSTER="TRILLIUM" ;;
		*klogin* | kn*) CLUSTER="KILLARNEY" ;;
		*tamia* | tg*) CLUSTER="TAMIA" ;;
		*nibi*) CLUSTER="NIBI" ;;
		*)
			echo "portable_env: unknown host '${host}', using CLUSTER=PORTABLE" >&2
			CLUSTER="PORTABLE"
			;;
		esac
	fi

	CLUSTER="${CLUSTER^^}"
	export CLUSTER

	if [[ -z "${RUNNING_MODE:-}" ]]; then
		if [[ "${CLUSTER}" == "TAMIA" ]]; then
			RUNNING_MODE="VENV"
		else
			RUNNING_MODE="APPTAINER"
		fi
	fi

	RUNNING_MODE="${RUNNING_MODE^^}"
	export RUNNING_MODE
}

# Every variable this library resolves. Used to enforce precedence around
# site.env; keep in sync with portable_set_paths and portable_set_offline.
PORTABLE_MANAGED_VARS=(
	CLUSTER RUNNING_MODE
	HF_HOME HF_HUB_CACHE TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE HF_DATASETS_CACHE
	HF_HUB_DISABLE_XET SIF_FILE VENV_LLAMAFACTORY APPTAINER_OVERLAY
	SCANNET_H5_DIR SPATIALSSRL_H5_DIR THINKER10K_H5_DIR MEDIA_DIR
	TRITON_CACHE_DIR TORCH_EXTENSIONS_DIR PYTORCH_KERNEL_CACHE_PATH MPLCONFIGDIR
	FLASHINFER_WORKSPACE_BASE WANDB_DIR WANDB_CACHE_DIR TORCH_CUDA_ARCH_LIST
	HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE WANDB_MODE
	DISABLE_VERSION_CHECK FORCE_TORCHRUN
)

# Source scripts/site.env if present, so operators can pin site paths without
# editing tracked files.
#
# Pre-set environment outranks site.env by contract. Enforce that here rather
# than trusting site.env to be written defensively: an operator will naturally
# write `export HF_HOME=/x`, and that plain form would otherwise clobber a value
# passed in by `sbatch --export` or set on the submit line. So snapshot the
# managed variables that were already set, source the file, then restore them.
# Variables site.env owns outright (PORTABLE_SRC_*, EXTRA_BINDS) are not in the
# managed list and pass through untouched.
portable_load_site_env() {
	[[ -n "${PORTABLE_SKIP_SITE_ENV:-}" ]] && return 0

	local site_env="${PORTABLE_SITE_ENV:-${PROJECT_DIR}/scripts/site.env}"
	[[ -f "${site_env}" ]] || return 0

	local -a preset_names=() preset_values=()
	local name
	for name in "${PORTABLE_MANAGED_VARS[@]}"; do
		# ${!name+x} tests "is set", so an intentionally empty value counts.
		if [[ -n "${!name+x}" ]]; then
			preset_names+=("${name}")
			preset_values+=("${!name}")
		fi
	done

	echo "portable_env: loading ${site_env}" >&2
	# shellcheck disable=SC1090
	source "${site_env}"

	# Guard the expansion: bash 4.2/4.3 error on an empty array under `set -u`.
	if ((${#preset_names[@]} > 0)); then
		local i
		for i in "${!preset_names[@]}"; do
			printf -v "${preset_names[i]}" '%s' "${preset_values[i]}"
			export "${preset_names[i]}"
		done
	fi
}

# Pull cluster values from sysconfig.json when available. Values equal to "None"
# or containing an unexpanded token are ignored.
_portable_sysconfig() {
	local key="$1" value=""

	command -v python3 >/dev/null 2>&1 || return 1

	value="$(PROJECT_DIR="${PROJECT_DIR}" PYTHONPATH="${PROJECT_DIR}/scripts${PYTHONPATH:+:${PYTHONPATH}}" \
		python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', '${key}') or '')" 2>/dev/null)" || return 1

	[[ -z "${value}" || "${value}" == "None" || "${value}" == *'${'* ]] && return 1
	printf '%s' "${value}"
}

# Assign a variable from, in order: existing env, sysconfig, repo-relative default.
_portable_default() {
	local name="$1" default="$2" from_sysconfig=""

	if [[ -n "${!name:-}" ]]; then
		export "${name}"
		return 0
	fi

	if from_sysconfig="$(_portable_sysconfig "${name}")"; then
		printf -v "${name}" '%s' "${from_sysconfig}"
	else
		printf -v "${name}" '%s' "${default}"
	fi

	export "${name}"
}

portable_set_paths() {
	local cache_base="${SLURM_TMPDIR:-${PROJECT_DIR}/.cache}"

	_portable_default HF_HOME "${PROJECT_DIR}/.cache/huggingface"
	_portable_default HF_HUB_CACHE "${HF_HOME}"
	export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
	export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
	export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HUB_CACHE}}"
	export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

	_portable_default SIF_FILE "${PROJECT_DIR}/containers/llamafactory.sif"
	_portable_default VENV_LLAMAFACTORY "${PROJECT_DIR}/.venv"
	export APPTAINER_OVERLAY="${APPTAINER_OVERLAY:-${PROJECT_DIR}/apptainer/overlay.img}"

	_portable_default SCANNET_H5_DIR "${PROJECT_DIR}/data/h5/ScanNet_h5/scans"
	_portable_default SPATIALSSRL_H5_DIR "${PROJECT_DIR}/data/h5/Spatial-SSRL_images_h5"
	_portable_default THINKER10K_H5_DIR "${PROJECT_DIR}/data/h5/3DThinker10K_images_h5"
	_portable_default MEDIA_DIR "${PROJECT_DIR}/data/h5/ScanNet_h5"

	export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${cache_base}/.triton_cache}"
	export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${cache_base}/torch_extensions}"
	export PYTORCH_KERNEL_CACHE_PATH="${PYTORCH_KERNEL_CACHE_PATH:-${cache_base}/torch/kernels}"
	export MPLCONFIGDIR="${MPLCONFIGDIR:-${cache_base}/matplotlib}"
	export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${cache_base}/flashinfer}"
	export WANDB_DIR="${WANDB_DIR:-${PROJECT_DIR}/wandb}"
	export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${cache_base}/.cache/wandb}"

	# L40S on Killarney is Ada (8.9); do not trust BEST_GPU=h100 there.
	if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
		if [[ "${CLUSTER}" == "KILLARNEY" ]]; then
			TORCH_CUDA_ARCH_LIST="8.9"
		elif [[ "$(_portable_sysconfig BEST_GPU || echo h100)" == "h100" ]]; then
			TORCH_CUDA_ARCH_LIST="9.0"
		else
			TORCH_CUDA_ARCH_LIST="8.0"
		fi
	fi
	export TORCH_CUDA_ARCH_LIST
}

portable_set_offline() {
	export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
	export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
	export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
	export WANDB_MODE="${WANDB_MODE:-offline}"
	export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
	export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"
	export PYTHONUNBUFFERED=1
	export PYTHONNOUSERSITE=1
}

portable_init() {
	portable_resolve_project_dir || return 1
	portable_detect_cluster
	portable_load_site_env
	portable_set_paths
	portable_set_offline

	# This tree's src must win over any editable install pointing elsewhere.
	export PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
}
