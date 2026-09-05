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

# Clusters this library knows how to configure.
PORTABLE_KNOWN_CLUSTERS=(RORQUAL TRILLIUM KILLARNEY TAMIA NIBI PORTABLE)

_portable_is_known_cluster() {
	local candidate="$1" known
	for known in "${PORTABLE_KNOWN_CLUSTERS[@]}"; do
		[[ "${candidate}" == "${known}" ]] && return 0
	done
	return 1
}

# Detect the cluster and default running mode. Never overwrites a value the
# caller already set, so CLUSTER=X in the environment (or in site.env, which is
# loaded before this runs) always wins.
portable_detect_cluster() {
	local host="${HOSTNAME:-$(hostname 2>/dev/null || echo unknown)}"

	if [[ -n "${CLUSTER:-}" ]]; then
		CLUSTER="${CLUSTER^^}"
		# Reject a typo rather than silently running with an unknown profile.
		if ! _portable_is_known_cluster "${CLUSTER}"; then
			echo "portable_env: unknown CLUSTER '${CLUSTER}' (expected one of:" \
				"${PORTABLE_KNOWN_CLUSTERS[*]}); using PORTABLE" >&2
			CLUSTER="PORTABLE"
		fi
	else
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
	DISABLE_VERSION_CHECK FORCE_TORCHRUN PYTHONUNBUFFERED PYTHONNOUSERSITE
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
		# Non-empty, matching _portable_default: an empty value counts as unset
		# everywhere in this library. Using "is set" semantics here instead would
		# preserve an empty value that _portable_default then overwrites anyway.
		if [[ -n "${!name:-}" ]]; then
			preset_names+=("${name}")
			preset_values+=("${!name}")
		fi
	done

	echo "portable_env: loading ${site_env}" >&2
	local source_rc=0
	# shellcheck disable=SC1090
	source "${site_env}" || source_rc=$?

	# Guard the expansion: bash 4.2/4.3 error on an empty array under `set -u`.
	if ((${#preset_names[@]} > 0)); then
		local i
		for i in "${!preset_names[@]}"; do
			printf -v "${preset_names[i]}" '%s' "${preset_values[i]}"
			export "${preset_names[i]}"
		done
	fi

	# Restore first, then fail. A site.env that errors partway has applied some of
	# its exports and not others; continuing from that state would burn GPU time
	# on a half-configured job, which is the exact failure this library prevents.
	if ((source_rc != 0)); then
		echo "portable_env: ${site_env} failed with status ${source_rc};" \
			"refusing to continue with a partially applied site config" >&2
		return "${source_rc}"
	fi
}

# Read a setting from the PORTABLE section of sysconfig.json.
#
# ONLY the PORTABLE section is consulted, never the legacy per-cluster sections.
# Those hold another user's absolute paths (/scratch/indrisch/...), which is
# precisely what this library exists to remove — inheriting them would recreate
# the non-portability on the very clusters we most need to run on. The detected
# CLUSTER still drives module loads and the CUDA arch list; it never supplies a
# path.
#
# A value is rejected when empty, literally "None", or still containing "${":
# an unexpanded token means the variable was unset, and using it would silently
# produce a path rooted at the filesystem root.
_portable_setting() {
	local key="$1" value=""

	command -v python3 >/dev/null 2>&1 || return 1

	value="$(PROJECT_DIR="${PROJECT_DIR}" PYTHONPATH="${PROJECT_DIR}/scripts${PYTHONPATH:+:${PYTHONPATH}}" \
		python3 -c "import sysconfigtool; print(sysconfigtool.read('PORTABLE', '${key}') or '')" 2>/dev/null)" || return 1

	[[ -z "${value}" || "${value}" == "None" || "${value}" == *'${'* ]] && return 1
	printf '%s' "${value}"
}

# BEST_GPU is a hardware fact about the detected cluster rather than a path, so
# this is the one lookup that reads a per-cluster section.
_portable_cluster_best_gpu() {
	local value=""

	command -v python3 >/dev/null 2>&1 || return 1

	value="$(PROJECT_DIR="${PROJECT_DIR}" PYTHONPATH="${PROJECT_DIR}/scripts${PYTHONPATH:+:${PYTHONPATH}}" \
		python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'BEST_GPU') or '')" 2>/dev/null)" || return 1

	[[ -z "${value}" || "${value}" == "None" ]] && return 1
	printf '%s' "${value}"
}

# Assign a variable from, in order: existing environment, the PORTABLE section of
# sysconfig.json, then the repo-relative default.
#
# An empty value counts as UNSET everywhere in this library. None of these
# variables has a meaningful empty value, and treating "" as set would strand a
# path at the filesystem root. Keep this consistent with the snapshot in
# portable_load_site_env.
_portable_default() {
	local name="$1" default="$2" from_config=""

	if [[ -n "${!name:-}" ]]; then
		export "${name}"
		return 0
	fi

	if from_config="$(_portable_setting "${name}")"; then
		printf -v "${name}" '%s' "${from_config}"
	else
		printf -v "${name}" '%s' "${default}"
	fi

	export "${name}"
}

# Resolve every managed path. Uniformly via _portable_default, so the documented
# precedence (environment > site.env > PORTABLE sysconfig > repo-relative
# default) holds for every variable rather than only for some of them.
portable_set_paths() {
	local cache_base="${SLURM_TMPDIR:-${PROJECT_DIR}/.cache}"

	_portable_default HF_HOME "${PROJECT_DIR}/.cache/huggingface"
	_portable_default HF_HUB_CACHE "${HF_HOME}"
	_portable_default TRANSFORMERS_CACHE "${HF_HUB_CACHE}"
	_portable_default HUGGINGFACE_HUB_CACHE "${HF_HUB_CACHE}"
	_portable_default HF_DATASETS_CACHE "${HF_HUB_CACHE}"
	_portable_default HF_HUB_DISABLE_XET "1"

	_portable_default SIF_FILE "${PROJECT_DIR}/containers/llamafactory.sif"
	_portable_default VENV_LLAMAFACTORY "${PROJECT_DIR}/.venv"
	_portable_default APPTAINER_OVERLAY "${PROJECT_DIR}/apptainer/overlay.img"

	_portable_default SCANNET_H5_DIR "${PROJECT_DIR}/data/h5/ScanNet_h5/scans"
	_portable_default SPATIALSSRL_H5_DIR "${PROJECT_DIR}/data/h5/Spatial-SSRL_images_h5"
	_portable_default THINKER10K_H5_DIR "${PROJECT_DIR}/data/h5/3DThinker10K_images_h5"
	_portable_default MEDIA_DIR "${PROJECT_DIR}/data/h5/ScanNet_h5"

	# Caches prefer $SLURM_TMPDIR: node-local scratch is much faster than shared
	# storage and the scheduler reaps it. They are deliberately absent from
	# sysconfig.json, because the right value is only known at runtime — but they
	# still route through _portable_default so an operator CAN pin them.
	_portable_default TRITON_CACHE_DIR "${cache_base}/.triton_cache"
	_portable_default TORCH_EXTENSIONS_DIR "${cache_base}/torch_extensions"
	_portable_default PYTORCH_KERNEL_CACHE_PATH "${cache_base}/torch/kernels"
	_portable_default MPLCONFIGDIR "${cache_base}/matplotlib"
	_portable_default FLASHINFER_WORKSPACE_BASE "${cache_base}/flashinfer"
	_portable_default WANDB_DIR "${PROJECT_DIR}/wandb"
	_portable_default WANDB_CACHE_DIR "${cache_base}/.cache/wandb"

	# L40S on Killarney is Ada (8.9); its sysconfig section wrongly says h100.
	local arch_default="9.0"
	if [[ "${CLUSTER}" == "KILLARNEY" ]]; then
		arch_default="8.9"
	elif [[ "$(_portable_cluster_best_gpu || echo h100)" != "h100" ]]; then
		arch_default="8.0"
	fi
	_portable_default TORCH_CUDA_ARCH_LIST "${arch_default}"
}

# Compute nodes have no network, so every hub client must be told up front.
# Keep this list in sync with PORTABLE_MANAGED_VARS.
portable_set_offline() {
	export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
	export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
	export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
	export WANDB_MODE="${WANDB_MODE:-offline}"
	export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
	export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"
	export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
	export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
}

portable_init() {
	portable_resolve_project_dir || return 1
	# site.env loads BEFORE detection so it can pin CLUSTER and RUNNING_MODE.
	# Detection would otherwise have already set them, and the loader's snapshot
	# would see them as pre-set and restore them over the operator's choice.
	portable_load_site_env || return 1
	portable_detect_cluster
	portable_set_paths
	portable_set_offline

	# This tree's src must win over any editable install pointing elsewhere.
	export PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
}

_portable_pf_row() {
	local status="$1" label="$2" path="$3"
	printf '%-4s %-24s %s\n' "${status}" "${label}" "${path}"
}

# A path still holding a ${...} token was built from an unset variable. Treating
# it as a real path would resolve it somewhere under the filesystem root, so it is
# never "present" — and never merely "missing" either, since the fix is to define
# the variable rather than to stage a file.
_portable_pf_unresolved() {
	[[ "$1" == *'${'* ]]
}

# Required artifact: absence, or an unresolved token, fails the job.
_portable_pf_require() {
	local label="$1" path="$2"
	if _portable_pf_unresolved "${path}"; then
		# Show the offending value: the operator needs it to find the culprit.
		_portable_pf_row "BAD" "${label}" "unexpanded token in ${path}"
		_PORTABLE_PF_RC=1
	elif [[ -n "${path}" && -e "${path}" ]]; then
		_portable_pf_row "OK" "${label}" "${path}"
	else
		_portable_pf_row "MISS" "${label}" "${path:-<unset>}"
		_PORTABLE_PF_RC=1
	fi
}

# Optional artifact: absence only warns. An unresolved token still fails, because
# it is a configuration error rather than an absent file.
_portable_pf_optional() {
	local label="$1" path="$2"
	if _portable_pf_unresolved "${path}"; then
		_portable_pf_row "BAD" "${label}" "unexpanded token in ${path}"
		_PORTABLE_PF_RC=1
	elif [[ -n "${path}" && -e "${path}" ]]; then
		_portable_pf_row "OK" "${label}" "${path}"
	else
		_portable_pf_row "WARN" "${label}" "${path:-<unset>}"
	fi
}

# Validate every resolved path before the job consumes GPU time. Compute nodes
# have no network, so a missing artifact can never be fetched at runtime.
portable_preflight() {
	local _PORTABLE_PF_RC=0

	echo "=== portable preflight ==="
	echo "PROJECT_DIR:  ${PROJECT_DIR}"
	echo "CLUSTER:      ${CLUSTER}"
	echo "RUNNING_MODE: ${RUNNING_MODE}"
	echo "---"

	_portable_pf_require "project_root" "${PROJECT_DIR}/setup.py"
	_portable_pf_require "llamafactory_src" "${PROJECT_DIR}/src/llamafactory"
	_portable_pf_require "deepspeed_config" "${PROJECT_DIR}/examples/deepspeed/ds_z2_config.json"
	_portable_pf_require "hf_cache" "${HF_HUB_CACHE}"
	_portable_pf_require "scannet_h5" "${SCANNET_H5_DIR}"
	_portable_pf_require "spatialssrl_h5" "${SPATIALSSRL_H5_DIR}"
	_portable_pf_require "thinker10k_h5" "${THINKER10K_H5_DIR}"
	_portable_pf_require "dataset_registry" "${PROJECT_DIR}/data/annotations/dataset_info.json"

	if [[ -n "${PORTABLE_YAML_FILE:-}" ]]; then
		_portable_pf_require "train_yaml" "${PORTABLE_YAML_FILE}"
	fi

	case "${RUNNING_MODE}" in
	APPTAINER | SHELL)
		_portable_pf_require "sif_image" "${SIF_FILE}"
		_portable_pf_optional "apptainer_overlay" "${APPTAINER_OVERLAY}"
		;;
	VENV)
		_portable_pf_require "venv_activate" "${VENV_LLAMAFACTORY}/bin/activate"
		;;
	esac

	_portable_pf_optional "media_dir" "${MEDIA_DIR}"

	echo "---"
	if [[ "${_PORTABLE_PF_RC}" -eq 0 ]]; then
		echo "preflight: PASS"
	else
		echo "preflight: FAIL — stage the MISS entries above (or set overrides in scripts/site.env);"
		echo "           a BAD entry means an unset variable, not a missing file"
		echo "           see docs/superpowers/specs/2026-09-05-portable-slurm-wrapper-design.md"
	fi
	echo "=========================="

	return "${_PORTABLE_PF_RC}"
}

# Create one repo-relative symlink.
#
# Returns 0 for success and for a deliberate skip (an unset or absent target --
# PORTABLE_SRC_* are all optional). Returns 1 only for a genuine failure, so
# portable_stage_assets can report that staging did not fully succeed instead of
# claiming a link it never made.
_portable_link() {
	local link="$1" target="$2"

	[[ -z "${target}" ]] && return 0

	if [[ ! -e "${target}" ]]; then
		echo "portable_env: stage target missing, skipping: ${target}" >&2
		return 0
	fi

	if [[ -L "${link}" ]]; then
		[[ "$(readlink -f "${link}")" == "$(readlink -f "${target}")" ]] && return 0
		rm -f "${link}" || {
			echo "portable_env: cannot replace stale link: ${link}" >&2
			return 1
		}
	elif [[ -e "${link}" ]]; then
		# Never clobber real data with a link.
		echo "portable_env: refusing to replace existing path: ${link}" >&2
		return 1
	fi

	if ! mkdir -p "$(dirname "${link}")"; then
		echo "portable_env: cannot create parent of ${link}" >&2
		return 1
	fi

	if ! ln -s "${target}" "${link}"; then
		echo "portable_env: cannot link ${link} -> ${target}" >&2
		return 1
	fi

	echo "portable_env: linked ${link} -> ${target}" >&2
}

# Create the repo-relative staging tree and regenerate the portable registry.
# Idempotent. Run explicitly with PORTABLE_STAGE=1; never called during training.
portable_stage_assets() {
	local rc=0

	if ! mkdir -p "${PROJECT_DIR}/data/h5" "${PROJECT_DIR}/data/annotations" \
		"${PROJECT_DIR}/containers" "${PROJECT_DIR}/.cache"; then
		echo "portable_env: cannot create the staging tree under ${PROJECT_DIR}" >&2
		return 1
	fi

	# Keep going after a failure so one bad link does not hide the rest, but
	# remember it: reporting success here would send a half-staged tree to a
	# compute node that cannot fetch what is missing.
	_portable_link "${PROJECT_DIR}/.cache/huggingface" "${PORTABLE_SRC_HF_CACHE:-}" || rc=1
	_portable_link "${PROJECT_DIR}/containers/llamafactory.sif" "${PORTABLE_SRC_SIF:-}" || rc=1
	_portable_link "${PROJECT_DIR}/data/h5/ScanNet_h5" "${PORTABLE_SRC_SCANNET_H5:-}" || rc=1
	_portable_link "${PROJECT_DIR}/data/h5/Spatial-SSRL_images_h5" "${PORTABLE_SRC_SPATIALSSRL_H5:-}" || rc=1
	_portable_link "${PROJECT_DIR}/data/h5/3DThinker10K_images_h5" "${PORTABLE_SRC_THINKER10K_H5:-}" || rc=1

	# Forward the site.env annotation redirects. The registry's recorded paths
	# belong to the original author and are unreadable for anyone else, so without
	# these two overrides Scene30k and SpatialSSRL_coldstart cannot be staged at
	# all -- and the design forbids editing data/dataset_info.json to fix it.
	local -a gen_args=(
		--source "${PROJECT_DIR}/data/dataset_info.json"
		--dest "${PROJECT_DIR}/data/annotations/dataset_info.json"
		--require "${PORTABLE_REQUIRED_DATASETS:-Scene30k,SpatialSSRL_coldstart,3DThinker10k}"
	)
	[[ -n "${PORTABLE_SRC_SCENE30K_ANNOTATION:-}" ]] &&
		gen_args+=(--override "Scene30k=${PORTABLE_SRC_SCENE30K_ANNOTATION}")
	[[ -n "${PORTABLE_SRC_SPATIALSSRL_ANNOTATION:-}" ]] &&
		gen_args+=(--override "SpatialSSRL_coldstart=${PORTABLE_SRC_SPATIALSSRL_ANNOTATION}")

	echo "portable_env: generating data/annotations/dataset_info.json" >&2
	python3 "${PROJECT_DIR}/scripts/make_portable_dataset_info.py" "${gen_args[@]}" || rc=1

	return "${rc}"
}
