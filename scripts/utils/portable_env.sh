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
