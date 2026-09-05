#!/usr/bin/env bash
# Bash test harness for scripts/utils/portable_env.sh
# Run: bash scripts/tests/test_portable_env.sh
set -uo pipefail

FAILED=0
PASS_COUNT=0

assert_eq() {
	local expected="$1" actual="$2" msg="$3"
	if [[ "${expected}" == "${actual}" ]]; then
		PASS_COUNT=$((PASS_COUNT + 1))
		echo "PASS: ${msg}"
	else
		FAILED=1
		echo "FAIL: ${msg}"
		echo "  expected: '${expected}'"
		echo "  actual:   '${actual}'"
	fi
}

assert_rc() {
	local expected="$1" actual="$2" msg="$3"
	assert_eq "${expected}" "${actual}" "${msg}"
}

# Locate the repo under test from this test file's own location.
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "${TEST_DIR}/../.." && pwd -P)"
RESOLVER="${REPO}/scripts/utils/portable_env.sh"

echo "=== portable_env.sh tests (repo: ${REPO}) ==="

# --- Task 1: project dir resolution ---

# Sourcing from an unrelated cwd still finds the repo root.
actual="$(cd /tmp && source "${RESOLVER}" >/dev/null 2>&1 && portable_resolve_project_dir >/dev/null 2>&1 && echo "${PROJECT_DIR}")"
assert_eq "${REPO}" "${actual}" "resolves PROJECT_DIR from unrelated cwd"

# LFS_PROJECT_DIR overrides detection.
actual="$(LFS_PROJECT_DIR="${REPO}" bash -c 'source "$1" >/dev/null 2>&1; portable_resolve_project_dir >/dev/null 2>&1; echo "$PROJECT_DIR"' _ "${RESOLVER}")"
assert_eq "${REPO}" "${actual}" "LFS_PROJECT_DIR override is honoured"

# Trailing slash in the override is normalized away.
actual="$(LFS_PROJECT_DIR="${REPO}/" bash -c 'source "$1" >/dev/null 2>&1; portable_resolve_project_dir >/dev/null 2>&1; echo "$PROJECT_DIR"' _ "${RESOLVER}")"
assert_eq "${REPO}" "${actual}" "trailing slash normalized"

# A bogus override fails loudly instead of silently guessing.
# NOTE: this harness never enables errexit, so capture the status with `|| rc=$?`
# rather than a set +e / set -e dance. Turning errexit ON mid-script would abort
# the run on the first intentionally-nonzero command (cp, grep -c, readlink).
rc=0
LFS_PROJECT_DIR="/nonexistent/path/xyz" bash -c 'source "$1" >/dev/null 2>&1; portable_resolve_project_dir >/dev/null 2>&1' _ "${RESOLVER}" || rc=$?
assert_rc "1" "${rc}" "bogus LFS_PROJECT_DIR returns nonzero"

# An override that exists but is NOT a repo root must also fail: every later
# task derives its paths from PROJECT_DIR, so accepting a non-root would produce
# a full set of plausible-looking wrong paths.
NOT_ROOT="$(mktemp -d)"
rc=0
LFS_PROJECT_DIR="${NOT_ROOT}" bash -c 'source "$1" >/dev/null 2>&1; portable_resolve_project_dir >/dev/null 2>&1' _ "${RESOLVER}" || rc=$?
assert_rc "1" "${rc}" "existing non-root LFS_PROJECT_DIR returns nonzero"
rm -rf "${NOT_ROOT}"

# PROJECT_DIR must be exported, not merely set: the job body hands it to child
# processes and to `apptainer --env`. Read it back from a grandchild process.
actual="$(cd /tmp && source "${RESOLVER}" >/dev/null 2>&1 && portable_resolve_project_dir >/dev/null 2>&1 && bash -c 'printf %s "${PROJECT_DIR}"')"
assert_eq "${REPO}" "${actual}" "PROJECT_DIR is exported to child processes"

# RELOCATION TEST: a copy under a different directory name must resolve to itself.
RELOC="$(mktemp -d)/renamed-checkout"
mkdir -p "${RELOC}/scripts/utils" "${RELOC}/src/llamafactory"
cp "${RESOLVER}" "${RELOC}/scripts/utils/portable_env.sh"
touch "${RELOC}/setup.py"
actual="$(cd /tmp && source "${RELOC}/scripts/utils/portable_env.sh" >/dev/null 2>&1 && portable_resolve_project_dir >/dev/null 2>&1 && echo "${PROJECT_DIR}")"
assert_eq "$(cd "${RELOC}" && pwd -P)" "${actual}" "relocated+renamed checkout resolves to itself"
rm -rf "$(dirname "${RELOC}")"

# --- Task 3: path defaults, precedence, cluster detection ---

# Repo-relative defaults when nothing is set. CLUSTER=PORTABLE avoids the
# site-specific sysconfig sections.
read -r hf sif scannet arch <<<"$(cd /tmp && env -u HF_HOME -u SIF_FILE -u SCANNET_H5_DIR -u SLURM_TMPDIR \
	CLUSTER=PORTABLE RUNNING_MODE=SHELL PORTABLE_SKIP_SITE_ENV=1 \
	bash -c 'source "$1" >/dev/null 2>&1; portable_init >/dev/null 2>&1; echo "$HF_HOME $SIF_FILE $SCANNET_H5_DIR $TORCH_CUDA_ARCH_LIST"' _ "${RESOLVER}")"
assert_eq "${REPO}/.cache/huggingface" "${hf}" "HF_HOME defaults repo-relative"
assert_eq "${REPO}/containers/llamafactory.sif" "${sif}" "SIF_FILE defaults repo-relative"
assert_eq "${REPO}/data/h5/ScanNet_h5/scans" "${scannet}" "SCANNET_H5_DIR defaults repo-relative"
assert_eq "9.0" "${arch}" "TORCH_CUDA_ARCH_LIST is 9.0 for h100"

# Pre-set environment wins over defaults.
actual="$(cd /tmp && SCANNET_H5_DIR=/custom/scannet CLUSTER=PORTABLE RUNNING_MODE=SHELL PORTABLE_SKIP_SITE_ENV=1 \
	bash -c 'source "$1" >/dev/null 2>&1; portable_init >/dev/null 2>&1; echo "$SCANNET_H5_DIR"' _ "${RESOLVER}")"
assert_eq "/custom/scannet" "${actual}" "pre-set env overrides default"

# site.env is honoured, but loses to pre-set env.
SITE_TMP="$(mktemp -d)"
cat >"${SITE_TMP}/site.env" <<'SITEEOF'
export SPATIALSSRL_H5_DIR=/from/site/env
export THINKER10K_H5_DIR=/from/site/env/thinker
SITEEOF
actual="$(cd /tmp && PORTABLE_SITE_ENV="${SITE_TMP}/site.env" THINKER10K_H5_DIR=/from/real/env \
	CLUSTER=PORTABLE RUNNING_MODE=SHELL \
	bash -c 'source "$1" >/dev/null 2>&1; portable_init >/dev/null 2>&1; echo "$SPATIALSSRL_H5_DIR|$THINKER10K_H5_DIR"' _ "${RESOLVER}")"
assert_eq "/from/site/env|/from/real/env" "${actual}" "site.env applies; pre-set env still wins"
rm -rf "${SITE_TMP}"

# Killarney gets the Ada arch list, not h100.
actual="$(cd /tmp && CLUSTER=KILLARNEY RUNNING_MODE=SHELL PORTABLE_SKIP_SITE_ENV=1 \
	bash -c 'source "$1" >/dev/null 2>&1; portable_init >/dev/null 2>&1; echo "$TORCH_CUDA_ARCH_LIST"' _ "${RESOLVER}")"
assert_eq "8.9" "${actual}" "KILLARNEY forces arch 8.9"

# Caches go to SLURM_TMPDIR when the scheduler provides one.
actual="$(cd /tmp && SLURM_TMPDIR=/tmp/fake_slurm CLUSTER=PORTABLE RUNNING_MODE=SHELL PORTABLE_SKIP_SITE_ENV=1 \
	env -u TRITON_CACHE_DIR bash -c 'source "$1" >/dev/null 2>&1; portable_init >/dev/null 2>&1; echo "$TRITON_CACHE_DIR"' _ "${RESOLVER}")"
assert_eq "/tmp/fake_slurm/.triton_cache" "${actual}" "TRITON_CACHE_DIR prefers SLURM_TMPDIR"

# Offline flags are all set.
actual="$(cd /tmp && CLUSTER=PORTABLE RUNNING_MODE=SHELL PORTABLE_SKIP_SITE_ENV=1 \
	bash -c 'source "$1" >/dev/null 2>&1; portable_init >/dev/null 2>&1; echo "$HF_HUB_OFFLINE$TRANSFORMERS_OFFLINE$HF_DATASETS_OFFLINE$FORCE_TORCHRUN$DISABLE_VERSION_CHECK $WANDB_MODE"' _ "${RESOLVER}")"
assert_eq "11111 offline" "${actual}" "offline flags set"

# No resolved path contains an unexpanded token or a hardcoded username.
actual="$(cd /tmp && CLUSTER=PORTABLE RUNNING_MODE=SHELL PORTABLE_SKIP_SITE_ENV=1 \
	bash -c 'source "$1" >/dev/null 2>&1; portable_init >/dev/null 2>&1; echo "$HF_HOME $SIF_FILE $VENV_LLAMAFACTORY $MEDIA_DIR $APPTAINER_OVERLAY"' _ "${RESOLVER}" | grep -cE '\$\{|indrisch')"
assert_eq "0" "${actual}" "no unexpanded tokens and no hardcoded username"

echo "=== ${PASS_COUNT} passed, failed=${FAILED} ==="
exit "${FAILED}"
