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
	bash -c 'source "$1" >/dev/null 2>&1; portable_init || exit 1; echo "$SCANNET_H5_DIR"' _ "${RESOLVER}" 2>/dev/null)"
assert_eq "/custom/scannet" "${actual}" "pre-set env overrides default"

# site.env is honoured, but loses to pre-set env.
SITE_TMP="$(mktemp -d)"
cat >"${SITE_TMP}/site.env" <<'SITEEOF'
export SPATIALSSRL_H5_DIR=/from/site/env
export THINKER10K_H5_DIR=/from/site/env/thinker
SITEEOF
actual="$(cd /tmp && PORTABLE_SITE_ENV="${SITE_TMP}/site.env" THINKER10K_H5_DIR=/from/real/env \
	CLUSTER=PORTABLE RUNNING_MODE=SHELL \
	bash -c 'source "$1" >/dev/null 2>&1; portable_init || exit 1; echo "$SPATIALSSRL_H5_DIR|$THINKER10K_H5_DIR"' _ "${RESOLVER}" 2>/dev/null)"
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
	bash -c 'source "$1" >/dev/null 2>&1; portable_init || exit 1; echo "$HF_HOME $SIF_FILE $VENV_LLAMAFACTORY $MEDIA_DIR $APPTAINER_OVERLAY"' _ "${RESOLVER}" 2>/dev/null | grep -cE '\$\{|indrisch')"
assert_eq "0" "${actual}" "no unexpanded tokens and no hardcoded username"

# portable_init succeeds. Without this, an assertion that merely echoes a
# variable can pass while portable_init is missing entirely.
rc=0
(cd /tmp && CLUSTER=PORTABLE RUNNING_MODE=SHELL PORTABLE_SKIP_SITE_ENV=1 \
	bash -c 'source "$1" >/dev/null 2>&1; portable_init >/dev/null 2>&1' _ "${RESOLVER}") || rc=$?
assert_rc "0" "${rc}" "portable_init returns 0"

# The sysconfig tier is really consulted, and it reads the PORTABLE section.
actual="$(cd /tmp && CLUSTER=PORTABLE PORTABLE_SKIP_SITE_ENV=1 \
	bash -c 'source "$1" >/dev/null 2>&1; portable_resolve_project_dir >/dev/null 2>&1; _portable_setting SIF_FILE' _ "${RESOLVER}")"
assert_eq "${REPO}/containers/llamafactory.sif" "${actual}" "_portable_setting reads expanded PORTABLE value"

# An unknown key is a miss, not an empty string, so _portable_default can fall
# through to its repo-relative default.
rc=0
(cd /tmp && CLUSTER=PORTABLE PORTABLE_SKIP_SITE_ENV=1 \
	bash -c 'source "$1" >/dev/null 2>&1; portable_resolve_project_dir >/dev/null 2>&1; _portable_setting NO_SUCH_KEY_XYZ' _ "${RESOLVER}" >/dev/null) || rc=$?
assert_rc "1" "${rc}" "_portable_setting returns nonzero for an unknown key"

# THE CENTRAL PORTABILITY GUARANTEE: on a legacy cluster whose sysconfig section
# is full of another user's absolute paths, none of them may leak in. Only the
# PORTABLE section is consulted for paths.
actual="$(cd /tmp && CLUSTER=TRILLIUM RUNNING_MODE=SHELL PORTABLE_SKIP_SITE_ENV=1 \
	bash -c 'source "$1" >/dev/null 2>&1; portable_init || exit 1; echo "$HF_HOME $HF_HUB_CACHE $SIF_FILE $VENV_LLAMAFACTORY $MEDIA_DIR $SCANNET_H5_DIR $TRITON_CACHE_DIR"' _ "${RESOLVER}" 2>/dev/null | grep -cE 'indrisch|def-wangcs')"
assert_eq "0" "${actual}" "legacy cluster sections never supply paths"

# site.env can pin CLUSTER, which requires it to load BEFORE detection.
CL_TMP="$(mktemp -d)"
printf 'export CLUSTER=KILLARNEY\n' >"${CL_TMP}/site.env"
actual="$(cd /tmp && env -u CLUSTER PORTABLE_SITE_ENV="${CL_TMP}/site.env" RUNNING_MODE=SHELL \
	bash -c 'source "$1" >/dev/null 2>&1; portable_init || exit 1; echo "$CLUSTER $TORCH_CUDA_ARCH_LIST"' _ "${RESOLVER}" 2>/dev/null)"
assert_eq "KILLARNEY 8.9" "${actual}" "site.env can pin CLUSTER before detection"
rm -rf "${CL_TMP}"

# A typo'd CLUSTER falls back to PORTABLE instead of running an unknown profile.
actual="$(cd /tmp && CLUSTER=TRILIUM RUNNING_MODE=SHELL PORTABLE_SKIP_SITE_ENV=1 \
	bash -c 'source "$1" >/dev/null 2>&1; portable_init 2>/dev/null || exit 1; echo "$CLUSTER"' _ "${RESOLVER}")"
assert_eq "PORTABLE" "${actual}" "unknown CLUSTER normalizes to PORTABLE"

# The Python hygiene vars are part of the managed set and get exported.
actual="$(cd /tmp && CLUSTER=PORTABLE RUNNING_MODE=SHELL PORTABLE_SKIP_SITE_ENV=1 \
	bash -c 'source "$1" >/dev/null 2>&1; portable_init || exit 1; echo "$PYTHONUNBUFFERED$PYTHONNOUSERSITE"' _ "${RESOLVER}" 2>/dev/null)"
assert_eq "11" "${actual}" "PYTHONUNBUFFERED and PYTHONNOUSERSITE are exported"

# TORCH_CUDA_ARCH_LIST is a pinnable location-style setting, not a hardcoded
# constant: site.env must be able to override the computed default.
ARCH_TMP="$(mktemp -d)"
printf 'export TORCH_CUDA_ARCH_LIST=8.6\n' >"${ARCH_TMP}/site.env"
actual="$(cd /tmp && env -u TORCH_CUDA_ARCH_LIST PORTABLE_SITE_ENV="${ARCH_TMP}/site.env" \
	CLUSTER=PORTABLE RUNNING_MODE=SHELL \
	bash -c 'source "$1" >/dev/null 2>&1; portable_init || exit 1; echo "$TORCH_CUDA_ARCH_LIST"' _ "${RESOLVER}" 2>/dev/null)"
assert_eq "8.6" "${actual}" "site.env can pin TORCH_CUDA_ARCH_LIST"
rm -rf "${ARCH_TMP}"

# A site.env that errors must abort the job, not leave it half-configured. Its
# exports before the failure have already applied; continuing would burn GPU time
# on a partially configured run.
BAD_TMP="$(mktemp -d)"
printf 'export SPATIALSSRL_H5_DIR=/from/bad/site\nfalse\n' >"${BAD_TMP}/site.env"
rc=0
(cd /tmp && PORTABLE_SITE_ENV="${BAD_TMP}/site.env" CLUSTER=PORTABLE RUNNING_MODE=SHELL \
	bash -c 'source "$1" >/dev/null 2>&1; portable_init >/dev/null 2>&1' _ "${RESOLVER}") || rc=$?
assert_rc "1" "${rc}" "a failing site.env aborts portable_init"
rm -rf "${BAD_TMP}"

# --- Task 4: preflight ---

# Everything missing under a bogus root: nonzero exit, MISS lines present.
PF_TMP="$(mktemp -d)/pfroot"
mkdir -p "${PF_TMP}/scripts/utils" "${PF_TMP}/src/llamafactory"
cp "${RESOLVER}" "${PF_TMP}/scripts/utils/portable_env.sh"
touch "${PF_TMP}/setup.py"
rc=0
out="$(cd /tmp && CLUSTER=PORTABLE RUNNING_MODE=VENV PORTABLE_SKIP_SITE_ENV=1 \
	bash -c 'source "$1" >/dev/null 2>&1; portable_init >/dev/null 2>&1; portable_preflight 2>&1' _ "${PF_TMP}/scripts/utils/portable_env.sh")" || rc=$?
assert_rc "1" "${rc}" "preflight fails when artifacts are missing"
assert_eq "yes" "$(grep -q 'MISS' <<<"${out}" && echo yes || echo no)" "preflight reports MISS lines"

# The real repo has the YAML and DeepSpeed config, so those specific rows pass.
out="$(cd /tmp && CLUSTER=PORTABLE RUNNING_MODE=VENV PORTABLE_SKIP_SITE_ENV=1 \
	PORTABLE_YAML_FILE="${REPO}/examples/train_lora/portable_qwen2_5vl_lora_sft_CoT_traineval.yaml" \
	bash -c 'source "$1" >/dev/null 2>&1; portable_init >/dev/null 2>&1; portable_preflight 2>&1' _ "${RESOLVER}")"
assert_eq "yes" "$(grep -qE '^OK +deepspeed_config' <<<"${out}" && echo yes || echo no)" "preflight finds deepspeed config"

# Require the table to have rendered. Without this, the token assertion below
# passes even when portable_preflight does not exist, because `out` is then just
# "command not found" and grep -c reports 0 on it.
assert_eq "yes" "$(grep -q '=== portable preflight ===' <<<"${out}" && echo yes || echo no)" "preflight renders its table"

# A correctly resolved environment leaks no token into the output.
assert_eq "0" "$(grep -cE '\$\{' <<<"${out}")" "preflight output has no unexpanded tokens"
rm -rf "$(dirname "${PF_TMP}")"

# A path carrying a literal ${...} must be refused, not reported OK because some
# ancestor happens to exist. _portable_default screens tokens out of sysconfig
# values only, so an environment or site.env value can still carry one.
TOKEN_TMP="$(mktemp -d)/token-root"
mkdir -p "${TOKEN_TMP}/scripts/utils" "${TOKEN_TMP}/src/llamafactory" \
	"${TOKEN_TMP}/examples/deepspeed" "${TOKEN_TMP}/data/annotations" \
	"${TOKEN_TMP}/.venv/bin" "${TOKEN_TMP}/\${NOT_SET_ANYWHERE}" \
	"${TOKEN_TMP}/\${OPTIONAL_NOT_SET}"
cp "${RESOLVER}" "${TOKEN_TMP}/scripts/utils/portable_env.sh"
touch "${TOKEN_TMP}/setup.py" "${TOKEN_TMP}/examples/deepspeed/ds_z2_config.json" \
	"${TOKEN_TMP}/data/annotations/dataset_info.json" "${TOKEN_TMP}/.venv/bin/activate"
rc=0
out="$(cd "${TOKEN_TMP}" && env -u PORTABLE_YAML_FILE HF_HUB_CACHE='${NOT_SET_ANYWHERE}' \
	SCANNET_H5_DIR="${TOKEN_TMP}" SPATIALSSRL_H5_DIR="${TOKEN_TMP}" THINKER10K_H5_DIR="${TOKEN_TMP}" \
	MEDIA_DIR="${TOKEN_TMP}" VENV_LLAMAFACTORY="${TOKEN_TMP}/.venv" \
	CLUSTER=PORTABLE RUNNING_MODE=VENV PORTABLE_SKIP_SITE_ENV=1 \
	bash -c 'source "$1" >/dev/null 2>&1; portable_init >/dev/null 2>&1; portable_preflight 2>&1' _ \
	"${TOKEN_TMP}/scripts/utils/portable_env.sh")" || rc=$?
assert_rc "1" "${rc}" "an unexpanded token fails preflight"
assert_eq "yes" "$(grep -qE '^BAD +hf_cache' <<<"${out}" && echo yes || echo no)" "an unexpanded token is reported as BAD"

rc=0
(cd "${TOKEN_TMP}" && env -u PORTABLE_YAML_FILE HF_HUB_CACHE="${TOKEN_TMP}" \
	SCANNET_H5_DIR="${TOKEN_TMP}" SPATIALSSRL_H5_DIR="${TOKEN_TMP}" THINKER10K_H5_DIR="${TOKEN_TMP}" \
	MEDIA_DIR='${OPTIONAL_NOT_SET}' VENV_LLAMAFACTORY="${TOKEN_TMP}/.venv" \
	CLUSTER=PORTABLE RUNNING_MODE=VENV PORTABLE_SKIP_SITE_ENV=1 \
	bash -c 'source "$1" >/dev/null 2>&1; portable_init >/dev/null 2>&1; portable_preflight >/dev/null 2>&1' _ \
	"${TOKEN_TMP}/scripts/utils/portable_env.sh") || rc=$?
assert_rc "1" "${rc}" "an unexpanded optional token fails preflight"
rm -rf "$(dirname "${TOKEN_TMP}")"

# --- _portable_link / portable_stage_assets ---------------------------------

# Earlier assertions source the resolver only inside subshells. Load it in this
# harness process before exercising its private staging helpers directly.
source "${RESOLVER}" || exit 1

stage_root="$(mktemp -d)"
mkdir -p "${stage_root}/src"
echo payload >"${stage_root}/src/file.bin"

rc=0
_portable_link "${stage_root}/link_a" "${stage_root}/src/file.bin" 2>/dev/null || rc=$?
assert_rc 0 "${rc}" "a fresh link succeeds"
assert_eq "payload" "$(cat "${stage_root}/link_a")" "the link resolves to the target"

rc=0
_portable_link "${stage_root}/link_a" "${stage_root}/src/file.bin" 2>/dev/null || rc=$?
assert_rc 0 "${rc}" "relinking the same target is idempotent"

echo other >"${stage_root}/src/other.bin"
rc=0
_portable_link "${stage_root}/link_a" "${stage_root}/src/other.bin" 2>/dev/null || rc=$?
assert_rc 0 "${rc}" "a stale link is repointed"
assert_eq "other" "$(cat "${stage_root}/link_a")" "the repointed link resolves to the new target"

# An unset or absent target is a deliberate skip: every PORTABLE_SRC_* is optional.
rc=0
_portable_link "${stage_root}/link_unset" "" 2>/dev/null || rc=$?
assert_rc 0 "${rc}" "an empty target is skipped, not failed"
assert_eq "absent" "$([[ -e "${stage_root}/link_unset" ]] && echo present || echo absent)" \
	"no link is made for an empty target"

rc=0
_portable_link "${stage_root}/link_missing" "${stage_root}/nope" 2>/dev/null || rc=$?
assert_rc 0 "${rc}" "an absent target is skipped, not failed"

# Real data in the link's place is never destroyed.
mkdir -p "${stage_root}/occupied"
echo precious >"${stage_root}/occupied/keep.txt"
rc=0
_portable_link "${stage_root}/occupied" "${stage_root}/src/file.bin" 2>/dev/null || rc=$?
assert_rc 1 "${rc}" "a real directory in the way is a failure, not a skip"
assert_eq "precious" "$(cat "${stage_root}/occupied/keep.txt")" "the real directory survives"

# A failing link must make staging report failure rather than a false success.
rc=0
(
	PROJECT_DIR="${stage_root}/proj"
	mkdir -p "${PROJECT_DIR}/scripts" "${PROJECT_DIR}/data"
	echo '{}' >"${PROJECT_DIR}/data/dataset_info.json"
	# A no-op stand-in so this assertion isolates the shell's rc accumulation from
	# the generator. Valid Python, since it is run as `python3 <file>`.
	echo 'pass' >"${PROJECT_DIR}/scripts/make_portable_dataset_info.py"
	# Occupy the SIF link path with a real directory so exactly one link fails.
	mkdir -p "${PROJECT_DIR}/containers/llamafactory.sif"
	PORTABLE_SRC_SIF="${stage_root}/src/file.bin"
	portable_stage_assets >/dev/null 2>&1
) || rc=$?
assert_rc 1 "${rc}" "one failing link makes portable_stage_assets return non-zero"

rm -rf "${stage_root}"

echo "=== ${PASS_COUNT} passed, failed=${FAILED} ==="
exit "${FAILED}"
