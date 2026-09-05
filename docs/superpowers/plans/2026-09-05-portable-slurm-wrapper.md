# Portable Repo-Relative SLURM Wrapper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a SLURM wrapper for the `qwen2_5vl_lora_sft_CoT_traineval` experiment that resolves every path relative to the repository root, so the checkout can be moved, renamed, or copied to another AllianceCan cluster and still run offline.

**Architecture:** A sourced bash resolver (`scripts/utils/portable_env.sh`) is the single component that turns a machine into a set of paths. A thin `#SBATCH` wrapper delegates to a portable body that sources the resolver, runs a preflight check, then dispatches to Apptainer / venv / shell. A `${PROJECT_DIR}`-aware `PORTABLE` section in `sysconfig.json` and a generated `data/annotations/dataset_info.json` remove the last absolute paths.

**Tech Stack:** bash 4+, SLURM (`sbatch`), Apptainer, Python 3.9+ (stdlib only: `json`, `os`, `argparse`, `pathlib`), pytest, LLaMA-Factory `llamafactory-cli train`.

**Spec:** `docs/superpowers/specs/2026-09-05-portable-slurm-wrapper-design.md`

## Global Constraints

- Do not modify existing `trillium_*`, `killarney_*`, `nibi_*`, `rorqual_*`, or unprefixed `slurm_*` scripts. All new behaviour goes in new files.
- Do not modify `data/dataset_info.json`. The portable registry is generated at `data/annotations/dataset_info.json`.
- New `.py` files under `scripts/`, `src/`, `tests/`, `tests_v1/` MUST have a first line containing all of `Copyright`, `2025`, `LlamaFactory` (enforced by `tests/check_license.py`).
- Ruff config (`pyproject.toml`): `line-length = 119`, `quote-style = "double"`, `target-version = "py39"`, isort `known-first-party = ["llamafactory"]`, `lines-after-imports = 2`, pydocstyle `convention = "google"`.
- Python floor is 3.9: no `match`, no `X | Y` runtime unions, no `dict[str, str]` in runtime-evaluated annotations without `from __future__ import annotations`.
- `shellcheck` is NOT installed in this environment. Shell verification uses `bash -n` only.
- `tests/check_license.py` asserts on the FIRST file it finds without a header and aborts, so
  it cannot be used as a pass/fail gate here. Two pre-existing offenders exist:
  `scripts/assign_question_ids.py` and `tests/hparams/test_resume_from_adapter.py`. Do not fix
  either. Verify new files individually instead, e.g.
  `head -1 <file> | grep -q Copyright && head -1 <file> | grep -q 2025 && head -1 <file> | grep -q LlamaFactory`.
- Never run GPU training on a login node. GPU work goes through `sbatch`.
- Offline env vars must be set exactly: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `WANDB_MODE=offline`, `DISABLE_VERSION_CHECK=1`, `FORCE_TORCHRUN=1`.
- Repo root sentinel: a directory is the repo root if it contains BOTH `setup.py` and `src/llamafactory`.
- Commit after every task. Do not commit to `main` (pre-commit `no-commit-to-branch` blocks it); work on `training_improvement`.

### Deviation from the spec, approved as part of this plan

The spec's risk section said "the plan adds no new `.py` files". This plan adds exactly one, `scripts/make_portable_dataset_info.py`, because rewriting JSON registry paths inside a bash heredoc would be untestable. It carries the required license header and is covered by unit tests, so `make license` and `make quality` do not regress.

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/utils/portable_env.sh` | NEW. Sourced resolver: repo root, `site.env`, cluster detection, repo-relative path defaults, `portable_preflight`, `portable_stage_assets`. Knows nothing about any experiment. |
| `scripts/site.env.example` | NEW. Documented override template; user copies to `scripts/site.env`. |
| `scripts/tests/test_portable_env.sh` | NEW. Bash test harness for the resolver. Not run by `make test` (keeps CI cross-platform). |
| `scripts/make_portable_dataset_info.py` | NEW. Rewrites `file_name` values relative to a new `dataset_dir`; optionally creates symlinks. |
| `tests/scripts/test_make_portable_dataset_info.py` | NEW. Unit tests for the rewriter. Pure stdlib, runs in CI. |
| `scripts/sysconfig.json` | MODIFY. Add a `PORTABLE` section using `${PROJECT_DIR}` tokens. |
| `scripts/sysconfigtool.py` | MODIFY. Expand `${VAR}` tokens in `read()` / `read_all()`. |
| `examples/train_lora/portable_qwen2_5vl_lora_sft_CoT_traineval.yaml` | NEW. No absolute paths. |
| `models/qwen2_5vl_lora_sft_CoT/portable_slurm_qwen2_5vl_lora_sft_CoT_traineval.sh` | NEW. `#SBATCH` header + `exec` the body. |
| `models/qwen2_5vl_lora_sft_CoT/portable_body_qwen2_5vl_lora_sft_CoT_traineval.sh` | NEW. Sources resolver, preflight, dispatches Apptainer / venv / shell. |
| `.gitignore` | MODIFY. Ignore `scripts/site.env`, `containers/`, `data/h5/`, `data/annotations/`. |
| `README_portable.md` section in `data/README.md` | MODIFY. Document staging and submission. |

---

### Task 1: Repo-root resolution + bash test harness

The single most important fix: the existing wrapper infers the repo root by matching the literal string `LLaMA-Factory-LFS` in `$PWD`, so renaming the checkout breaks it.

**Files:**
- Create: `scripts/utils/portable_env.sh`
- Test: `scripts/tests/test_portable_env.sh`

**Interfaces:**
- Consumes: nothing.
- Produces: `portable_resolve_project_dir()` — sets and exports `PROJECT_DIR` (absolute, symlink-resolved, no trailing slash); returns 0 on success, 1 if no repo root can be found. Honours `LFS_PROJECT_DIR` override. Later tasks add functions to this same file.

- [ ] **Step 1: Write the failing test**

Create `scripts/tests/test_portable_env.sh`:

```bash
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

echo "=== ${PASS_COUNT} passed, failed=${FAILED} ==="
exit "${FAILED}"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `bash scripts/tests/test_portable_env.sh`

Expected: FAIL. Every assertion fails because `scripts/utils/portable_env.sh` does not exist, so `source` errors and `portable_resolve_project_dir` is undefined. Final line shows `failed=1`, exit code 1.

- [ ] **Step 3: Write the minimal implementation**

Create `scripts/utils/portable_env.sh`:

```bash
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `bash scripts/tests/test_portable_env.sh`

Expected: PASS — `7 passed, failed=0`, exit code 0.

Also run: `bash -n scripts/utils/portable_env.sh && bash -n scripts/tests/test_portable_env.sh`
Expected: no output, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/utils/portable_env.sh scripts/tests/test_portable_env.sh
git commit -m "feat(scripts): add portable repo-root resolver with relocation test"
```

---

### Task 2: `${VAR}` expansion in sysconfigtool + PORTABLE section

**Files:**
- Modify: `scripts/sysconfigtool.py` (add expansion to `read` and `read_all`)
- Modify: `scripts/sysconfig.json` (append a `PORTABLE` section)
- Test: `tests/scripts/test_make_portable_dataset_info.py` is Task 5; this task's test is `tests/scripts/test_sysconfigtool.py`
- Create: `tests/scripts/test_sysconfigtool.py`

**Interfaces:**
- Consumes: `PROJECT_DIR` from Task 1 (read from the environment at call time).
- Produces: `sysconfigtool.read(system, key)` and `sysconfigtool.read_all(system)` now expand `${VAR}` tokens against `os.environ`. Unset tokens are left verbatim. `sysconfigtool.write()` is unchanged and stores raw values.

- [ ] **Step 1: Write the failing test**

Create `tests/scripts/test_sysconfigtool.py`:

```python
# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import sysconfigtool  # noqa: E402


def test_expands_project_dir_token(monkeypatch):
    monkeypatch.setenv("PROJECT_DIR", "/tmp/anywhere")
    assert sysconfigtool.expand_value("${PROJECT_DIR}/containers/x.sif") == "/tmp/anywhere/containers/x.sif"


def test_leaves_unset_token_verbatim(monkeypatch):
    monkeypatch.delenv("PORTABLE_UNSET_TOKEN", raising=False)
    assert sysconfigtool.expand_value("${PORTABLE_UNSET_TOKEN}/x") == "${PORTABLE_UNSET_TOKEN}/x"


def test_non_string_values_pass_through():
    assert sysconfigtool.expand_value(7) == 7
    assert sysconfigtool.expand_value(None) is None


def test_read_expands_portable_section(monkeypatch):
    monkeypatch.setenv("PROJECT_DIR", "/tmp/anywhere")
    value = sysconfigtool.read("PORTABLE", "SIF_FILE")
    assert value == "/tmp/anywhere/containers/llamafactory.sif"


def test_read_all_expands_every_value(monkeypatch):
    monkeypatch.setenv("PROJECT_DIR", "/tmp/anywhere")
    values = sysconfigtool.read_all("PORTABLE")
    assert values
    assert not any("${PROJECT_DIR}" in str(v) for v in values.values())


def test_existing_sections_unchanged(monkeypatch):
    monkeypatch.setenv("PROJECT_DIR", "/tmp/anywhere")
    assert sysconfigtool.read("TRILLIUM", "BEST_GPU") == "h100"
    assert sysconfigtool.read("TRILLIUM", "HF_HOME") == "/scratch/indrisch/huggingface/hub"


def test_missing_system_returns_empty():
    assert sysconfigtool.read_all("NO_SUCH_CLUSTER") == {}
    assert sysconfigtool.read("NO_SUCH_CLUSTER", "HF_HOME") is None


@pytest.mark.parametrize("key", ["HF_HOME", "SIF_FILE", "VENV_LLAMAFACTORY", "MEDIA_DIR"])
def test_portable_keys_present(key, monkeypatch):
    monkeypatch.setenv("PROJECT_DIR", "/tmp/anywhere")
    assert sysconfigtool.read("PORTABLE", key) is not None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `CUDA_VISIBLE_DEVICES= python -m pytest tests/scripts/test_sysconfigtool.py -v`

Expected: FAIL — `AttributeError: module 'sysconfigtool' has no attribute 'expand_value'`, and the `PORTABLE` tests fail because the section does not exist.

- [ ] **Step 3: Write the minimal implementation**

Replace the whole of `scripts/sysconfigtool.py` with:

```python
# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Read and write per-cluster settings from sysconfig.json.

Values may contain ``${VAR}`` tokens, which are expanded against the process
environment on read. This lets a section express repo-relative paths such as
``${PROJECT_DIR}/containers/llamafactory.sif``. Unset tokens are left verbatim
so a caller can detect and report them.
"""

import json
import os
import re


# Get the directory where this script is located.
_script_dir = os.path.dirname(os.path.abspath(__file__))
# The JSON file is in the same directory.
_config_file = os.path.join(_script_dir, "sysconfig.json")

_TOKEN_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _load_config():
    """Load the configuration from sysconfig.json."""
    if not os.path.exists(_config_file):
        return {}

    with open(_config_file, encoding="utf-8") as f:
        return json.load(f)


def _save_config(data):
    """Save the configuration to sysconfig.json."""
    with open(_config_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def expand_value(value):
    """Expand ``${VAR}`` tokens in a value against the environment.

    Non-string values are returned unchanged. Tokens whose variable is unset are
    left verbatim rather than replaced with an empty string, so that a missing
    ``PROJECT_DIR`` surfaces as a visible ``${PROJECT_DIR}`` instead of a path
    that silently starts at the filesystem root.

    Args:
        value: The raw configuration value.

    Returns:
        The value with known tokens expanded.
    """
    if not isinstance(value, str):
        return value

    def _sub(match):
        return os.environ.get(match.group(1), match.group(0))

    return _TOKEN_RE.sub(_sub, value)


def read(system, key):
    """Read a value from the sysconfig.json file.

    Args:
        system (str): The system name (e.g. "RORQUAL", "PORTABLE").
        key (str): The configuration key (e.g. "HF_HOME").

    Returns:
        The expanded value of the configuration key, or None if not found.
    """
    config = _load_config()
    return expand_value(config.get(system, {}).get(key))


def read_all(system):
    """Read all configuration key-value pairs for a system.

    Args:
        system (str): The system name (e.g. "RORQUAL", "PORTABLE").

    Returns:
        dict: Mapping of configuration keys to expanded values for the system,
            or an empty dict if the system is not found.
    """
    config = _load_config()
    return {key: expand_value(value) for key, value in config.get(system, {}).items()}


def write(system, key, value):
    """Write a raw value to the sysconfig.json file.

    The value is stored verbatim; ``${VAR}`` tokens are preserved so they expand
    on read.

    Args:
        system (str): The system name (e.g. "RORQUAL").
        key (str): The configuration key (e.g. "HF_HOME").
        value: The value to write.
    """
    config = _load_config()
    if system not in config:
        config[system] = {}

    config[system][key] = value
    _save_config(config)
```

- [ ] **Step 4: Add the PORTABLE section to `scripts/sysconfig.json`**

The file currently ends with the `TAMIA` block followed by `}`. Change the closing of `TAMIA` from `    }` to `    },` and append a `PORTABLE` block before the final `}`:

```json
    "PORTABLE": {
        "HF_HOME": "${PROJECT_DIR}/.cache/huggingface",
        "HF_HUB_CACHE": "${PROJECT_DIR}/.cache/huggingface",
        "HF_HUB_DISABLE_XET": "1",
        "TRITON_CACHE_DIR": "${PROJECT_DIR}/.cache/triton",
        "TORCH_EXTENSIONS_DIR": "${PROJECT_DIR}/.cache/torch_extensions",
        "BEST_GPU": "h100",
        "FLASHINFER_WORKSPACE_BASE": "${PROJECT_DIR}/.cache/flashinfer",
        "LLAMAFACTORY_HOME": "${PROJECT_DIR}",
        "SIF_FILE": "${PROJECT_DIR}/containers/llamafactory.sif",
        "MEDIA_DIR": "${PROJECT_DIR}/data/h5/ScanNet_h5",
        "VENV_LLAMAFACTORY": "${PROJECT_DIR}/.venv",
        "SCANNET_H5_DIR": "${PROJECT_DIR}/data/h5/ScanNet_h5/scans",
        "SPATIALSSRL_H5_DIR": "${PROJECT_DIR}/data/h5/Spatial-SSRL_images_h5",
        "THINKER10K_H5_DIR": "${PROJECT_DIR}/data/h5/3DThinker10K_images_h5"
    }
```

Note `MEDIA_DIR` is spelled uppercase here (unlike `media_dir` in the legacy sections) so `get_sysconfig_settings`-style uppercasing is a no-op.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES= python -m pytest tests/scripts/test_sysconfigtool.py -v`
Expected: PASS — 11 passed (7 tests + 4 parametrized cases).

Run: `python -c "import json; json.load(open('scripts/sysconfig.json'))" && echo JSON_OK`
Expected: `JSON_OK`

Run: `ruff check scripts/sysconfigtool.py tests/scripts/test_sysconfigtool.py && ruff format --check scripts/sysconfigtool.py tests/scripts/test_sysconfigtool.py`
Expected: `All checks passed!` and `2 files already formatted`.

Verify the license header on the new file directly (`check_license.py` aborts on a
pre-existing offender before it reaches ours):

```bash
for kw in Copyright 2025 LlamaFactory; do
  head -1 tests/scripts/test_sysconfigtool.py | grep -q "$kw" || echo "MISSING $kw"
done; echo LICENSE_HEADER_OK
```
Expected: `LICENSE_HEADER_OK` with no `MISSING` lines.

- [ ] **Step 6: Commit**

```bash
git add scripts/sysconfigtool.py scripts/sysconfig.json tests/scripts/test_sysconfigtool.py
git commit -m "feat(scripts): expand \${VAR} in sysconfig reads and add PORTABLE section"
```

---

### Task 3: Repo-relative path defaults, `site.env`, cluster detection

**Files:**
- Modify: `scripts/utils/portable_env.sh` (append functions)
- Create: `scripts/site.env.example`
- Test: `scripts/tests/test_portable_env.sh` (append assertions)

**Interfaces:**
- Consumes: `portable_resolve_project_dir()` from Task 1; `sysconfigtool.read_all` from Task 2.
- Produces:
  - `portable_detect_cluster()` — exports `CLUSTER` (one of `RORQUAL`, `TRILLIUM`, `KILLARNEY`, `TAMIA`, `NIBI`, `PORTABLE`) and `RUNNING_MODE` (`APPTAINER`, `VENV`, or `SHELL`), never overwriting values already set.
  - `portable_set_paths()` — exports `HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`, `HUGGINGFACE_HUB_CACHE`, `HF_DATASETS_CACHE`, `SIF_FILE`, `APPTAINER_OVERLAY`, `VENV_LLAMAFACTORY`, `SCANNET_H5_DIR`, `SPATIALSSRL_H5_DIR`, `THINKER10K_H5_DIR`, `TRITON_CACHE_DIR`, `TORCH_EXTENSIONS_DIR`, `PYTORCH_KERNEL_CACHE_PATH`, `FLASHINFER_WORKSPACE_BASE`, `WANDB_DIR`, `TORCH_CUDA_ARCH_LIST`.
  - `portable_set_offline()` — exports the six offline vars from Global Constraints.
  - `portable_init()` — convenience: resolve dir, detect cluster, load `site.env`, set paths, set offline.
- Precedence, highest first: pre-set environment, `scripts/site.env`, `sysconfig.json` for the detected `CLUSTER`, repo-relative defaults.

- [ ] **Step 1: Write the failing test**

Append to `scripts/tests/test_portable_env.sh`, immediately before the final `echo "=== ${PASS_COUNT} passed..."` line:

```bash
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
```

- [ ] **Step 2: Run the test to verify the new assertions fail**

Run: `bash scripts/tests/test_portable_env.sh`
Expected: the 7 Task 1 assertions still PASS; the 10 new assertions FAIL because `portable_init` is undefined. Exit code 1.

- [ ] **Step 3: Write the minimal implementation**

Append to `scripts/utils/portable_env.sh`:

```bash
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
```

- [ ] **Step 4: Create `scripts/site.env.example`**

```bash
# Site overrides for portable LLaMA-Factory-LFS SLURM jobs.
#
# Copy to scripts/site.env (gitignored) and edit. Sourced by
# scripts/utils/portable_env.sh.
#
# A plain `export X=value` is fine here: portable_load_site_env restores any
# variable that was already set before this file was sourced, so anything passed
# in via `sbatch --export` or on the submit line still wins. You do not need the
# defensive ${X:-value} form.
#
# Everything here is OPTIONAL. With no site.env, all paths default to
# repo-relative locations and you stage them with symlinks:
#   PORTABLE_STAGE=1 models/qwen2_5vl_lora_sft_CoT/portable_body_qwen2_5vl_lora_sft_CoT_traineval.sh

# --- Where the real artifacts live (targets for repo-relative symlinks) ---
# export PORTABLE_SRC_HF_CACHE="/scratch/$USER/huggingface/hub"
# export PORTABLE_SRC_SIF="/scratch/$USER/containers/llamafactory.sif"
# export PORTABLE_SRC_SCANNET_H5="/scratch/$USER/ScanNet_h5"
# export PORTABLE_SRC_SPATIALSSRL_H5="/scratch/$USER/Spatial-SSRL_images_h5"
# export PORTABLE_SRC_THINKER10K_H5="/scratch/$USER/3DThinker10K_images_h5"
# export PORTABLE_SRC_SCENE30K_ANNOTATION="/scratch/$USER/huggingface/hub/datasets--cvis-tmu--Scene30K/snapshots/13b41da710700aed32c928c81b8f5e433134eb75/data/train-00000-of-00001.parquet"
# export PORTABLE_SRC_SPATIALSSRL_ANNOTATION="/scratch/$USER/huggingface/hub/datasets--internlm--Spatial-SSRL-81k/snapshots/54b82086060a5612f95588b4979446da2282bcd9/SFT-coldstart.json"

# --- Or point directly at absolute paths and skip staging entirely ---
# export HF_HOME="/scratch/$USER/huggingface/hub"
# export SIF_FILE="/scratch/$USER/containers/llamafactory.sif"
# export SCANNET_H5_DIR="/scratch/$USER/ScanNet_h5/scans"
# export SPATIALSSRL_H5_DIR="/scratch/$USER/Spatial-SSRL_images_h5"
# export THINKER10K_H5_DIR="/scratch/$USER/3DThinker10K_images_h5"
# export VENV_LLAMAFACTORY="/scratch/$USER/venv_llamafactory_cu126"
# export APPTAINER_OVERLAY="/scratch/$USER/apptainer/overlay.img"

# --- Extra Apptainer binds, space separated ---
# export EXTRA_BINDS="-B /project/def-someone/shared -B /scratch/$USER"
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `bash scripts/tests/test_portable_env.sh`
Expected: PASS — `17 passed, failed=0`, exit code 0.

Run: `bash -n scripts/utils/portable_env.sh && bash -n scripts/site.env.example && echo SYNTAX_OK`
Expected: `SYNTAX_OK`

- [ ] **Step 6: Commit**

```bash
git add scripts/utils/portable_env.sh scripts/site.env.example scripts/tests/test_portable_env.sh
git commit -m "feat(scripts): add repo-relative path defaults, site.env, cluster detection"
```

---

### Task 4: Preflight check

**Files:**
- Modify: `scripts/utils/portable_env.sh` (append `portable_preflight`)
- Test: `scripts/tests/test_portable_env.sh` (append assertions)

**Interfaces:**
- Consumes: `portable_init()` from Task 3.
- Produces: `portable_preflight()` — prints one line per checked artifact in the form `OK   <label>: <path>` or `MISS <label>: <path>` or `WARN <label>: <path>`, then a summary. Returns 0 when all required artifacts exist, 1 otherwise. Reads `PORTABLE_YAML_FILE` for the config to validate.

- [ ] **Step 1: Write the failing test**

Append to `scripts/tests/test_portable_env.sh` before the final summary line:

```bash
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

# Preflight output never leaks an unexpanded token.
assert_eq "0" "$(grep -cE '\$\{' <<<"${out}")" "preflight output has no unexpanded tokens"
rm -rf "$(dirname "${PF_TMP}")"
```

- [ ] **Step 2: Run the test to verify the new assertions fail**

Run: `bash scripts/tests/test_portable_env.sh`
Expected: the 17 earlier assertions PASS; the 4 new ones FAIL with `portable_preflight: command not found`. Exit code 1.

- [ ] **Step 3: Write the minimal implementation**

Append to `scripts/utils/portable_env.sh`:

```bash
_PORTABLE_PF_RC=0

_portable_pf_row() {
	local status="$1" label="$2" path="$3"
	printf '%-4s %-24s %s\n' "${status}" "${label}" "${path}"
}

# Required artifact: absence fails the job.
_portable_pf_require() {
	local label="$1" path="$2"
	if [[ -n "${path}" && -e "${path}" ]]; then
		_portable_pf_row "OK" "${label}" "${path}"
	else
		_portable_pf_row "MISS" "${label}" "${path:-<unset>}"
		_PORTABLE_PF_RC=1
	fi
}

# Optional artifact: absence is reported but does not fail the job.
_portable_pf_optional() {
	local label="$1" path="$2"
	if [[ -n "${path}" && -e "${path}" ]]; then
		_portable_pf_row "OK" "${label}" "${path}"
	else
		_portable_pf_row "WARN" "${label}" "${path:-<unset>}"
	fi
}

# Validate every resolved path before the job consumes GPU time. Compute nodes
# have no network, so a missing artifact can never be fetched at runtime.
portable_preflight() {
	_PORTABLE_PF_RC=0

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
		echo "preflight: FAIL — stage the MISS entries above, or set overrides in scripts/site.env"
		echo "           see docs/superpowers/specs/2026-09-05-portable-slurm-wrapper-design.md"
	fi
	echo "=========================="

	return "${_PORTABLE_PF_RC}"
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `bash scripts/tests/test_portable_env.sh`

Expected: the `deepspeed_config` row passes; the whole suite reports `21 passed, failed=0`, exit code 0. Note the two assertions that exercise the real repo tolerate a `MISS` on `dataset_registry` because Task 5 has not generated it yet — they only assert on the `deepspeed_config` row and token absence.

Run: `bash -n scripts/utils/portable_env.sh && echo SYNTAX_OK`
Expected: `SYNTAX_OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/utils/portable_env.sh scripts/tests/test_portable_env.sh
git commit -m "feat(scripts): add portable_preflight to validate paths before GPU time"
```

---

### Task 5: Portable dataset registry + asset staging

`data/dataset_info.json` has absolute hub-snapshot paths for `Scene30k` and `SpatialSSRL_coldstart`. A generated registry under `data/annotations/` makes all three CoT datasets resolve relative to the repo.

**Files:**
- Create: `scripts/make_portable_dataset_info.py`
- Create: `tests/scripts/test_make_portable_dataset_info.py`
- Modify: `scripts/utils/portable_env.sh` (append `portable_stage_assets`)

**Interfaces:**
- Consumes: `PROJECT_DIR` and the `PORTABLE_SRC_*` variables from `site.env`.
- Produces:
  - `scripts/make_portable_dataset_info.py` CLI: `--source PATH` (default `data/dataset_info.json`), `--dest PATH` (default `data/annotations/dataset_info.json`), `--no-symlinks`. Exit 0 on success, 1 when an absolute source file is missing.
  - Python API: `rewrite_registry(registry: dict, source_dir: str, dest_dir: str) -> tuple` returning `(new_registry, links)` where `links` is a list of `(link_relpath, absolute_target)` pairs.
  - `portable_stage_assets()` in the resolver — creates repo-relative symlinks from `PORTABLE_SRC_*` and regenerates the registry. Idempotent; never overwrites a real directory.

- [ ] **Step 1: Write the failing test**

Create `tests/scripts/test_make_portable_dataset_info.py`:

```python
# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import make_portable_dataset_info as mpdi  # noqa: E402


def test_absolute_file_name_becomes_dataset_scoped_relative_path():
    registry = {"Scene30k": {"file_name": "/abs/hub/snap/data/train-00000-of-00001.parquet"}}
    new_registry, links = mpdi.rewrite_registry(registry, "/repo/data", "/repo/data/annotations")

    assert new_registry["Scene30k"]["file_name"] == "Scene30k/train-00000-of-00001.parquet"
    assert links == [("Scene30k/train-00000-of-00001.parquet", "/abs/hub/snap/data/train-00000-of-00001.parquet")]


def test_relative_file_name_is_reanchored_to_new_dir():
    registry = {"3DThinker10k": {"file_name": "3DThinker-10K/out/3dthinker10k_cot.jsonl"}}
    new_registry, links = mpdi.rewrite_registry(registry, "/repo/data", "/repo/data/annotations")

    assert new_registry["3DThinker10k"]["file_name"] == "../3DThinker-10K/out/3dthinker10k_cot.jsonl"
    assert links == []


def test_hub_url_entries_are_left_alone():
    registry = {"alpaca_en_demo": {"hf_hub_url": "llamafactory/alpaca_en"}}
    new_registry, links = mpdi.rewrite_registry(registry, "/repo/data", "/repo/data/annotations")

    assert new_registry == registry
    assert links == []


def test_source_registry_is_not_mutated():
    registry = {"Scene30k": {"file_name": "/abs/x.parquet"}}
    mpdi.rewrite_registry(registry, "/repo/data", "/repo/data/annotations")

    assert registry["Scene30k"]["file_name"] == "/abs/x.parquet"


def test_other_keys_are_preserved():
    registry = {
        "Scene30k": {
            "file_name": "/abs/x.parquet",
            "formatting": "alpaca",
            "columns": {"prompt": "q", "response": "cot"},
        }
    }
    new_registry, _ = mpdi.rewrite_registry(registry, "/repo/data", "/repo/data/annotations")

    assert new_registry["Scene30k"]["formatting"] == "alpaca"
    assert new_registry["Scene30k"]["columns"] == {"prompt": "q", "response": "cot"}


def test_main_writes_dest_registry(tmp_path):
    source = tmp_path / "dataset_info.json"
    source.write_text(json.dumps({"D": {"file_name": "sub/x.jsonl"}}), encoding="utf-8")
    dest = tmp_path / "annotations" / "dataset_info.json"

    rc = mpdi.main(["--source", str(source), "--dest", str(dest), "--no-symlinks"])

    assert rc == 0
    written = json.loads(dest.read_text(encoding="utf-8"))
    assert written["D"]["file_name"] == "../sub/x.jsonl"


def test_main_reports_missing_absolute_target(tmp_path, capsys):
    source = tmp_path / "dataset_info.json"
    source.write_text(json.dumps({"D": {"file_name": "/definitely/missing/x.parquet"}}), encoding="utf-8")
    dest = tmp_path / "annotations" / "dataset_info.json"

    rc = mpdi.main(["--source", str(source), "--dest", str(dest), "--no-symlinks"])

    assert rc == 1
    assert "missing" in capsys.readouterr().out.lower()


def test_main_creates_symlink(tmp_path):
    target = tmp_path / "real" / "x.parquet"
    target.parent.mkdir(parents=True)
    target.write_text("data", encoding="utf-8")
    source = tmp_path / "dataset_info.json"
    source.write_text(json.dumps({"D": {"file_name": str(target)}}), encoding="utf-8")
    dest = tmp_path / "annotations" / "dataset_info.json"

    rc = mpdi.main(["--source", str(source), "--dest", str(dest)])

    assert rc == 0
    assert (tmp_path / "annotations" / "D" / "x.parquet").resolve() == target.resolve()


def test_main_is_idempotent(tmp_path):
    target = tmp_path / "real" / "x.parquet"
    target.parent.mkdir(parents=True)
    target.write_text("data", encoding="utf-8")
    source = tmp_path / "dataset_info.json"
    source.write_text(json.dumps({"D": {"file_name": str(target)}}), encoding="utf-8")
    dest = tmp_path / "annotations" / "dataset_info.json"

    assert mpdi.main(["--source", str(source), "--dest", str(dest)]) == 0
    assert mpdi.main(["--source", str(source), "--dest", str(dest)]) == 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `CUDA_VISIBLE_DEVICES= python -m pytest tests/scripts/test_make_portable_dataset_info.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'make_portable_dataset_info'`.

- [ ] **Step 3: Write the minimal implementation**

Create `scripts/make_portable_dataset_info.py`:

```python
# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate a portable dataset_info.json whose file_name values are repo-relative.

LLaMA-Factory joins ``dataset_dir`` with each ``file_name``. Moving the registry
from ``data/`` to ``data/annotations/`` therefore requires rewriting every
``file_name``:

* Absolute paths become ``<dataset_name>/<basename>`` and are reached through a
  symlink created next to the generated registry, so no large file is copied.
* Already-relative paths are re-anchored with ``..`` so they keep resolving.

Usage:
    python scripts/make_portable_dataset_info.py
    python scripts/make_portable_dataset_info.py --source A --dest B --no-symlinks
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple


def rewrite_registry(
    registry: Dict[str, Any], source_dir: str, dest_dir: str
) -> Tuple[Dict[str, Any], List[Tuple[str, str]]]:
    """Rewrite every ``file_name`` so it resolves against ``dest_dir``.

    Args:
        registry: Parsed contents of a dataset_info.json file.
        source_dir: Directory the original relative file names resolve against.
        dest_dir: Directory the rewritten file names must resolve against.

    Returns:
        A tuple of the new registry and a list of ``(link_relpath, target)``
        pairs describing the symlinks needed for absolute entries.
    """
    new_registry: Dict[str, Any] = {}
    links: List[Tuple[str, str]] = []

    for name, attrs in registry.items():
        if not isinstance(attrs, dict) or "file_name" not in attrs:
            new_registry[name] = attrs
            continue

        new_attrs = dict(attrs)
        file_name = attrs["file_name"]

        if os.path.isabs(file_name):
            link_relpath = "{}/{}".format(name, os.path.basename(file_name))
            new_attrs["file_name"] = link_relpath
            links.append((link_relpath, file_name))
        else:
            absolute = os.path.join(source_dir, file_name)
            new_attrs["file_name"] = os.path.relpath(absolute, dest_dir).replace(os.sep, "/")

        new_registry[name] = new_attrs

    return new_registry, links


def _create_symlink(link_path: str, target: str) -> None:
    """Create or refresh a single symlink, never clobbering a real file."""
    os.makedirs(os.path.dirname(link_path), exist_ok=True)

    if os.path.islink(link_path):
        if os.path.realpath(link_path) == os.path.realpath(target):
            return

        os.unlink(link_path)
    elif os.path.exists(link_path):
        raise RuntimeError("refusing to replace non-symlink: {}".format(link_path))

    os.symlink(target, link_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point. Returns a process exit code."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=os.path.join(repo_root, "data", "dataset_info.json"))
    parser.add_argument("--dest", default=os.path.join(repo_root, "data", "annotations", "dataset_info.json"))
    parser.add_argument("--no-symlinks", action="store_true", help="rewrite paths without creating symlinks")
    args = parser.parse_args(argv)

    with open(args.source, encoding="utf-8") as f:
        registry = json.load(f)

    source_dir = os.path.dirname(os.path.abspath(args.source))
    dest_dir = os.path.dirname(os.path.abspath(args.dest))
    new_registry, links = rewrite_registry(registry, source_dir, dest_dir)

    exit_code = 0
    for link_relpath, target in links:
        if not os.path.exists(target):
            print("missing annotation source for {}: {}".format(link_relpath, target))
            exit_code = 1
            continue

        if not args.no_symlinks:
            _create_symlink(os.path.join(dest_dir, link_relpath), target)

    os.makedirs(dest_dir, exist_ok=True)
    with open(args.dest, "w", encoding="utf-8") as f:
        json.dump(new_registry, f, indent=2)
        f.write("\n")

    print("wrote {} ({} entries, {} symlinked)".format(args.dest, len(new_registry), len(links)))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES= python -m pytest tests/scripts/test_make_portable_dataset_info.py -v`
Expected: PASS — 9 passed.

Run: `ruff check scripts/make_portable_dataset_info.py tests/scripts/test_make_portable_dataset_info.py && ruff format --check scripts/make_portable_dataset_info.py tests/scripts/test_make_portable_dataset_info.py`
Expected: `All checks passed!` and `2 files already formatted`.

Verify the license header on both new files directly (`check_license.py` aborts on a
pre-existing offender before it reaches ours):

```bash
for f in scripts/make_portable_dataset_info.py tests/scripts/test_make_portable_dataset_info.py; do
  for kw in Copyright 2025 LlamaFactory; do
    head -1 "$f" | grep -q "$kw" || echo "MISSING $kw in $f"
  done
done; echo LICENSE_HEADERS_OK
```
Expected: `LICENSE_HEADERS_OK` with no `MISSING` lines.

- [ ] **Step 5: Append `portable_stage_assets` to `scripts/utils/portable_env.sh`**

```bash
# Create one repo-relative symlink, skipping when the target is unset and
# refusing to clobber a real directory.
_portable_link() {
	local link="$1" target="$2"

	[[ -z "${target}" ]] && return 0

	if [[ ! -e "${target}" ]]; then
		echo "portable_env: stage target missing, skipping: ${target}" >&2
		return 0
	fi

	if [[ -L "${link}" ]]; then
		[[ "$(readlink -f "${link}")" == "$(readlink -f "${target}")" ]] && return 0
		rm -f "${link}"
	elif [[ -e "${link}" ]]; then
		echo "portable_env: refusing to replace existing path: ${link}" >&2
		return 0
	fi

	mkdir -p "$(dirname "${link}")"
	ln -s "${target}" "${link}"
	echo "portable_env: linked ${link} -> ${target}" >&2
}

# Create the repo-relative staging tree and regenerate the portable registry.
# Idempotent. Run explicitly with PORTABLE_STAGE=1; never called during training.
portable_stage_assets() {
	mkdir -p "${PROJECT_DIR}/data/h5" "${PROJECT_DIR}/data/annotations" \
		"${PROJECT_DIR}/containers" "${PROJECT_DIR}/.cache"

	_portable_link "${PROJECT_DIR}/.cache/huggingface" "${PORTABLE_SRC_HF_CACHE:-}"
	_portable_link "${PROJECT_DIR}/containers/llamafactory.sif" "${PORTABLE_SRC_SIF:-}"
	_portable_link "${PROJECT_DIR}/data/h5/ScanNet_h5" "${PORTABLE_SRC_SCANNET_H5:-}"
	_portable_link "${PROJECT_DIR}/data/h5/Spatial-SSRL_images_h5" "${PORTABLE_SRC_SPATIALSSRL_H5:-}"
	_portable_link "${PROJECT_DIR}/data/h5/3DThinker10K_images_h5" "${PORTABLE_SRC_THINKER10K_H5:-}"

	echo "portable_env: generating data/annotations/dataset_info.json" >&2
	python3 "${PROJECT_DIR}/scripts/make_portable_dataset_info.py" \
		--source "${PROJECT_DIR}/data/dataset_info.json" \
		--dest "${PROJECT_DIR}/data/annotations/dataset_info.json"
}
```

- [ ] **Step 6: Verify staging is idempotent and syntax-clean**

Run:
```bash
bash -n scripts/utils/portable_env.sh
cd /tmp && bash -c 'source "$1"; portable_init >/dev/null; portable_stage_assets' _ \
  "$(git -C "$OLDPWD" rev-parse --show-toplevel 2>/dev/null || echo .)/scripts/utils/portable_env.sh" 2>&1 | tail -3
```
Expected: no syntax errors; `wrote .../data/annotations/dataset_info.json (N entries, 2 symlinked)`. Exit code may be 1 if the two hub annotation snapshots are absent on this machine — that is the correct "missing" report, not a bug.

Run twice in a row and confirm the second run prints no new `linked` lines (idempotence).

- [ ] **Step 7: Commit**

```bash
git add scripts/make_portable_dataset_info.py tests/scripts/test_make_portable_dataset_info.py scripts/utils/portable_env.sh
git commit -m "feat(scripts): generate portable dataset registry and stage assets via symlinks"
```

---

### Task 6: Portable YAML, wrapper, and job body

This task produces the actually submittable job.

**Files:**
- Create: `examples/train_lora/portable_qwen2_5vl_lora_sft_CoT_traineval.yaml`
- Create: `models/qwen2_5vl_lora_sft_CoT/portable_body_qwen2_5vl_lora_sft_CoT_traineval.sh`
- Create: `models/qwen2_5vl_lora_sft_CoT/portable_slurm_qwen2_5vl_lora_sft_CoT_traineval.sh`

**Interfaces:**
- Consumes: `portable_init()`, `portable_preflight()`, `portable_stage_assets()` from Tasks 1, 3, 4, 5.
- Produces: a `sbatch`-able wrapper. The body honours `PREFLIGHT=1` (check and exit), `PORTABLE_STAGE=1` (stage and exit), and `RUNNING_MODE` in `APPTAINER` / `VENV` / `SHELL`. It forwards `"$@"` to `llamafactory-cli train` so `key=value` overrides work.

- [ ] **Step 1: Create the portable YAML**

Hyperparameters are copied verbatim from `examples/train_lora/trillium_qwen2_5vl_lora_sft_CoT_traineval.yaml` so results stay comparable. Only paths differ, plus a distinct `output_dir` so a portable run cannot clobber the Trillium run's checkpoints (both set `overwrite_output_dir: true`).

```yaml
### model
# No cache_dir: the job exports HF_HOME / HF_HUB_CACHE, which the hub respects.
model_name_or_path: Qwen/Qwen2.5-VL-7B-Instruct
image_max_pixels: 65536
video_max_pixels: 16384
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### dataset
# CoT mix: Scene30k + SpatialSSRL_coldstart + 3DThinker10k (full concat each epoch).
#
# Every path here is relative to the repo root, which the job body sets as the
# working directory. dataset_dir points at the GENERATED registry written by
# `PORTABLE_STAGE=1 ...portable_body_...sh`, whose file_name values resolve
# repo-relative. The original data/dataset_info.json is never modified.
#
# H5 image roots come from the environment, not from media_dir:
#   SCANNET_H5_DIR, SPATIALSSRL_H5_DIR, THINKER10K_H5_DIR
# h5_image_store dispatches by path shape, so a single unified media tree is not
# required. media_dir below is only a filesystem join prefix for Scene30k-style
# relative paths. Override either on the CLI, e.g. `... media_dir=/other/root`.
dataset: Scene30k,SpatialSSRL_coldstart,3DThinker10k
dataset_dir: data/annotations
mix_strategy: concat
media_dir: data/h5/ScanNet_h5
template: qwen2_vl
cutoff_len: 131072
overwrite_cache: false
preprocessing_num_workers: 32
dataloader_num_workers: 4
dataloader_prefetch_factor: 1
dataloader_pin_memory: false
low_cpu_mem_usage: true

### output
output_dir: saves/qwen2_5vl-7b/lora/sft/CoT_traineval_portable
logging_steps: 10
save_steps: 620
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: wandb

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 5.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null

# debugging level (kept identical to the Trillium config for comparability;
# set debug_mm_training: false for cleaner logs on long runs)
debug: underflow_overflow
log_level: debug
log_level_replica: debug
print_param_status: true
debug_mm_training: true
debug_mm_steps: 1

# acceleration
flash_attn: fa2
enable_liger_kernel: true

# distribution
deepspeed: examples/deepspeed/ds_z2_config.json

## eval
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 620
```

- [ ] **Step 2: Create the job body**

Create `models/qwen2_5vl_lora_sft_CoT/portable_body_qwen2_5vl_lora_sft_CoT_traineval.sh` (make it executable):

```bash
#!/usr/bin/env bash
# Portable body for CoT SFT: Scene30k + SpatialSSRL_coldstart + 3DThinker10k.
#
# Every path is resolved relative to the repo root by
# scripts/utils/portable_env.sh, so this tree can be moved or renamed.
#
# Modes:
#   PREFLIGHT=1 <this script>        check paths and exit (safe on a login node)
#   PORTABLE_STAGE=1 <this script>   create repo-relative symlinks + registry, exit
#   RUNNING_MODE=APPTAINER           run llamafactory-cli inside the SIF (default)
#   RUNNING_MODE=VENV                run llamafactory-cli from VENV_LLAMAFACTORY
#   RUNNING_MODE=SHELL               open a shell inside the SIF
#
# Extra args are forwarded to llamafactory-cli, e.g.:
#   sbatch portable_slurm_...sh num_train_epochs=1.0
set -euo pipefail

BODY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=../../scripts/utils/portable_env.sh
source "${BODY_DIR}/../../scripts/utils/portable_env.sh"

portable_init

EXPERIMENT_NAME="qwen2_5vl_lora_sft_CoT_traineval"
export PORTABLE_YAML_FILE="${PROJECT_DIR}/examples/train_lora/portable_${EXPERIMENT_NAME}.yaml"

mkdir -p "${BODY_DIR}/out" "${WANDB_DIR}"
cd "${PROJECT_DIR}"

if [[ -n "${PORTABLE_STAGE:-}" ]]; then
	portable_stage_assets
	exit $?
fi

if [[ -n "${PREFLIGHT:-}" ]]; then
	portable_preflight
	exit $?
fi

portable_preflight || exit 1

# AllianceCan module stack. Absent on a workstation, which is fine.
if command -v module >/dev/null 2>&1; then
	module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 || true
	module load python/3.12 cuda/12.6 opencv/4.12.0 || true
	module load arrow || true
	[[ "${RUNNING_MODE}" != "VENV" ]] && { module load apptainer || true; }
fi

echo "=== host diagnostics (${CLUSTER}) ==="
echo "HOSTNAME:             $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi || true
echo "====================================="

run_in_apptainer() {
	local program="$1"
	local binds=()
	local nv_lib_dir=""

	# Bind only paths that exist; Apptainer fails on a missing bind source.
	local candidate
	for candidate in "${PROJECT_DIR}" "${HF_HUB_CACHE}" "${SCANNET_H5_DIR}" \
		"${SPATIALSSRL_H5_DIR}" "${THINKER10K_H5_DIR}" "${MEDIA_DIR}" "${HOME}"; do
		[[ -n "${candidate}" && "${candidate}" != "None" && -e "${candidate}" ]] && binds+=(-B "${candidate}")
	done
	[[ -d /dev/shm ]] && binds+=(-B /dev/shm:/dev/shm)
	[[ -d /etc/ssl/certs ]] && binds+=(-B /etc/ssl/certs:/etc/ssl/certs:ro)
	[[ -d /etc/pki ]] && binds+=(-B /etc/pki:/etc/pki:ro)

	# shellcheck disable=SC2206
	[[ -n "${EXTRA_BINDS:-}" ]] && binds+=(${EXTRA_BINDS})

	nv_lib_dir="$(dirname "$(ldconfig -p 2>/dev/null | awk '/libcuda\.so /{print $NF}' | head -1)" 2>/dev/null || true)"
	[[ -n "${nv_lib_dir}" && -d "${nv_lib_dir}" ]] && binds+=(-B "${nv_lib_dir}")

	local overlay=()
	[[ -f "${APPTAINER_OVERLAY}" ]] && overlay=(--overlay "${APPTAINER_OVERLAY}")

	# Use the CUDA toolkit inside the image, not a host module path.
	export APPTAINERENV_CUDA_HOME=/usr/local/cuda

	apptainer run --nv "${overlay[@]}" \
		"${binds[@]}" \
		-W "${SLURM_TMPDIR:-/tmp}" \
		--env HF_HOME="${HF_HOME}" \
		--env HF_HUB_CACHE="${HF_HUB_CACHE}" \
		--env HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" \
		--env TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE}" \
		--env HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE}" \
		--env SCANNET_H5_DIR="${SCANNET_H5_DIR}" \
		--env SPATIALSSRL_H5_DIR="${SPATIALSSRL_H5_DIR}" \
		--env THINKER10K_H5_DIR="${THINKER10K_H5_DIR}" \
		--env MPLCONFIGDIR="${MPLCONFIGDIR}" \
		--env TRITON_CACHE_DIR="${TRITON_CACHE_DIR}" \
		--env TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR}" \
		--env PYTORCH_KERNEL_CACHE_PATH="${PYTORCH_KERNEL_CACHE_PATH}" \
		--env FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE}" \
		--env TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
		--env DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK}" \
		--env FORCE_TORCHRUN="${FORCE_TORCHRUN}" \
		--env WANDB_MODE="${WANDB_MODE}" \
		--env WANDB_DIR="${WANDB_DIR}" \
		--env WANDB_CACHE_DIR="${WANDB_CACHE_DIR}" \
		--env PYTHONNOUSERSITE=1 \
		--env PYTHONUNBUFFERED=1 \
		--env PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}/scripts" \
		--env NCCL_DEBUG=INFO \
		--env NCCL_IB_DISABLE=0 \
		--env NCCL_P2P_DISABLE=0 \
		--env NCCL_SOCKET_IFNAME=^docker0,lo \
		--env CUDA_HOME="${APPTAINERENV_CUDA_HOME}" \
		--pwd "${PROJECT_DIR}" \
		"${SIF_FILE}" \
		${program}
}

case "${RUNNING_MODE}" in
APPTAINER)
	run_in_apptainer "llamafactory-cli train ${PORTABLE_YAML_FILE} $*"
	;;
SHELL)
	run_in_apptainer "bash"
	;;
VENV)
	# shellcheck disable=SC1091
	source "${VENV_LLAMAFACTORY}/bin/activate"
	# Re-assert after activation: an editable install may point at another tree.
	export PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}/scripts"
	llamafactory-cli train "${PORTABLE_YAML_FILE}" "$@"
	;;
*)
	echo "Invalid RUNNING_MODE: ${RUNNING_MODE} (expected APPTAINER, VENV, or SHELL)" >&2
	exit 1
	;;
esac
```

- [ ] **Step 3: Create the SBATCH wrapper**

Create `models/qwen2_5vl_lora_sft_CoT/portable_slurm_qwen2_5vl_lora_sft_CoT_traineval.sh` (make it executable). Resources match the Trillium wrapper. No `--mail-user` and no `--account`: pass those at submit time.

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=out/%N-qwen2_5vl_lora_sft_CoT_traineval-%j.out
#SBATCH --cpus-per-task=96
#SBATCH --time=1-00:00:00
#SBATCH --gpus-per-node=h100:4

# Portable wrapper for CoT SFT (Scene30k + SpatialSSRL_coldstart + 3DThinker10k).
#
# Unlike the cluster-specific wrappers, this one derives the repo root from its
# OWN location, so the checkout can be renamed or moved anywhere.
#
# Submit from this directory so SLURM out/ lands next to the script:
#   sbatch portable_slurm_qwen2_5vl_lora_sft_CoT_traineval.sh
# Add site flags as needed:
#   sbatch -A <account> --mail-user=<you> --mail-type=ALL portable_slurm_...sh
#
# One-time setup on a login node:
#   PORTABLE_STAGE=1 ./portable_body_qwen2_5vl_lora_sft_CoT_traineval.sh
#   PREFLIGHT=1      ./portable_body_qwen2_5vl_lora_sft_CoT_traineval.sh
set -euo pipefail

WRAPPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
mkdir -p "${WRAPPER_DIR}/out"

exec "${WRAPPER_DIR}/portable_body_qwen2_5vl_lora_sft_CoT_traineval.sh" "$@"
```

- [ ] **Step 4: Verify syntax and preflight wiring**

```bash
chmod +x models/qwen2_5vl_lora_sft_CoT/portable_slurm_qwen2_5vl_lora_sft_CoT_traineval.sh \
         models/qwen2_5vl_lora_sft_CoT/portable_body_qwen2_5vl_lora_sft_CoT_traineval.sh
bash -n models/qwen2_5vl_lora_sft_CoT/portable_slurm_qwen2_5vl_lora_sft_CoT_traineval.sh
bash -n models/qwen2_5vl_lora_sft_CoT/portable_body_qwen2_5vl_lora_sft_CoT_traineval.sh
python -c "import yaml,sys; yaml.safe_load(open('examples/train_lora/portable_qwen2_5vl_lora_sft_CoT_traineval.yaml')); print('YAML_OK')"
```
Expected: no syntax errors, then `YAML_OK`.

- [ ] **Step 5: Run preflight from an unrelated directory**

```bash
cd /tmp && PREFLIGHT=1 RUNNING_MODE=VENV \
  "$OLDPWD/models/qwen2_5vl_lora_sft_CoT/portable_body_qwen2_5vl_lora_sft_CoT_traineval.sh"; echo "rc=$?"
```
Expected: the preflight table prints with `PROJECT_DIR` equal to the repo root even though the cwd is `/tmp`, `OK` on `project_root`, `llamafactory_src`, `deepspeed_config`, and `train_yaml`, and `MISS` on the un-staged artifacts. `rc=1` on this machine is the correct result; `rc=0` only on a fully staged cluster.

- [ ] **Step 6: Confirm the YAML has no absolute paths**

Run: `grep -nE '^[^#]*: */' examples/train_lora/portable_qwen2_5vl_lora_sft_CoT_traineval.yaml || echo NO_ABSOLUTE_PATHS`
Expected: `NO_ABSOLUTE_PATHS`

- [ ] **Step 7: Commit**

```bash
git add examples/train_lora/portable_qwen2_5vl_lora_sft_CoT_traineval.yaml \
        models/qwen2_5vl_lora_sft_CoT/portable_body_qwen2_5vl_lora_sft_CoT_traineval.sh \
        models/qwen2_5vl_lora_sft_CoT/portable_slurm_qwen2_5vl_lora_sft_CoT_traineval.sh
git commit -m "feat(models): add portable repo-relative CoT SFT wrapper, body, and YAML"
```

---

### Task 7: Ignore staged artifacts, document, and run the acceptance test

**Files:**
- Modify: `.gitignore`
- Modify: `data/README.md`
- Test: `scripts/tests/test_portable_env.sh` (append the end-to-end relocation assertion)

**Interfaces:**
- Consumes: everything from Tasks 1-6.
- Produces: no new interfaces. Final gate: a full-repo relocation test plus the repo's own quality gates.

- [ ] **Step 1: Write the failing acceptance test**

Append to `scripts/tests/test_portable_env.sh` before the final summary line:

```bash
# --- Task 7: end-to-end relocation acceptance ---
# Copy the files the portable job needs into a differently-named tree and confirm
# the body resolves entirely within that copy, with no reference to the original.

E2E_BASE="$(mktemp -d)"
E2E="${E2E_BASE}/some-other-name"
mkdir -p "${E2E}/scripts/utils" "${E2E}/src/llamafactory" \
	"${E2E}/models/qwen2_5vl_lora_sft_CoT" "${E2E}/examples/train_lora" \
	"${E2E}/examples/deepspeed" "${E2E}/data"
touch "${E2E}/setup.py"
cp "${REPO}/scripts/utils/portable_env.sh" "${E2E}/scripts/utils/"
cp "${REPO}/scripts/sysconfigtool.py" "${REPO}/scripts/sysconfig.json" \
	"${REPO}/scripts/make_portable_dataset_info.py" "${E2E}/scripts/"
cp "${REPO}/examples/deepspeed/ds_z2_config.json" "${E2E}/examples/deepspeed/"
cp "${REPO}/examples/train_lora/portable_qwen2_5vl_lora_sft_CoT_traineval.yaml" "${E2E}/examples/train_lora/"
cp "${REPO}/models/qwen2_5vl_lora_sft_CoT/portable_body_qwen2_5vl_lora_sft_CoT_traineval.sh" \
	"${REPO}/models/qwen2_5vl_lora_sft_CoT/portable_slurm_qwen2_5vl_lora_sft_CoT_traineval.sh" \
	"${E2E}/models/qwen2_5vl_lora_sft_CoT/"
echo '{}' >"${E2E}/data/dataset_info.json"

out="$(cd /tmp && PREFLIGHT=1 CLUSTER=PORTABLE RUNNING_MODE=VENV PORTABLE_SKIP_SITE_ENV=1 \
	"${E2E}/models/qwen2_5vl_lora_sft_CoT/portable_body_qwen2_5vl_lora_sft_CoT_traineval.sh" 2>&1)" || true

assert_eq "yes" "$(grep -q "PROJECT_DIR:  ${E2E}" <<<"${out}" && echo yes || echo no)" \
	"relocated body resolves PROJECT_DIR to the copy"
assert_eq "0" "$(grep -c "${REPO}/" <<<"${out}")" \
	"relocated body never references the original checkout"
assert_eq "yes" "$(grep -qE '^OK +train_yaml' <<<"${out}" && echo yes || echo no)" \
	"relocated body finds its own YAML"
rm -rf "${E2E_BASE}"
```

- [ ] **Step 2: Run the test to verify the new assertions fail if anything is non-portable**

Run: `bash scripts/tests/test_portable_env.sh`
Expected: all assertions PASS — `24 passed, failed=0`, exit 0. A failure here means a path is still tied to the original tree; fix the offending default in `portable_env.sh` rather than the test.

- [ ] **Step 3: Add ignore rules**

Append to `.gitignore`:

```gitignore
# portable job staging (see docs/superpowers/specs/2026-09-05-portable-slurm-wrapper-design.md)
scripts/site.env
containers/
data/h5/
data/annotations/
```

- [ ] **Step 4: Verify staged artifacts are ignored**

```bash
mkdir -p containers data/h5 data/annotations && touch scripts/site.env containers/x.sif data/h5/x data/annotations/x
git status --porcelain | grep -E 'site\.env|containers/|data/h5/|data/annotations/' && echo LEAKED || echo IGNORED_OK
rm -f scripts/site.env containers/x.sif data/h5/x data/annotations/x
```
Expected: `IGNORED_OK`

- [ ] **Step 5: Document usage in `data/README.md`**

Insert this section immediately after the `## Project multimodal datasets (this fork)` table:

```markdown
### Portable (repo-relative) CoT job

`models/qwen2_5vl_lora_sft_CoT/portable_slurm_qwen2_5vl_lora_sft_CoT_traineval.sh`
runs the same CoT SFT mix as the cluster-specific wrappers, but derives every
path from the repo root, so the checkout can be renamed or moved to another
cluster. Large artifacts are reached through symlinks; nothing is copied.

One-time setup on a **login** node (has network; compute nodes do not):

```bash
cp scripts/site.env.example scripts/site.env
# edit scripts/site.env: point PORTABLE_SRC_* at the real HF cache, SIF, and H5 trees

cd models/qwen2_5vl_lora_sft_CoT
PORTABLE_STAGE=1 ./portable_body_qwen2_5vl_lora_sft_CoT_traineval.sh   # create symlinks + registry
PREFLIGHT=1      ./portable_body_qwen2_5vl_lora_sft_CoT_traineval.sh   # verify, exits nonzero on gaps
```

Submit once preflight passes:

```bash
sbatch portable_slurm_qwen2_5vl_lora_sft_CoT_traineval.sh
sbatch -A <account> --mail-user=<you> --mail-type=ALL portable_slurm_qwen2_5vl_lora_sft_CoT_traineval.sh
sbatch portable_slurm_qwen2_5vl_lora_sft_CoT_traineval.sh num_train_epochs=1.0   # CLI overrides
```

Staging writes `data/annotations/dataset_info.json`, a generated registry whose
`file_name` values are repo-relative. `data/dataset_info.json` is never modified,
so the existing cluster jobs are unaffected. Outputs land in
`saves/qwen2_5vl-7b/lora/sft/CoT_traineval_portable`, separate from the
Trillium run.
```

- [ ] **Step 6: Run the full quality gate**

```bash
bash scripts/tests/test_portable_env.sh
make style && make quality
CUDA_VISIBLE_DEVICES= WANDB_DISABLED=true python -m pytest tests/scripts -v
git diff --stat
```
Expected: bash suite `24 passed, failed=0`; ruff reports all checks passed; `tests/scripts` shows 20 passed. Confirm `git diff --stat` lists no `trillium_*`, `killarney_*`, `nibi_*`, `rorqual_*`, or unprefixed `slurm_*` file, and no change to `data/dataset_info.json`.

- [ ] **Step 7: Commit**

```bash
git add .gitignore data/README.md scripts/tests/test_portable_env.sh
git commit -m "docs: document portable CoT job and ignore staged artifacts"
```

---

## Post-implementation: cluster verification (requires a login node)

These cannot run in the authoring environment. Run them on a real cluster before trusting the job:

1. `PORTABLE_STAGE=1` then `PREFLIGHT=1` until preflight reports `PASS`.
2. `RUNNING_MODE=SHELL sbatch ...` (or an interactive `salloc`) and inside the container check that `python -c "import llamafactory; print(llamafactory.__file__)"` points into **this** tree's `src/`.
3. A truncated smoke submission: `sbatch portable_slurm_...sh max_samples=8 max_steps=4 num_train_epochs=1.0`, confirming the three datasets concatenate and the first batch decodes H5 images.
4. The full job.
5. Repeat step 1 in a renamed copy of the checkout to confirm portability on real storage.

---

## Self-Review

**Spec coverage.** Every spec section maps to a task: five-layer path inventory → Tasks 1-6; chosen approach A → the file structure; architecture's 5 new files + 2 edits → Tasks 1, 2, 3, 5, 6; path resolution contract → Task 3; override precedence → Task 3 Step 1 assertions; data staging → Task 5; the `media_dir` / `dataset_dir` YAML-not-env correction → Task 6 Step 1; cluster and runtime detection → Task 3; error handling / preflight → Task 4; testing (bash syntax, `PREFLIGHT=1`, relocation, `SHELL`, smoke) → Tasks 1, 4, 7 and the post-implementation section; risks → Global Constraints plus the documented one-`.py`-file deviation.

**Placeholder scan.** No `TBD`, `TODO`, "implement later", "add error handling", or "similar to Task N". Every code step contains complete file content, and each step that mentions a command gives the exact command and expected output.

**Type and name consistency.** `portable_resolve_project_dir`, `portable_detect_cluster`, `portable_load_site_env`, `portable_set_paths`, `portable_set_offline`, `portable_init`, `portable_preflight`, `portable_stage_assets`, and the private `_portable_*` helpers are each defined once and referenced with the same spelling in later tasks. `rewrite_registry(registry, source_dir, dest_dir) -> (dict, list)` and `main(argv) -> int` match between the implementation and its tests. `PORTABLE_YAML_FILE` is exported in Task 6 and consumed by `portable_preflight` from Task 4. `MEDIA_DIR` is spelled uppercase in the `PORTABLE` sysconfig section to match `_portable_default MEDIA_DIR`.
