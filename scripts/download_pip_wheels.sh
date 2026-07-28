#!/bin/bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: download_pip_wheels.sh [--no-deps|--with-deps] [requirements-file] [wheelhouse-dir]

Downloads binary pip artifacts for offline venv creation on the login node.

Arguments:
  --no-deps          Download only the named requirements, not their
                     dependencies.
  --with-deps        Download requirements and their dependencies. This is the
                     default.
    requirements-file  Requirements file to mirror. This may be a regular file
                     path or a readable file descriptor path such as /dev/fd/63
                     from process substitution. Defaults to
                     requirements_venv_llamafactory_cu126_qwen35.txt.
  wheelhouse-dir     Directory where downloaded wheels are stored. Defaults to
                     ./wheels/<requirements-basename>.
EOF
}

DOWNLOAD_DEPS=1

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ "${1:-}" == "--no-deps" ]]; then
    DOWNLOAD_DEPS=0
    shift
elif [[ "${1:-}" == "--with-deps" ]]; then
    DOWNLOAD_DEPS=1
    shift
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_REQ_FILE="${PROJECT_DIR}/requirements_venv_llamafactory_cu126_qwen35.txt"

REQ_FILE="${1:-${DEFAULT_REQ_FILE}}"
if [[ -n "${2:-}" ]]; then
    WHEELHOUSE_DIR="$2"
elif [[ -f "${REQ_FILE}" ]]; then
    WHEELHOUSE_DIR="${PROJECT_DIR}/wheels/$(basename "${REQ_FILE}" .txt)"
else
    WHEELHOUSE_DIR="${PROJECT_DIR}/wheels/requirements_fd"
fi

if [[ ! -r "${REQ_FILE}" ]]; then
    echo "ERROR: requirements file not found: ${REQ_FILE}"
    exit 1
fi

if command -v module >/dev/null 2>&1; then
    module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
    module load python/3.12 cuda/12.6 opencv/4.12.0
    module load arrow
fi

mkdir -p "${WHEELHOUSE_DIR}"

filtered_requirements="$(mktemp)"
trap 'rm -f "${filtered_requirements}"' EXIT

# pip download cannot consume the repo's editable install entry, so mirror only
# the resolvable package specs from the requirements file.
grep -Ev '^[[:space:]]*($|#|-e[[:space:]]|--editable[[:space:]]|git\+|https?://|file:|@ )' "${REQ_FILE}" > "${filtered_requirements}"

python3 -m pip download \
    --only-binary=:all: \
    $(if [[ "${DOWNLOAD_DEPS}" -eq 0 ]]; then echo "--no-deps"; fi) \
    --dest "${WHEELHOUSE_DIR}" \
    -r "${filtered_requirements}"

echo "Downloaded wheelhouse: ${WHEELHOUSE_DIR}"
find "${WHEELHOUSE_DIR}" -maxdepth 1 -type f \( -name '*.whl' -o -name '*.tar.gz' -o -name '*.zip' \) | sort