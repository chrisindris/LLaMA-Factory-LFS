#!/usr/bin/env bash
# Pull the LLaMA-Factory Docker image and convert it to an Apptainer SIF.
#
# Usage (from repo root or scripts/):
#   bash scripts/build_apptainer.sh
#   IMAGE=docker://hiyouga/llamafactory:latest bash scripts/build_apptainer.sh
#   FORCE_RECREATE=1 bash scripts/build_apptainer.sh
#
# Cache/tmp live under APPTAINER_DIR (scratch) so home-quota Docker layers
# do not fill $HOME. Overlay install is a separate step:
#   bash scripts/build_apptainer_overlay.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/utils/env.sh"

# --- setting environment ---

export APPTAINER_NUM_THREADS="${APPTAINER_NUM_THREADS:-4}"

APPTAINER_DIR="${APPTAINER_DIR:-${PROJECT_DIR}/apptainer}"
IMAGE="${IMAGE:-docker://hiyouga/llamafactory:latest-910b-ubuntu}"
SIF="${SIF:-${APPTAINER_DIR}/llamafactory.sif}"
FORCE_RECREATE="${FORCE_RECREATE:-0}"

# AllianceCan home quotas cannot hold OCI layers; keep cache/tmp on scratch.
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${APPTAINER_DIR}/cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${APPTAINER_DIR}/tmp}"

mkdir -p "${APPTAINER_DIR}" "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

# --- get the container ---

module load apptainer 2>/dev/null || true
if ! command -v apptainer >/dev/null 2>&1; then
	echo "ERROR: apptainer not found (module load apptainer failed and it is not on PATH)"
	exit 1
fi

echo "APPTAINER_DIR: ${APPTAINER_DIR}"
echo "APPTAINER_CACHEDIR: ${APPTAINER_CACHEDIR}"
echo "APPTAINER_TMPDIR: ${APPTAINER_TMPDIR}"
echo "IMAGE: ${IMAGE}"
echo "SIF: ${SIF}"

if [[ -f "${SIF}" && "${FORCE_RECREATE}" != "1" ]]; then
	echo "SIF already exists ($(du -h "${SIF}" | awk '{print $1}')). Skipping pull."
	echo "  To rebuild: FORCE_RECREATE=1 $0"
	exit 0
fi

echo "Pulling ${IMAGE} -> ${SIF}"
apptainer pull --force "${SIF}" "${IMAGE}"

echo "Done. SIF ready: ${SIF}"
