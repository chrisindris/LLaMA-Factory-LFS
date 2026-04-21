#!/bin/bash
# Helper script to launch apptainer shell with MPI libraries properly mounted
# Usage: ./apptainer_shell_with_mpi.sh

module load apptainer

# MPI library paths for bind mounting
MPI_LIB_PATH="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcc12/openmpi/4.1.5/lib"
HWLOC_LIB_PATH="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/hwloc/2.9.1/lib"

# Set LD_LIBRARY_PATH: container libraries first (for compatibility), then MPI libraries
# This ensures container's compatible libraries are found first, avoiding GLIBC issues
export LD_LIBRARY_PATH="${MPI_LIB_PATH}:${HWLOC_LIB_PATH}:${LD_LIBRARY_PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_SCRIPT="${SCRIPT_DIR}/install_mpi_deps.sh"

# Common bind mount options
BIND_MOUNTS=(
    -B /project/aip-wangcs/indrisch/LLaMA-Factory
    -B /home/indrisch
    -B /dev/shm:/dev/shm
    -B /etc/ssl/certs:/etc/ssl/certs:ro
    -B /etc/pki:/etc/pki:ro
    -B "${MPI_LIB_PATH}:${MPI_LIB_PATH}:ro"
    -B "${HWLOC_LIB_PATH}:${HWLOC_LIB_PATH}:ro"
)

# Common environment variables
ENV_VARS=(
    --env LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/lib64:/lib/x86_64-linux-gnu:/lib64:${LD_LIBRARY_PATH}"
)

# First, run the installation script using exec
echo "Installing missing MPI dependencies..."
apptainer exec --nv --writable-tmpfs \
    "${BIND_MOUNTS[@]}" \
    "${ENV_VARS[@]}" \
    /project/aip-wangcs/indrisch/easyr1_verl_sif/llamafactory.sif \
    bash "${INSTALL_SCRIPT}"

# Then launch interactive shell
echo "Starting interactive shell..."
apptainer shell --nv --writable-tmpfs \
    "${BIND_MOUNTS[@]}" \
    "${ENV_VARS[@]}" \
    /project/aip-wangcs/indrisch/easyr1_verl_sif/llamafactory.sif
