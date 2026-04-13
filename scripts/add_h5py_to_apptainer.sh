# Paths
SIF=/scratch/indrisch/huggingface/hub/datasets--cvis-tmu--compute_canada_sif_files/snapshots/be3f0f117608208681a73c5564ce39aebc41f718/llamafactory.sif
WHEELHOUSE=/scratch/indrisch/wheels/llamafactory_py311

mkdir -p "${WHEELHOUSE}"

# Confirm container Python version (should match job container runtime)
apptainer exec "${SIF}" python -c "import sys; print(sys.version)"

# Download h5py wheel using the container's Python/pip resolver
apptainer exec "${SIF}" python -m pip download \
  --only-binary=:all: \
  --no-deps \
  --dest "${WHEELHOUSE}" \
  "h5py==3.16.0"

# Verify wheel exists
ls -lh "${WHEELHOUSE}" | grep h5py