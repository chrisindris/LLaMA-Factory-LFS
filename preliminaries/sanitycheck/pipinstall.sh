# regarding memory; trillium doesn't use a --mem option and the default of 256M on nibi is needed, so we can leave it out.

module load apptainer

BASE_SIF="/scratch/indrisch/easyr1_verl_sif/llamafactory.sif"
OUTPUT_SIF="/scratch/indrisch/easyr1_verl_sif/llamafactory_unsloth.sif"
SANDBOX_DIR="/scratch/indrisch/easyr1_verl_sif/llamafactory_unsloth_sandbox"

# Clean up any existing sandbox directory
if [ -d "$SANDBOX_DIR" ]; then
    rm -rf "$SANDBOX_DIR"
fi

# Create a sandbox (writable directory) from the base .sif file
echo "Creating sandbox from base image..."
apptainer build --sandbox "$SANDBOX_DIR" "$BASE_SIF"

# Install unsloth in the sandbox
echo "Installing unsloth in sandbox..."
apptainer exec --nv \
    -B /scratch/indrisch/LLaMA-Factory \
    -B /home/indrisch \
    -B /project/def-wangcs/indrisch \
    -B /dev/shm:/dev/shm \
    -B /etc/ssl/certs:/etc/ssl/certs:ro \
    -B /etc/pki:/etc/pki:ro \
    --writable "$SANDBOX_DIR" \
    python -m pip install unsloth

# Build the new .sif file from the sandbox
# The sandbox approach ensures sufficient space since it's a writable directory
echo "Building new .sif file from sandbox..."
apptainer build "$OUTPUT_SIF" "$SANDBOX_DIR"

# Clean up the sandbox directory
echo "Cleaning up sandbox..."
rm -rf "$SANDBOX_DIR"

echo "Done! New image created at: $OUTPUT_SIF"