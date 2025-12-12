#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:05:00
#SBATCH --mem=8GB
#SBATCH --gpus-per-node=1
#SBATCH --output=%N-qwen2_5vl_lora_sft_SQA3Devery24_traineval_2node-%j.out

# 1. Load modules
module load python


# 2. & 3. Create a virtualenv and install Ray on all nodes
#         Then, activate that virtualenv in this shell.
echo "Setting up virtual environment and installing Ray on all nodes..."
srun -N $SLURM_NNODES -n $SLURM_NNODES config_env.sh
source $SLURM_TMPDIR/ENV/bin/activate
echo "Virtual environment activated."


# Define Ray head node address (this node) & port (any free port will do)
export HEAD_NODE=$(hostname)
export RAY_PORT=$(
    python -c 'import socket;             \
               s=socket.socket();         \
               s.bind(("", 0));           \
               print(s.getsockname()[1]); \
               s.close()'
)


# Craft --multi-prog configuration for Ray
ray_multiprog_config="$SLURM_TMPDIR/ray-multiprog.conf"
cat <<EOF >"$ray_multiprog_config"
0   ray start --block --head --node-ip-address=$HEAD_NODE --port=$RAY_PORT
*   ray start --block --address=${HEAD_NODE}:${RAY_PORT}
EOF

#
# Some older Ray versions malfunctioned if the worker nodes started before
# the head node. In that case, please modify the worker-node launch configuration
# above to add a 10-second delay, using the following alternative lines:
#
# cat <<EOF >"$ray_multiprog_config"
# 0   ray start --block --head --node-ip-address=$HEAD_NODE --port=$RAY_PORT
# *   sh -c 'sleep 10; exec "\$@"' sh ray start --block --address=${HEAD_NODE}:${RAY_PORT}
# EOF
#


# Set up cleanup trap
cleanup_ray() {
    if [ -n "$ray_cluster_pid" ] && kill -0 $ray_cluster_pid 2>/dev/null; then
        echo "Cleaning up Ray cluster..." >&2
        srun -N $SLURM_NNODES -n $SLURM_NNODES ray stop --force || true
        kill -TERM $ray_cluster_pid 2>/dev/null || true
        sleep 1
        kill -KILL $ray_cluster_pid 2>/dev/null || true
    fi
}
trap cleanup_ray EXIT TERM INT

# Launch head and workers on any/all nodes allocated to the job.
echo "Starting Ray cluster on all nodes..."
srun -N $SLURM_NNODES -n $SLURM_NNODES                       \
     --multi-prog "$ray_multiprog_config"                    \
     ${SLURM_CPUS_ON_NODE:+--num-cpus} ${SLURM_CPUS_ON_NODE} \
     ${SLURM_GPUS_ON_NODE:+--num-gpus} ${SLURM_GPUS_ON_NODE} &
ray_cluster_pid=$!

# Wait a bit for Ray cluster to start
echo "Waiting for Ray cluster to initialize..."
sleep 5
echo "Ray cluster initialization wait complete."

# Run your own script here
echo "Running Python script..."
python -u slurm_qwen2_5vl_lora_sft_SQA3Devery24_traineval_2node_ray.py "$@"
script_exit_code=$?
echo "Python script completed with exit code: $script_exit_code"

# Shut down Ray worker nodes after the Python script exits.
echo "Shutting down Ray cluster..."

# Disable trap during shutdown to avoid double cleanup
trap - EXIT TERM INT

# Stop Ray gracefully - this will cause ray start --block to exit
# Run in background with timeout to prevent hanging
echo "Attempting graceful Ray stop..."
(srun -N $SLURM_NNODES -n $SLURM_NNODES ray stop || true) &
stop_pid=$!
(sleep 3 && kill $stop_pid 2>/dev/null || true) &
stop_timeout=$!

# Wait for srun process (running ray start --block) to exit
# Ray should stop when we call ray stop above, causing ray start --block to exit
max_wait=5
for i in $(seq 1 $max_wait); do
    if ! kill -0 $ray_cluster_pid 2>/dev/null; then
        echo "Ray cluster stopped gracefully."
        kill $stop_timeout 2>/dev/null || true
        wait $stop_pid 2>/dev/null || true
        exit $script_exit_code
    fi
    sleep 1
done

# If still running, try force stop
kill $stop_timeout 2>/dev/null || true
wait $stop_pid 2>/dev/null || true

echo "Trying force stop..."
(srun -N $SLURM_NNODES -n $SLURM_NNODES ray stop --force || true) &
force_stop_pid=$!
(sleep 2 && kill $force_stop_pid 2>/dev/null || true) &
force_timeout=$!

# Wait a bit more for srun to exit
for i in {1..3}; do
    if ! kill -0 $ray_cluster_pid 2>/dev/null; then
        echo "Ray cluster stopped after force stop."
        kill $force_timeout 2>/dev/null || true
        wait $force_stop_pid 2>/dev/null || true
        exit $script_exit_code
    fi
    sleep 1
done

# Last resort: kill the srun process directly
kill $force_timeout 2>/dev/null || true
wait $force_stop_pid 2>/dev/null || true

if kill -0 $ray_cluster_pid 2>/dev/null; then
    echo "Force killing Ray cluster processes..."
    kill -TERM $ray_cluster_pid 2>/dev/null || true
    sleep 1
    if kill -0 $ray_cluster_pid 2>/dev/null; then
        kill -KILL $ray_cluster_pid 2>/dev/null || true
    fi
fi

echo "Ray cluster shutdown complete."

# Exit with the script's exit code
exit $script_exit_code


# Stage out your results, if any
# cp $SLURM_TMPDIR/results.json ~/scratch/

# # This script will be identical to slurm_qwen2_5vl_lora_sft_SQA3Devery24_traineval.sh, but we will distribute 8 GPUs across 2 nodes.

# # 1. Load modules
# module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
# module load cuda/12.6 opencv/4.11 python/3.11
# module load arrow
# module load apptainer

# export TORCH_CUDA_ARCH_LIST="9.0" # for clusters with h100 GPUs

# # 2. & 3. Create a virtualenv and install Ray on all nodes
# #         Then, activate that virtualenv in this shell.
# SCRIPT_DIR="/project/aip-wangcs/indrisch/LLaMA-Factory/preliminaries/qwen2_5vl_lora_sft_SQA3D"
# srun -N $SLURM_NNODES -n $SLURM_NNODES "$SCRIPT_DIR/config_env.sh"
# source $SLURM_TMPDIR/ENV/bin/activate

# # Define Ray head node address (this node) & port (any free port will do)
# export HEAD_NODE=$(hostname)
# export RAY_PORT=$(
#     python -c 'import socket;             \
#                s=socket.socket();         \
#                s.bind(("", 0));           \
#                print(s.getsockname()[1]); \
#                s.close()'
# )


# # Craft --multi-prog configuration for Ray
# ray_multiprog_config="$SLURM_TMPDIR/ray-multiprog.conf"
# cat <<EOF >"$ray_multiprog_config"
# 0   ray start --block --head --node-ip-address=$HEAD_NODE --port=$RAY_PORT
# *   ray start --block --address=${HEAD_NODE}:${RAY_PORT}
# EOF

# #
# # Some older Ray versions malfunctioned if the worker nodes started before
# # the head node. In that case, please modify the worker-node launch configuration
# # above to add a 10-second delay, using the following alternative lines:
# #
# # cat <<EOF >"$ray_multiprog_config"
# # 0   ray start --block --head --node-ip-address=$HEAD_NODE --port=$RAY_PORT
# # *   sh -c 'sleep 10; exec "\$@"' sh ray start --block --address=${HEAD_NODE}:${RAY_PORT}
# # EOF
# #


# # Launch head and workers on any/all nodes allocated to the job.
# srun -N $SLURM_NNODES -n $SLURM_NNODES                       \
#      --multi-prog "$ray_multiprog_config"                    \
#      ${SLURM_CPUS_ON_NODE:+--num-cpus} ${SLURM_CPUS_ON_NODE} \
#      ${SLURM_GPUS_ON_NODE:+--num-gpus} ${SLURM_GPUS_ON_NODE} &
# ray_cluster_pid=$!

# # Wait for Ray cluster to be ready
# echo "Waiting for Ray cluster to initialize..."
# sleep 5
# # Verify Ray cluster is accessible
# python -u -c "
# import ray
# import sys
# try:
#     ray.init(address='${HEAD_NODE}:${RAY_PORT}', ignore_reinit_error=True)
#     print('Ray cluster connected successfully', flush=True)
#     print(f'Available resources: {ray.available_resources()}', flush=True)
#     # Try to disconnect if method exists (newer Ray versions), otherwise just exit
#     if hasattr(ray, 'disconnect'):
#         ray.disconnect()
#     elif hasattr(ray, 'shutdown'):
#         # Don't shutdown the cluster, just exit - the cluster should keep running
#         pass
#     print('Ray verification complete', flush=True)
#     sys.exit(0)
# except Exception as e:
#     print(f'Warning: Ray cluster may not be fully ready yet: {e}', file=sys.stderr, flush=True)
#     sys.exit(0)
# " || echo "Warning: Ray cluster verification had issues, but continuing..."

# echo "Ray cluster verification complete. Starting training..."

# # Run your own script here
# # STEP 1: RUN THE TRAINING AND EVALUATION

# # better to have triton cache on a non-nfs file system for speed
# # if we are offline, we need to indicate this
# # Run training on both nodes in parallel using srun with multi-prog
# # ray_train_config="$SLURM_TMPDIR/ray-train-multiprog.conf"
# # cat <<EOF >"$ray_train_config"
# # 0   $SCRIPT_DIR/slurm_qwen2_5vl_lora_sft_SQA3Devery24_traineval_2node_ray.sh 0 $HEAD_NODE $RAY_PORT
# # 1   $SCRIPT_DIR/slurm_qwen2_5vl_lora_sft_SQA3Devery24_traineval_2node_ray.sh 1 $HEAD_NODE $RAY_PORT
# # EOF

# # echo "Launching training on both nodes..."
# # srun -N $SLURM_NNODES -n $SLURM_NNODES --multi-prog "$ray_train_config"
# # training_exit_code=$?

# # if [ $training_exit_code -ne 0 ]; then
# #     echo "Warning: Training srun command exited with code $training_exit_code" >&2
# # fi

# python $SCRIPT_DIR/slurm_qwen2_5vl_lora_sft_SQA3Devery24_traineval_2node_ray.py "$@"


# # Shut down Ray worker nodes after the Python script exits.
# # Await its exit.
# kill $ray_cluster_pid
# wait


# # Stage out your results, if any
# # cp $SLURM_TMPDIR/results.json ~/scratch/