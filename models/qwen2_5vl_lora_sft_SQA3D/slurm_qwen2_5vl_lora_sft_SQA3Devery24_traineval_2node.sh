#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=0-00:02:00
#SBATCH --gpus-per-node=h100:4
#SBATCH --output=%N-qwen2_5vl_lora_sft_SQA3Devery24_traineval_2node-%j.out

# This script will be identical to slurm_qwen2_5vl_lora_sft_SQA3Devery24_traineval.sh, but we will distribute 8 GPUs across 2 nodes.

# 1. Load modules
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
module load cuda/12.6 opencv/4.11 python/3.11
module load arrow
module load apptainer

TORCH_CUDA_ARCH_LIST="9.0" # for clusters with h100 GPUs

# 2. & 3. Create a virtualenv and install Ray on all nodes
#         Then, activate that virtualenv in this shell.
srun -N $SLURM_NNODES -n $SLURM_NNODES config_env.sh
source /scratch/indrisch/venv_llamafactory/bin/activate

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


# Launch head and workers on any/all nodes allocated to the job.
srun -N $SLURM_NNODES -n $SLURM_NNODES                       \
     --multi-prog "$ray_multiprog_config"                    \
     ${SLURM_CPUS_ON_NODE:+--num-cpus} ${SLURM_CPUS_ON_NODE} \
     ${SLURM_GPUS_ON_NODE:+--num-gpus} ${SLURM_GPUS_ON_NODE} &
ray_cluster_pid=$!


# Run your own script here
# STEP 1: RUN THE TRAINING AND EVALUATION

# better to have triton cache on a non-nfs file system for speed
# if we are offline, we need to indicate this
/scratch/indrisch/LLaMA-Factory/preliminaries/qwen2_5vl_lora_sft_SQA3D/slurm_qwen2_5vl_lora_sft_SQA3Devery24_traineval_2node_ray.sh 0 $HEAD_NODE $RAY_PORT
/scratch/indrisch/LLaMA-Factory/preliminaries/qwen2_5vl_lora_sft_SQA3D/slurm_qwen2_5vl_lora_sft_SQA3Devery24_traineval_2node_ray.sh 1 $HEAD_NODE $RAY_PORT

sleep 10


# Shut down Ray worker nodes after the Python script exits.
# Await its exit.
kill $ray_cluster_pid
wait


# Stage out your results, if any
# cp $SLURM_TMPDIR/results.json ~/scratch/