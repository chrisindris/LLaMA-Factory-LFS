#!/bin/bash

NODE_RANK=$1
HEAD_NODE=$2
RAY_PORT=$3

echo "NODE_RANK: $NODE_RANK"
echo "HEAD_NODE: $HEAD_NODE"
echo "RAY_PORT: $RAY_PORT"

# # Connect to Ray cluster
# python -c "import ray; ray.init(address=f"{os.environ['HEAD_NODE']}:{os.environ['RAY_PORT']}",_node_ip_address=os.environ['HEAD_NODE'])"

# # Check that Ray sees two nodes and their status is 'Alive'
# python -c "print("Nodes in the Ray cluster:")"
# python -c "import ray; print(ray.nodes())"

# # Check that Ray sees 12 CPUs and 2 GPUs over 2 Nodes
# python -c "import ray; print(ray.available_resources())"



# apptainer run --nv --writable-tmpfs \
#     -B /project/aip-wangcs/indrisch/LLaMA-Factory \
#     -B /home/indrisch \
#     -B /dev/shm:/dev/shm \
#     -B /etc/ssl/certs:/etc/ssl/certs:ro \
#     -B /etc/pki:/etc/pki:ro \
#     -B /etc/pki/tls/certs/ca-bundle.crt \
#     -B /project/aip-wangcs/indrisch/ \
#     -B /project/aip-wangcs/indrisch/apptainer-home/ \
#     -B /project/aip-wangcs/indrisch/apptainer-tmp/ \
#     -W ${SLURM_TMPDIR} \
#     --env RAY_ADDRESS=${HEAD_NODE}:${RAY_PORT} \
#     --env RAY_NODE_IP_ADDRESS=${HEAD_NODE} \
#     --env RAY_PORT=${RAY_PORT} \
#     --env HEAD_NODE=${HEAD_NODE} \
#     --env HF_HUB_OFFLINE=1 \
#     --env MPLCONFIGDIR="${SLURM_TMPDIR}/.config/matplotlib" \
#     --env HF_HOME="/project/aip-wangcs/indrisch/huggingface/hub" \
#     --env HF_HUB_CACHE="/project/aip-wangcs/indrisch/huggingface/hub" \
#     --env TRITON_CACHE_DIR="${SLURM_TMPDIR}/.triton_cache" \
#     --env FLASHINFER_WORKSPACE_BASE="/project/aip-wangcs/indrisch/" \
#     --env TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
#     --env TORCH_EXTENSIONS_DIR="${SLURM_TMPDIR}/.cache/torch_extensions" \
#     --env PYTORCH_KERNEL_CACHE_PATH="${SLURM_TMPDIR}/.cache/torch/kernels" \
#     --env FORCE_TORCHRUN=1 \
#     --env NNODES=2 \
#     --env NODE_RANK=$NODE_RANK \
#     --env MASTER_ADDR=${HEAD_NODE} \
#     --env MASTER_PORT=${RAY_PORT} \
#     --env WANDB_MODE=offline \
#     --env WANDB_DIR="/project/aip-wangcs/indrisch/LLaMA-Factory/wandb/" \
#     --pwd /project/aip-wangcs/indrisch/LLaMA-Factory \
#     /project/aip-wangcs/indrisch/easyr1_verl_sif/llamafactory_wandb.sif \
#     llamafactory-cli train /project/aip-wangcs/indrisch/LLaMA-Factory/examples/train_lora/qwen2_5vl_lora_sft_SQA3Devery24_2node.yaml