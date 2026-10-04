---
name: alliancecan-distributed
description: >
  LLaMA-Factory distributed training: choose and launch NativeDDP, DeepSpeed (ZeRO), FSDP/FSDP2,
  or Ray; single-node multi-GPU and multi-node multi-card via FORCE_TORCHRUN, torchrun, accelerate,
  or deepspeed. Use when the user mentions multi-GPU, multi-node, DDP, FSDP, FSDP2, ZeRO stage,
  FORCE_TORCHRUN, MASTER_ADDR, accelerate launch, hostfile, USE_RAY, or runs /alliancecan-distributed.
  Sibling of /alliancecan (cluster env) and /alliancecan-deepspeed (AllianceCan DeepSpeed SLURM).
---

# LLaMA-Factory distributed training

Use for configuring and launching multi-GPU / multi-node LLaMA-Factory jobs. On AllianceCan, also
apply **`alliancecan`** (login vs compute, modules, venv). For cluster-specific DeepSpeed + SLURM
`torchrun`/`srun` patterns, load **`alliancecan-deepspeed`**.

Source: `references/llamafactory.readthedocs.io-Distributed Training.pdf` (docs:
https://llamafactory.readthedocs.io/en/latest/advanced/distributed.html).

## Engine comparison

| Engine | Data shard | Model shard | Optimizer shard | Param offload |
|--------|------------|-------------|-----------------|---------------|
| **DDP** (NativeDDP) | Yes | No | No | No |
| **DeepSpeed** | Yes | Yes | Yes | Yes |
| **FSDP** | Yes | Yes | Yes | Yes |
| **FSDP2** | Yes | Yes | Yes | Yes |

- **DDP**: full model + optimizer replica per GPU; accelerate via data/model parallel process group.
- **DeepSpeed**: ZeRO + offload + other Microsoft optimizations.
- **FSDP**: shards params/grads/optimizer; can offload params to CPU.
- **FSDP2**: FSDP1 features with per-parameter DTensor sharding (better compute/comm overlap); used
  by current LLaMA-Factory Accelerate integration for FSDP2.

## Choosing ZeRO (DeepSpeed)

Estimate memory, then pick stage:

| Stage | Shards | Memory | Speed |
|-------|--------|--------|-------|
| **ZeRO-1** | Optimizer | Higher | Faster |
| **ZeRO-2** | Optimizer + grads | Medium | Medium |
| **ZeRO-3** | Optimizer + grads + params | Lowest | Slowest |

Rule of thumb from LLaMA-Factory docs:

- Enough memory → prefer **ZeRO-1** and **`offload_param=none`**.
- Tight memory → raise stage; `offload_param=cpu` saves GPU memory but **slows training a lot**.

Example JSON configs under the repo (when present):

- `examples/deepspeed/ds_z0_config.json` — stage 0 (ZeRO disabled)
- `examples/deepspeed/ds_z2_config.json` — stage 2
- `examples/deepspeed/ds_z2_offload_config.json` — stage 2 + optimizer offload CPU
- `examples/deepspeed/ds_z3_config.json` — stage 3
- `examples/deepspeed/ds_z3_offload_config.json` — stage 3 + optimizer/param offload CPU
- AutoTP (ZeRO-1/2 + tensor parallel): needs **DeepSpeed ≥ 0.16.4**; limited model support

In training YAML, point DeepSpeed at the JSON:

```yaml
deepspeed: examples/deepspeed/ds_z3_config.json
```

### Config snippets

**ZeRO-0 base fields** (then change `stage` / offload):

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "zero_allow_untested_optimizer": true,
  "fp16": { "enabled": "auto", "loss_scale": 0, "loss_scale_window": 1000,
            "initial_scale_power": 16, "hysteresis": 2, "min_loss_scale": 1 },
  "bf16": { "enabled": "auto" },
  "zero_optimization": {
    "stage": 0,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true,
    "round_robin_gradients": true
  }
}
```

- ZeRO-2: set `"stage": 2`
- ZeRO-2 offload: add `"offload_optimizer": { "device": "cpu", "pin_memory": true }`
- ZeRO-3: stage 3 + stage3-specific knobs (`overlap_comm`, `contiguous_gradients`,
  `sub_group_size`, `reduce_bucket_size`, prefetch/persistence, etc.)
- ZeRO-3 offload: add both `offload_optimizer` and `offload_param` to CPU
- AutoTP example: `"tensor_parallel": { "autotp_size": 4 }` with stage 1 or 2

## Launch methods

### 1) NativeDDP — single-node multi-GPU

**llamafactory-cli** (recommended simple path):

```bash
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/llama3_full_sft_ds3.yaml
# optional GPU subset:
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train config/config1.yaml
```

If `CUDA_VISIBLE_DEVICES` is unset, all visible GPUs are used.

**torchrun**:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=8 src/train.py \
  --stage sft --model_name_or_path ... --do_train ... --bf16
```

**accelerate**: generate config (`accelerate config`) or edit YAML, then:

```bash
accelerate launch --config_file accelerate_singleNode_config.yaml \
  src/train.py training_config.yaml
```

Key accelerate fields: `num_machines: 1`, `num_processes: <num_gpus>`,
`distributed_type: MULTI_GPU`, `mixed_precision: fp16|bf16`.

### 2) NativeDDP — multi-node multi-card

**llamafactory-cli** (run appropriate rank on each node):

```bash
FORCE_TORCHRUN=1 NNODES=2 NODE_RANK=0 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 \
  llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml

FORCE_TORCHRUN=1 NNODES=2 NODE_RANK=1 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 \
  llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
```

| Variable | Meaning |
|----------|---------|
| `FORCE_TORCHRUN` | Force torchrun launch path |
| `NNODES` | Number of nodes |
| `NODE_RANK` | Rank of this node (0 = master) |
| `MASTER_ADDR` | Master node address |
| `MASTER_PORT` | Master port |

**torchrun** equivalent: set `--nnodes`, `--nproc_per_node`, `--node_rank`, `--master_addr`,
`--master_port` on each node.

**accelerate** multi-node: set `num_machines`, `num_processes` (total GPUs), `main_process_ip`,
`main_process_port`, `machine_rank` (per node), then `accelerate launch --config_file ...`.

On AllianceCan SLURM, map these from `SLURM_NNODES`, `SLURM_NODEID`, head hostname, etc.; prefer
patterns in **`alliancecan-deepspeed`** for rendezvous + `srun`.

### 3) DeepSpeed — single-node

Via **llamafactory-cli** + YAML `deepspeed: path/to/ds_*.json`:

```bash
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/llama3_full_sft_ds3.yaml
```

Via **deepspeed** CLI:

```bash
deepspeed --num_gpus 8 src/train.py --deepspeed examples/deepspeed/ds_z3_config.json ...
```

Notes:

- With the `deepspeed` CLI, **do not use `CUDA_VISIBLE_DEVICES`** to pick GPUs; use
  `deepspeed --include localhost:1 your_program.py ...` (e.g. only gpu1).
- On AllianceCan, **prefer torchrun-style launch** (see **`alliancecan-deepspeed`**) over
  DeepSpeed’s multi-node launcher when docs conflict.

### 4) DeepSpeed — multi-node

**llamafactory-cli** with `FORCE_TORCHRUN` + `NNODES` / `NODE_RANK` / `MASTER_*` and a
`*_ds*.yaml` that references a ZeRO config.

**deepspeed** CLI multi-node (generic docs):

```bash
deepspeed --num_gpus 8 --num_nodes 2 --hostfile hostfile \
  --master_addr hostname1 --master_port=9901 \
  your_program.py ... --deepspeed ds_config.json
```

Hostfile format:

```text
worker-1 slots=4
worker-2 slots=4
```

If `hostfile` is omitted, DeepSpeed may look for `/job/hostfile`, else use all local GPUs.

**accelerate** + DeepSpeed: config with `distributed_type: DEEPSPEED`, `zero_stage`,
`offload_optimizer_device` / `offload_param_device`, `num_machines`, `num_processes`,
`main_process_ip` / `main_process_port`, then `accelerate launch --config_file deepspeed_config.yaml ...`.

### 5) FSDP / FSDP2

**ShardingStrategy** (FSDP1-style):

| Strategy | Meaning | ~ZeRO |
|----------|---------|-------|
| `FULL_SHARD` | Params + grads + optimizer | ZeRO-3 |
| `SHARD_GRAD_OP` | Grads + optimizer; full params per GPU | ZeRO-2 |
| `NO_SHARD` | No sharding | ZeRO-0 |

**llamafactory-cli** path for FSDP+QLoRA examples:

```bash
bash examples/extras/fsdp_qlora/train.sh
# edit examples/accelerate/fsdp_config.yaml and examples/extras/fsdp_qlora/*.yaml as needed
```

**accelerate**:

```bash
accelerate launch --config_file fsdp_config.yaml src/train.py llm_config.yaml
```

Critical: **`num_processes` must equal total GPUs used**. Multi-node FSDP also needs
`main_process_ip`, `main_process_port`, `machine_rank`, `num_machines`, and total
`num_processes = num_machines * gpus_per_machine`.

FSDP2 example config sets `fsdp_version: 2` and related keys (`fsdp_reshard_after_forward`, etc.)
under `examples/accelerate/fsdp2_config.yaml`.

**Warning:** Do **not** use GPTQ/AWQ models with **FSDP+QLoRA**.

### 6) Ray

Enable with `USE_RAY=1`.

Single-node NativeDDP/DeepSpeed:

```bash
USE_RAY=1 llamafactory-cli train training_config.yaml
```

FSDP with Ray: set `num_processes: 1` in `fsdp_config.yaml`, then:

```bash
USE_RAY=1 accelerate launch --config_file fsdp_config.yaml src/train.py training_config.yaml
```

Multi-node: start Ray on each node first:

```bash
# master
ray start --head --port=6379
# workers
ray start --address='<master_ip>:6379'
```

Then train on master, or:

```bash
RAY_API_SERVER_ADDRESS='http://dashboard-host:dashboard-port' \
  ray job submit -- llamafactory-cli train training_config.yaml
```

For multi-machine Ray, you generally do **not** set multi-machine accelerate fields; Ray schedules workers.

## AllianceCan mapping tips

1. Use **`alliancecan`** module + venv activation inside the SLURM script (or container).
2. Prefer **prebuilt venv / Apptainer** over `pip install` on compute (no internet).
3. Map master address to head-node hostname; map ranks from `SLURM_NODEID` / `SLURM_PROCID`.
4. For DeepSpeed multi-node on this cluster, follow **`alliancecan-deepspeed`** (`torchrun`,
   `NCCL_ASYNC_ERROR_HANDLING`, `$SLURM_TMPDIR` for offload).
5. Stage datasets and models on `/scratch` or `/project` before submit; no downloads on compute.
6. Match `num_processes` / `--nproc_per_node` to **allocated GPUs**, not login-node counts.

## Agent checklist

1. Pick engine from memory/model size: DDP (simple) → DeepSpeed ZeRO → FSDP/FSDP2.
2. Ensure training YAML and ds/fsdp accelerate configs agree (batch, precision, offload).
3. Single-node: `FORCE_TORCHRUN=1` + correct GPU count is often enough.
4. Multi-node: set **same** `MASTER_ADDR`/`MASTER_PORT` on all ranks; unique `NODE_RANK` /
   `machine_rank`.
5. Verify total process count = total GPUs.
6. Avoid GPTQ/AWQ + FSDP+QLoRA; avoid DeepSpeed launcher on AllianceCan when torchrun works.
7. Point users at repo examples under `examples/deepspeed/`, `examples/accelerate/`,
   `examples/extras/fsdp_qlora/` when they exist.
