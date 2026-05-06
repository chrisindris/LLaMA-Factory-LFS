#!/bin/bash
set -euo pipefail

# Run this on a compute node allocation (or inside your SLURM job), not on login.
PROJECT_DIR="/project/aip-wangcs/indrisch/LLaMA-Factory"
BASE_YAML="${PROJECT_DIR}/examples/train_lora/qwen2_5vl_lora_sft_SQA3Devery24_traineval_resumefromcheckpoint_epoch2.yaml"
CHECKPOINT_DIR="${PROJECT_DIR}/saves/qwen2_5vl-7b/lora/sft/SQA3Devery24_traineval_native8gpu_evalsteps200_continued/checkpoint-200"
OUTPUT_DIR="${PROJECT_DIR}/saves/qwen2_5vl-7b/lora/sft/SQA3Devery24_resume_smoke_from_checkpoint200"
RUN_LOG="${OUTPUT_DIR}/resume_smoke.log"

module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
module load python/3.12 cuda/12.6 opencv/4.12.0
module load arrow
source /project/aip-wangcs/indrisch/venv_llamafactory_cu126/bin/activate

export DISABLE_VERSION_CHECK=1
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_DIR}"

# Phase 1: parser-level preflight. This confirms auto-resume is selected before launching training.
python - <<'PY'
from llamafactory.hparams import get_train_args

project_dir = "/project/aip-wangcs/indrisch/LLaMA-Factory"
base_yaml = f"{project_dir}/examples/train_lora/qwen2_5vl_lora_sft_SQA3Devery24_traineval_resumefromcheckpoint_epoch2.yaml"
checkpoint_dir = f"{project_dir}/saves/qwen2_5vl-7b/lora/sft/SQA3Devery24_traineval_native8gpu_evalsteps200_continued/checkpoint-200"
output_dir = f"{project_dir}/saves/qwen2_5vl-7b/lora/sft/SQA3Devery24_resume_smoke_from_checkpoint200"

args = [
    base_yaml,
    f"adapter_name_or_path={checkpoint_dir}",
    f"output_dir={output_dir}",
    "overwrite_output_dir=true",
    "max_steps=202",
    "save_steps=202",
    "logging_steps=1",
    "report_to=none",
]

_, _, training_args, _, _ = get_train_args(args)
print(f"Resolved resume_from_checkpoint: {training_args.resume_from_checkpoint}")
if training_args.resume_from_checkpoint != checkpoint_dir:
    raise SystemExit("Auto-resume preflight failed: parser did not resolve checkpoint-200")
PY

# Phase 2: actual smoke run. Should perform only a couple of steps if resume works.
llamafactory-cli train \
    "${BASE_YAML}" \
    adapter_name_or_path="${CHECKPOINT_DIR}" \
    output_dir="${OUTPUT_DIR}" \
    overwrite_output_dir=true \
    max_steps=202 \
    save_steps=202 \
    logging_steps=1 \
    report_to=none \
    2>&1 | tee "${RUN_LOG}"

# Phase 3: continuity checks in logs/artifacts.
if ! grep -q "Resuming training from adapter checkpoint" "${RUN_LOG}"; then
    echo "ERROR: adapter auto-resume log line not found in ${RUN_LOG}"
    exit 1
fi

python - <<'PY'
import json
from pathlib import Path

output_dir = Path("/project/aip-wangcs/indrisch/LLaMA-Factory/saves/qwen2_5vl-7b/lora/sft/SQA3Devery24_resume_smoke_from_checkpoint200")
trainer_state_path = output_dir / "trainer_state.json"
trainer_log_path = output_dir / "trainer_log.jsonl"

if not trainer_state_path.exists():
    raise SystemExit(f"Missing {trainer_state_path}")

state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
final_step = int(state.get("global_step", -1))
if final_step < 202:
    raise SystemExit(f"Expected global_step >= 202 after smoke run, got {final_step}")

if not trainer_log_path.exists():
    raise SystemExit(f"Missing {trainer_log_path}")

lr_records = []
for line in trainer_log_path.read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    if "learning_rate" in row:
        lr_records.append(row)

if not lr_records:
    raise SystemExit("No learning_rate records found in trainer_log.jsonl")

print(f"Final global_step: {final_step}")
print(f"First logged LR record: {lr_records[0]}")
print(f"Last logged LR record: {lr_records[-1]}")
print("Smoke resume checks passed.")
PY

echo "Done. Resume smoke run completed successfully."