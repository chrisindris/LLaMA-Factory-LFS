#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=out/%N-qwen2_5vl_lora_sft_CoT_prediction_dump_smoke-%j.out
#SBATCH --cpus-per-task=24
#SBATCH --time=0-00:10:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --mail-user=christopher.indris@torontomu.ca
#SBATCH --mail-type=ALL

# Short GPU smoke for QUESTION_ID prediction dumps.
# Submit from models/qwen2_5vl_lora_sft_CoT/:
#   sbatch trillium_slurm_qwen2_5vl_lora_sft_CoT_prediction_dump_smoke.sh
#
# Uses examples/train_lora/trillium_qwen2_5vl_lora_sft_CoT_prediction_dump_smoke.yaml
# which sets max_samples: 8 (per dataset), max_steps: 4, and
# save_train_predictions / save_eval_predictions with teacher_forced mode.
#
# After the job, check:
#   saves/qwen2_5vl-7b/lora/sft/CoT_prediction_dump_smoke/train_predictions.json
#   saves/qwen2_5vl-7b/lora/sft/CoT_prediction_dump_smoke/eval_predictions.json

set -euo pipefail

# Prefer the longest matching project root (copy / LFS / main).
if [[ "$PWD" == *LLaMA-Factory-LFS* ]]; then
	PROJECT_DIR="${PWD%%LLaMA-Factory-LFS*}/LLaMA-Factory-LFS"
elif [[ "$PWD" == *LLaMA-Factory-copy* ]]; then
	PROJECT_DIR="${PWD%%LLaMA-Factory-copy*}/LLaMA-Factory-copy"
elif [[ "$PWD" == *LLaMA-Factory* ]]; then
	PROJECT_DIR="${PWD%%LLaMA-Factory*}/LLaMA-Factory"
else
	echo "Error: Could not find 'LLaMA-Factory' (or -copy / -LFS) in the current path."
	exit 1
fi

# Normalize (avoid double slashes like /scratch/indrisch//LLaMA-Factory-copy)
PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd)"

SYSCONFIG_DIR_PATH="$PROJECT_DIR/scripts"
# CRITICAL: the shared venv is usually an editable install of /scratch/.../LLaMA-Factory,
# not this tree. Prepend THIS project's src so prediction-dump flags resolve here.
export PYTHONPATH="${PROJECT_DIR}/src:${SYSCONFIG_DIR_PATH}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

mkdir -p "${PROJECT_DIR}/models/qwen2_5vl_lora_sft_CoT/out"
mkdir -p "${PROJECT_DIR}/saves/qwen2_5vl-7b/lora/sft/CoT_prediction_dump_smoke"

# Cluster / H5 roots (same env pattern as full CoT jobs)
if [[ "${HOSTNAME:-}" == *"trig"* ]] || [[ "${PS1:-}" == *"trig"* ]]; then
	CLUSTER="TRILLIUM"
elif [[ "${HOSTNAME:-}" == *"nibi"* ]] || [[ "${PS1:-}" == *"nibi"* ]]; then
	CLUSTER="NIBI"
elif [[ "${HOSTNAME:-}" == *"rg"* ]] || [[ "${HOSTNAME:-}" == *"rorqual"* ]]; then
	CLUSTER="RORQUAL"
else
	CLUSTER="TRILLIUM"
fi

export RUNNING_MODE="SMOKE"

export SCANNET_H5_DIR="${SCANNET_H5_DIR:-/scratch/indrisch/ScanNet_h5/scans}"
export SPATIALSSRL_H5_DIR="${SPATIALSSRL_H5_DIR:-/scratch/indrisch/Spatial-SSRL_images_h5}"
export THINKER10K_H5_DIR="${THINKER10K_H5_DIR:-/scratch/indrisch/3DThinker10K_images_h5}"

# --- Offline Hugging Face (compute nodes have no outbound net) ---
# Defaults match the local hub tree; override via sbatch --export if needed.
export HF_HOME="${HF_HOME:-/scratch/indrisch/huggingface/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/scratch/indrisch/huggingface/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HUB_CACHE}}"
# Optional cluster-specific overrides from sysconfig (only when import works).
if command -v python3 >/dev/null 2>&1 && python3 -c "import sysconfigtool" 2>/dev/null; then
	_hf_home="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'HF_HOME'))" 2>/dev/null || true)"
	_hf_cache="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'HF_HUB_CACHE'))" 2>/dev/null || true)"
	_triton="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'TRITON_CACHE_DIR'))" 2>/dev/null || true)"
	[[ -n "${_hf_home}" && "${_hf_home}" != "None" ]] && export HF_HOME="${_hf_home}"
	[[ -n "${_hf_cache}" && "${_hf_cache}" != "None" ]] && export HF_HUB_CACHE="${_hf_cache}"
	[[ -n "${_triton}" && "${_triton}" != "None" ]] && export TRITON_CACHE_DIR="${_triton}"
	export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
	export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
fi
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/scratch/indrisch/.triton_cache}"
# Force pure local reads — no HEAD/GET to huggingface.co
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
# Avoid accidental wandb online calls if someone flips report_to
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

cd "$PROJECT_DIR"

module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
module load python/3.12 cuda/12.6 opencv/4.12.0
module load arrow

# Prefer project venv if present
if [[ -f /scratch/indrisch/venv_llamafactory_cu126/bin/activate ]]; then
	# shellcheck disable=SC1091
	source /scratch/indrisch/venv_llamafactory_cu126/bin/activate
fi

# Re-assert PYTHONPATH after module/venv (they often prepend site-packages).
export PYTHONPATH="${PROJECT_DIR}/src:${SYSCONFIG_DIR_PATH}${PYTHONPATH:+:$PYTHONPATH}"

export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
export FORCE_TORCHRUN=1

CONFIG="${PROJECT_DIR}/examples/train_lora/trillium_qwen2_5vl_lora_sft_CoT_prediction_dump_smoke.yaml"

echo "=== prediction dump smoke ==="
echo "PROJECT_DIR=$PROJECT_DIR"
echo "CLUSTER=$CLUSTER"
echo "CONFIG=$CONFIG"
echo "PYTHONPATH=$PYTHONPATH"
echo "HF_HOME=$HF_HOME"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
echo "SCANNET_H5_DIR=$SCANNET_H5_DIR"
echo "SPATIALSSRL_H5_DIR=$SPATIALSSRL_H5_DIR"
echo "THINKER10K_H5_DIR=$THINKER10K_H5_DIR"
python - <<'PY'
import llamafactory, pathlib
print("llamafactory=", llamafactory.__file__)
# Fail early if the shared venv still wins over this tree.
path = pathlib.Path(llamafactory.__file__).resolve()
if "LLaMA-Factory-copy" not in str(path) and "LLaMA-Factory-LFS" not in str(path):
    # Allow plain LLaMA-Factory only when that tree itself has the flags.
    fa = path.parent / "hparams" / "finetuning_args.py"
    text = fa.read_text(encoding="utf-8") if fa.is_file() else ""
    if "save_train_predictions" not in text:
        raise SystemExit(
            f"Loaded llamafactory without prediction-dump flags: {path}\n"
            "Fix: ensure PYTHONPATH starts with $PROJECT_DIR/src"
        )
print("flags_ok=True")
PY
echo "============================="

# Optional: stamp IDs if not already present (writes sibling files; does not mutate hub snapshots
# unless you pass --in-place / --update-dataset-info). Uncomment when ready:
# python scripts/assign_question_ids.py \
#   --from-dataset-info data/dataset_info.json \
#   --datasets Scene30k,SpatialSSRL_coldstart,3DThinker10k

# Prefer invoking the local package entrypoint so torchrun child processes inherit PYTHONPATH.
llamafactory-cli train "$CONFIG" "$@"

OUT_DIR="${PROJECT_DIR}/saves/qwen2_5vl-7b/lora/sft/CoT_prediction_dump_smoke"
echo "=== dump artifacts ==="
ls -la "${OUT_DIR}/train_predictions.json" "${OUT_DIR}/eval_predictions.json" 2>/dev/null ||
	echo "Prediction JSON not found yet (check logs / question_id column)."
echo "======================"
