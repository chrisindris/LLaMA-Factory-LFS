import json
from pathlib import Path
import pandas as pd
import wandb
import re

# Standalone eval dump for the 3-epoch merged checkpoint (no train_predictions.json here).
EVAL_DIR = Path(
    "/scratch/i/indrisch/LLaMA-Factory-LFS/models/"
    "Qwen2.5-VL-7B-Instruct/lora/eval"
)
eval_predictions_path = EVAL_DIR / "eval_predictions.json"
eval_results_path = EVAL_DIR / "eval_results.json"
# 3 epochs × 620 steps/epoch from the CoT train YAML (eval_steps / save_steps).
EVAL_STEP = 1860


def get_json_from_path(pth):
    with open(pth, 'r') as f:
        jsn = json.load(f)
    return jsn


e_json = get_json_from_path(eval_predictions_path)
r_json = get_json_from_path(eval_results_path)


# QUESTION_IDs are often hub paths + trailing row index, e.g.
#   .../datasets--cvis-tmu--Scene30K/.../train-00000-of-00001.parquet_4102
#   .../datasets--internlm--Spatial-SSRL-81k/.../SFT-coldstart.json_3350
#   3DThinker-10K/out/3dthinker10k_cot.jsonl_4039
_DATASET_PATTERNS = (
    (re.compile(r"scene30k", re.I), "Scene30k"),
    (re.compile(r"spatial[-_]?ssrl|sft-coldstart", re.I), "SpatialSSRL_coldstart"),
    (re.compile(r"3dthinker10k_cot|3dthinker", re.I), "3DThinker10k"),
)
_QID_RE = re.compile(r"_(\d+)$")


def extract_dataset_and_qid(question: str):
    qid_m = _QID_RE.search(question)
    qid = qid_m.group(1) if qid_m else None
    for pat, name in _DATASET_PATTERNS:
        if pat.search(question):
            return name, qid
    return "UNKNOWN", qid


def hf_eval_metrics_to_wandb(metrics: dict) -> dict:
    """Map HF trainer keys (`eval_loss`) to wandb panel keys (`eval/loss`).

    W&B groups charts by the prefix before `/`, so these land in the eval section
    as line plots (same rewrite HuggingFace's wandb callback uses).
    """
    out = {}
    for key, value in metrics.items():
        if key.startswith("eval_"):
            out["eval/" + key[len("eval_"):]] = value
        elif key.startswith("eval/"):
            out[key] = value
        else:
            out["eval/" + key] = value
    return out


K, V, datasets, qids = [], [], [], []
for k, v in e_json.items():
    K.append(k)
    d, q = extract_dataset_and_qid(k)
    datasets.append(d)
    qids.append(q)
    V.append(v)


e_df = pd.DataFrame({'ID': K,
                     'dataset': datasets,
                     'qid': qids,
                     'prediction': V})

e_table = wandb.Table(dataframe=e_df)
eval_metrics = hf_eval_metrics_to_wandb(r_json)

run = wandb.init(
    entity="cvis_tmu",
    project="llamafactory",
    id="Qwen2.5-VL-7B-Instruct_CoT_eval",
    resume="allow",
)

# One step so the table and eval scalars share a history row (x = global step).
run.log({"eval_predictions": e_table, **eval_metrics}, step=EVAL_STEP)

run.finish()

