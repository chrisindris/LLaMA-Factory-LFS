import json
from numpy import datetime_as_string
import pandas as pd
import wandb
import re

eval_path="/scratch/i/indrisch/LLaMA-Factory-LFS/saves/qwen2_5vl-7b/lora/sft/CoT_traineval_resume_ep1/eval_predictions_ep1.json"
train_path="/scratch/i/indrisch/LLaMA-Factory-LFS/saves/qwen2_5vl-7b/lora/sft/CoT_traineval_resume_ep1/train_predictions_ep1.json"


def get_json_from_path(pth):
    with open(pth, 'r') as f:
        jsn = json.load(f)
    return jsn


e_json = get_json_from_path(eval_path)
t_json = get_json_from_path(train_path)



def extract_dataset_and_qid(question: str):
    r = re.match(r'.*(Scene30k|SpatialSSRL_coldstart|3dthinker10k_cot).*_(\d+)', question)
    return r.group(1), r.group(2)


K, V, datasets, qids = [], [], [], []
for k,v in e_json.items():
    K.append(k)
    d, q = extract_dataset_and_qid(k)
    datasets.append(d)
    qids.append(q)
    V.append(v)


e_df = pd.DataFrame({'ID': K,
                     'dataset': datasets,
                     'qid': qids,
                     'prediction': V})


K, V, datasets, qids, steps = [], [], [], [], []
for k,v in t_json.items():
    K.append(k)
    d, q = extract_dataset_and_qid(k)
    datasets.append(d)
    qids.append(q)
    step, response = list(v.items())[0]
    steps.append(step)
    V.append(response)


t_df = pd.DataFrame({'ID': K,
                     'dataset': datasets,
                     'qid': qids,
                     'step': steps,
                     'prediction': V})


e_table = wandb.Table(dataframe=e_df)
t_table = wandb.Table(dataframe=t_df)

run = wandb.init(
    entity="cvis_tmu",
    project="llamafactory",
    id="qwen2_5vl-7b-lora-sft-CoT_traineval_1epochs_traininglog",  # The unique 8-character ID from your first run
    resume="allow",
)

run.log({"eval_predictions_ep1": e_table})
run.log({"train_predictions_ep1": t_table})

run.finish()
