---
name: llamafactory-cli-eval
description: >
  Configure and run LLaMA-Factory evaluation via `llamafactory-cli train` (not
  `llamafactory-cli eval`, which is unsupported): eval-only YAML with do_train
  false, do_eval/do_predict, adapter loading, val_size/eval_dataset, generation
  metrics, and prediction dumps. Use when the user mentions model evaluation,
  eval-only jobs, do_eval, do_predict, nlg_eval, predict_with_generate,
  standalone eval YAML, or runs /llamafactory-cli-eval.
---

# llamafactory-cli eval (via train)

Help Grok build correct **eval-only** invocations for this repo.

## Hard rule

**Never run `llamafactory-cli eval`.** In this codebase it always raises:

```text
NotImplementedError: Evaluation will be deprecated in the future.
```

(see `src/llamafactory/launcher.py`). Benchmark / MMLU-style docs that still say
`llamafactory-cli eval …` are outdated.

**How to evaluate now:** call **`llamafactory-cli train`** with a YAML (or CLI
flags) that **does not enable training** — typically `do_train: false` plus
`do_eval: true` and/or `do_predict: true`.

Shared model/data/LoRA/distributed flag details live in **`llamafactory-cli-train`**.
Full CLI dump: `references/llamafactory-cli_train_-h.txt` (or re-run
`llamafactory-cli train -h` with the project venv). Cluster launch:
**`alliancecan`**, **`alliancecan-deepspeed`**, **`alliancecan-distributed`**.

## Eval vs train+eval vs predict

| Intent | Command | Key YAML knobs |
|--------|---------|----------------|
| **Standalone eval** (this skill) | `llamafactory-cli train eval.yaml` | `do_train: false`, `do_eval: true` and/or `do_predict: true` |
| **Train + mid-run eval** | same CLI | `do_train: true` + `do_eval` / `eval_strategy` / `eval_steps` → use **`llamafactory-cli-train`** |
| **Legacy MMLU CLI** | ~~`llamafactory-cli eval`~~ | unsupported; do not use `task` / `save_dir` / `batch_size` eval-only schema for the dead path |

There is **no** `stage: eval`. `stage` remains a training stage:
`pt` \| `sft` \| `rm` \| `ppo` \| `dpo` \| `kto` (default **sft** for SFT eval).

## How to invoke

Prefer a **YAML config** (repo examples under `examples/train_lora/*eval*.yaml`
and `examples/extras/nlg_eval/`):

```bash
# single process / auto torchrun depending on env
llamafactory-cli train path/to/eval.yaml

# multi-GPU (common)
FORCE_TORCHRUN=1 llamafactory-cli train path/to/eval.yaml

# optional: pin GPUs
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train path/to/eval.yaml
```

CLI flags override / supplement YAML. Hyphen and underscore forms are accepted.
Optional env often used in this project: `DISABLE_VERSION_CHECK=1`.

Activate the same venv/modules the job uses before dry checks (see **`alliancecan`**).
Do **not** run GPU eval on login nodes; put the CLI in SLURM.

## Mode A — `do_eval` (loss / trainer metrics)

Canonical project pattern: load base (+ optional LoRA), **no training**, evaluate
on a holdout from `dataset` via `val_size`, or on an explicit `eval_dataset`.

Minimal shape (see e.g. `examples/train_lora/qwen2_5vl_eval_X1_on_X1.yaml`):

```yaml
### model
model_name_or_path: Qwen/Qwen2.5-VL-7B-Instruct
# adapter_name_or_path: /path/to/lora/adapter   # LoRA checkpoint to evaluate
trust_remote_code: true

### method
stage: sft
do_train: false
do_eval: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### dataset
dataset: my_eval_source          # with val_size below; OR use eval_dataset instead
# eval_dataset: my_holdout
template: qwen2_vl
cutoff_len: 2048
media_dir: /path/to/media        # multimodal

### output
output_dir: saves/.../eval/run1
overwrite_output_dir: true

### train (vestigial when do_train is false; still often set in project YAMLs)
bf16: true
learning_rate: 0.0
num_train_epochs: 0.0

### eval
val_size: 0.1                    # required if eval_dataset is unset
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 1
```

**Parser requirements** (`src/llamafactory/hparams/parser.py`):

- `do_eval` or `do_predict` needs **`eval_dataset` set** **or** **`val_size` > 0**.
- With only `dataset` + `val_size` and `do_train: false`, the holdout split is the eval set (project style).
- Prefer **`bf16` / `bf16_full_eval`** when hardware supports it.

CLI sketch:

```bash
FORCE_TORCHRUN=1 llamafactory-cli train \
  --stage sft \
  --do_train false \
  --do_eval \
  --model_name_or_path /path/to/base \
  --adapter_name_or_path /path/to/lora \
  --finetuning_type lora \
  --dataset my_data \
  --val_size 0.1 \
  --template qwen2_vl \
  --output_dir saves/eval/run1 \
  --per_device_eval_batch_size 1 \
  --bf16
```

(Boolean disable form may vary; prefer YAML for `do_train: false`.)

## Mode B — `do_predict` (generation / BLEU·ROUGE-style NLG)

Use when you need **decoded generations** and predict metrics (see
`examples/extras/nlg_eval/llama3_lora_predict.yaml`):

```yaml
### method
stage: sft
do_predict: true
# do_train omitted / false — do not train
finetuning_type: lora

### dataset
eval_dataset: identity,alpaca_en_demo   # required for predict_with_generate
template: llama3
cutoff_len: 2048
max_samples: 50                         # optional cap for smoke runs

### output
output_dir: saves/.../predict
overwrite_output_dir: true

### eval
per_device_eval_batch_size: 1
predict_with_generate: true
ddp_timeout: 180000000
```

**Hard constraints:**

- `predict_with_generate` requires **`eval_dataset`** (not only `val_size` split).
- **`predict_with_generate` is incompatible with DeepSpeed ZeRO-3.**
- Cannot combine `predict_with_generate` with `compute_accuracy`.
- Batch generation can be slow; docs recommend `scripts/vllm_infer.py` for speed when appropriate.

Generation knobs (when predicting): `--do_sample` / `--no_do_sample`,
`--temperature`, `--top_p`, `--top_k`, `--num_beams`, `--max_new_tokens`,
`--repetition_penalty`, `--generation_max_length`, etc. (see train skill /
help dump).

## Loading the model under test

| Artifact | Keys |
|----------|------|
| Base / full FT | `model_name_or_path` |
| LoRA / adapters | `adapter_name_or_path` (comma-separated), optional `adapter_folder` |
| Finetuning type | `finetuning_type: lora` (or matching freeze/full) even for eval-only |

Match **`template`**, multimodal pixel/fps/`media_dir`, and freeze flags to the
checkpoint’s training setup. Wrong template → garbage metrics.

## Prediction dumps (repo feature)

This fork can log text outputs during eval without the old MMLU CLI:

- `save_eval_predictions: true` → JSON keyed by `QUESTION_ID` (dataset must have that column; see `scripts/assign_question_ids.py` / data README)
- `eval_prediction_mode: generate` (default) or `teacher_forced`
- `eval_predictions_file:` optional path (default `{output_dir}/eval_predictions.json`)

Smoke reference: `examples/train_lora/trillium_qwen2_5vl_lora_sft_CoT_prediction_dump_smoke.yaml`
(that file also trains; for pure eval, keep `do_train: false` and enable dump flags).

## Distributed notes for eval

- Multi-GPU: `FORCE_TORCHRUN=1` as with training.
- DeepSpeed for **eval-only** project jobs often uses ZeRO-3 config
  (`examples/deepspeed/ds_z3_config.json`) when **not** using `predict_with_generate`.
- If you need `predict_with_generate`, drop ZeRO-3 (or use non-Z3 / non-DS path).
- Multinode: same rank env as training → **`alliancecan-distributed`**.

## What not to copy from old docs

`references/llamafactory-eval-docs.pdf` and upstream “General Capability Evaluation”
still show `llamafactory-cli eval` + fields like `task`, `n_shot`, `save_dir`,
`batch_size`. That path is **dead in this tree**. Do not emit those configs for
CLI runs.

`examples/train_lora/llama3_lora_eval.yaml` is the **legacy** MMLU-style file
(for the removed command). Prefer Mode A/B YAMLs above.

`references/eval-deprecation-and-alternatives.md` is background; if it suggests
`stage: eval`, **ignore that** — stages do not include `eval`.

WebUI Evaluate tab (`llamafactory-cli webui`) still exists separately; prefer
YAML + `train` for reproducible cluster jobs.

## Agent workflow

1. **Refuse / rewrite** any request that uses `llamafactory-cli eval`.
2. **Pick mode:** metrics/loss → Mode A (`do_eval`); free-form generations → Mode B (`do_predict` + `predict_with_generate`).
3. **Clone nearest example:** `examples/train_lora/*eval*.yaml` or `examples/extras/nlg_eval/*`.
4. **Set model + adapter + template + output_dir**; keep `do_train: false`.
5. **Wire data:** `eval_dataset` **or** `dataset` + `val_size > 0` (Mode B needs `eval_dataset`).
6. **Multimodal:** `media_dir`, image/video caps, VL template.
7. **DS / generate conflict:** no ZeRO-3 with `predict_with_generate`.
8. **Validate** flags against `references/llamafactory-cli_train_-h.txt` or `train -h`.
9. **Launch** via SLURM on compute nodes; reuse cluster skills for multi-GPU/node.

## Quick troubleshooting

| Symptom | Checks |
|---------|--------|
| `NotImplementedError: Evaluation will be deprecated` | You ran `eval`; switch to `train` + eval-only YAML |
| `Please specify dataset for evaluation` | Set `eval_dataset` or `val_size > 0` |
| `predict_with_generate` + ZeRO-3 error | Remove DeepSpeed Z3 or disable generate predict |
| `Cannot use predict_with_generate if eval_dataset is None` | Set `eval_dataset` explicitly |
| Wrong chat / VL formatting | Fix `template` / `media_dir` / pixel limits |
| OOM at eval | Lower `per_device_eval_batch_size`, shrink pixels / `cutoff_len`, fewer samples |
| Metrics look like training run | Confirm `do_train: false` and no nonzero train schedule intent |
| Legacy `task` / `save_dir` ignored or fail | Old MMLU CLI schema; convert to Mode A/B |

When in doubt, copy a working `*_eval_*.yaml` from `examples/train_lora/` and
change paths — do not invent a separate `llamafactory-cli eval` flow.
