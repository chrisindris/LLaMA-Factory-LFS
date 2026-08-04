---
name: llamafactory-cli-train
description: >
  Configure and run LLaMA-Factory training via `llamafactory-cli train`: YAML configs,
  CLI flags for model/data/finetuning/trainer, LoRA/full/freeze, stages (pt/sft/rm/ppo/dpo/kto),
  eval/save, DeepSpeed/FSDP, multimodal, and resume. Use when the user mentions llamafactory-cli
  train, training YAML, --stage, --finetuning_type, LoRA train args, or runs /llamafactory-cli-train.
---

# llamafactory-cli train

Help Grok build correct **`llamafactory-cli train`** invocations and training configs for this repo.

Full CLI dump (source of truth for flags/defaults):  
`references/llamafactory-cli_train_-h.txt`  
(or re-run `llamafactory-cli train -h` with the project venv).

For cluster env, DeepSpeed SLURM, and multi-node launch, also use **`alliancecan`**,
**`alliancecan-deepspeed`**, and **`alliancecan-distributed`** when those skills are available.

## How to invoke

Prefer a **YAML config** (repo style under `examples/` and job-specific paths):

```bash
# single process / auto torchrun depending on env
llamafactory-cli train path/to/train.yaml

# multi-GPU (common)
FORCE_TORCHRUN=1 llamafactory-cli train path/to/train.yaml

# optional: pin GPUs
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train path/to/train.yaml
```

CLI flags override / supplement YAML. Hyphen and underscore forms are accepted  
(e.g. `--learning_rate` / `--learning-rate`).

Boolean pattern from the help:

- Enable: `--bf16`, `--do_train`, `--plot_loss`
- Disable counterpart when present: `--no_use_fast_tokenizer`, `--no_enable_thinking`, etc.

Optional env often used in this project: `DISABLE_VERSION_CHECK=1`.

Activate the same venv/modules the job uses before running help or dry checks (see **`alliancecan`**).

## Minimal training recipe

Always set (or inherit from YAML):

| Concern | Flags / keys | Notes |
|---------|--------------|--------|
| Model | `--model_name_or_path` | HF/ModelScope id or local path |
| Stage | `--stage` | `pt` \| `sft` \| `rm` \| `ppo` \| `dpo` \| `kto` (default **sft**) |
| Method | `--finetuning_type` | `lora` \| `oft` \| `freeze` \| `full` (default **lora**) |
| Data | `--dataset`, `--dataset_dir`, `--template` | Comma-separated multi-dataset; default `dataset_dir=data` |
| Output | `--output_dir` | Checkpoints and logs |
| Train loop | `--do_train`, batch/epochs/lr | See below |

Example SFT LoRA sketch:

```bash
FORCE_TORCHRUN=1 llamafactory-cli train \
  --stage sft \
  --do_train \
  --model_name_or_path /path/to/base \
  --finetuning_type lora \
  --lora_rank 8 \
  --lora_target all \
  --dataset my_dataset \
  --dataset_dir data \
  --template qwen2_vl \
  --output_dir saves/run1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --num_train_epochs 2.0 \
  --lr_scheduler_type cosine \
  --bf16 \
  --cutoff_len 2048 \
  --logging_steps 10 \
  --save_steps 500 \
  --plot_loss
```

Prefer matching an existing example under `examples/train_lora/`, `examples/train_full/`, etc., then diff flags.

## Core flag groups

### Model & tokenizer

- `--model_name_or_path`, `--adapter_name_or_path` (comma-separated adapters), `--adapter_folder`
- `--cache_dir`, `--model_revision` (default `main`), `--trust_remote_code`
- Tokenizer: `--use_fast_tokenizer` / `--no_use_fast_tokenizer`, `--resize_vocab`,
  `--add_tokens`, `--add_special_tokens`, `--new_special_tokens_config`, `--init_special_tokens`
- Memory/load: `--low_cpu_mem_usage` / `--no_low_cpu_mem_usage`, `--offload_folder`
- Attention: `--flash_attn {auto,disabled,sdpa,fa2}`, `--shift_attn`, `--rope_scaling {linear,dynamic,yarn,llama3}`
- Extras: `--use_unsloth`, `--enable_liger_kernel` / `--use_liger_kernel`, `--train_from_scratch`,
  `--disable_gradient_checkpointing`, `--use_reentrant_gc` / `--no_use_reentrant_gc`
- Quantization (on-the-fly): `--quantization_method {bnb,gptq,awq,...}`, `--quantization_bit`,
  `--quantization_type {fp4,nf4}`, `--double_quantization` / `--no_double_quantization`
- FP8: `--fp8`, `--fp8_backend`, `--fp8_enable_fsdp_float8_all_gather`
- Hub tokens: `--hf_hub_token`, `--ms_hub_token`, `--om_hub_token`

### Multimodal (image / video / audio)

- Image: `--image_max_pixels` (default 589824), `--image_min_pixels`, `--image_do_pan_and_scan`,
  `--crop_to_patches`, `--image_sample_stride`
- Video: `--video_max_pixels`, `--video_min_pixels`, `--video_fps` (default 2.0),
  `--video_maxlen` (default 128), `--use_audio_in_video`
- Audio: `--audio_sampling_rate` (default 16000)
- Media root: `--media_dir` (defaults to `dataset_dir`)
- MLLM freezes: `--freeze_vision_tower` (default True), `--freeze_multi_modal_projector` (default True),
  `--freeze_language_model`; use `--no_freeze_*` to unfreeze
- Debug: `--debug_mm_training`, `--debug_mm_steps`

### Data & template

- `--template`, `--dataset`, `--eval_dataset`, `--dataset_dir`, `--cutoff_len` (default 2048)
- `--train_on_prompt`, `--mask_history`, `--streaming`, `--buffer_size`
- Mix: `--mix_strategy {concat,interleave_under,interleave_over}`, `--interleave_probs`
- Cache: `--overwrite_cache`, `--preprocessing_batch_size`, `--preprocessing_num_workers`,
  `--tokenized_path`, `--data_shared_file_system`
- Eval split: `--val_size` (int or float in `[0,1)`), `--eval_on_each_dataset`, `--max_samples`
- Packing: `--packing`, `--neat_packing`
- Thinking models: `--enable_thinking` / `--no_enable_thinking` (default enabled)
- Loss: `--ignore_pad_token_for_loss` (default True)

### Training hyperparameters

- Batch: `--per_device_train_batch_size` (default 8), `--gradient_accumulation_steps` (default 1)  
  Effective batch ≈ `per_device * num_devices * grad_accum`
- Length: `--num_train_epochs` (default 3.0) or `--max_steps` (−1 = use epochs)
- LR: `--learning_rate` (default 5e-5), `--lr_scheduler_type` (default `linear`; often use `cosine`),
  `--lr_scheduler_kwargs`, `--warmup_steps` (int or ratio in `[0,1)`), deprecated `--warmup_ratio`
- Optim: `--optim` (default `adamw_torch_fused`), `--optim_args`, `--weight_decay`,
  `--adam_beta1/2`, `--adam_epsilon`, `--max_grad_norm` (default 1.0)
- Precision: prefer **`--bf16`** over `--fp16` when hardware supports; `--pure_bf16` (no AMP);
  eval: `--bf16_full_eval` / `--fp16_full_eval`; `--tf32`
- Memory/speed: `--gradient_checkpointing`, `--auto_find_batch_size`, `--torch_compile*`,
  `--neftune_noise_alpha`, `--torch_empty_cache_steps`
- Sampling: `--train_sampling_strategy {random,sequential,group_by_length}`, `--seed`, `--data_seed`,
  `--disable_shuffling`

### Logging, eval, save, resume

- Log: `--logging_strategy {no,steps,epoch}`, `--logging_steps` (default 500), `--logging_first_step`,
  `--report_to` (default `none`), `--run_name`, SwanLab/Trackio flags
- Eval: `--eval_strategy {no,steps,epoch}`, `--eval_steps`, `--per_device_eval_batch_size`,
  `--do_eval`, `--prediction_loss_only`, `--compute_accuracy`
- Save: `--save_strategy {no,steps,epoch,best}`, `--save_steps` (default 500),
  `--save_total_limit`, `--save_only_model`, `--save_on_each_node`
- Best model: `--load_best_model_at_end`, `--metric_for_best_model`, `--greater_is_better`,
  `--early_stopping_steps`
- Resume: `--resume_from_checkpoint PATH`  
  Related: `--ignore_data_skip`, `--restore_callback_states_from_checkpoint`,
  `--enable_jit_checkpoint` (SIGTERM graceful save)
- Modes: `--do_train`, `--do_eval`, `--do_predict`
- Curves: `--plot_loss`

### LoRA / freeze / full / OFT

**LoRA** (`--finetuning_type lora`):

- `--lora_rank` (default 8), `--lora_alpha` (default `rank*2` if unset), `--lora_dropout`
- `--lora_target` (default `all`; comma-separated module names)
- `--additional_target`, `--create_new_adapter`
- Variants: `--use_rslora`, `--use_dora`, `--loraplus_lr_ratio`, `--loraplus_lr_embedding`
- PiSSA: `--pissa_init`, `--pissa_iter`, `--pissa_convert`
- Continue: `--adapter_name_or_path` existing adapters

**Freeze**:

- `--freeze_trainable_layers` (default 2; +last n / −first n), `--freeze_trainable_modules`,
  `--freeze_extra_modules`

**OFT**: `--oft_rank`, `--oft_block_size`, `--oft_target`, `--module_dropout`

**Full**: `--finetuning_type full` (and usually higher memory; consider DeepSpeed/FSDP).

### Preference / RL stages

Use matching `--stage` and related models:

| Stage | Typical extras |
|-------|----------------|
| `dpo` / prefs | `--pref_beta`, `--pref_ftx`, `--pref_loss {sigmoid,hinge,ipo,kto_pair,orpo,simpo}`, `--ref_model*` |
| `kto` | `--kto_chosen_weight`, `--kto_rejected_weight` |
| `ppo` | `--reward_model*`, `--ppo_epochs`, `--ppo_buffer_size`, `--ppo_target`, … |
| `rm` | reward-model training |
| `pt` | pretrain; packing often auto-enabled |

### Distributed (brief)

- DeepSpeed: `--deepspeed path/to/ds_config.json` (or dict)
- FSDP: `--fsdp` options + `--fsdp_config`
- DDP knobs: `--ddp_backend`, `--ddp_timeout` (default 1800), `--ddp_find_unused_parameters`, …
- Ray: `--ray_num_workers`, `--resources_per_worker`, etc., often with `USE_RAY=1`

Do **not** invent multi-node wiring here; use **`alliancecan-distributed`** /
**`alliancecan-deepspeed`**. On AllianceCan prefer `FORCE_TORCHRUN=1` + correct rank env over
DeepSpeed’s own multi-node launcher when cluster docs say so.

### Generation (eval / predict)

When `--predict_with_generate` or generative metrics:

- `--do_sample` / `--no_do_sample`, `--temperature`, `--top_p`, `--top_k`
- `--num_beams`, `--max_length`, `--max_new_tokens`, `--repetition_penalty`, `--length_penalty`
- `--skip_special_tokens` / `--no_skip_special_tokens`
- `--generation_max_length`, `--generation_num_beams`, `--generation_config`

### Export (often separate export command, but flags exist on train parser)

`--export_dir`, `--export_size`, `--export_device`, quantization export flags, `--export_hub_model_id`.

## Agent workflow

1. **Find a nearest example YAML** in `examples/` or the user’s `models/**` job folder; copy/adapt.
2. **Confirm stage + finetuning_type + model + dataset + template + output_dir**.
3. **Set train/eval/save** cadence and precision (`bf16` preferred).
4. **LoRA**: rank/alpha/target; load prior adapters if resuming adapters only.
5. **Large models**: add `--deepspeed` JSON from `examples/deepspeed/` or FSDP config; align with
   cluster skills.
6. **Multimodal**: set `media_dir`, pixel/fps caps, freeze flags, and a VL-capable `template`.
7. **Resume**: `--resume_from_checkpoint` to a valid checkpoint dir (not only adapter path unless
   that is how the job is structured).
8. **Validate flags** against `references/llamafactory-cli_train_-h.txt` or `train -h` if unsure
   of a default/enum.
9. **Do not run GPU training on login nodes** (AllianceCan); put the CLI in a SLURM script.

## Quick troubleshooting

| Symptom | Checks |
|---------|--------|
| Unknown arg | Flag name typo; consult full help reference |
| OOM | Lower `per_device_train_batch_size`, raise `gradient_accumulation_steps`, enable GC,
  ZeRO/FSDP, quantize, shrink `cutoff_len` / vision pixels |
| Wrong chat format | Fix `--template` for the base model |
| Dataset not found | `--dataset_dir` + `dataset_info.json` registration |
| Multi-GPU idle | `FORCE_TORCHRUN=1`, GPU count, DeepSpeed/FSDP config |
| Resume restarts data | `--ignore_data_skip` behavior; checkpoint completeness |

When in doubt, prefer YAML + documented example over long one-off CLI lines.
