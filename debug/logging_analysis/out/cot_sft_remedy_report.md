# CoT SFT grammar crash: diagnosis and LlamaFactory flag advice

Analysis only. No training/eval YAML or SLURM files were changed.

**Scope:** current job `models/qwen2_5vl_lora_sft_CoT/tamia_slurm_qwen2_5vl_lora_sft_CoT_traineval.sh` overlaying `examples/train_lora/trillium_qwen2_5vl_lora_sft_CoT_traineval_resume_epoch2.yaml`; proposed changes in `models/qwen2_5vl_lora_sft_CoT/tamia_slurm_qwen2_5vl_lora_sft_CoT_traineval_evalevery10trainsteps.sh`; flags from `.grok/skills/llamafactory-cli-train/references/llamafactory-cli_train_-h.txt`.

**Evidence:** `debug/logging_analysis/out/eval_compare/` — same 4384 CoT holdout IDs for base Qwen2.5-VL-7B-Instruct vs merged ep1–ep3 (`report.md`, `run_summary.csv`, `dataset_run_summary.csv`, example CSVs). External-benchmark crash (base → ep1 cliff, slow recovery, still below base at ep4) is taken from the training/eval runs, not from that folder.

---

## 1. Two evaluations, two stories

The CoT holdout and the external benches are not in conflict. They measure different things, and together they describe one failure mode: **the model quickly absorbs the CoT template and gold answers, while English fluency and the original Instruct policy are damaged in the same first epoch.**

### 1.1 CoT holdout (`eval_compare/report.md`) — “it is learning”

Same 4384 questions at every epoch (Jaccard 1.0). Pooled:

| run | eval_loss | canonical tags | usable tags | norm EM | median think tok | median `repetition_score` |
| --- | --- | --- | --- | --- | --- | --- |
| base | 1.671 | 0.0% | 0.0% | 9.9% | 0 | 0.157 |
| ep1 | 0.955 | 44.4% | 62.3% | 48.5% | 237 | 0.154 |
| ep2 | 0.913 | 46.5% | 65.1% | 49.1% | 242 | 0.153 |
| ep3 | 0.892 | 47.6% | **66.7%** | 49.6% | 248 | 0.153 |

Empty-answer rate 78% → 13%. Tag-pair score 0.11 → 0.78. Almost **all of the gain is epoch 0 → 1**; ep2/ep3 are small increments. That matches “training curves look normal.”

Caveats on those numbers:

- Dumps were `eval_prediction_mode: teacher_forced`. Median **total** tokens stay ~300 at every epoch (think tokens appear by re-parsing the same budget). This is next-token accuracy on **gold prefixes**, not free generation. External benches are free `generate` and will look worse.
- `normalized_exact_match` is lexical match to the reference, not spatial correctness.
- Grammar scoring was **disabled** in that run (`Grammar: disabled`). The qualitative CSVs still show broken English.

### 1.2 Same CoT holdout — a third is still a mess, and fluent traces still stutter

**Usable format at ep3 is 66.7%** (`usable_format` = ordered `<think>…</think><answer>…</answer>` with non-empty bodies; a prefix such as `<output_3D>` is allowed). That is **1460 / 4384 predictions that cannot even be parsed as CoT**. The 66.7% is not leftover noise; it is the plateau after three full epochs.

Breakdown of that 33.3% (from `dataset_run_summary.csv` at ep3):

| Dataset | n | usable | canonical | What “unusable” means here |
| --- | --- | --- | --- | --- |
| Scene30k | 3009 | **69.3%** | 69.3% | ~31% missing/broken tags (`malformed_tags` 30.7%) |
| 3DThinker10k | 1007 | **83.4%** | **0%** | Prefix `<output_3D>` blocks canonical; ~17% still fail pairs |
| SpatialSSRL_coldstart | 368 | **0%** | 0% | Never emits think/answer; often `\boxed{…}` / “Analysis analysis:” |

SpatialSSRL alone is 368 always-unusable rows (8.4% of the pool). Scene30k contributes ~923 failures. 3DThinker ~167. Format is **not one language**; concat SFT is averaging three.

**Usable ≠ grammatical.** Examples in `examples_largest_grammar_deterioration.csv` / `examples_biggest_repetition_increase.csv` / `examples_largest_reasoning_collapse.csv` are full of local stutter even when tags parse:

- “I am constructed constructed a 3D scene based a room … images images”
- “figure out out … FirstFirst … mentioned mentioned … gray gray … table table”
- “radiator radiator … gather gather … brown brown”

The pooled **`repetition_score` median is almost flat** (0.157 → 0.153). That scalar is a mix of adjacent identity, trigram diversity, zlib, and unigram concentration over the whole string. Long CoT dilutes local doubling, so the table will say “repetition is fine” while the traces are unreadable. **Trust the examples and adjacent-duplicate rate, not the global median.** Trigram repetition actually ticked **up** (0.060 → 0.064).

So: CoT SFT is succeeding as **template + answer-span fitting**, and failing as **language modeling**. A 66.7% parse rate after 3 epochs, with stutter inside the parsed third, is not a model that should be expected to beat base Qwen on clean Instruct benches.

### 1.3 External benchmarks — the cliff

Base Instruct English/policy is still the right prior for SQA3D / X62-style scoring. After ep1 the model prefers long tagged CoT, often ungrammatical, often with extra prefixes (`<output_3D>`) or boxed SpatialSSRL style. Scores crash at ep1 (peak-LR checkpoint), then drift up a little as cosine decays, and still sit below base at ep4.

Before attributing 100% of the bench drop to capability loss, confirm scorers **extract `<answer>`** (and ignore think). Format mismatch will look like a crash even if the boxed/short answer is right. After extraction, leftover drop is the fluency/forgetting problem.

---

## 2. What the current run actually does

Effective recipe after the TamIA wrapper copies the epoch-2 template and overrides paths/batch:

| Knob | Current value | Effect |
| --- | --- | --- |
| Model | `Qwen/Qwen2.5-VL-7B-Instruct` + LoRA | Instruct model, not a native thinking model |
| Stage / method | `sft` / `lora` | CE on assistant tokens |
| LoRA | `rank=8`, `alpha` default `16`, `dropout=0`, `target=all` | Language-linears only (`freeze_vision_tower` default True) |
| LR | `1.0e-4`, cosine, `warmup_ratio=0.1`, 5-epoch horizon | Peak LR at step ~310; **ep1 ckpt still near peak** |
| Batch | 4×H100, `bs=2`, `ga=8` | Effective batch **64**; ~620 steps/epoch |
| Data | `Scene30k,SpatialSSRL_coldstart,3DThinker10k`, `mix_strategy=concat` | ~43.9k rows; Scene30k ~68% |
| Split | `val_size=0.1` | The 4384-row holdout above; train is 90% of mix |
| Context | `cutoff_len=131072`, `image_max_pixels=65536` | Full context; vision ~256² |
| Mid-run eval | `eval_steps=620`, dumps **`teacher_forced`** | Matches `eval_compare`, hides free-gen failure |
| Decode | `max_new_tokens=2048`, `repetition_penalty=1.0` | No anti-stutter at decode |

`template: qwen2_vl` is ChatML, not `ReasoningTemplate`. `enable_thinking` default True is a no-op. This run is teaching a **new** think/answer contract to an Instruct model.

---

## 3. Why this recipe produces exactly that pair of curves

**Teacher-forced CE on mixed CoT is easy to reduce and is the wrong health metric.** eval_loss 1.67 → 0.89 says gold-prefix next-token NLL improved. It does not say the model can generate grammatical CoT or short Instruct answers.

**Epoch-1 checkpoint is the strongest shock.** `warmup_ratio=0.1` of 5 epochs ≈ 310 steps = half of epoch 1. Cosine then decays from `1e-4` toward 0. The ep1 save (`save_steps=620`) is taken just after peak LR. That is when CoT tags jump 0% → 62% usable **and** when external benches fall off a cliff. Later epochs: LR falls, tags/EM inch up (62.3 → 66.7% usable), benches recover slightly, never to base.

Cutting warmup to 0.02 **moves the shock earlier**; it does not lower peak LR. `num_cycles: 0.4` **keeps LR higher later**, which continues the damage. Neither is the right first lever.

**Three output languages in one CE.** Scene30k wants bare think/answer; 3DThinker prefixes `<output_3D>` (hence 0% canonical forever); SpatialSSRL never uses those tags (0% usable at every epoch, `\boxed{}`). Concat+shuffle still trains one head on all three. Interpolation → missing tags on 1/3 of CoT eval, stutter, and a policy that no longer looks like base Qwen.

**Five epochs of ~40k LoRA at 1e-4 is a lot** for preserving Instruct English. Most of the CoT metric movement is done by ep1; extra epochs buy ~4 points of usable format and do not repair grammar.

---

## 4. Item-by-item on the evalevery10 comments

### Keep / do (instrumentation)

**Fixed 16-example *generate* eval every 10 steps.** Necessary, because teacher-forced CoT eval is what made the holdout look healthier than benches. Frozen IDs: 8 Scene30k + 4 SpatialSSRL + 4 3DThinker.

Flags:

- tiny `eval_dataset` (or dedicated 16-row set); `val_size=0` if training on the full mix
- `eval_strategy=steps`, `eval_steps=10`, `eval_on_start=true`
- `save_eval_predictions=true`
- **`eval_prediction_mode=generate`**
- `prediction_loss_only=false`
- `max_new_tokens=2048`, `per_device_eval_batch_size=1`

Score those 16 traces for: canonical, usable, **adjacent-duplicate / stutter rate**, empty answer, and a quick read of think text. Do not rely on pooled `repetition_score`.

Do **not** `do_predict` the full 4384-row split every 10 steps (~90 min).

**`eval_on_each_dataset=true`:** worth a **separate** cheap diagnostic, not the 10-step loop. Pooled 66.7% usable hides SpatialSSRL at 0% and Scene30k at 69%.

**`compute_accuracy=true`:** cheap teacher-forced token acc, useful next to eval_loss. Incompatible with HF `predict_with_generate`; fine with this fork’s prediction dumps.

**`repetition_penalty=1.1`:** decode-only. **Use it on benchmark and generate-dump YAML.** It will not train away stutter, but it is justified by the CoT examples. Try 1.05–1.15; too high causes bland loops.

### Change, but not as the comments currently aim

**`warmup_ratio: 0.02`.** Better than 310 steps, but `warmup_ratio` is deprecated — use `warmup_steps` (int or float in `[0,1)`). Hitting `1e-4` at step 62 still causes the ep1 cliff.

Prefer: **`learning_rate: 2e-5`** (try `5e-5` only if tags never appear) + `warmup_steps: 0.03` + **`lr_scheduler_type: cosine_with_min_lr`** with `--lr_scheduler_kwargs '{"min_lr_rate": 0.1}'`. That avoids slamming to 0 **and** avoids `num_cycles: 0.4`.

**Epochs:** after the LR cut, default to **1–2 epochs**. CoT holdout already plateaus after ep1; 5 epochs did not fix the 33% unusable pile or the stutter.

### Special tokens: high risk, wrong file, do later

Comments 3–6 (`new_special_tokens_config`, `init_special_tokens=desc_init_w_noise`, `additional_target=embed_tokens,lm_head`, `skip_special_tokens false`) are **not** the first retry. Atomic tags might help the 31% Scene30k malformed pile **after** LR and format unify, not before.

1. **`data/control_tokens.json` will not load as intended.** Loader expects a **dict** `{ "<think>": "description…", ... }`. The file is a list of AddedToken-style objects. `desc_init*` needs descriptions.
2. Adding tokens auto-sets `resize_vocab=True`. PEFT `modules_to_save` then trains the **entire** embedding/lm_head at the **main** LR. At `1e-4` that is a standard way to destroy grammar. `loraplus_lr_embedding` (1e-6) does **not** apply to `modules_to_save`.
3. Tags already tokenize as subwords; most CoT SFT never needs new specials.
4. `--no_skip_special_tokens` only matters once tags are special tokens.

If they are added later: YAML descriptions; `learning_rate ≤ 2e-5`; same tokenizer extras at merge/eval.

### LoRA extras and optimizer shopping

| Flag | Verdict |
| --- | --- |
| `lora_dropout=0.05` (try 0.1) | **Yes** — default 0 overfits template/stutter |
| `use_rslora=true` | **Yes** if raising rank |
| `lora_rank=16` | Maybe, **with lower LR**; rank 8 is not why 1/3 is unusable |
| `loraplus_lr_ratio=16` | Optional second try |
| `use_dora` | Skip for now |
| GaLore / APOLLO / BAdam | **Memory**, not fluency |
| Keep AdamW | Default is fine |
| `neftune_noise_alpha=5` | Optional, secondary |
| `label_smoothing_factor=0.05` | Optional; don’t combine with DFT at first |

### Loss functions

CE overfits easy high-frequency tokens (tags, doubled words). This fork:

- **`--use_dft_loss`**: best loss experiment after LR + format unify (designed to stop SFT from flattening the base).
- `--use_asft_loss` / `--use_eaft_loss`: only if DFT is inconclusive.
- DPO/KTO: need rejected traces that are not in this mix.

Do not combine DFT with full-embedding `additional_target` on the first retry.

---

## 5. Data and eval protocol (higher leverage than most flags)

1. **Unify traces** to one schema: `\n<think>\n...\n</think>\n<answer>\n...\n</answer>\n`. Move `<output_3D>` into the system prompt. Convert or drop SpatialSSRL until it uses the same tags — it is currently a 0% usable sink and a competing `\boxed{}` policy.
2. **Filter stuttering gold.** If references already contain “images images” / doubled function words, CE will copy it. That is the cheapest explanation of the example CSVs.
3. Train on 100% of the mix once the 16-row eval set is frozen.
4. **Replay a little Instruct-style data** (short answers, no tags) as a fourth mix member. Standard anti-forgetting for the external-bench cliff.
5. Benchmark scorers must extract `<answer>`.
6. Generate-mode 16-probe loop as in §4.
7. `image_max_pixels=65536` is low vs SpatialSSRL’s own yaml (`262144`). Raise later; it is a spatial-gap issue, not the grammar issue.

---

## 6. Recommended first retry (one job, few knobs)

Keep: model, `qwen2_vl`, ZeRO-2, LoRA on language, `bf16`, `flash_attn=fa2`, cutoff, DeepSpeed, resume/`stop_at_global_step` machinery.

```text
learning_rate: 2.0e-5
lr_scheduler_type: cosine_with_min_lr
lr_scheduler_kwargs: {"min_lr_rate": 0.1}
warmup_steps: 0.03
lora_dropout: 0.05
num_train_epochs: 2.0          # or keep 5 + stop at 2 epochs
eval_steps: 10
eval_on_start: true
eval_prediction_mode: generate
compute_accuracy: true
# 16 frozen eval rows; val_size: 0
# generate dumps: repetition_penalty 1.1
# no new special tokens, no additional_target, no num_cycles 0.4, no optim swap
```

Optional on the same job **if traces are already unified**: `--use_dft_loss`.

**Success on the 16 generate probes + one external bench:** (1) usable tags without word-salad by ~200 steps; (2) adjacent-duplicate rate well below the current example CSVs; (3) usable format not stuck near 2/3 because SpatialSSRL/`<output_3D>` still disagree; (4) external drop vs base much smaller than today’s ep1 cliff. If tags never appear, raise LR to `5e-5` before raising rank.

---

## 7. Later ablations (one change at a time)

1. DFT vs CE at the same LR
2. `lora_rank=16` + `use_rslora` at `2e-5`
3. Instruct replay mix
4. Special-token YAML + `desc_init_w_noise` only at ≤2e-5
5. `image_max_pixels` 262144+
6. `neftune_noise_alpha=5`
7. Decode `repetition_penalty` sweep on benches only

---

## 8. What not to do

- Do not treat falling CoT `eval_loss` / teacher-forced EM / “66.7% usable and climbing” as proof the reasoning model is healthy. The same epoch that produced those curves produced the bench cliff and the stuttering traces.
- Do not trust pooled `repetition_score` when the examples stutter.
- Do not unfreeze full `embed_tokens`/`lm_head` at `1e-4`.
- Do not point `new_special_tokens_config` at list-shaped `control_tokens.json`.
- Do not use GaLore/APOLLO/BAdam to fix grammar.
- Do not `do_predict` 4384 rows every 10 steps.
- Do not lengthen high-LR cosine (`num_cycles=0.4`) to recover from a crash caused by that LR.
