# Plan: Toggleable train/eval prediction JSON dumps + QUESTION_ID assignment

## Goal

Add **optional** debugging flags that dump model text outputs to JSON, keyed by stable `QUESTION_ID`s:

1. **Train dump**: `D[QUESTION_ID][STEP] = MODEL_OUTPUT` where `STEP` is the trainer **optimizer step** (`state.global_step`, matching the existing `mm_debug ... optimizer_step=` logs).
2. **Eval dump**: `D[QUESTION_ID] = MODEL_OUTPUT` for the val/eval set (assume one meaningful eval per run; later evals overwrite).
3. **Offline script** to stamp annotations with `QUESTION_ID = f"{DATASET_NAME}_{NUMBER}"` by order of appearance (annotations currently lack this column).

This builds on the existing multimodal debug surface (`debug_mm_training` / `debug_mm_steps` → collator `batch_samples` + trainer `mm_debug pre_step` memory lines) but targets **text predictions**, not memory.

## Design choices (confirmed)

| Choice | Decision |
|--------|----------|
| Train `MODEL_OUTPUT` | **Both modes**, flag-selectable: `teacher_forced` (cheap) and `generate` (true free-form) |
| Train cadence | **Subsampled**: every **N** optimizer steps and/or first **K** samples |
| ID files | Default write **new** files; optional `--in-place` |
| Eval generation | **Custom** path that works with `val_size` (stock `predict_with_generate` currently requires a separate `eval_dataset`) |

## JSON shapes

**Train** (`train_predictions.json`, default under `output_dir`):

```json
{
  "Scene30k_12": {
    "620": "model text ...",
    "1240": "model text ..."
  },
  "SpatialSSRL_coldstart_3": {
    "621": "..."
  }
}
```

- Outer key: `QUESTION_ID` (string).
- Inner key: optimizer step as **string** (JSON object keys must be strings).
- Value: decoded model text only (no prompt, no images, no gold label).

**Eval** (`eval_predictions.json`):

```json
{
  "Scene30k_12": "model text ...",
  "3DThinker10k_99": "..."
}
```

Optional metadata file or top-level `"_meta"` is **not** required unless useful for debugging; keep the user-facing contract as pure `QUESTION_ID → …` maps.

## New CLI / YAML flags (`FinetuningArguments`)

Add next to existing `debug_mm_*` in `src/llamafactory/hparams/finetuning_args.py`:

| Flag | Default | Meaning |
|------|---------|---------|
| `save_train_predictions` | `false` | Enable train JSON dump |
| `train_prediction_mode` | `"teacher_forced"` | `"teacher_forced"` \| `"generate"` |
| `train_prediction_interval` | `1` | Log when `global_step % interval == 0` (and after resume, still use live `global_step`) |
| `train_prediction_max_samples` | `0` | Cap total `(QUESTION_ID, step)` records written this run; `0` = unlimited |
| `train_predictions_file` | `null` | Override path; default `{output_dir}/train_predictions.json` |
| `save_eval_predictions` | `false` | Enable eval JSON dump |
| `eval_prediction_mode` | `"generate"` | `"teacher_forced"` \| `"generate"` |
| `eval_predictions_file` | `null` | Default `{output_dir}/eval_predictions.json` |

Generation knobs reuse existing `GeneratingArguments` / trainer `gen_kwargs` (`max_new_tokens`, temperature, etc.) when mode is `generate`.

Example YAML fragment (for a debug CoT job):

```yaml
save_train_predictions: true
train_prediction_mode: teacher_forced   # or generate
train_prediction_interval: 50
train_prediction_max_samples: 200

save_eval_predictions: true
eval_prediction_mode: generate
```

## Critical pipeline work: preserve `question_id`

Today, converters strip all raw columns and supervised preprocessing only keeps tensors + media (+ `sample_idx` / `sample_media` for mm debug). `question_id` must survive end-to-end.

### 1) Offline assignment script

**New** `scripts/assign_question_ids.py` (plus short usage in a comment / README blurb under `data/README.md` if appropriate):

```bash
# single file → new file (default)
python scripts/assign_question_ids.py \
  --dataset-name Scene30k \
  --input /path/to/train.parquet \
  --output /path/to/train.with_question_id.parquet

# in-place
python scripts/assign_question_ids.py \
  --dataset-name SpatialSSRL_coldstart \
  --input /path/to/SFT-coldstart.json \
  --in-place

# batch from dataset_info
python scripts/assign_question_ids.py \
  --from-dataset-info data/dataset_info.json \
  --datasets Scene30k,SpatialSSRL_coldstart,3DThinker10k
```

Behavior:

- Supported formats: **JSON list**, **JSONL**, **Parquet**.
- ID format: `{dataset_name}_{i}` with `i = 0..n-1` in **file order** (stable, reproducible).
- Column name default: `question_id` (override with `--column`).
- Do **not** reuse existing fields like Thinker’s `idx` (those are not `DATASET_NUMBER` order IDs).
- Default output path: insert `.with_question_id` before the extension; `--in-place` overwrites input.
- Print a suggested `dataset_info.json` `columns` snippet: `"question_id": "question_id"`.
- After batch mode, optionally print/update notes for pointing `file_name` at the new paths (prefer not silently rewriting hub snapshot paths without `--update-dataset-info`).

### 2) Dataset wiring

- Extend `DatasetAttr` / `parser.py` column list with optional `question_id` (mapped from `columns.question_id` in `dataset_info.json`).
- `AlpacaDatasetConverter` / `SharegptDatasetConverter`: pass through `_question_id` when the column exists; otherwise `None`.
- `SupervisedDatasetProcessor` (and packed variant if used): append `question_id` string (or empty) per example, analogous to `sample_idx`.
- Unsupervised / eval-generate processors: same field if those code paths are hit.
- Multi-dataset concat stays valid because IDs are globally unique by construction (`Scene30k_0` vs `3DThinker10k_0`).
- `val_size` train/test split preserves the column (HF split keeps features).

### 3) Collator

In `MultiModalDataCollatorForSeq2Seq.__call__` (same place as `sample_idx` / `debug_samples`):

- `pop("question_id")` from each feature **before** `super().__call__` (parent collator only tolerates tensorizable fields).
- Attach `batch["question_ids"] = list[str]` (ragged non-tensor side channel, like `debug_samples`).
- Pop `question_ids` in trainer before model forward so the model never sees them.

## Trainer / workflow implementation

Primary files:

- `src/llamafactory/train/sft/trainer.py` — dump logic
- `src/llamafactory/train/sft/workflow.py` — wire flags; ensure eval path runs dump
- Small helper e.g. `src/llamafactory/train/prediction_dump.py` — nested-dict merge + atomic JSON write (keep trainer thinner)

### Train dump

Hook in `compute_loss` / `training_step` (rank-local, then reduce to rank0):

1. If `not save_train_predictions` → no-op.
2. Let `step = state.global_step` (optimizer step; same number as current `mm_debug pre_step optimizer_step=`).
3. Log only when `step > 0` and `step % train_prediction_interval == 0` (document: microbatches within the same optim step share one `STEP` key; last write for a `QUESTION_ID` at that step wins).
4. Stop once `train_prediction_max_samples` records have been stored (if `> 0`).
5. Require `question_ids` on the batch; if missing/empty, `warning_rank0_once` and skip (do not invent unstable IDs mid-run).
6. Decode `MODEL_OUTPUT`:
   - **`teacher_forced`**: from the same forward used for loss, take `argmax` over vocab at positions where `labels != IGNORE_INDEX`, decode contiguous response tokens per sample. Cheap; not free-form generation.
   - **`generate`**: build prompt-only sequences (truncate at first non-`IGNORE_INDEX` label position), run `model.generate` with trainer `gen_kwargs` + multimodal tensors, decode new tokens only. Expensive; rely on interval/max_samples.
7. Aggregate across ranks with `accelerator.gather_object` (or equivalent) so the JSON is complete, not rank0-only.
8. Update in-memory `dict[str, dict[str, str]]` on world process zero; **atomic** rewrite of the JSON file periodically (e.g. every logged step) and on train end / stop callback so crashes don’t lose everything.

**DeepSpeed note**: free `generate` during training must use the wrapped model carefully (`self.model` vs unwrapped); mirror patterns from existing `prediction_step`. Document that `generate` mode may be fragile under some ZeRO settings; ZeRO-2 (current CoT jobs) is the target.

### Eval dump (val_size compatible)

Do **not** require `predict_with_generate=true` or a separate `eval_dataset`.

When `save_eval_predictions`:

- Override/extend `prediction_step` (or a thin `evaluate` wrapper) so each eval batch:
  1. Still computes eval **loss** as today.
  2. Additionally produces text via `eval_prediction_mode` (`teacher_forced` or `generate`).
  3. Maps each sample’s `question_id` → text into a rank-local buffer.
- After eval loop completes (or in `on_evaluate` callback), gather to rank0 and write `eval_predictions.json` once.
- If multiple evals occur in one run, later writes **overwrite** per `QUESTION_ID` (matches “one eval per run” assumption).

Avoid forcing the stock `predict_with_generate` constraints in `hparams/parser.py` (those currently reject `val_size`-only setups).

### Interaction with existing debug

- Independent of `debug_mm_training`; can be combined.
- Reuse the same side-channel style as `debug_samples` so collator/model forward stay clean.

## Dataset ops for the CoT mix (user-facing steps after code lands)

For `Scene30k`, `SpatialSSRL_coldstart`, `3DThinker10k`:

1. Run `assign_question_ids.py` on each annotation file (prefer new files, not mutating HF hub snapshots in place unless intended).
2. Point `data/dataset_info.json` `file_name` at the stamped files (or use `--update-dataset-info` if implemented).
3. Add to each dataset’s `columns`:

```json
"question_id": "question_id"
```

4. Clear/rebuild tokenized cache if needed (`overwrite_cache: true` once) so the new column is re-aligned.

## Files to touch

| Area | Files |
|------|--------|
| Flags | `src/llamafactory/hparams/finetuning_args.py` |
| Dataset attr/parser | `src/llamafactory/data/parser.py` |
| Converter | `src/llamafactory/data/converter.py` |
| Supervised processor | `src/llamafactory/data/processor/supervised.py` (+ unsupervised if needed) |
| Collator | `src/llamafactory/data/collator.py` |
| Dump helper | **new** `src/llamafactory/train/prediction_dump.py` |
| Trainer / workflow | `src/llamafactory/train/sft/trainer.py`, `workflow.py` |
| ID script | **new** `scripts/assign_question_ids.py` |
| Dataset registry | `data/dataset_info.json` (after stamps; optional in same PR) |
| Docs | short note in `data/README.md` or example YAML comments |
| Help dump | regenerate/update `.grok/skills/llamafactory-cli-train/references/llamafactory-cli_train_-h.txt` if you keep that in sync |

## Testing / verification

1. **Unit**: script assigns `Foo_0..n-1` on tiny JSON/JSONL fixtures; parquet if env has deps.
2. **Unit**: nested dict merge + atomic write; step keys as strings.
3. **Dry data path**: load one stamped sample through converter → processor → collator and assert `question_ids` present.
4. **Smoke (1 GPU, tiny max_samples)**: enable `save_train_predictions` + `teacher_forced`, `train_prediction_max_samples=4`, confirm JSON shape.
5. **Smoke eval**: `save_eval_predictions` with `val_size` split and `eval_prediction_mode=teacher_forced` (faster than generate).
6. Optional generate-mode smoke only if GPU time allows; document expected slowdown.

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Free `generate` OOM / huge slowdown on VL CoT | Default train mode `teacher_forced`; require interval + max_samples for generate |
| Multi-GPU incomplete dumps | `gather_object` all ranks before write |
| Huge JSON rewrites | Cap via `max_samples`; atomic write; consider JSONL sidecar later if needed |
| Missing `question_id` column | Clear warning; skip dump rather than invent IDs |
| `predict_with_generate` parser conflicts | Custom eval dump path; leave stock flag alone |
| Tokenized cache stale after stamping | Document one-time `overwrite_cache: true` |

## Implementation order

1. `assign_question_ids.py` + stamp CoT annotation copies (or document user run).
2. Propagate `question_id` through parser → converter → processor → collator.
3. Add finetuning flags + JSON writer helper.
4. Train dump (`teacher_forced` first, then `generate`).
5. Eval dump with `val_size` support.
6. Wire example YAML comments on a CoT config; light tests / smoke notes.
7. Update `dataset_info.json` columns once stamped files exist.

## Out of scope

- Logging gold labels or prompts into the prediction JSON (explicitly excluded by design).
- Changing training loss, LR schedule, or checkpointing.
- Full-time free generation on every microbatch without subsampling.
- Making stock `predict_with_generate` work with `val_size` (we bypass that constraint with a dedicated dump path).
