# Eval prediction comparison

This report is a **deterministic statistical summary**. It is not a judge of
reasoning quality, semantic correctness, or training success.

## Inputs

- **base** · step 0 · eval_loss=1.6715
  - dir: `models/Qwen2.5-VL-7B-Instruct/lora/eval`
  - predictions: `models/Qwen2.5-VL-7B-Instruct/lora/eval/eval_predictions.json`
  - trainer_log: `models/Qwen2.5-VL-7B-Instruct/lora/eval/trainer_log.jsonl`
- **ep1** · step 1 · eval_loss=0.9554
  - dir: `models/qwen2_5vl-7b-lora-sft-CoT_traineval_1epochs_merged/lora/eval`
  - predictions: `models/qwen2_5vl-7b-lora-sft-CoT_traineval_1epochs_merged/lora/eval/eval_predictions.json`
  - trainer_log: `models/qwen2_5vl-7b-lora-sft-CoT_traineval_1epochs_merged/lora/eval/trainer_log.jsonl`
- **ep2** · step 2 · eval_loss=0.9134
  - dir: `models/qwen2_5vl-7b-lora-sft-CoT_traineval_2epochs_merged/lora/eval`
  - predictions: `models/qwen2_5vl-7b-lora-sft-CoT_traineval_2epochs_merged/lora/eval/eval_predictions.json`
  - trainer_log: `models/qwen2_5vl-7b-lora-sft-CoT_traineval_2epochs_merged/lora/eval/trainer_log.jsonl`
- **ep3** · step 3 · eval_loss=0.8919
  - dir: `models/qwen2_5vl-7b-lora-sft-CoT_traineval_3epochs_merged/lora/eval`
  - predictions: `models/qwen2_5vl-7b-lora-sft-CoT_traineval_3epochs_merged/lora/eval/eval_predictions.json`
  - trainer_log: `models/qwen2_5vl-7b-lora-sft-CoT_traineval_3epochs_merged/lora/eval/trainer_log.jsonl`

- Predictions: **17536**
- Datasets: 3DThinker10k, Scene30k, SpatialSSRL_coldstart
- Steps: 0 … 3 (4 distinct)
- Annotation lookup: {"success": 17536, "failure": 0, "rate": 1.0, "files": {"3DThinker10k": "/project/aip-wangcs/indrisch/huggingface/hub/datasets--cvis-tmu--3dthinker-10k-mcq/snapshots/c0392e4172ddf9c106b7066c584724dd7ae04144/3dthinker10k_cot.with_question_id.jsonl", "Scene30k": "/project/aip-wangcs/indrisch/huggingface/hub/datasets--cvis-tmu--Scene30K/snapshots/4be0f2eadaf440b9fe9392fdeca790c4edfd68fd/data/train-00000-of-00001.with_question_id.parquet", "SpatialSSRL_coldstart": "/project/aip-wangcs/indrisch/huggingface/hub/datasets--cvis-tmu--Spatial-SSRL-81k/snapshots/c6bce21bad8cb7d751a47f7bb91dca7875115c96/SFT-coldstart.with_question_id.json"}}
- Grammar: disabled
- Checkpoint probe loss: disabled

Each eval save folder is a separate run. Assigned steps keep them from being
pooled. `--matched-questions` intersects IDs across those runs.


| run | step | n | canonical | usable | norm EM | repetition | think tok | eval_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `base` | 0 | 4384 | 0.0% | 0.0% | 9.9% | 0.157 | 0.0 | 1.671 |
| `ep1` | 1 | 4384 | 44.4% | 62.3% | 48.5% | 0.154 | 237.0 | 0.955 |
| `ep2` | 2 | 4384 | 46.5% | 65.1% | 49.1% | 0.153 | 242.0 | 0.913 |
| `ep3` | 3 | 4384 | 47.6% | 66.7% | 49.6% | 0.153 | 248.0 | 0.892 |

## Sampling

Matched-question strategy: `unmatched_all_observations` (matched IDs: 0).

Mean consecutive-step question-ID Jaccard: 1.000.
Intersection of question IDs across **all** steps: 4384 / union 4384.

If Jaccard is ~0, step trends mix different questions. That can masquerade as learning.

## Tag adherence

Canonical-format rate 0.0% at step 0 vs 47.6% at step 3.
Mean tag-pair score 0.11 at step 0 vs 0.78 at step 3.
Usable-format rate 0.0% at step 0 vs 66.7% at step 3.

Canonical-format rate by dataset (all steps pooled):

- `3DThinker10k`: 0.0% (n=4028)
- `Scene30k`: 50.4% (n=12036)
- `SpatialSSRL_coldstart`: 0.0% (n=1472)

## Length

median think tokens: 0.0 at step 0 vs 248.0 at step 3.
median answer tokens: 0.0 at step 0 vs 9.0 at step 3.
median total tokens: 300.0 at step 0 vs 300.5 at step 3.
Empty-answer rate: 78.3% at step 0 vs 13.4% at step 3.

## Repetition

median repetition_score: 0.157 at step 0 vs 0.153 at step 3.
median trigram repetition fraction: 0.060 at step 0 vs 0.064 at step 3.
median adjacent-identical fraction: 0.024 at step 0 vs 0.018 at step 3.

`repetition_score` is an uncalibrated weighted mix of adjacent-token identity,
trigram distinct-n, zlib compression, and unigram concentration. Components are in the tables.

## Surface match vs reference (not semantic correctness)

Normalized exact-match rate: 9.9% at step 0 vs 49.6% at step 3.

## Combined views (read the tables/plots; no causal claims)

- Formatting up + repetition up: the model may learn tags while degenerating.
- Probe loss down + think tokens collapsing: optimization vs generation mismatch.
- Think tokens up + normalized EM flat: longer reasoning without lexical match gains.
- Per-dataset rows: one mix member can drive a global trend.

## Flags

None.

Warnings written: 13508 (see `warnings.csv`).

