# Train prediction log analysis

This report is a **deterministic statistical summary**. It is not a judge of
reasoning quality, semantic correctness, or training success.

## Inputs

- `saves/qwen2_5vl-7b/lora/sft/CoT_traineval_resume_ep1/train_predictions_ep1.json`

- Predictions: **408**
- Datasets: 3DThinker10k, Scene30k, SpatialSSRL_coldstart
- Steps: 10 … 510 (51 distinct)
- Annotation lookup: {"success": 408, "failure": 0, "rate": 1.0, "files": {"3DThinker10k": "/project/aip-wangcs/indrisch/huggingface/hub/datasets--cvis-tmu--3dthinker-10k-mcq/snapshots/c0392e4172ddf9c106b7066c584724dd7ae04144/3dthinker10k_cot.with_question_id.jsonl", "Scene30k": "/project/aip-wangcs/indrisch/huggingface/hub/datasets--cvis-tmu--Scene30K/snapshots/4be0f2eadaf440b9fe9392fdeca790c4edfd68fd/data/train-00000-of-00001.with_question_id.parquet", "SpatialSSRL_coldstart": "/project/aip-wangcs/indrisch/huggingface/hub/datasets--cvis-tmu--Spatial-SSRL-81k/snapshots/c6bce21bad8cb7d751a47f7bb91dca7875115c96/SFT-coldstart.with_question_id.json"}}
- Grammar: disabled
- Checkpoint probe loss: disabled

## Teacher-forced dumps

The training YAML in this repo defaults to `train_prediction_mode: teacher_forced`.
Those texts are **argmax next-token decodes of response positions**, not
`model.generate` outputs. Tag / repetition / surface-match numbers still
describe the dumped strings, but they are **not** free-form generation quality.

## Sampling

Matched-question strategy: `unmatched_all_observations` (matched IDs: 0).

Mean consecutive-step question-ID Jaccard: 0.000.
Intersection of question IDs across **all** steps: 0 / union 408.

If Jaccard is ~0, step trends mix different questions. That can masquerade as learning.

No question ID appears at more than one step, so per-question trajectories are empty.

## Tag adherence

Canonical-format rate 0.0% at step 10.0 vs 62.5% at step 510.0.
Mean tag-pair score 0.25 at step 10.0 vs 0.88 at step 510.0.
Usable-format rate 0.0% at step 10.0 vs 75.0% at step 510.0.

Canonical-format rate by dataset (all steps pooled):

- `3DThinker10k`: 0.0% (n=90)
- `Scene30k`: 59.1% (n=279)
- `SpatialSSRL_coldstart`: 0.0% (n=39)

## Length

median think tokens: 0.0 at step 10.0 vs 277.0 at step 510.0.
median answer tokens: 1.0 at step 10.0 vs 11.5 at step 510.0.
median total tokens: 277.5 at step 10.0 vs 314.5 at step 510.0.
Empty-answer rate: 50.0% at step 10.0 vs 0.0% at step 510.0.

## Repetition

median repetition_score: 0.151 at step 10.0 vs 0.159 at step 510.0.
median trigram repetition fraction: 0.048 at step 10.0 vs 0.080 at step 510.0.
median adjacent-identical fraction: 0.022 at step 10.0 vs 0.019 at step 510.0.

`repetition_score` is an uncalibrated weighted mix of adjacent-token identity,
trigram distinct-n, zlib compression, and unigram concentration. Components are in the tables.

## Surface match vs reference (not semantic correctness)

Normalized exact-match rate: 37.5% at step 10.0 vs 37.5% at step 510.0.

## Combined views (read the tables/plots; no causal claims)

- Formatting up + repetition up: the model may learn tags while degenerating.
- Probe loss down + think tokens collapsing: optimization vs generation mismatch.
- Think tokens up + normalized EM flat: longer reasoning without lexical match gains.
- Per-dataset rows: one mix member can drive a global trend.

## Flags

step 110.0: canonical_format_rate_drop_25.0pp; step 120.0: canonical_format_rate_drop_25.0pp; step 140.0: canonical_format_rate_drop_25.0pp; step 170.0: canonical_format_rate_drop_37.5pp; step 230.0: canonical_format_rate_drop_37.5pp; step 260.0: median_think_tokens_x4.6; step 290.0: median_think_tokens_x3.4; step 320.0: canonical_format_rate_drop_37.5pp; step 330.0: median_think_tokens_x4.7; step 350.0: canonical_format_rate_drop_25.0pp; step 390.0: canonical_format_rate_drop_25.0pp; step 410.0: canonical_format_rate_drop_50.0pp; step 440.0: canonical_format_rate_drop_25.0pp; step 450.0: canonical_format_rate_drop_25.0pp; step 490.0: canonical_format_rate_drop_25.0pp

Warnings written: 15 (see `warnings.csv`).

