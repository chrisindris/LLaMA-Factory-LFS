You are an expert Python/ML engineer familiar with Hugging Face Transformers, Qwen2.5-VL, multimodal supervised fine-tuning, and LLaMAFactory.

Create a production-quality Python script named:

`analyze_reasoning_training_log.py`

The script is intended to **watch, diagnose, and visualize the evolution of a Qwen2.5-VL-7B-Instruct model during SFT on 3D scene-understanding reasoning datasets**.

The training mixture contains:

- `Scene30k`
- `3dthinker10k`
- `SpatialSSRL_coldstart`

These correspond conceptually to Scene30K, 3DThinker, and SpatialSSRL.

The model is being trained to become a reasoning model whose desirable output format is approximately:

```text
<think>
reasoning...
</think>
<answer>
answer...
</answer>
```

The script must perform **non-AI preliminary analysis only**.

## Hard constraint: no AI analysis

Do **not**:

- call OpenAI, Anthropic, Gemini, or any other API;
- call another LLM;
- use an LLM-as-a-judge;
- use an embedding model;
- use a classifier or learned evaluator to assess response quality;
- download or invoke another generative model merely to analyze the predictions.

Ordinary deterministic/rule-based NLP/statistical tools are fine.

In particular, `language_tool_python` may be used for grammar diagnostics.

Loading the **actual Qwen training checkpoints** is permitted **only for the explicitly optional loss-analysis functionality described below**, because this is numerical evaluation of the model being debugged rather than asking another AI system to judge its responses.

The main analysis must work without loading any model.

---

# 1. Inputs

The principal input is a JSON log of predictions generated periodically during SFT.

Its structure is:

```python
{
    "<annotation_file_path>_<question_index>": {
        "<global_training_step>": "<model_prediction>"
    },
    ...
}
```

For example:

```json
{
  "/tmp/cot_stage/annotations/SpatialSSRL_coldstart.json_1661": {
    "1200": "<think>...</think><answer>...</answer>"
  }
}
```

The outer key identifies the original annotation and question.

The inner dictionary normally contains exactly one key-value pair:

- key = global training step at which the prediction was collected;
- value = model prediction at that step.

However, write the parser defensively. If an entry unexpectedly contains multiple step/prediction pairs, process each one rather than crashing.

Training-step keys may be JSON strings and must be converted to integers.

---

# 2. Determine the source dataset

The paths stored in the prediction log may no longer exist because training may have used temporary storage.

Therefore, do **not** assume the path in the prediction-log key is readable.

Determine dataset membership from the outer key.

At minimum support these aliases:

```python
DATASET_ALIASES = {
    "Scene30k": ["Scene30k"],
    "3dthinker10k": ["3dthinker10k", "3DThinker"],
    "SpatialSSRL_coldstart": ["SpatialSSRL_coldstart", "SpatialSSRL"],
}
```

Matching should preferably be case-insensitive but deterministic.

Store a canonical dataset name for each prediction.

If no dataset can be determined:

- label it `UNKNOWN`;
- emit a warning;
- continue processing rather than crashing.

Also detect ambiguous matches and report them.

---

# 3. Determine the question index

The final integer following the last underscore in the log key is the question index.

For example:

```text
/tmp/cot_stage/annotations/SpatialSSRL_coldstart.json_1661
```

has:

```text
question_index = 1661
```

Use a robust regex such as the conceptual equivalent of:

```regex
_(\d+)$
```

Do not make assumptions about underscores appearing earlier in the path or filename.

Construct the expected dataset question ID as:

```python
f"{dataset_name}_{question_index}"
```

For example:

```text
SpatialSSRL_coldstart_1661
```

---

# 4. Recover the original annotations

Accept a command-line argument for `dataset_info.json`.

It is loaded as a dictionary `D`.

For a dataset:

```python
F = D[dataset_name]["file_name"]
```

identifies the real annotation file.

`file_name` may refer to either:

- JSON; or
- Parquet.

The script should load the annotation file and find the record whose:

```python
question_id == f"{dataset_name}_{question_index}"
```

For the example above, search for:

```text
SpatialSSRL_coldstart_1661
```

This record provides the ground truth and potentially other useful metadata.

### Be robust to annotation structure

JSON files may contain:

- a top-level list of dictionaries;
- a dictionary mapping IDs to records;
- a dictionary containing a list under another key.

Do not silently guess if the structure is genuinely ambiguous. Implement reasonable common cases and produce an informative error/warning otherwise.

Parquet should be handled with pandas/pyarrow.

### Efficiency

Do **not** reread an annotation file once per prediction.

Load each dataset annotation source once and create an index such as:

```python
question_id -> annotation_record
```

Use that index for subsequent lookups.

If the referenced question cannot be found:

- retain the prediction;
- set annotation-derived fields to missing;
- warn/count the failure;
- continue.

At the end, report lookup success/failure statistics.

---

# 5. Ground-truth field discovery

The exact field containing the desired assistant response may vary across these datasets.

Inspect common possibilities such as:

- `answer`
- `answers`
- `response`
- `output`
- `assistant`
- `conversations`
- `messages`
- other obvious LLaMAFactory-style fields

Implement the extraction in a modular function.

Do not recklessly choose an arbitrary string if several fields plausibly represent different things.

Provide CLI overrides allowing the user to explicitly specify a ground-truth field per dataset if necessary.

Preserve the complete original annotation record or selected useful metadata in an optional enriched JSONL/Parquet output.

---

# 6. Normalize everything into a dataframe

Create one row per:

```text
(question, prediction step)
```

At minimum include:

```text
dataset
question_index
question_id
log_key
step
prediction
ground_truth
annotation_found
```

Sort sensibly by:

```text
step, dataset, question_id
```

All subsequent analyses should operate on this normalized representation.

---

# 7. Reasoning-format / tag analysis

One central purpose of this tool is determining whether the model is learning the desired reasoning format.

Analyze `<think>` and `<answer>` tags carefully.

For every prediction calculate at least:

```text
has_think_open
has_think_close
has_answer_open
has_answer_close

think_open_count
think_close_count
answer_open_count
answer_close_count

has_complete_think_pair
has_complete_answer_pair
has_both_tag_pairs

think_before_answer
proper_tag_order
tags_non_overlapping
think_is_empty
answer_is_empty

text_before_think
text_between_think_and_answer
text_after_answer

think_char_count
answer_char_count
think_word_count
answer_word_count
think_token_count
answer_token_count
```

Use a deterministic tokenizer for the basic token-count analysis. If the Qwen tokenizer is locally available, optionally support it, but the ordinary analysis must not require downloading a tokenizer/model. A simple whitespace/regex tokenizer should be available as a fallback.

Define several useful summary scores rather than reducing everything to one boolean.

For example:

### `tag_presence_score`

A simple score in `[0, 1]` based only on the presence of the four expected tags:

```text
0.25 each:
<think>
</think>
<answer>
</answer>
```

### `tag_pair_score`

```text
0.5 for a complete think pair
0.5 for a complete answer pair
```

### `canonical_format`

Boolean requiring something approximately like:

1. exactly one `<think>` opening and closing tag;
2. exactly one `<answer>` opening and closing tag;
3. all tags correctly ordered;
4. no illegal nesting/overlap;
5. non-empty think content;
6. non-empty answer content.

Also report a slightly less strict `usable_format` that tolerates harmless whitespace/text formatting where appropriate.

Do **not** hide the component metrics behind the aggregate score.

We want to be able to tell *why* format adherence failed.

---

# 8. Extract reasoning and final answer

Implement robust deterministic functions such as:

```python
extract_think(prediction)
extract_answer(prediction)
```

Return missing values when extraction is impossible rather than inventing content.

If malformed tags make several interpretations possible, flag the case.

Keep:

```text
think_text
answer_text
```

as columns in the detailed output.

Do the same to the ground truth where applicable.

---

# 9. Response-length analysis

Track basic length statistics because reasoning collapse or runaway generation should be visible.

For the whole response, `<think>` section, and `<answer>` section, compute where available:

- characters;
- words;
- tokens;
- lines;
- sentences.

Analyze ratios such as:

```text
think_tokens / total_tokens
answer_tokens / total_tokens
think_tokens / max(answer_tokens, 1)
```

Flag:

- completely empty predictions;
- extremely short reasoning;
- extremely long reasoning;
- empty final answers;
- suspiciously abrupt outputs.

Avoid hardcoding universal notions of "too long" or "too short". Make thresholds configurable and also report continuous values.

---

# 10. Repetition / degeneration analysis

This is particularly important.

Poorly trained or unstable LLMs often degenerate into patterns such as:

```text
the the the the the the
```

or:

```text
left of the chair, left of the chair, left of the chair...
```

Implement several **non-AI repetition metrics**, because no single metric captures all forms of degeneration.

Calculate them separately for:

- the complete prediction;
- `<think>`;
- `<answer>`.

At minimum include the following.

## A. Consecutive identical-token repetition

For token sequence:

```python
tokens
```

measure:

```text
number of adjacent identical-token pairs
fraction of adjacent token pairs that are identical
maximum run length of an identical token
number of runs >= 3
number of runs >= 5
```

Example:

```text
over and over and over
```

may not trigger identical-token runs strongly, which is why other metrics are also required.

## B. Repeated n-grams

For at least:

```text
n = 2, 3, 4, 5
```

calculate:

```text
total ngrams
unique ngrams
repeated ngram count
repeated ngram fraction
maximum frequency of any ngram
```

Useful derived metric:

```text
1 - unique_ngrams / total_ngrams
```

when defined.

## C. Distinct-n

Calculate:

```text
distinct_1
distinct_2
distinct_3
distinct_4
```

where:

```text
distinct_n = unique ngrams / total ngrams
```

Lower values can indicate repetition.

Do not automatically interpret low distinct-n as model failure for very short answers.

## D. Token-frequency concentration

Calculate things such as:

```text
most_common_token_fraction
top_3_token_fraction
top_5_token_fraction
```

Exclude pure whitespace.

Optionally provide versions excluding common punctuation.

## E. Repeated spans

Detect repeated multi-token spans, particularly repeated contiguous phrases.

A practical implementation could identify repeated n-grams or longer repeated substrings and record:

```text
longest_repeated_token_span
most_repeated_span
most_repeated_span_count
```

Do this efficiently; avoid a naive algorithm that becomes quadratic or worse on very long generations.

## F. Compression-based repetition proxy

Optionally compute a deterministic compression ratio using a standard-library compressor such as `zlib`.

Highly repetitive text compresses unusually well.

For example report:

```text
compressed_bytes / original_utf8_bytes
```

Treat this only as an additional heuristic, not a semantic quality score.

## G. Composite degeneration indicator

It is acceptable to create an interpretable composite `repetition_score`, but:

- document the formula;
- keep every raw component;
- do not imply that this is a scientifically calibrated metric;
- make thresholds/weights configurable.

The individual metrics matter more than the composite.

---

# 11. Grammar analysis with language_tool_python

Use `language_tool_python` as an optional local/rule-based diagnostic.

Do not make grammar analysis a hard dependency of the entire script.

Support something like:

```bash
--grammar
```

When enabled, analyze separately:

- full response after stripping XML-like tags;
- think text;
- answer text.

At minimum record:

```text
grammar_issue_count
grammar_issues_per_100_words
grammar_issues_per_100_tokens
```

Also aggregate by LanguageTool category/rule where practical.

For example useful fields might include counts for:

```text
grammar
typos/spelling
punctuation
style
capitalization
```

depending on what LanguageTool exposes.

Store the most common triggered rule IDs in summary output.

### Important caveat

Grammar errors are a **diagnostic**, not a correctness metric.

Scene-understanding answers may legitimately be fragments such as:

```text
left of the sofa
```

and short final answers may receive misleading grammar warnings.

Therefore:

- analyze `<think>` and `<answer>` separately;
- report normalized issue rates;
- do not call this an "accuracy score";
- allow grammar checking to be disabled;
- allow configurable language, defaulting to English.

Avoid repeatedly starting LanguageTool for each row. Initialize it once and reuse it.

Handle LanguageTool/Java unavailability gracefully.

---

# 12. Surface-level comparison with ground truth

Where ground truth can be recovered, perform useful deterministic comparisons.

These are **not semantic judging**.

For the final `<answer>` section, calculate where possible:

```text
exact_match
case_insensitive_exact_match
normalized_exact_match
```

Define and document normalization, for example:

- Unicode normalization;
- strip leading/trailing whitespace;
- collapse internal whitespace;
- lowercase;
- optionally strip surrounding punctuation/articles as a separately named metric.

Do not silently make aggressive transformations.

Also calculate lexical quantities such as:

```text
token_precision
token_recall
token_f1
Jaccard similarity
```

if meaningful.

Optionally calculate character/token edit distance or normalized Levenshtein distance using a lightweight library or a small implementation.

These are surface diagnostics only.

Do **not** characterize them as semantic correctness.

If an annotation has multiple valid answers, support that explicitly by computing comparison against each and selecting the best surface match while retaining the fact that multiple references existed.

---

# 13. Compare prediction structure with ground-truth structure

If the ground truth itself contains `<think>` and `<answer>` sections, calculate useful structural differences such as:

```text
prediction_think_tokens - gt_think_tokens
prediction_answer_tokens - gt_answer_tokens
prediction_total_tokens - gt_total_tokens

prediction/GT reasoning-length ratio
prediction/GT answer-length ratio
```

Compare repetition and grammar diagnostics between prediction and reference as well.

This can help distinguish:

- behavior learned from the training data;
- model-specific degeneration.

Do not assume the ground truth is perfect.

---

# 14. Dataset-level analysis

Every metric must be aggregatable separately for:

- Scene30k;
- 3dthinker10k;
- SpatialSSRL_coldstart;
- all datasets combined.

This is critical because one dataset may be responsible for an observed training pathology.

For each dataset and step, calculate sensible statistics such as:

```text
N
mean
median
standard deviation
min
max
25th percentile
75th percentile
```

where meaningful.

For boolean metrics report both:

```text
count
fraction/percentage
```

Examples:

```text
canonical format rate
think-tag pair rate
answer-tag pair rate
empty-answer rate
mean think length
median think length
mean grammar issues / 100 words
median repetition metric
normalized-exact-match rate
```

---

# 15. Training-step analysis

The central visualization axis is **global training step**.

Aggregate the diagnostic metrics by:

```text
step
```

and:

```text
dataset × step
```

The script is intended to answer questions such as:

- Is `<think>`/`<answer>` adherence improving over training?
- At what step does the model start reliably emitting reasoning?
- Does reasoning length grow, stabilize, or collapse?
- Does the answer section disappear at some stage?
- Does repetition increase at later steps?
- Does grammar improve or worsen?
- Are certain datasets responsible for unusual behavior?
- Does lexical agreement with the reference improve?
- Does loss decrease while generation quality appears to degenerate?
- Does one dataset's loss improve while another dataset's loss worsens?

Make the tables and plots support these questions.

---

# 16. Important sampling consideration

The same exact questions may or may not have been sampled at every logged training step.

This can seriously confound trends.

Therefore calculate and report:

```text
number of predictions per step
number per dataset per step
number of unique question IDs per step
```

Also determine whether the evaluation question set is stable across steps.

For every pair/consecutive set of steps, compute useful overlap statistics such as:

```text
question-ID intersection size
question-ID union size
Jaccard overlap
```

Most importantly, provide an option for **matched-question analysis**:

```bash
--matched-questions
```

or equivalent.

In matched-question mode, compare steps only using questions observed at all relevant steps, or provide a clearly documented strategy for matched comparisons.

This prevents a change in question composition from masquerading as learning progress.

This feature is important.

---

# 17. Optional per-question trajectories

For questions that occur at several checkpoints/steps, create trajectories showing how their outputs evolve.

Record deltas between consecutive observations such as:

```text
tag score delta
reasoning length delta
answer length delta
repetition delta
grammar delta
surface-match delta
```

Allow selecting interesting examples such as:

- biggest improvement in tag adherence;
- biggest increase in repetition;
- largest reasoning-length explosion;
- answer becoming empty;
- answer changing repeatedly;
- largest grammar deterioration;
- largest normalized-exact-match improvement.

Export these examples to a human-readable report for manual inspection.

Do not use an AI model to pick examples.

---

# 18. OPTIONAL checkpoint-based loss analysis

Implement this as a clearly separated optional feature.

## Critical statistical/ML requirement

**Do not pretend SFT loss can be reconstructed from the prediction strings in the log.**

The generated prediction alone is insufficient.

To calculate the actual teacher-forced cross-entropy associated with a checkpoint, we need things such as:

- the corresponding model checkpoint;
- original example;
- Qwen processor/tokenizer;
- exact chat template;
- multimodal inputs;
- masking convention;
- LLaMAFactory preprocessing;
- label construction;
- potentially the same training configuration.

Therefore loss analysis should only run if the user supplies the required model/checkpoint and LLaMAFactory information.

Provide CLI options along the lines of:

```text
--compute-loss
--checkpoint-root PATH
--llamafactory-root PATH
--training-config PATH
--model-name-or-path PATH
--loss-batch-size N
--loss-max-samples-per-dataset N
```

Adapt names if a better interface is obvious.

### Checkpoint mapping

If checkpoints are stored as:

```text
checkpoint-500/
checkpoint-1000/
checkpoint-1500/
```

associate logged step `1000` with `checkpoint-1000`.

Do not silently substitute a different checkpoint.

Provide an explicit optional nearest-checkpoint policy only if useful, and clearly label such results.

---

# 19. Reuse LLaMAFactory loss/data logic where practical

Do **not** casually reimplement LLaMAFactory's SFT preprocessing and label masking.

Prefer to reuse the **installed/source-tree version corresponding to the actual training run**, because LLaMAFactory internals evolve over time.

Before implementing this integration:

1. inspect the supplied LLaMAFactory source/version;
2. identify its SFT dataset preprocessing/template path;
3. identify its multimodal data collator;
4. identify how labels and ignored tokens are constructed;
5. identify how its SFT trainer computes ordinary loss;
6. reuse those components where practical.

Do not hardcode assumptions based on a random version from the Internet.

Keep this integration isolated so that changes to LLaMAFactory internals do not break all log-only analyses.

If direct internal imports are too version-fragile, create a clean adapter layer and produce a helpful error explaining what API was expected.

### Important: what loss should mean

For SFT diagnosis, the most useful loss is generally the checkpoint's **teacher-forced loss on the reference assistant target**, using the same masking/template conventions as training.

Do **not** compute loss on the generated prediction and label that "SFT loss".

If you also support generated-output negative log-likelihood, give it a completely different and unambiguous name.

### Per-dataset loss

For each available checkpoint, evaluate examples separately for:

- Scene30k;
- 3dthinker10k;
- SpatialSSRL_coldstart.

Report:

```text
mean loss
median per-example loss
standard deviation
number of examples
```

where technically valid.

If feasible, calculate:

```text
per-example mean target-token NLL
target-token count
```

and aggregate the token-weighted loss separately from the example-weighted mean.

This distinction matters when response lengths vary substantially.

### "Training loss" versus diagnostic probe loss

Be very precise with terminology.

If evaluating recovered annotation examples after the fact, call this something like:

```text
checkpoint probe loss
per-dataset evaluation loss
reference-target loss
```

unless it is genuinely identical to the batches used by the trainer at that training step.

Do **not** call post-hoc evaluation of sampled questions the original training-step batch loss.

Document this distinction.

---

# 20. Multimodal loss failures

Because this is Qwen2.5-VL on 3D scene data, an annotation may reference:

- images;
- multiple images;
- video/frame sequences;
- depth images;
- paths that moved after training.

Loss mode must therefore gracefully handle missing visual assets.

For each skipped example record a reason such as:

```text
missing_image
missing_video
unresolvable_path
processor_error
annotation_missing
checkpoint_missing
OOM
other
```

Report counts by dataset and checkpoint.

Do not allow skipped examples to disappear silently.

The normal non-model log analysis must remain usable even if *all* original visual files are unavailable.

---

# 21. Correlation analysis

For numerical diagnostics, calculate simple descriptive correlations with training step.

At minimum support Spearman correlation for suitable metrics such as:

```text
step vs tag score
step vs think length
step vs answer length
step vs repetition
step vs grammar issue density
step vs normalized surface match
step vs loss
```

Calculate these:

- globally;
- separately per dataset.

Pearson correlation can additionally be provided where useful.

Do not overinterpret correlations as causal.

For metrics with strong grouping by question, matched-question trends are preferable.

---

# 22. Rolling/smoothed trends

Training diagnostics can be noisy.

Plots should show raw step aggregates and optionally a smoothed trend.

Support a configurable rolling window where sufficiently many distinct steps exist.

Do not smooth away the raw data; retain both.

---

# 23. Plots

Use `matplotlib` (and optionally seaborn if desired) to produce publication/debug-friendly plots.

Save plots to an output directory rather than merely displaying them.

At minimum create plots for:

1. **Tag adherence vs training step**
   - canonical format rate
   - tag-pair score
   - separate dataset lines where useful

2. **Reasoning length vs training step**
   - think token count
   - preferably median plus an indication of spread

3. **Answer length vs training step**

4. **Total response length vs training step**

5. **Repetition vs training step**
   - choose several interpretable repetition metrics
   - do not jam every metric into one unreadable graph

6. **Grammar issues vs training step**
   - when grammar mode is enabled

7. **Reference surface-match metrics vs training step**
   - when GT is available

8. **Number of samples per training step**
   - dataset breakdown

9. **Question-set overlap across steps**

10. **Per-dataset loss vs training step/checkpoint**
    - only when loss analysis is available

11. **Loss versus generation diagnostics**
    - e.g. loss vs repetition
    - loss vs tag adherence
    - only where observations can be meaningfully aligned

12. **Dataset comparison plots**
    - enough to make differences between Scene30k, 3DThinker, and SpatialSSRL immediately apparent.

Avoid deceptive plotting.

Label axes, metrics, denominators, and units clearly.

When the number of examples differs greatly across steps, make that visible.

---

# 24. Output files

Create a structured output directory, for example:

```text
analysis_output/
    detailed_predictions.parquet
    detailed_predictions.csv
    step_summary.csv
    dataset_step_summary.csv
    dataset_summary.csv
    question_overlap.csv
    question_trajectories.csv
    loss_by_checkpoint_dataset.csv
    warnings.csv
    analysis_summary.json
    report.md
    plots/
        ...
```

Not every file must exist if its analysis was disabled, but naming should be stable and documented.

### `detailed_predictions`

This should contain one row per prediction observation and all practical scalar diagnostics.

Long text fields may be kept in Parquet/JSONL if CSV becomes awkward.

### `analysis_summary.json`

Include:

- input paths;
- execution timestamp;
- dependency/version information;
- datasets found;
- steps found;
- row counts;
- annotation lookup rates;
- tag-adherence summary;
- repetition summary;
- grammar summary if enabled;
- surface-comparison summary if possible;
- loss summary if enabled;
- skipped/failure counts;
- configuration/thresholds used.

This makes experiment runs reproducible and machine-comparable.

### `report.md`

Generate a concise deterministic human-readable report summarizing notable facts without AI interpretation.

For example:

```text
Canonical-format adherence increased from 42.1% at step 500
to 94.8% at step 3000.
```

is acceptable because it is a direct statistical statement.

Avoid subjective statements such as:

```text
The model's reasoning became much smarter.
```

---

# 25. Automatically flag suspicious training behavior

Create deterministic flags for debugging.

Potential examples:

```text
missing_think_tags
missing_answer_tags
malformed_tags
empty_think
empty_answer
very_high_token_repetition
very_high_ngram_repetition
length_explosion
length_collapse
grammar_spike
loss_spike
prediction_empty
prediction_identical_to_previous_step
```

Thresholds should be:

- reasonable defaults;
- CLI-configurable;
- recorded in output metadata.

For step-level diagnostics, optionally identify changes relative to the previous logged step.

Example:

```text
canonical format rate drops by > 20 percentage points
```

or:

```text
median reasoning length grows > 3x
```

Again, call these diagnostic flags rather than proof of training failure.

---

# 26. Detect exact prediction stagnation

For questions appearing at multiple training steps, determine whether the generated prediction is unchanged.

Calculate:

```text
same_as_previous_prediction
number_of_unique_predictions_for_question
fraction_of_observations_identical_to_previous
```

Also separately compare:

```text
think_text
answer_text
```

after conservative normalization.

This may identify checkpoints producing effectively frozen outputs.

---

# 27. Basic vocabulary diagnostics

Without using any learned model, calculate useful lexical statistics such as:

```text
unique token count
type-token ratio
Hapax count/fraction
mean token length
```

Do this mainly for `<think>`.

Aggregate per step and per dataset.

Because type-token ratio depends heavily on text length, do not overinterpret it and consider using fixed-size/matched samples where possible.

---

# 28. Punctuation / malformed-generation diagnostics

Track common degeneration symptoms such as:

- excessive repeated punctuation;
- extremely long runs of the same character;
- unclosed brackets/parentheses where straightforward to identify;
- weirdly repeated XML tags;
- repeated `<think>` or `<answer>` blocks;
- text continuing after `</answer>`;
- `<answer>` occurring inside `<think>`;
- `<think>` restarting after `<answer>`.

Keep these rule-based.

---

# 29. Performance

The prediction log may become large.

Design accordingly.

Requirements:

- avoid unnecessary copies of giant strings;
- annotation files loaded once;
- use vectorized pandas aggregation where sensible;
- grammar analysis may be slow, so provide progress reporting;
- optionally cache LanguageTool results by exact text hash;
- checkpoint/model analysis must be batched;
- use `torch.inference_mode()` for loss evaluation where appropriate;
- clean up model/checkpoint GPU memory between checkpoints when necessary.

Use `tqdm` for longer operations if installed, with a fallback if not.

---

# 30. Dependencies and graceful degradation

Core functionality should preferably depend on common packages such as:

```text
python >= 3.10
pandas
numpy
matplotlib
```

Optional functionality may use:

```text
pyarrow
language_tool_python
scipy
rapidfuzz
torch
transformers
llamafactory
tqdm
```

Do not fail the entire program because an optional dependency is absent.

For example:

```text
--grammar
```

without `language_tool_python` should produce a clear installation/error message.

Loss analysis may naturally require the exact training environment.

---

# 31. CLI

Use `argparse`.

Design a clean CLI along the lines of:

```bash
python analyze_reasoning_training_log.py \
    --log predictions.json \
    --dataset-info dataset_info.json \
    --output-dir analysis_output \
    --grammar
```

Optional loss example:

```bash
python analyze_reasoning_training_log.py \
    --log predictions.json \
    --dataset-info dataset_info.json \
    --output-dir analysis_output \
    --compute-loss \
    --checkpoint-root /path/to/training/output \
    --llamafactory-root /path/to/LLaMA-Factory \
    --training-config /path/to/train.yaml
```

Support useful switches such as:

```text
--grammar
--language en-US
--matched-questions
--plots
--no-plots
--compute-loss
--loss-max-samples-per-dataset
--seed
--verbose
```

Choose sensible defaults.

Print a useful `--help`.

---

# 32. Reproducibility

Set a seed wherever sampling occurs.

Record:

- seed;
- exact CLI options;
- package versions;
- LLaMAFactory version/commit if discoverable;
- Transformers version;
- Torch version;
- model/checkpoint paths;
- analysis timestamp.

When loss samples are subsampled, save the exact question IDs that were selected so future checkpoints can use the **same probe examples**.

This is important: per-dataset loss curves should preferably use a fixed probe set across checkpoints rather than resampling examples independently at every step.

---

# 33. Validation / data-integrity checks

Before running the main analysis, validate the input.

Report:

```text
number of outer records
number of prediction observations
number of datasets recognized/unrecognized
training-step range
number of distinct steps
number of unique questions
duplicate question/step observations
malformed outer keys
malformed inner dictionaries
non-string predictions
annotation lookup failures
```

Do not silently drop malformed data.

Write issues to `warnings.csv` with enough identifying information to debug them.

---

# 34. Tests / self-check

Include either:

- a `--self-test` mode; or
- a small accompanying test module if you strongly prefer.

At minimum test:

### Dataset/key parsing

```text
/tmp/cot_stage/annotations/SpatialSSRL_coldstart.json_1661
```

must become:

```text
dataset = SpatialSSRL_coldstart
question_index = 1661
question_id = SpatialSSRL_coldstart_1661
```

### Correct tags

```text
<think>There is a chair left of the table.</think>
<answer>chair</answer>
```

must be recognized as canonical.

### Missing tag

```text
<think>reasoning</think>
chair
```

must fail answer-tag checks.

### Empty reasoning

```text
<think></think><answer>chair</answer>
```

must flag empty reasoning.

### Malformed ordering

```text
<answer>chair</answer><think>reasoning</think>
```

must fail canonical ordering.

### Token repetition

```text
chair chair chair chair chair chair
```

must have:

```text
max_identical_token_run == 6
```

under the simple tokenizer.

### Phrase repetition

```text
left of chair left of chair left of chair
```

must produce high repeated-ngram diagnostics even though identical-adjacent-token repetition is low.

### Multiple steps in one dictionary

Ensure they become multiple dataframe rows.

---

# 35. Code quality

The finished script should be maintainable rather than one giant procedural block.

Use:

- type hints;
- docstrings;
- `pathlib.Path`;
- `logging`;
- dataclasses where they genuinely help;
- small focused functions;
- clear separation between parsing, metrics, aggregation, plotting, grammar, GT lookup, and model-loss evaluation.

A reasonable conceptual structure is:

```python
parse_args()
load_prediction_log()
parse_log_key()
identify_dataset()
load_dataset_info()
load_annotation_indices()
normalize_predictions()

analyze_tags()
analyze_lengths()
analyze_repetition()
analyze_grammar()
compare_to_ground_truth()
analyze_question_overlap()
analyze_trajectories()

optional_compute_checkpoint_losses()

aggregate_by_step()
aggregate_by_dataset_step()
calculate_correlations()
generate_plots()
generate_report()
save_outputs()

main()
```

You may improve this organization.

Do not overengineer it into a large package unless there is a compelling reason.

---

# 36. Statistical interpretation

Be careful about what each metric actually establishes.

The script must never imply:

- grammaticality = reasoning correctness;
- exact lexical match = semantic correctness;
- long reasoning = good reasoning;
- short reasoning = bad reasoning;
- repetition = necessarily incorrect;
- decreasing loss = necessarily better generated reasoning;
- correlation with training step = causation.

This is a **debugging/monitoring tool**, not an automated scientific judge.

The purpose is to surface suspicious or interesting changes for subsequent human inspection.

---

# 37. Particularly useful combined views

Please implement or strongly prioritize analyses that let us catch tradeoffs.

For example:

### Formatting improves, degeneration worsens

Plot/tag:

```text
canonical_format_rate ↑
repetition_score ↑
```

A model could successfully learn the XML-like format while becoming less linguistically stable.

### Loss decreases, reasoning collapses

Compare:

```text
reference-target checkpoint loss ↓
median think tokens ↓ sharply
```

### Reasoning grows without answer improvement

Compare:

```text
think_tokens ↑
normalized exact-match ≈ flat
```

### Dataset interference

For example:

```text
Scene30k loss ↓
SpatialSSRL loss ↑
```

or:

```text
3DThinker tag adherence ↑
Scene30k tag adherence ↓
```

These cross-metric trends are particularly valuable when debugging mixed-dataset SFT.

---

# 38. Optional composite health dashboard

Create a compact table per step with fields such as:

```text
step
n
canonical_format_pct
median_think_tokens
median_answer_tokens
empty_answer_pct
repetition_metric
grammar_issues_per_100_words
normalized_exact_match_pct
Scene30k_loss
3dthinker10k_loss
SpatialSSRL_coldstart_loss
```

Only include metrics that are actually available.

This should be one of the easiest outputs to inspect quickly during training.

Do not collapse everything into a mysterious single "quality score".

---

# 39. Important distinction between observation levels

Be statistically careful about:

1. **per-prediction metrics**;
2. **per-question trajectories**;
3. **per-step aggregates**;
4. **per-dataset-per-step aggregates**.

Avoid pseudoreplication where possible.

For example, if one step logs many more examples than another, do not hide that fact behind two equally weighted points without reporting sample sizes.

For paired/matched analysis, pair using `question_id`.

---

# 40. Documentation

At the top of the script include a substantial module docstring explaining:

- purpose;
- expected log structure;
- expected dataset resolution behavior;
- output files;
- what can be run without a model;
- what requires LanguageTool;
- what requires checkpoints/LLaMAFactory;
- why generated prediction logs alone cannot provide SFT loss.

Also include usage examples.

---

# 41. Deliverable

Output the **complete contents of `analyze_reasoning_training_log.py`**, not pseudocode.

The program must be runnable.

If the exact LLaMAFactory version/API cannot be known from the information provided, do **not** fabricate an internal API.

Instead:

1. fully implement all log-only and annotation-based diagnostics;
2. isolate the checkpoint-loss functionality behind a clean adapter;
3. inspect an available local LLaMAFactory checkout if one is available;
4. implement against that version if possible;
5. otherwise make the loss adapter fail with a detailed actionable message describing what information/API must be supplied.

Do not leave the main requested analysis as TODOs simply because optional checkpoint loss is version-dependent.

At the end of your response, after the code, briefly state:

- dependencies;
- example invocation;
- output files;
- any assumptions made;
- exactly how checkpoint loss is defined if implemented.

The priority is a **trustworthy diagnostic instrument for observing reasoning-model SFT**, not merely generating a few plots.