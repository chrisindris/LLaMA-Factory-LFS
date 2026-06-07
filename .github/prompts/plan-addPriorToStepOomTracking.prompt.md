## Plan: Add Prior-to-Step OOM Frame & Memory Tracking

The issue with the existing `mm_debug pre_forward` logs not appearing is that they are injected into `compute_loss`. HuggingFace's `Trainer` pushes the batch inputs to the GPU (via `_prepare_inputs`) inside `training_step`, which happens *before* `compute_loss`. If the Out-Of-Memory (OOM) error occurs while moving massive image tensors to the GPU or preparing the initial context, the code never reaches `compute_loss` and fails silently.

To capture the frame count and memory *just before* the payload hits the GPU, we must intercept the loop earlier by overriding `training_step`.

**Steps**
1. Modify `CustomSeq2SeqTrainer` to override the `training_step` method. 
2. Inside the override, safely parse `inputs.get("debug_samples", [])` (which the collator automatically builds due to `debug_mm_training: true`) to deduce the total number of images/video frames attempting to load.
3. Calculate current memory usage with `torch.cuda.memory_allocated()` and `torch.cuda.max_memory_allocated()`.
4. Execute `print(..., flush=True)` exclusively on rank 0 to immediately emit the requested output prior to calling `super().training_step(...)`. The buffer flush guarantees the logs write to the `.out` file before an OS/CUDA process kill.
5. Pass execution back to the default `Trainer.training_step` to continue standard processing.

**Relevant files**
- `src/llamafactory/train/sft/trainer.py` — We will add `def training_step(...)` to the `CustomSeq2SeqTrainer` class.

**Verification**
1. Initiate the job via `sbatch models/qwen3_5_4b_lora_sft_Scene30k/killarney_slurm_2nodes_qwen3_5_4b_lora_sft_Scene30k_traineval_5epochs.sh`.
2. Inspect the latest `%j.out` file.
3. Confirm that the number of frames and memory usage prints natively before each micro-batch. (Note: Since `gradient_accumulation_steps: 16` is set in the `.yaml`, the log will print 16 times per visual tick of the progress bar — perfectly capturing the exact micro-batch responsible for the OOM.)

**Decisions**
- `print(..., flush=True)` is purposefully chosen over `logger.info` to avoid any Python buffering delays that often devour terminal outputs preceding strict memory crashes.
- Intercepting at `training_step` guarantees we catch the exact frame payload *before* HuggingFace commands the GPU memory allocation.