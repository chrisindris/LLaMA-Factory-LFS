## Plan: Reduce Killarney OOM

Confirm the exact runtime config and DeepSpeed offload path used by the sbatch run, then reduce the biggest activation drivers (sequence length and image/video resolution), followed by LoRA rank/target trimming, validating each change with a short run until GPU headroom is stable.

**Steps**
1. Inspect the latest sbatch output to confirm the run uses the Killarney YAML and capture the effective values of `image_max_pixels`, `video_max_pixels`, `cutoff_len`, `per_device_train_batch_size`, `gradient_checkpointing`, and `deepspeed`. Confirm the OOM happens during the train step and note peak GPU usage to set a headroom target. *depends on none*
2. Validate that ZeRO-3 with CPU offload is active and CPU Adam builds cleanly on Killarney: confirm `ds_z3_offload_config.json` is loaded, and check for CPU Adam build errors or `TORCH_EXTENSIONS_DIR` misconfiguration. If offload is not active, fix that before tuning input sizes. *depends on step 1*
3. Reduce activation memory in the YAML by stepping down `cutoff_len` and multimodal pixels (one change per run): start with `cutoff_len` 16384 -> 8192; then `image_max_pixels` 16384 -> 8192 and `video_max_pixels` 4096 -> 2048; if still OOM, drop `cutoff_len` to 4096. *depends on step 1*
4. Reduce LoRA footprint if OOM persists: drop `lora_rank` 8 -> 4 and restrict `lora_target` from `all` to attention-only modules (e.g., q/k/v/o projections) to reduce optimizer states and gradients. *depends on step 3*
5. Re-run a short debug job (keep `max_samples` small) after each step to verify max GPU allocation stays below ~40 GB on every rank; then scale to the full run once headroom is consistent. *depends on steps 3-4*

**Relevant files**
- [examples/train_lora/killarney_2nodes_qwen3_5_4b_lora_sft_Scene30k_traineval_5epochs.yaml](examples/train_lora/killarney_2nodes_qwen3_5_4b_lora_sft_Scene30k_traineval_5epochs.yaml#L1-L68) — current training config for pixels, cutoff length, LoRA, eval, and DeepSpeed
- [examples/deepspeed/ds_z3_offload_config.json](examples/deepspeed/ds_z3_offload_config.json#L1-L37) — ZeRO-3 CPU offload settings used by the run
- [examples/deepspeed/ds_z3_config.json](examples/deepspeed/ds_z3_config.json#L1-L28) — fallback if offload must be disabled
- [models/qwen3_5_4b_lora_sft_Scene30k/killarney_slurm_2nodes_qwen3_5_4b_lora_sft_Scene30k_traineval_5epochs.sh](models/qwen3_5_4b_lora_sft_Scene30k/killarney_slurm_2nodes_qwen3_5_4b_lora_sft_Scene30k_traineval_5epochs.sh#L1-L40) — sbatch wrapper used for the run
- [models/qwen3_5_4b_lora_sft_Scene30k/slurm_2nodes_qwen3_5_4b_lora_sft_Scene30k_traineval_5epochs.sh](models/qwen3_5_4b_lora_sft_Scene30k/slurm_2nodes_qwen3_5_4b_lora_sft_Scene30k_traineval_5epochs.sh#L83-L93) — YAML path and CUDA allocation config
- [models/qwen3_5_4b_lora_sft_Scene30k/slurm_2nodes_qwen3_5_4b_lora_sft_Scene30k_traineval_5epochs.sh](models/qwen3_5_4b_lora_sft_Scene30k/slurm_2nodes_qwen3_5_4b_lora_sft_Scene30k_traineval_5epochs.sh#L446-L470) — CPU Adam build and dataset cache env vars
- [models/qwen2_5vl_lora_sft_SQA3D/cursor_adjust_variables_to_prevent_oom.md](models/qwen2_5vl_lora_sft_SQA3D/cursor_adjust_variables_to_prevent_oom.md#L27-L96) — repo guidance on OOM adjustments

**Verification**
1. Run a short debug sbatch (use `max_samples` and 1-2 eval steps) and confirm each rank reports stable GPU headroom (e.g., `nvidia-smi` in logs stays under ~40 GB for L40s).
2. Confirm DeepSpeed prints ZeRO-3 offload in the log and no CPU Adam build errors.
3. Scale to the full 5-epoch run once two consecutive short runs complete without OOM.

**Decisions**
- Prioritize lowering `cutoff_len` and multimodal pixel limits, then reduce LoRA rank/targets, with CPU offload allowed even if slower.
