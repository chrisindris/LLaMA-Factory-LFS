# Adjust variables to prevent OOM errors
_Exported on 11/19/2025 at 10:53:37 EST from Cursor (2.0.77)_

---

**User**

@qwen2_5vl_lora_sft_SQA3D.yaml 



Suggest adjustments to the variables seen here to prevent OOM errors. 



To see further arguments, check https://llamafactory.readthedocs.io/en/latest/advanced/arguments.html .

---

**Cursor**


Analyzing the configuration and searching for memory-saving options to prevent OOM errors.



Suggestions to prevent OOM errors:

## Critical adjustments

### 1. Reduce batch size and increase gradient accumulation
```yaml
per_device_train_batch_size: 1  # Reduce from 2 to 1
gradient_accumulation_steps: 16  # Increase from 8 to 16 (maintains effective batch size)
```

### 2. Reduce image resolution
```yaml
image_max_pixels: 32768  # Reduce from 65536 to half (or even 16384 if still OOM)
video_max_pixels: 8192   # Reduce from 16384 to half
```

### 3. Reduce sequence length
```yaml
cutoff_len: 65536  # Reduce from 131072 to half (or even 32768 if needed)
```

### 4. Enable gradient checkpointing
Add this to reduce memory during forward pass:
```yaml
# Add under model section or training section
disable_gradient_checkpointing: false  # Ensure gradient checkpointing is enabled (default)
```

### 5. Reduce dataloader workers
```yaml
dataloader_num_workers: 4  # Reduce from 16 to 4 (reduces memory overhead)
preprocessing_num_workers: 32  # Reduce from 96 to 32
```

### 6. Additional memory optimizations
```yaml
# Add these settings:
save_only_model: true  # Change from false - saves only model weights, not optimizer states
dataloader_pin_memory: false  # Disable pin memory if OOM persists
```

## Recommended configuration changes

Here's a summary of the key changes to make:

```yaml
### model
model_name_or_path: Qwen/Qwen2.5-VL-7B-Instruct
image_max_pixels: 32768  # Reduced from 65536
video_max_pixels: 8192   # Reduced from 16384
trust_remote_code: true

### dataset
cutoff_len: 65536  # Reduced from 131072
preprocessing_num_workers: 32  # Reduced from 96
dataloader_num_workers: 4  # Reduced from 16

### train
per_device_train_batch_size: 1  # Reduced from 2
gradient_accumulation_steps: 16  # Increased from 8 (maintains effective batch size of 16)

### output
save_only_model: true  # Changed from false
```

## Notes

1. Effective batch size: `per_device_train_batch_size: 1` × `gradient_accumulation_steps: 16` = 16 (same as before: 2 × 8).
2. DeepSpeed ZeRO-3 with offload is already configured, which helps.
3. If OOM persists, further reduce `image_max_pixels` to `16384` or `cutoff_len` to `32768`.
4. Monitor memory usage and adjust incrementally.

These changes should reduce memory usage while maintaining training dynamics. Start with the batch size and image resolution reductions, as they typically have the largest impact.

```yaml
per_device_train_batch_size: 1  # Reduce from 2 to 1
gradient_accumulation_steps: 16  # Increase from 8 to 16 (maintains effective batch size)
```

```yaml
image_max_pixels: 32768  # Reduce from 65536 to half (or even 16384 if still OOM)
video_max_pixels: 8192   # Reduce from 16384 to half
```

```yaml
cutoff_len: 65536  # Reduce from 131072 to half (or even 32768 if needed)
```

```yaml
# Add under model section or training section
disable_gradient_checkpointing: false  # Ensure gradient checkpointing is enabled (default)
```

```yaml
dataloader_num_workers: 4  # Reduce from 16 to 4 (reduces memory overhead)
preprocessing_num_workers: 32  # Reduce from 96 to 32
```

```yaml
# Add these settings:
save_only_model: true  # Change from false - saves only model weights, not optimizer states
dataloader_pin_memory: false  # Disable pin memory if OOM persists
```

```yaml
### model
model_name_or_path: Qwen/Qwen2.5-VL-7B-Instruct
image_max_pixels: 32768  # Reduced from 65536
video_max_pixels: 8192   # Reduced from 16384
trust_remote_code: true

### dataset
cutoff_len: 65536  # Reduced from 131072
preprocessing_num_workers: 32  # Reduced from 96
dataloader_num_workers: 4  # Reduced from 16

### train
per_device_train_batch_size: 1  # Reduced from 2
gradient_accumulation_steps: 16  # Increased from 8 (maintains effective batch size of 16)

### output
save_only_model: true  # Changed from false
```

